// Copyright © 2026 macMLX. English comments only.

/// Translates a Responses API request into the inputs of the chat path: the
/// system prompt, the `[ChatMessage]` conversation, the chat-shaped tool specs
/// the template consumes, and the decoded structured-output constraint.
///
/// Stateless by design: macMLX stores no responses, so the client always sends
/// the whole conversation as `input` items, and anything that points at a
/// stored object (`previous_response_id`, an `item_reference` item, a
/// `file_id`) is a 400. Every rejection is a `TranslationError` carrying the
/// message the handler returns with HTTP 400 — nothing is silently dropped
/// except what the Responses wire format makes optional to honor: hosted
/// tools (Codex always declares `web_search`) and `reasoning` input items
/// (the conversation model has no reasoning field yet). An image in a tool
/// output or a system message, which the template cannot render, is replaced
/// by a placeholder saying so.
///
/// Pure apart from decoding an inline image into its temporary file, which the
/// chat path does the same way.
enum ResponsesInputTranslator {

    /// Everything the handler needs to build a `GenerateRequest`.
    struct Translation: Sendable {
        let systemPrompt: String?
        let messages: [ChatMessage]
        /// Chat-shaped function specs (`{"type":"function","function":{…}}`),
        /// nil when the request offers none or `tool_choice` is `"none"`.
        let tools: [JSONValue]?
        /// Maps a name the model calls back to the declared tool.
        let toolNames: ResponsesToolNameMap
        let responseFormat: ResponseFormat?
        /// `reasoning.effort` asked for no thinking (`"none"` / `"minimal"`).
        let disablesThinking: Bool
        /// The `type` of every hosted or unknown tool that was not offered, for
        /// the handler's debug log.
        let droppedToolTypes: [String]
    }

    /// A request macMLX cannot serve as sent; `message` is the 400's text.
    struct TranslationError: Error, Equatable, Sendable {
        let message: String
    }

    /// The most images one message may carry — the chat path's cap.
    static let maxImagesPerMessage = 4

    static func translate(_ request: ResponsesRequest) throws -> Translation {
        guard let model = request.model, !model.isEmpty else {
            throw TranslationError(message: "`model` is required.")
        }
        if request.previous_response_id != nil {
            throw TranslationError(
                message: "`previous_response_id` is not supported: macMLX keeps no responses, "
                    + "so send the whole conversation in `input` instead.")
        }
        if request.background == true {
            throw TranslationError(
                message: "`background` is not supported: macMLX keeps no responses to poll.")
        }

        let (specs, toolNames, dropped) = try toolSpecs(request.tools)
        let offersTools: Bool
        if case .string("none") = request.tool_choice {
            offersTools = false
        } else {
            offersTools = !specs.isEmpty
        }

        guard let input = request.input, input != .null else {
            throw TranslationError(message: "`input` is required.")
        }
        var conversation: [ChatMessage] = []
        if let instructions = request.instructions, !instructions.isEmpty {
            conversation.append(ChatMessage(role: .system, content: instructions))
        }
        switch input {
        case .string(let text):
            conversation.append(ChatMessage(role: .user, content: text))
        case .array(let items):
            conversation.append(contentsOf: try messages(fromItems: items, toolNames: toolNames))
        default:
            throw TranslationError(message: "`input` must be a string or an array of input items.")
        }
        let (systemPrompt, messages) = MidConversationSystemFolding.fold(conversation)
        do {
            try validateToolCallPairing(messages)
        } catch let error as ToolHistoryDecodeError {
            throw TranslationError(message: error.message)
        }

        let responseFormat = try decodeResponseFormat(request.text)
        if responseFormat != nil, messages.contains(where: { !$0.images.isEmpty }) {
            throw TranslationError(
                message: "`text.format` (structured output) is not supported together "
                    + "with image inputs (vision) in this version.")
        }
        if responseFormat != nil, offersTools {
            throw TranslationError(
                message: "`tools` is not supported together with `text.format` "
                    + "(structured output) in this version.")
        }

        return Translation(
            systemPrompt: systemPrompt,
            messages: messages,
            tools: offersTools ? specs : nil,
            toolNames: toolNames,
            responseFormat: responseFormat,
            disablesThinking: disablesThinking(request.reasoning),
            droppedToolTypes: dropped
        )
    }

    // MARK: - Tools

    /// One declared tool before exposed names are known.
    private struct DeclaredTool {
        let entry: ResponsesToolNameMap.Entry
        let description: String
        let parameters: JSONValue?
    }

    /// The chat-shaped specs for every function, custom and namespaced tool,
    /// the name map, and the types that were dropped.
    private static func toolSpecs(
        _ tools: [JSONValue]?
    ) throws -> (specs: [JSONValue], names: ResponsesToolNameMap, dropped: [String]) {
        guard let tools else { return ([], .empty, []) }
        var declared: [DeclaredTool] = []
        var dropped: [String] = []
        for (index, tool) in tools.enumerated() {
            guard case .object(let fields) = tool else {
                throw TranslationError(message: "`tools[\(index)]` must be an object.")
            }
            let type = string(fields["type"]) ?? "unknown"
            if type == "namespace" {
                guard let namespace = string(fields["name"]), !namespace.isEmpty else {
                    throw TranslationError(message: "The namespace tool `tools[\(index)]` is missing its `name`.")
                }
                guard case .array(let nested)? = fields["tools"] else {
                    throw TranslationError(message: "The namespace tool '\(namespace)' is missing its `tools` array.")
                }
                for (nestedIndex, nestedTool) in nested.enumerated() {
                    guard case .object(let nestedFields) = nestedTool else {
                        throw TranslationError(
                            message: "`tools[\(index)].tools[\(nestedIndex)]` must be an object.")
                    }
                    if let tool = try declaredTool(nestedFields, namespace: namespace) {
                        declared.append(tool)
                    } else {
                        dropped.append("\(namespace).\(string(nestedFields["type"]) ?? "unknown")")
                    }
                }
            } else if let tool = try declaredTool(fields, namespace: nil) {
                declared.append(tool)
            } else {
                dropped.append(type)
            }
        }

        let names = ResponsesToolNameMap(declared.map(\.entry))
        let specs: [JSONValue] = declared.map { tool in
            let exposed = names.exposedName(namespace: tool.entry.namespace, name: tool.entry.name)
                ?? tool.entry.name
            return .object([
                "type": .string("function"),
                "function": .object([
                    "name": .string(exposed),
                    "description": .string(tool.description),
                    "parameters": tool.parameters ?? .object([:]),
                ]),
            ])
        }
        return (specs, names, dropped)
    }

    /// A `function` or `custom` tool; nil for any other (hosted) type.
    private static func declaredTool(_ fields: [String: JSONValue], namespace: String?) throws -> DeclaredTool? {
        let kind: ResponsesToolNameMap.Kind
        switch string(fields["type"]) {
        case "function": kind = .function
        case "custom": kind = .custom
        default: return nil
        }
        guard let name = string(fields["name"]), !name.isEmpty else {
            let type = kind == .function ? "function" : "custom"
            throw TranslationError(message: "A `\(type)` tool is missing its `name`.")
        }
        let description = string(fields["description"]) ?? ""
        switch kind {
        case .function:
            return DeclaredTool(
                entry: .init(namespace: namespace, name: name, kind: .function),
                description: description,
                parameters: fields["parameters"].flatMap { $0 == .null ? nil : $0 })
        case .custom:
            // A freeform tool takes one raw string. Offered as a function with a
            // single required `input` argument; the call is turned back into a
            // `custom_tool_call` whose `input` is that argument. The template
            // has no slot for the tool's input grammar, so its definition is
            // given to the model in the description instead.
            return DeclaredTool(
                entry: .init(namespace: namespace, name: name, kind: .custom),
                description: customToolDescription(description, format: fields["format"]),
                parameters: customToolParameters)
        }
    }

    /// A custom tool's description, followed after a blank line by its input
    /// format when `format` carries a `definition` (a grammar, such as the
    /// Lark grammar of Codex's `apply_patch`).
    static func customToolDescription(_ description: String, format: JSONValue?) -> String {
        guard case .object(let fields)? = format, let definition = string(fields["definition"]) else {
            return description
        }
        let heading = string(fields["syntax"]).map { "Input format (\($0)):" } ?? "Input format:"
        let block = "\(heading)\n\(definition)"
        return description.isEmpty ? block : "\(description)\n\n\(block)"
    }

    /// What a tool result's image part becomes: the template renders tool
    /// results as text, so the model is told an image was there rather than
    /// seeing an empty result it might retry forever.
    static let omittedToolImage = "[image omitted: macMLX does not pass images in tool results]"

    /// The parameters a custom tool is offered with.
    static let customToolParameters: JSONValue = .object([
        "type": .string("object"),
        "properties": .object([
            "input": .object([
                "type": .string("string"),
                "description": .string("The raw input for the tool, in the form its description asks for."),
            ]),
        ]),
        "required": .array([.string("input")]),
    ])

    // MARK: - Input items

    private static func messages(
        fromItems items: [JSONValue],
        toolNames: ResponsesToolNameMap
    ) throws -> [ChatMessage] {
        var out: [ChatMessage] = []
        for (index, item) in items.enumerated() {
            guard case .object(let fields) = item else {
                throw TranslationError(message: "`input[\(index)]` must be an object.")
            }
            let type = string(fields["type"]) ?? (fields["role"] != nil ? "message" : nil)
            switch type {
            case "message":
                out.append(try message(fields))
            case "function_call":
                appendToolCall(try functionCall(fields, toolNames: toolNames), to: &out)
            case "custom_tool_call":
                appendToolCall(try customToolCall(fields, toolNames: toolNames), to: &out)
            case "function_call_output", "custom_tool_call_output":
                out.append(try toolOutput(fields, type: type ?? ""))
            case "reasoning":
                // Not carried across turns yet: `ChatMessage` has no reasoning
                // field, and the template re-renders history without it.
                continue
            case "item_reference":
                throw TranslationError(
                    message: "`item_reference` input items are not supported: macMLX keeps no "
                        + "responses, so send the referenced items themselves.")
            case .some(let other):
                throw TranslationError(message: "Unsupported input item type '\(other)'.")
            case .none:
                throw TranslationError(message: "`input[\(index)]` has neither a `type` nor a `role`.")
            }
        }
        return out
    }

    /// A `message` item: roles `user`, `assistant`, and `system`/`developer`
    /// (both become `.system`, placed by `MidConversationSystemFolding`). An
    /// image on a system turn is never decoded: the system prompt is text, so
    /// each one becomes the placeholder in place, whatever its URL holds.
    private static func message(_ fields: [String: JSONValue]) throws -> ChatMessage {
        let role: MessageRole
        switch string(fields["role"]) {
        case "user": role = .user
        case "assistant": role = .assistant
        case "system", "developer": role = .system
        case .some(let other):
            throw TranslationError(message: "Unsupported message role '\(other)'.")
        case .none:
            throw TranslationError(message: "A `message` input item is missing its `role`.")
        }
        var texts: [String] = []
        var images: [ImageAttachment] = []
        switch fields["content"] {
        case .none, .some(.null):
            break
        case .some(.string(let text)):
            texts.append(text)
        case .some(.array(let parts)):
            for part in parts {
                guard case .object(let partFields) = part else {
                    throw TranslationError(message: "A message content part must be an object.")
                }
                switch string(partFields["type"]) {
                case "input_text", "output_text", "text":
                    texts.append(string(partFields["text"]) ?? "")
                case "refusal":
                    texts.append(string(partFields["refusal"]) ?? "")
                case "input_image" where role == .system:
                    texts.append(MidConversationSystemFolding.omittedImage)
                case "input_image":
                    images.append(try image(partFields))
                case .some(let other):
                    throw TranslationError(message: "Content part type '\(other)' is not supported.")
                case .none:
                    throw TranslationError(message: "A message content part is missing its `type`.")
                }
            }
        case .some:
            throw TranslationError(message: "A message's `content` must be a string or an array of parts.")
        }
        guard images.count <= maxImagesPerMessage else {
            throw TranslationError(message: "At most \(maxImagesPerMessage) images per message are supported.")
        }
        return ChatMessage(role: role, content: texts.joined(separator: "\n"), images: images)
    }

    /// An `input_image` part. Only an inline base64 data URL can be served:
    /// macMLX fetches no URLs and stores no files.
    private static func image(_ fields: [String: JSONValue]) throws -> ImageAttachment {
        if fields["file_id"].map({ $0 != .null }) == true {
            throw TranslationError(
                message: "`input_image.file_id` is not supported: send the image as a base64 data URL in `image_url`.")
        }
        guard let url = string(fields["image_url"]) else {
            throw TranslationError(message: "An `input_image` part is missing its `image_url`.")
        }
        guard url.hasPrefix("data:") else {
            throw TranslationError(
                message: "Only base64 data URLs are supported for `input_image.image_url`; macMLX fetches no URLs.")
        }
        guard let attachment = MultimodalContent.decodeDataURL(url) else {
            throw TranslationError(
                message: "The `input_image` data URL could not be decoded: expected a base64 JPEG, PNG, "
                    + "WebP, GIF, HEIC or BMP image of at most 10 MB.")
        }
        return attachment
    }

    /// A `function_call` item, its name mapped back to the one the model was
    /// offered (a namespaced call carries the bare name plus `namespace`).
    private static func functionCall(
        _ fields: [String: JSONValue],
        toolNames: ResponsesToolNameMap
    ) throws -> ToolCallRequest {
        let (callID, name) = try callIdentity(fields, type: "function_call")
        let arguments: [String: JSONValue]
        switch fields["arguments"] {
        case .none, .some(.null):
            arguments = [:]
        case .some(.string(let raw)):
            do {
                arguments = try decodeToolArgumentsJSON(raw, tool: name)
            } catch let error as ToolHistoryDecodeError {
                throw TranslationError(message: error.message)
            }
        case .some:
            throw TranslationError(message: "The `arguments` of function call '\(name)' must be a JSON string.")
        }
        let exposed = toolNames.exposedName(namespace: string(fields["namespace"]), name: name) ?? name
        return ToolCallRequest(id: callID, name: exposed, arguments: arguments)
    }

    /// A `custom_tool_call` item: its raw `input` becomes the single `input`
    /// argument the tool was offered with.
    private static func customToolCall(
        _ fields: [String: JSONValue],
        toolNames: ResponsesToolNameMap
    ) throws -> ToolCallRequest {
        let (callID, name) = try callIdentity(fields, type: "custom_tool_call")
        let input = string(fields["input"]) ?? ""
        let exposed = toolNames.exposedName(namespace: string(fields["namespace"]), name: name) ?? name
        return ToolCallRequest(id: callID, name: exposed, arguments: ["input": .string(input)])
    }

    private static func callIdentity(
        _ fields: [String: JSONValue],
        type: String
    ) throws -> (callID: String, name: String) {
        guard let name = string(fields["name"]), !name.isEmpty else {
            throw TranslationError(message: "A `\(type)` input item is missing its `name`.")
        }
        // The call id is what the following output pairs on, so a synthesised
        // one would mispair: it is required, unlike the chat path's lenient id.
        guard let callID = string(fields["call_id"]), !callID.isEmpty else {
            throw TranslationError(message: "The `\(type)` input item for '\(name)' is missing its `call_id`.")
        }
        return (callID, name)
    }

    /// Consecutive calls, and an assistant text message directly before them,
    /// form one assistant turn — the shape `validateToolCallPairing` and the
    /// chat templates expect.
    private static func appendToolCall(_ call: ToolCallRequest, to out: inout [ChatMessage]) {
        guard let last = out.last, last.role == .assistant else {
            out.append(ChatMessage(role: .assistant, content: "", toolCalls: [call]))
            return
        }
        out[out.count - 1] = ChatMessage(
            id: last.id,
            role: .assistant,
            content: last.content,
            images: last.images,
            toolCalls: (last.toolCalls ?? []) + [call]
        )
    }

    /// A `function_call_output` / `custom_tool_call_output` item → a `.tool`
    /// message. `output` is a string, or parts joined line by line: a text part
    /// as its text, any other part as a placeholder naming what was left out.
    private static func toolOutput(_ fields: [String: JSONValue], type: String) throws -> ChatMessage {
        guard let callID = string(fields["call_id"]), !callID.isEmpty else {
            throw TranslationError(message: "A `\(type)` input item is missing its `call_id`.")
        }
        let content: String
        switch fields["output"] {
        case .none, .some(.null):
            content = ""
        case .some(.string(let text)):
            content = text
        case .some(.array(let parts)):
            content = parts.map { part -> String in
                guard case .object(let partFields) = part else { return "[part omitted]" }
                switch string(partFields["type"]) {
                case "input_text", "output_text", "text": return string(partFields["text"]) ?? ""
                case "input_image": return omittedToolImage
                case .some(let other): return "[\(other) omitted]"
                case .none: return "[part omitted]"
                }
            }.joined(separator: "\n")
        case .some:
            throw TranslationError(message: "The `output` of a `\(type)` item must be a string or an array of parts.")
        }
        return ChatMessage(role: .tool, content: content, toolCallID: callID)
    }

    // MARK: - Structured output and reasoning

    /// `text.format` rebuilt as the chat path's `response_format`, so the same
    /// decoder validates it and reports the same errors. The Responses
    /// `json_schema` format is flat (`{type,name,schema,strict}`); the chat one
    /// nests those under `json_schema`. Nil when no format is set.
    static func chatResponseFormat(fromText text: JSONValue?) throws -> JSONValue? {
        switch text {
        case .none, .some(.null):
            return nil
        case .some(.object(let fields)):
            guard let format = fields["format"], format != .null else { return nil }
            guard case .object(let formatFields) = format,
                  formatFields["type"] == .string("json_schema")
            else { return format }
            let envelope = formatFields.filter { ["name", "schema", "strict", "description"].contains($0.key) }
            return .object(["type": .string("json_schema"), "json_schema": .object(envelope)])
        case .some:
            throw TranslationError(message: "`text` must be an object.")
        }
    }

    private static func decodeResponseFormat(_ text: JSONValue?) throws -> ResponseFormat? {
        do {
            return try ResponseFormatDecoder.decode(try chatResponseFormat(fromText: text))
        } catch let error as ResponseFormatError {
            throw TranslationError(message: error.description)
        }
    }

    private static func disablesThinking(_ reasoning: JSONValue?) -> Bool {
        guard case .object(let fields)? = reasoning else { return false }
        switch string(fields["effort"]) {
        case "none", "minimal": return true
        default: return false
        }
    }

    private static func string(_ value: JSONValue?) -> String? {
        guard case .string(let text)? = value else { return nil }
        return text
    }
}
