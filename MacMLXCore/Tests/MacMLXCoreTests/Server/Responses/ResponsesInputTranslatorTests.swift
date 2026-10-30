import Foundation
import Testing

@testable import MacMLXCore

// MARK: - ResponsesInputTranslator Tests
//
// Pure, MLX-free: a Responses API request → the chat path's inputs. The
// fixtures are synthetic but keep the shape Codex 0.162.1 sends (developer
// messages first, flat function tools, a `namespace` tool, a hosted
// `web_search`, and history replayed as reasoning / function_call /
// function_call_output items).

@Suite("ResponsesInputTranslator")
struct ResponsesInputTranslatorTests {

    // MARK: Helpers

    private func request(_ body: [String: Any]) throws -> ResponsesRequest {
        let data = try JSONSerialization.data(withJSONObject: body)
        return try JSONDecoder().decode(ResponsesRequest.self, from: data)
    }

    private func translate(_ body: [String: Any]) throws -> ResponsesInputTranslator.Translation {
        try ResponsesInputTranslator.translate(request(body))
    }

    /// The 400 message a request is rejected with, or nil when it is accepted.
    private func rejection(_ body: [String: Any]) throws -> String? {
        do {
            _ = try translate(body)
            return nil
        } catch let error as ResponsesInputTranslator.TranslationError {
            return error.message
        }
    }

    private func developer(_ text: String) -> [String: Any] {
        ["type": "message", "role": "developer", "content": [["type": "input_text", "text": text]]]
    }

    private func user(_ text: String) -> [String: Any] {
        ["type": "message", "role": "user", "content": [["type": "input_text", "text": text]]]
    }

    private func function(_ name: String) -> [String: Any] {
        [
            "type": "function", "name": name, "description": "Runs \(name).", "strict": false,
            "parameters": ["type": "object", "properties": ["cmd": ["type": "string"]], "required": ["cmd"]],
        ]
    }

    /// The `function.name` of each chat-shaped spec.
    private func specNames(_ tools: [JSONValue]?) -> [String] {
        (tools ?? []).compactMap { spec in
            guard case .object(let fields) = spec, case .object(let function)? = fields["function"],
                  case .string(let name)? = function["name"] else { return nil }
            return name
        }
    }

    private var codexTools: [[String: Any]] {
        [
            function("exec_command"),
            ["type": "namespace", "name": "multi_agent_v1", "description": "Agents.", "tools": [function("wait_agent")]],
            ["type": "web_search", "external_web_access": false],
        ]
    }

    // MARK: Codex-shaped turns

    @Test
    func codexFirstTurnBuildsSystemPromptToolsAndUserMessage() throws {
        let translation = try translate([
            "model": "m", "stream": true, "store": false,
            "input": [developer("dev one"), developer("dev two"), user("Run echo")],
            "tools": codexTools, "tool_choice": "auto", "parallel_tool_calls": true,
            "reasoning": ["summary": "auto"], "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "k", "client_metadata": ["a": "b"],
        ])
        #expect(translation.systemPrompt == "dev one\n\ndev two")
        #expect(translation.messages.map(\.role) == [.user])
        #expect(translation.messages.first?.content == "Run echo")
        #expect(specNames(translation.tools) == ["exec_command", "wait_agent"])
        #expect(translation.droppedToolTypes == ["web_search"])
        #expect(translation.toolNames.entry(forExposedName: "wait_agent")?.namespace == "multi_agent_v1")
        #expect(translation.responseFormat == nil)
        #expect(!translation.disablesThinking)
    }

    /// Codex replays a reasoning item with `summary` and an `encrypted_content`
    /// another provider wrote: neither is reasoning macMLX can read.
    @Test
    func codexSecondTurnReplaysToolHistoryAndDropsForeignReasoning() throws {
        let translation = try translate([
            "model": "m",
            "input": [
                developer("dev"),
                user("Run echo"),
                ["type": "reasoning", "id": "rs_1", "summary": [["type": "summary_text", "text": "thinking"]],
                 "content": NSNull(), "encrypted_content": "opaque"],
                ["type": "function_call", "id": "fc_1", "name": "exec_command",
                 "arguments": "{\"cmd\":\"echo hi\"}", "call_id": "call_1"],
                ["type": "function_call_output", "id": "fco_1", "call_id": "call_1", "output": "Output:\nhi\n"],
            ],
            "tools": codexTools,
        ])
        #expect(translation.systemPrompt == "dev")
        #expect(translation.messages.map(\.role) == [.user, .assistant, .tool])
        let calls = try #require(translation.messages[1].toolCalls)
        #expect(calls == [ToolCallRequest(id: "call_1", name: "exec_command", arguments: ["cmd": .string("echo hi")])])
        #expect(translation.messages[1].content == "")
        #expect(translation.messages[1].reasoning == nil)
        #expect(translation.messages[2].toolCallID == "call_1")
        #expect(translation.messages[2].content == "Output:\nhi\n")
    }

    @Test
    func assistantTextAndConsecutiveCallsMergeIntoOneTurn() throws {
        let translation = try translate([
            "model": "m",
            "input": [
                user("go"),
                ["type": "message", "role": "assistant", "content": [["type": "output_text", "text": "Checking."]]],
                ["type": "function_call", "name": "exec_command", "arguments": "{}", "call_id": "a"],
                ["type": "reasoning", "summary": []],
                ["type": "function_call", "name": "exec_command", "arguments": "", "call_id": "b"],
                ["type": "function_call_output", "call_id": "a", "output": "1"],
                ["type": "function_call_output", "call_id": "b",
                 "output": [["type": "input_text", "text": "two"], ["type": "input_image", "image_url": "x"],
                            ["type": "output_text", "text": "lines"]]],
            ],
        ])
        #expect(translation.messages.map(\.role) == [.user, .assistant, .tool, .tool])
        #expect(translation.messages[1].content == "Checking.")
        #expect(translation.messages[1].toolCalls?.map(\.id) == ["a", "b"])
        // Text parts are kept; an image is replaced by a placeholder, not dropped.
        #expect(translation.messages[3].content
            == "two\n[image omitted: macMLX does not pass images in tool results]\nlines")
    }

    @Test
    func namespacedCallInHistoryMapsToTheOfferedName() throws {
        let translation = try translate([
            "model": "m",
            "input": [
                user("wait"),
                ["type": "function_call", "name": "wait_agent", "namespace": "multi_agent_v1",
                 "arguments": "{}", "call_id": "c1"],
                ["type": "function_call_output", "call_id": "c1", "output": "done"],
            ],
            "tools": codexTools,
        ])
        #expect(translation.messages[1].toolCalls?.first?.name == "wait_agent")
    }

    // MARK: System prompt and input forms

    @Test
    func instructionsComeBeforeLeadingDeveloperMessages() throws {
        let translation = try translate([
            "model": "m", "instructions": "be brief",
            "input": [developer("dev"), ["role": "system", "content": "sys"], user("hi")],
        ])
        #expect(translation.systemPrompt == "be brief\n\ndev\n\nsys")
        #expect(translation.messages.map(\.content) == ["hi"])
    }

    @Test
    func developerMessageAfterAnAssistantTurnOpensAUserTurnTheNextUserJoins() throws {
        let translation = try translate([
            "model": "m",
            "input": [user("one"), ["role": "assistant", "content": "ok"], developer("reminder"), user("two")],
        ])
        #expect(translation.systemPrompt == nil)
        #expect(translation.messages.map(\.role) == [.user, .assistant, .user])
        #expect(translation.messages[2].content == "reminder\n\ntwo")
    }

    @Test
    func developerMessageStaysAtItsPosition() throws {
        // After a user turn it is appended to that turn; after a tool output
        // it opens a user turn there, never reaching back to an earlier turn
        // (an instruction stays out of the tool response, unlike
        // `/v1/messages`' metadata turn).
        let translation = try translate([
            "model": "m",
            "input": [
                user("one"), developer("after user"),
                ["type": "function_call", "name": "exec_command", "arguments": "{}", "call_id": "c1"],
                ["type": "function_call_output", "call_id": "c1", "output": "done"],
                developer("after tool"),
            ],
        ])
        #expect(translation.systemPrompt == nil)
        try #require(translation.messages.map(\.role) == [.user, .assistant, .tool, .user])
        #expect(translation.messages[0].content == "one\n\nafter user")
        #expect(translation.messages[2].content == "done")
        #expect(translation.messages[3].content == "after tool")
    }

    @Test
    func stringInputIsOneUserMessage() throws {
        let translation = try translate(["model": "m", "input": "hello"])
        #expect(translation.messages.map(\.role) == [.user])
        #expect(translation.messages.first?.content == "hello")
        #expect(translation.tools == nil)
    }

    @Test
    func dataURLImageBecomesAnAttachment() throws {
        // A 1x1 PNG.
        let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        let translation = try translate([
            "model": "m",
            "input": [["role": "user", "content": [
                ["type": "input_text", "text": "what is this"],
                ["type": "input_image", "image_url": "data:image/png;base64,\(png)"],
            ]]],
        ])
        let message = try #require(translation.messages.first)
        #expect(message.content == "what is this")
        #expect(message.images.count == 1)
        #expect(message.images.first?.mimeType == "image/png")
    }

    // MARK: Tools

    @Test
    func collidingNamesAreQualifiedAndMapBack() throws {
        let translation = try translate([
            "model": "m",
            "input": [
                user("go"),
                ["type": "function_call", "name": "wait", "namespace": "b", "arguments": "{}", "call_id": "c1"],
                ["type": "function_call_output", "call_id": "c1", "output": "ok"],
            ],
            "tools": [
                function("wait"),
                ["type": "namespace", "name": "a", "tools": [function("wait"), function("solo")]],
                ["type": "namespace", "name": "b", "tools": [function("wait")]],
            ],
        ])
        #expect(specNames(translation.tools) == ["wait", "a__wait", "solo", "b__wait"])
        let names = translation.toolNames
        #expect(names.entry(forExposedName: "wait") == .init(namespace: nil, name: "wait", kind: .function))
        #expect(names.entry(forExposedName: "a__wait") == .init(namespace: "a", name: "wait", kind: .function))
        #expect(names.entry(forExposedName: "solo") == .init(namespace: "a", name: "solo", kind: .function))
        #expect(names.entry(forExposedName: "b__wait") == .init(namespace: "b", name: "wait", kind: .function))
        #expect(translation.messages[1].toolCalls?.first?.name == "b__wait")
    }

    @Test
    func customToolIsOfferedAsAFunctionTakingInput() throws {
        let translation = try translate([
            "model": "m",
            "input": [
                user("patch it"),
                ["type": "custom_tool_call", "name": "apply_patch", "input": "*** Begin Patch", "call_id": "p1"],
                ["type": "custom_tool_call_output", "call_id": "p1", "output": "Success"],
            ],
            "tools": [["type": "custom", "name": "apply_patch", "description": "Apply a patch.",
                       "format": ["type": "grammar", "syntax": "lark", "definition": "start: /.+/"]]],
        ])
        let spec = try #require(translation.tools?.first)
        #expect(spec == .object([
            "type": .string("function"),
            "function": .object([
                "name": .string("apply_patch"),
                "description": .string("Apply a patch.\n\nInput format (lark):\nstart: /.+/"),
                "parameters": ResponsesInputTranslator.customToolParameters,
            ]),
        ]))
        #expect(translation.toolNames.entry(forExposedName: "apply_patch")?.kind == .custom)
        #expect(translation.messages[1].toolCalls?.first?.arguments == ["input": .string("*** Begin Patch")])
        #expect(translation.messages[2].toolCallID == "p1")
    }

    @Test
    func customToolFormatDefinitionIsAppendedToTheDescription() throws {
        let grammar = "start: begin_patch hunk+ end_patch\nbegin_patch: \"*** Begin Patch\" LF"
        let translation = try translate([
            "model": "m", "input": "hi",
            "tools": [
                ["type": "custom", "name": "apply_patch", "description": "Use the `apply_patch` tool to edit files.",
                 "format": ["type": "grammar", "syntax": "lark", "definition": grammar]],
                ["type": "custom", "name": "free", "description": "No grammar.", "format": ["type": "text"]],
                ["type": "custom", "name": "bare", "format": ["type": "grammar", "definition": "x"]],
            ],
        ])
        let descriptions = (translation.tools ?? []).compactMap { spec -> String? in
            guard case .object(let fields) = spec, case .object(let function)? = fields["function"],
                  case .string(let description)? = function["description"] else { return nil }
            return description
        }
        #expect(descriptions == [
            "Use the `apply_patch` tool to edit files.\n\nInput format (lark):\n\(grammar)",
            "No grammar.",
            "Input format:\nx",
        ])
    }

    @Test
    func toolOutputPartsOtherThanTextBecomePlaceholders() throws {
        let translation = try translate([
            "model": "m",
            "input": [
                user("look"),
                ["type": "function_call", "name": "view_image", "arguments": "{}", "call_id": "v1"],
                ["type": "function_call_output", "call_id": "v1",
                 "output": [["type": "input_image", "image_url": "data:image/png;base64,AAAA"],
                            ["type": "input_file", "file_id": "f"]]],
            ],
        ])
        #expect(translation.messages[2].content
            == "[image omitted: macMLX does not pass images in tool results]\n[input_file omitted]")
    }

    @Test
    func imageOnADeveloperMessageBecomesAPlaceholder() throws {
        let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        let translation = try translate([
            "model": "m",
            "input": [
                ["role": "developer", "content": [
                    ["type": "input_text", "text": "look at this"],
                    ["type": "input_image", "image_url": "data:image/png;base64,\(png)"],
                ]],
                user("ok"),
            ],
        ])
        #expect(translation.systemPrompt == "look at this\n[image omitted from a system message]")
        #expect(translation.messages.allSatisfy { $0.images.isEmpty })
    }

    @Test
    func undecodableImageOnASystemMessageIsNotRejected() throws {
        // The image is never decoded for a system turn, so a URL the user turn
        // would reject only adds the placeholder.
        let translation = try translate([
            "model": "m",
            "input": [
                ["role": "system", "content": [
                    ["type": "input_image", "image_url": "data:application/pdf;base64,AAAA"],
                    ["type": "input_text", "text": "after"],
                    ["type": "input_image", "image_url": "https://example.com/x.png"],
                ]],
                user("ok"),
            ],
        ])
        #expect(translation.systemPrompt
            == "[image omitted from a system message]\nafter\n[image omitted from a system message]")
    }

    @Test
    func hostedToolsAreDroppedNotRejected() throws {
        let hosted = ["web_search", "web_search_preview", "file_search", "code_interpreter",
                      "computer_use_preview", "image_generation", "mcp", "local_shell", "tool_search", "brand_new"]
        let translation = try translate([
            "model": "m", "input": "hi", "tools": hosted.map { ["type": $0] },
        ])
        #expect(translation.tools == nil)
        #expect(translation.droppedToolTypes == hosted)
    }

    @Test
    func toolChoiceNoneOffersNoTools() throws {
        let none = try translate(["model": "m", "input": "hi", "tools": [function("f")], "tool_choice": "none"])
        #expect(none.tools == nil)
        let forced = try translate([
            "model": "m", "input": "hi", "tools": [function("f")], "tool_choice": ["type": "function", "name": "f"],
        ])
        #expect(specNames(forced.tools) == ["f"])
    }

    // MARK: Structured output and reasoning

    @Test
    func textFormatJSONSchemaDecodesLikeResponseFormat() throws {
        let translation = try translate([
            "model": "m", "input": "hi",
            "text": ["format": ["type": "json_schema", "name": "answer", "strict": true, "schema": [
                "type": "object", "properties": ["n": ["type": "integer"]], "required": ["n"],
                "additionalProperties": false,
            ]]],
        ])
        let expected = try ResponseFormatDecoder.decode(.object([
            "type": .string("json_schema"),
            "json_schema": .object(["schema": .object([
                "type": .string("object"), "properties": .object(["n": .object(["type": .string("integer")])]),
                "required": .array([.string("n")]), "additionalProperties": .bool(false),
            ])]),
        ]))
        #expect(translation.responseFormat == expected)
        #expect(try translate(["model": "m", "input": "hi", "text": ["format": ["type": "json_object"]]])
            .responseFormat == .jsonObject)
        #expect(try translate(["model": "m", "input": "hi", "text": ["format": ["type": "text"]]])
            .responseFormat == nil)
    }

    @Test
    func unsupportedSchemaIsRejectedWithTheDecodersMessage() throws {
        let schema: [String: Any] = ["type": "object", "properties": ["s": ["type": "string", "pattern": "a+"]]]
        let message = try rejection(["model": "m", "input": "hi",
                                     "text": ["format": ["type": "json_schema", "name": "x", "schema": schema]]])
        var expected: String?
        do {
            _ = try ResponseFormatDecoder.decode(.object([
                "type": .string("json_schema"),
                "json_schema": .object(["schema": .object([
                    "type": .string("object"),
                    "properties": .object(["s": .object(["type": .string("string"), "pattern": .string("a+")])]),
                ])]),
            ]))
        } catch let error as ResponseFormatError {
            expected = error.description
        }
        #expect(message != nil)
        #expect(message == expected)
    }

    @Test
    func structuredOutputWithToolsIsRejected() throws {
        let message = try rejection(["model": "m", "input": "hi", "tools": [function("f")],
                                     "text": ["format": ["type": "json_object"]]])
        #expect(message?.contains("`tools` is not supported together with `text.format`") == true)
        // Suppressed tools do not conflict.
        #expect(try rejection(["model": "m", "input": "hi", "tools": [function("f")], "tool_choice": "none",
                               "text": ["format": ["type": "json_object"]]]) == nil)
    }

    @Test
    func reasoningEffortNoneOrMinimalDisablesThinking() throws {
        #expect(try translate(["model": "m", "input": "hi", "reasoning": ["effort": "none"]]).disablesThinking)
        #expect(try translate(["model": "m", "input": "hi", "reasoning": ["effort": "minimal"]]).disablesThinking)
        #expect(try !translate(["model": "m", "input": "hi", "reasoning": ["effort": "high"]]).disablesThinking)
    }

    // MARK: Rejections

    @Test
    func previousResponseIDIsRejected() throws {
        let message = try rejection(["model": "m", "input": "hi", "previous_response_id": "resp_1"])
        #expect(message?.contains("`previous_response_id` is not supported") == true)
        // An explicit null is the stateless default.
        #expect(try rejection(["model": "m", "input": "hi", "previous_response_id": NSNull()]) == nil)
    }

    @Test
    func itemReferenceIsRejected() throws {
        let message = try rejection(["model": "m", "input": [["type": "item_reference", "id": "msg_1"]]])
        #expect(message?.contains("`item_reference`") == true)
    }

    @Test
    func unsupportedContentPartsAreRejected() throws {
        let file = try rejection(["model": "m", "input": [["role": "user", "content": [
            ["type": "input_file", "file_id": "file_1"],
        ]]]])
        #expect(file == "Content part type 'input_file' is not supported.")
        let audio = try rejection(["model": "m", "input": [["role": "user", "content": [
            ["type": "input_audio", "input_audio": ["data": "", "format": "wav"]],
        ]]]])
        #expect(audio == "Content part type 'input_audio' is not supported.")
    }

    @Test
    func nonDataImagesAreRejected() throws {
        let url = try rejection(["model": "m", "input": [["role": "user", "content": [
            ["type": "input_image", "image_url": "https://example.com/cat.png"],
        ]]]])
        #expect(url?.contains("Only base64 data URLs") == true)
        let fileID = try rejection(["model": "m", "input": [["role": "user", "content": [
            ["type": "input_image", "file_id": "file_1"],
        ]]]])
        #expect(fileID?.contains("`input_image.file_id` is not supported") == true)
        let undecodable = try rejection(["model": "m", "input": [["role": "user", "content": [
            ["type": "input_image", "image_url": "data:application/pdf;base64,AAAA"],
        ]]]])
        #expect(undecodable?.contains("could not be decoded") == true)
    }

    @Test
    func unknownItemTypeIsRejected() throws {
        let message = try rejection(["model": "m", "input": [["type": "web_search_call", "id": "ws_1"]]])
        #expect(message == "Unsupported input item type 'web_search_call'.")
    }

    @Test
    func missingFieldsAreRejected() throws {
        #expect(try rejection(["model": "m"]) == "`input` is required.")
        #expect(try rejection(["model": "m", "input": NSNull()]) == "`input` is required.")
        #expect(try rejection(["input": "hi"]) == "`model` is required.")
        #expect(try rejection(["model": "m", "input": "hi", "background": true])?.contains("`background`") == true)
    }

    @Test
    func brokenToolHistoryIsRejected() throws {
        let orphan = try rejection(["model": "m", "input": [
            user("hi"), ["type": "function_call_output", "call_id": "nope", "output": "x"],
        ]])
        #expect(orphan == "Tool result references unknown tool_call id 'nope'.")
        let noCallID = try rejection(["model": "m", "input": [
            user("hi"), ["type": "function_call", "name": "f", "arguments": "{}"],
        ]])
        #expect(noCallID?.contains("missing its `call_id`") == true)
        let badArguments = try rejection(["model": "m", "input": [
            user("hi"), ["type": "function_call", "name": "f", "arguments": "[1]", "call_id": "c"],
        ]])
        #expect(badArguments == "Malformed `arguments` for tool 'f': expected a JSON object string.")
    }

    // MARK: Reasoning history

    private func reasoning(
        content: [String]? = nil, summary: String? = nil, encrypted: String? = nil
    ) -> [String: Any] {
        var item: [String: Any] = ["type": "reasoning", "id": "rs_1"]
        item["content"] = content.map { $0.map { ["type": "reasoning_text", "text": $0] } } ?? NSNull()
        item["summary"] = summary.map { [["type": "summary_text", "text": $0]] } ?? []
        if let encrypted { item["encrypted_content"] = encrypted }
        return item
    }

    private func call(_ id: String) -> [String: Any] {
        ["type": "function_call", "name": "exec_command", "arguments": "{}", "call_id": id]
    }

    private func output(_ id: String) -> [String: Any] {
        ["type": "function_call_output", "call_id": id, "output": "ok"]
    }

    private func reasoningPerMessage(_ input: [[String: Any]]) throws -> [String?] {
        try translate(["model": "m", "input": input]).messages.map(\.reasoning)
    }

    @Test
    func reasoningContentAttachesToTheFollowingCall() throws {
        // Codex replays the item verbatim: summary, content and encrypted_content.
        let item = reasoning(
            content: ["Run it."], summary: "a summary", encrypted: ReasoningCarrier.encode("not used: content wins"))
        #expect(try reasoningPerMessage([user("go"), item, call("a"), output("a")]) == [nil, "Run it.", nil])
    }

    @Test
    func encryptedContentIsReadWhenThereIsNoContent() throws {
        let item = reasoning(encrypted: ReasoningCarrier.encode("From the blob."))
        #expect(try reasoningPerMessage([user("go"), item, call("a"), output("a")]) == [nil, "From the blob.", nil])
    }

    /// A provider's summary is not the model's reasoning, and another
    /// provider's encrypted content is opaque: neither is read.
    @Test
    func summaryAndForeignEncryptedContentAreNotReasoning() throws {
        let item = reasoning(summary: "a summary", encrypted: "gAAAAABo-opaque")
        #expect(try reasoningPerMessage([user("go"), item, call("a"), output("a")]) == [nil, nil, nil])
    }

    @Test
    func reasoningAttachesToAnAssistantMessageAndSurvivesTheMerge() throws {
        let messages = try translate(["model": "m", "input": [
            user("go"), reasoning(content: ["one"]),
            ["type": "message", "role": "assistant", "content": [["type": "output_text", "text": "Checking."]]],
            reasoning(content: ["two"]), call("a"), call("b"), output("a"), output("b"),
        ]]).messages
        try #require(messages.map(\.role) == [.user, .assistant, .tool, .tool])
        #expect(messages[1].content == "Checking.")
        #expect(messages[1].toolCalls?.map(\.id) == ["a", "b"])
        #expect(messages[1].reasoning == "one\n\ntwo")
    }

    @Test
    func severalItemsBeforeOneTurnJoin() throws {
        let input = [user("go"), reasoning(content: ["one", "two"]), reasoning(content: ["three"]), call("a"), output("a")]
        #expect(try reasoningPerMessage(input) == [nil, "one\n\ntwo\n\nthree", nil])
    }

    /// Reasoning belongs to the assistant turn it precedes: a user, system or
    /// tool item first drops it.
    @Test
    func reasoningBeforeANonAssistantItemIsDropped() throws {
        #expect(try reasoningPerMessage([reasoning(content: ["lost"]), user("go"), call("a"), output("a")])
            == [nil, nil, nil])
        #expect(try reasoningPerMessage([user("go"), call("a"), reasoning(content: ["lost"]), output("a"), user("more")])
            == [nil, nil, nil, nil])
        #expect(try reasoningPerMessage([user("go"), reasoning(content: ["lost"]), developer("note"), call("a"), output("a")])
            == [nil, nil, nil])
    }
}
