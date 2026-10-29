// Copyright © 2026 macMLX. English comments only.

import Foundation

/// Builds the JSON of Responses API output items, usage and the response
/// object, as the `[String: Any]` dictionaries the server serializes with
/// `JSONSerialization`. Pure: every id, timestamp and echoed field is passed in,
/// so the shapes are unit-testable without a server.
enum ResponsesOutputBuilder {

    // MARK: - Ids

    /// A fresh id: `prefix` (`resp`, `rs`, `msg`, `fc`, `ctc`), an underscore
    /// and 32 random hex digits.
    static func newID(_ prefix: String) -> String {
        let hex = UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
        return "\(prefix)_\(hex)"
    }

    // MARK: - Items

    /// A `reasoning` item. The reasoning is the model's raw chain of thought,
    /// so it goes into `content` as `reasoning_text`. `summary` is empty:
    /// macMLX writes no summaries. An empty `text` gives empty `content`
    /// (the shape `response.output_item.added` carries).
    static func reasoningItem(id: String, text: String?) -> [String: Any] {
        [
            "type": "reasoning",
            "id": id,
            "summary": [] as [[String: Any]],
            "content": text.map { [reasoningTextPart($0)] } ?? [],
        ]
    }

    static func reasoningTextPart(_ text: String) -> [String: Any] {
        ["type": "reasoning_text", "text": text]
    }

    /// The assistant's visible text as a `message` item. A nil `text` gives
    /// empty `content` and `in_progress` (the shape `output_item.added` carries).
    static func messageItem(id: String, text: String?) -> [String: Any] {
        [
            "type": "message",
            "id": id,
            "status": text == nil ? "in_progress" : "completed",
            "role": "assistant",
            "content": text.map { [outputTextPart($0)] } ?? [],
        ]
    }

    static func outputTextPart(_ text: String) -> [String: Any] {
        ["type": "output_text", "text": text, "annotations": [] as [Any]]
    }

    /// One tool call as a `function_call` item, or a `custom_tool_call` when
    /// the model called a declared freeform tool. The name is the declared one
    /// (`namespace` added when the tool lives in one), not the name the model
    /// was offered. `done == false` gives the `output_item.added` shape: empty
    /// arguments/input and `in_progress`.
    static func toolCallItem(
        id: String,
        call: ToolCallRequest,
        toolNames: ResponsesToolNameMap,
        done: Bool
    ) -> [String: Any] {
        let entry = toolNames.entry(forExposedName: call.name)
        let name = entry?.name ?? call.name
        var item: [String: Any] = [
            "id": id,
            "call_id": call.id,
            "name": name,
            "status": done ? "completed" : "in_progress",
        ]
        if entry?.kind == .custom {
            item["type"] = "custom_tool_call"
            item["input"] = done ? customToolInput(call) : ""
        } else {
            item["type"] = "function_call"
            item["arguments"] = done ? openAIToolArgumentsJSON(call.arguments) : ""
        }
        if let namespace = entry?.namespace {
            item["namespace"] = namespace
        }
        return item
    }

    /// `call` with a call id: the engine always sets one, but an empty id
    /// would leave the client nothing to pair its output with, so one is
    /// minted (`call_<uuid>`, the engine's own fallback shape).
    static func withCallID(_ call: ToolCallRequest) -> ToolCallRequest {
        guard call.id.isEmpty else { return call }
        return ToolCallRequest(id: "call_\(UUID().uuidString)", name: call.name, arguments: call.arguments)
    }

    /// Whether `call` comes back as a `custom_tool_call`.
    static func isCustomToolCall(_ call: ToolCallRequest, toolNames: ResponsesToolNameMap) -> Bool {
        toolNames.entry(forExposedName: call.name)?.kind == .custom
    }

    /// The raw input of a custom tool call: its `input` argument, or — if the
    /// model sent something else — the whole arguments object as JSON.
    static func customToolInput(_ call: ToolCallRequest) -> String {
        if case .string(let input)? = call.arguments["input"], call.arguments.count == 1 {
            return input
        }
        return openAIToolArgumentsJSON(call.arguments)
    }

    /// The JSON-string arguments of a function call, or a custom call's input.
    static func toolCallPayload(_ call: ToolCallRequest, toolNames: ResponsesToolNameMap) -> String {
        isCustomToolCall(call, toolNames: toolNames) ? customToolInput(call) : openAIToolArgumentsJSON(call.arguments)
    }

    // MARK: - Status and usage

    /// How a generation ended, in Responses terms.
    enum Outcome: Equatable, Sendable {
        /// The engine finished with this reason (nil: the stream ended
        /// without a terminal chunk, which reads as a normal stop).
        case finished(FinishReason?)
        /// The generation failed: a thrown error or a stall.
        case failed(String)
    }

    /// `status`, `incomplete_details` and `error` for an outcome: a stop or a
    /// tool call completes, the length cap is `incomplete` with reason
    /// `max_output_tokens`, an engine error or failure is `failed` with a
    /// `server_error`.
    static func statusFields(_ outcome: Outcome) -> (status: String, incompleteDetails: Any, error: Any) {
        switch outcome {
        case .finished(.length):
            return ("incomplete", ["reason": "max_output_tokens"] as [String: Any], NSNull())
        case .finished(.error):
            return ("failed", NSNull(), errorObject("The model failed to generate a response."))
        case .finished(.stop), .finished(.toolCalls), .finished(.none):
            return ("completed", NSNull(), NSNull())
        case .failed(let message):
            return ("failed", NSNull(), errorObject(message))
        }
    }

    private static func errorObject(_ message: String) -> [String: Any] {
        ["code": "server_error", "message": message]
    }

    /// The Responses `usage` object. `input_tokens` includes the cached
    /// prompt tokens, as OpenAI's does; `cached_tokens` is 0 when the engine
    /// reported no figure or the caller may not learn it (see
    /// `HummingbirdServer.exposeCacheFigures(_:)`). macMLX does not count
    /// reasoning tokens separately, so `reasoning_tokens` is 0. A nil usage
    /// gives zeros.
    static func usageObject(_ usage: TokenUsage?, exposeCacheFigures: Bool) -> [String: Any] {
        let input = usage?.promptTokens ?? 0
        let output = usage?.completionTokens ?? 0
        let cached = exposeCacheFigures ? min(max(usage?.cachedPromptTokens ?? 0, 0), input) : 0
        return [
            "input_tokens": input,
            "input_tokens_details": ["cached_tokens": cached] as [String: Any],
            "output_tokens": output,
            "output_tokens_details": ["reasoning_tokens": 0] as [String: Any],
            "total_tokens": input + output,
        ]
    }

    // MARK: - Response object

    /// The request-derived fields every response object of one request
    /// carries, fixed when the request is accepted.
    struct Envelope: Sendable {
        let id: String
        let createdAt: Int
        let model: String
        /// Echoed request fields, defaults filled in.
        let echo: [String: JSONValue]

        init(id: String, createdAt: Int, request: ResponsesRequest, parameters: GenerationParameters) {
            self.id = id
            self.createdAt = createdAt
            self.model = request.model ?? ""
            self.echo = [
                "instructions": request.instructions.map { .string($0) } ?? .null,
                "tools": .array(request.tools ?? []),
                "tool_choice": request.tool_choice ?? .string("auto"),
                "parallel_tool_calls": .bool(request.parallel_tool_calls ?? true),
                "temperature": .double(parameters.temperature),
                "top_p": .double(parameters.topP),
                "max_output_tokens": .int(parameters.maxTokens),
                "text": request.text ?? .object(["format": .object(["type": .string("text")])]),
                "reasoning": request.reasoning ?? .object(["effort": .null, "summary": .null]),
                "metadata": request.metadata ?? .object([:]),
            ]
        }
    }

    /// The response object. `usage` is nil until the response is terminal
    /// (`response.created` / `in_progress` carry `null`).
    static func responseObject(
        _ envelope: Envelope,
        status: String,
        output: [[String: Any]],
        usage: [String: Any]?,
        incompleteDetails: Any = NSNull(),
        error: Any = NSNull()
    ) -> [String: Any] {
        var object: [String: Any] = envelope.echo.mapValues { $0.toSendable() }
        object["id"] = envelope.id
        object["object"] = "response"
        object["created_at"] = envelope.createdAt
        object["status"] = status
        object["model"] = envelope.model
        object["output"] = output
        object["usage"] = usage ?? NSNull()
        object["error"] = error
        object["incomplete_details"] = incompleteDetails
        object["store"] = false
        object["previous_response_id"] = NSNull()
        return object
    }
}
