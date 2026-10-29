import Foundation
import Testing

@testable import MacMLXCore

// MARK: - ResponsesOutputBuilder Tests
//
// Pure, MLX-free: the shapes of Responses output items, status mapping, usage
// and the response object.

@Suite("ResponsesOutputBuilder")
struct ResponsesOutputBuilderTests {

    private let namespaced = ResponsesToolNameMap([
        .init(namespace: "multi_agent_v1", name: "wait_agent", kind: .function),
        .init(namespace: nil, name: "apply_patch", kind: .custom),
        .init(namespace: nil, name: "exec_command", kind: .function),
    ])

    @Test
    func idsCarryTheirPrefixAndAreUnique() {
        let first = ResponsesOutputBuilder.newID("resp")
        let second = ResponsesOutputBuilder.newID("resp")
        #expect(first.hasPrefix("resp_"))
        #expect(first.count == "resp_".count + 32)
        #expect(first.dropFirst("resp_".count).allSatisfy { $0.isHexDigit })
        #expect(first != second)
    }

    @Test
    func functionCallItemCarriesItsNamespaceAndJSONArguments() throws {
        let call = ToolCallRequest(id: "call_1", name: "wait_agent", arguments: ["id": .int(3)])
        let item = ResponsesOutputBuilder.toolCallItem(id: "fc_1", call: call, toolNames: namespaced, done: true)
        #expect(item["type"] as? String == "function_call")
        #expect(item["id"] as? String == "fc_1")
        #expect(item["call_id"] as? String == "call_1")
        #expect(item["name"] as? String == "wait_agent")
        #expect(item["namespace"] as? String == "multi_agent_v1")
        #expect(item["arguments"] as? String == "{\"id\":3}")
        #expect(item["status"] as? String == "completed")

        let added = ResponsesOutputBuilder.toolCallItem(id: "fc_1", call: call, toolNames: namespaced, done: false)
        #expect(added["arguments"] as? String == "")
        #expect(added["status"] as? String == "in_progress")
    }

    @Test
    func topLevelFunctionCallHasNoNamespace() {
        let call = ToolCallRequest(id: "call_1", name: "exec_command", arguments: [:])
        let item = ResponsesOutputBuilder.toolCallItem(id: "fc_1", call: call, toolNames: namespaced, done: true)
        #expect(item["namespace"] == nil)
        #expect(item["arguments"] as? String == "{}")
    }

    @Test
    func qualifiedNameMapsBackToTheDeclaredName() {
        let names = ResponsesToolNameMap([
            .init(namespace: nil, name: "wait", kind: .function),
            .init(namespace: "a", name: "wait", kind: .function),
        ])
        let call = ToolCallRequest(id: "c", name: "a__wait", arguments: [:])
        let item = ResponsesOutputBuilder.toolCallItem(id: "fc_1", call: call, toolNames: names, done: true)
        #expect(item["name"] as? String == "wait")
        #expect(item["namespace"] as? String == "a")
    }

    @Test
    func customToolCallCarriesTheRawInput() {
        let call = ToolCallRequest(id: "call_p", name: "apply_patch", arguments: ["input": .string("*** Begin Patch")])
        #expect(ResponsesOutputBuilder.isCustomToolCall(call, toolNames: namespaced))
        let item = ResponsesOutputBuilder.toolCallItem(id: "ctc_1", call: call, toolNames: namespaced, done: true)
        #expect(item["type"] as? String == "custom_tool_call")
        #expect(item["input"] as? String == "*** Begin Patch")
        #expect(item["arguments"] == nil)
        // Anything but a lone string `input` falls back to the arguments JSON.
        let odd = ToolCallRequest(id: "c", name: "apply_patch", arguments: ["patch": .string("x")])
        #expect(ResponsesOutputBuilder.customToolInput(odd) == "{\"patch\":\"x\"}")
    }

    @Test
    func emptyCallIDIsMinted() {
        let minted = ResponsesOutputBuilder.withCallID(ToolCallRequest(id: "", name: "f", arguments: [:]))
        #expect(minted.id.hasPrefix("call_"))
        let kept = ResponsesOutputBuilder.withCallID(ToolCallRequest(id: "call_x", name: "f", arguments: [:]))
        #expect(kept.id == "call_x")
    }

    @Test
    func messageAndReasoningItems() throws {
        let message = ResponsesOutputBuilder.messageItem(id: "msg_1", text: "hi")
        #expect(message["role"] as? String == "assistant")
        #expect(message["status"] as? String == "completed")
        let content = try #require(message["content"] as? [[String: Any]])
        #expect(content.first?["type"] as? String == "output_text")
        #expect(content.first?["text"] as? String == "hi")
        #expect((content.first?["annotations"] as? [Any])?.isEmpty == true)
        #expect((ResponsesOutputBuilder.messageItem(id: "msg_1", text: nil)["content"] as? [Any])?.isEmpty == true)

        let reasoning = ResponsesOutputBuilder.reasoningItem(id: "rs_1", text: "think")
        #expect(reasoning["type"] as? String == "reasoning")
        #expect((reasoning["summary"] as? [Any])?.isEmpty == true)
        let parts = try #require(reasoning["content"] as? [[String: Any]])
        #expect(parts.first?["type"] as? String == "reasoning_text")
        #expect(parts.first?["text"] as? String == "think")
    }

    @Test
    func statusFollowsTheFinishReason() {
        #expect(ResponsesOutputBuilder.statusFields(.finished(.stop)).status == "completed")
        #expect(ResponsesOutputBuilder.statusFields(.finished(.toolCalls)).status == "completed")
        #expect(ResponsesOutputBuilder.statusFields(.finished(nil)).status == "completed")
        #expect(ResponsesOutputBuilder.statusFields(.finished(.stop)).error is NSNull)

        let length = ResponsesOutputBuilder.statusFields(.finished(.length))
        #expect(length.status == "incomplete")
        #expect((length.incompleteDetails as? [String: Any])?["reason"] as? String == "max_output_tokens")

        let engineError = ResponsesOutputBuilder.statusFields(.finished(.error))
        #expect(engineError.status == "failed")
        #expect((engineError.error as? [String: Any])?["code"] as? String == "server_error")

        let failed = ResponsesOutputBuilder.statusFields(.failed("stalled"))
        #expect(failed.status == "failed")
        #expect((failed.error as? [String: Any])?["message"] as? String == "stalled")
        #expect(failed.incompleteDetails is NSNull)
    }

    @Test
    func usageReportsCachedTokensOnlyWhenExposed() throws {
        let usage = TokenUsage(promptTokens: 10, completionTokens: 4, cachedPromptTokens: 6)
        let shown = ResponsesOutputBuilder.usageObject(usage, exposeCacheFigures: true)
        #expect(shown["input_tokens"] as? Int == 10)
        #expect(shown["output_tokens"] as? Int == 4)
        #expect(shown["total_tokens"] as? Int == 14)
        #expect((shown["input_tokens_details"] as? [String: Any])?["cached_tokens"] as? Int == 6)
        #expect((shown["output_tokens_details"] as? [String: Any])?["reasoning_tokens"] as? Int == 0)

        let hidden = ResponsesOutputBuilder.usageObject(usage, exposeCacheFigures: false)
        #expect((hidden["input_tokens_details"] as? [String: Any])?["cached_tokens"] as? Int == 0)

        let none = ResponsesOutputBuilder.usageObject(nil, exposeCacheFigures: true)
        #expect(none["total_tokens"] as? Int == 0)
        #expect((none["input_tokens_details"] as? [String: Any])?["cached_tokens"] as? Int == 0)
    }

    @Test
    func responseObjectEchoesTheRequest() throws {
        let body: [String: Any] = [
            "model": "qwen", "input": "hi", "instructions": "be brief", "tool_choice": "required",
            "parallel_tool_calls": false, "metadata": ["k": "v"], "store": true,
            "tools": [["type": "web_search"]], "text": ["format": ["type": "json_object"]],
        ]
        let request = try JSONDecoder().decode(
            ResponsesRequest.self, from: JSONSerialization.data(withJSONObject: body))
        let envelope = ResponsesOutputBuilder.Envelope(
            id: "resp_1", createdAt: 1_700_000_000, request: request,
            parameters: GenerationParameters(temperature: 0.2, topP: 0.9, maxTokens: 64))
        let object = ResponsesOutputBuilder.responseObject(envelope, status: "in_progress", output: [], usage: nil)

        #expect(object["id"] as? String == "resp_1")
        #expect(object["object"] as? String == "response")
        #expect(object["created_at"] as? Int == 1_700_000_000)
        #expect(object["model"] as? String == "qwen")
        #expect(object["status"] as? String == "in_progress")
        #expect(object["usage"] is NSNull)
        #expect(object["error"] is NSNull)
        #expect(object["incomplete_details"] is NSNull)
        #expect(object["instructions"] as? String == "be brief")
        #expect(object["tool_choice"] as? String == "required")
        #expect(object["parallel_tool_calls"] as? Bool == false)
        #expect(object["temperature"] as? Double == 0.2)
        #expect(object["top_p"] as? Double == 0.9)
        #expect(object["max_output_tokens"] as? Int == 64)
        #expect((object["metadata"] as? [String: Any])?["k"] as? String == "v")
        #expect((object["tools"] as? [[String: Any]])?.first?["type"] as? String == "web_search")
        #expect(((object["text"] as? [String: Any])?["format"] as? [String: Any])?["type"] as? String == "json_object")
        // Stateless whatever the client asked for.
        #expect(object["store"] as? Bool == false)
        #expect(object["previous_response_id"] is NSNull)
        // Serializable as is.
        _ = try JSONSerialization.data(withJSONObject: object)
    }

    @Test
    func responseObjectDefaultsWhenTheRequestIsMinimal() throws {
        let request = try JSONDecoder().decode(
            ResponsesRequest.self, from: Data(#"{"model":"m","input":"hi"}"#.utf8))
        let envelope = ResponsesOutputBuilder.Envelope(
            id: "resp_1", createdAt: 0, request: request, parameters: GenerationParameters())
        let object = ResponsesOutputBuilder.responseObject(envelope, status: "completed", output: [], usage: [:])
        #expect(object["instructions"] is NSNull)
        #expect(object["tool_choice"] as? String == "auto")
        #expect(object["parallel_tool_calls"] as? Bool == true)
        #expect((object["tools"] as? [Any])?.isEmpty == true)
        #expect(((object["text"] as? [String: Any])?["format"] as? [String: Any])?["type"] as? String == "text")
        #expect((object["metadata"] as? [String: Any])?.isEmpty == true)
    }
}
