import Foundation
import Testing

@testable import MacMLXCore

// MARK: - /v1/responses endpoint tests
//
// A real HummingbirdServer over HTTP, backed by a scripted stub engine — no
// model, no Metal. Covers the Responses wire format end to end: the response
// object, the SSE event sequence, reasoning and tool-call items, history
// replay into the engine's request, terminal statuses and request errors.
//
// Port assignments (21_100 range, spaced by 30 for port-retry headroom):
//   nonStreamingTextCompletes                       : 21_100
//   streamingTextEmitsTheEventSequence              : 21_130
//   streamingReasoningFromAnOpenThinkPrompt         : 21_160
//   nonStreamingReasoningFromThinkTags              : 21_190
//   toolCallNonStreaming                            : 21_220
//   toolCallStreaming                               : 21_250
//   textThenToolCallStreaming                       : 21_280
//   namespacedToolCallCarriesItsNamespace           : 21_310
//   customToolCallStreaming                         : 21_340
//   historyReachesTheEngineAsChatMessages           : 21_370
//   lengthCapIsIncomplete                           : 21_400
//   unknownModelIs404BeforeTheStream                : 21_430
//   requestErrorsAre400                             : 21_460
//   batchSeamIsNeverConsulted                       : 21_490
//   engineErrorIsResponseFailed                     : 21_520
//   stallIsResponseFailed                           : 21_550
//   whitespaceBeforeAToolCallMakesNoMessageItem     : 21_580
//   (21_600 is ResponsesAgentLoopE2ETests)
//   maxOutputTokensDefaultsTo32768                  : 21_640
//   lockIsFreedAfterAClientDisconnect               : 21_670
//   coldSwapFailureAfterHeadersIsResponseFailed     : 21_730
//   reasoningEffortNoneTurnsThinkingOff             : 21_760
//   originHeaderHidesCachedTokens                   : 21_820
//   bodyOverTwoMegabytesIsAccepted                  : 21_850
//   badImageOnADeveloperMessageIsAPlaceholder       : 21_880

@Suite("ResponsesEndpoint")
struct ResponsesEndpointTests {

    // MARK: Stubs

    private struct StubFailure: Error, LocalizedError {
        var errorDescription: String? { "stub engine failed" }
    }

    /// Plays back a fixed chunk script (optionally failing after it, never
    /// finishing, or streaming "x" forever until the consumer goes away),
    /// refuses to load `brokenModel`, reports whether the prompt opens a think
    /// block, and records the request it was asked to generate.
    private actor ScriptedEngine: InferenceEngine {
        nonisolated let engineID: EngineID = .mlxSwift
        private(set) var status: EngineStatus = .idle
        private(set) var loadedModel: LocalModel?
        let version = "responses-stub-1"
        private(set) var capturedRequest: GenerateRequest?

        private let chunks: [GenerateChunk]
        private let opensThink: Bool
        private let failsAfterScript: Bool
        private let hangsAfterScript: Bool
        private let dripsForever: Bool

        init(
            chunks: [GenerateChunk],
            opensThink: Bool = false,
            failsAfterScript: Bool = false,
            hangsAfterScript: Bool = false,
            dripsForever: Bool = false
        ) {
            self.chunks = chunks
            self.opensThink = opensThink
            self.failsAfterScript = failsAfterScript
            self.hangsAfterScript = hangsAfterScript
            self.dripsForever = dripsForever
        }

        func load(_ model: LocalModel) async throws {
            if model.id == ResponsesEndpointTests.brokenModel { throw StubFailure() }
            loadedModel = model
            status = .ready(model: model.id)
        }

        func unload() async throws {
            loadedModel = nil
            status = .idle
        }

        nonisolated func generate(_ request: GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error> {
            let chunks = self.chunks
            let failsAfterScript = self.failsAfterScript
            let hangsAfterScript = self.hangsAfterScript
            let dripsForever = self.dripsForever
            return AsyncThrowingStream { continuation in
                Task {
                    await self.capture(request)
                    for chunk in chunks {
                        continuation.yield(chunk)
                    }
                    while dripsForever {
                        if case .terminated = continuation.yield(GenerateChunk(text: "x")) { return }
                        try? await Task.sleep(nanoseconds: 20_000_000)
                    }
                    if failsAfterScript {
                        continuation.finish(throwing: StubFailure())
                    } else if !hangsAfterScript {
                        continuation.finish()
                    }
                }
            }
        }

        func promptOpensThinkBlock(_ request: GenerateRequest) async -> Bool { opensThink }

        private func capture(_ request: GenerateRequest) { capturedRequest = request }

        func healthCheck() async -> Bool { true }
    }

    /// A batching seam that records every submit and serves nothing.
    private actor RecordingBatchServing: BatchGenerationServing {
        private(set) var submitted: [String] = []

        func submit(_ request: GenerateRequest) async -> AsyncThrowingStream<GenerateChunk, Error>? {
            submitted.append(request.model)
            return AsyncThrowingStream { continuation in
                continuation.yield(GenerateChunk(text: "batched", finishReason: .stop))
                continuation.finish()
            }
        }

        func drainForModelChange() async {}
    }

    // MARK: Helpers

    private static let model = "stub-model"
    fileprivate static let brokenModel = "broken-model"

    private func localModel(_ id: String) -> LocalModel {
        LocalModel(
            id: id, displayName: id, directory: URL(filePath: "/tmp"), sizeBytes: 0,
            format: .mlx, quantization: nil, parameterCount: nil, architecture: nil)
    }

    /// Polls until the server's generation lock is free (the release runs in
    /// a detached task, so it lands shortly after the response ends), for at
    /// most 60 s. True when it was released.
    private func lockIsReleased(_ server: HummingbirdServer) async throws -> Bool {
        for _ in 0..<1_200 {
            if await !server.generationLockIsHeld { return true }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        return false
    }

    private func loadedEngine(_ engine: ScriptedEngine) async throws -> ScriptedEngine {
        try await engine.load(LocalModel(
            id: Self.model, displayName: "Stub", directory: URL(filePath: "/tmp"), sizeBytes: 0,
            format: .mlx, quantization: nil, parameterCount: nil, architecture: nil))
        return engine
    }

    private func start(_ engine: ScriptedEngine, port: Int) async throws -> (HummingbirdServer, Int) {
        let server = HummingbirdServer(engine: try await loadedEngine(engine))
        return (server, try await server.start(preferredPort: port))
    }

    private func post(port: Int, body: Any, headers: [String: String] = [:]) async throws -> (Data, HTTPURLResponse) {
        try await postRaw(port: port, data: JSONSerialization.data(withJSONObject: body), headers: headers)
    }

    private func urlRequest(port: Int, data: Data, headers: [String: String] = [:]) throws -> URLRequest {
        var request = URLRequest(url: try #require(URL(string: "http://127.0.0.1:\(port)/v1/responses")))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        for (field, value) in headers { request.setValue(value, forHTTPHeaderField: field) }
        request.httpBody = data
        return request
    }

    private func postRaw(port: Int, data: Data, headers: [String: String] = [:]) async throws -> (Data, HTTPURLResponse) {
        let request = try urlRequest(port: port, data: data, headers: headers)
        let (body, response) = try await URLSession.shared.data(for: request)
        return (body, try #require(response as? HTTPURLResponse))
    }

    private func json(_ data: Data) throws -> [String: Any] {
        try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    /// The SSE body as (event name, payload) pairs, checking each payload's
    /// `type` matches its event line.
    private func events(_ data: Data) throws -> [(type: String, payload: [String: Any])] {
        let text = String(decoding: data, as: UTF8.self)
        #expect(!text.contains("[DONE]"))
        return try text.components(separatedBy: "\n\n").filter { !$0.isEmpty }.map { frame in
            let lines = frame.components(separatedBy: "\n")
            #expect(lines.count == 2)
            let type = String(try #require(lines.first).dropFirst("event: ".count))
            let payload = try json(Data(try #require(lines.last).dropFirst("data: ".count).utf8))
            #expect(payload["type"] as? String == type)
            return (type, payload)
        }
    }

    private func streamRequest(_ extra: [String: Any] = [:]) -> [String: Any] {
        ["model": Self.model, "input": "hi", "stream": true].merging(extra) { $1 }
    }

    private func terminalResponse(_ events: [(type: String, payload: [String: Any])]) throws -> [String: Any] {
        try #require(events.last?.payload["response"] as? [String: Any])
    }

    private func outputTypes(_ response: [String: Any]) -> [String] {
        (response["output"] as? [[String: Any]] ?? []).compactMap { $0["type"] as? String }
    }

    private let weatherCall = ToolCallRequest(id: "call_abc123", name: "get_weather", arguments: ["city": .string("Paris")])

    private func toolChunk(_ calls: [ToolCallRequest]) -> GenerateChunk {
        GenerateChunk(text: "", finishReason: .toolCalls, usage: TokenUsage(promptTokens: 3, completionTokens: 4), toolCalls: calls)
    }

    private let weatherTool: [String: Any] = [
        "type": "function", "name": "get_weather", "description": "Weather.",
        "parameters": ["type": "object", "properties": ["city": ["type": "string"]]],
    ]

    // MARK: Text

    @Test
    func nonStreamingTextCompletes() async throws {
        let (server, port) = try await start(ScriptedEngine(chunks: [
            GenerateChunk(text: "Hello"),
            GenerateChunk(text: " world", finishReason: .stop,
                          usage: TokenUsage(promptTokens: 5, completionTokens: 2, cachedPromptTokens: 3)),
        ]), port: 21_100)
        let (data, response) = try await post(port: port, body: ["model": Self.model, "input": "hi"])
        await server.stop()

        #expect(response.statusCode == 200)
        let body = try json(data)
        #expect(body["object"] as? String == "response")
        #expect((body["id"] as? String)?.hasPrefix("resp_") == true)
        #expect(body["status"] as? String == "completed")
        #expect(body["model"] as? String == Self.model)
        #expect(body["output_text"] as? String == "Hello world")
        let output = try #require(body["output"] as? [[String: Any]])
        #expect(output.count == 1)
        #expect(output[0]["type"] as? String == "message")
        #expect(((output[0]["content"] as? [[String: Any]])?.first)?["text"] as? String == "Hello world")
        let usage = try #require(body["usage"] as? [String: Any])
        #expect(usage["input_tokens"] as? Int == 5)
        #expect(usage["output_tokens"] as? Int == 2)
        #expect(usage["total_tokens"] as? Int == 7)
        #expect((usage["input_tokens_details"] as? [String: Any])?["cached_tokens"] as? Int == 3)
    }

    @Test
    func streamingTextEmitsTheEventSequence() async throws {
        let (server, port) = try await start(ScriptedEngine(chunks: [
            GenerateChunk(text: "Hello"),
            GenerateChunk(text: " world", finishReason: .stop, usage: TokenUsage(promptTokens: 5, completionTokens: 2)),
        ]), port: 21_130)
        let (data, response) = try await post(port: port, body: streamRequest())
        await server.stop()

        #expect(response.statusCode == 200)
        #expect(response.value(forHTTPHeaderField: "Content-Type") == "text/event-stream")
        let events = try events(data)
        #expect(events.map(\.type) == [
            "response.created", "response.in_progress",
            "response.output_item.added", "response.content_part.added",
            "response.output_text.delta", "response.output_text.delta",
            "response.output_text.done", "response.content_part.done", "response.output_item.done",
            "response.completed",
        ])
        #expect(events.map { $0.payload["sequence_number"] as? Int } == Array(0..<events.count))
        let deltas = events.filter { $0.type == "response.output_text.delta" }
        #expect(deltas.compactMap { $0.payload["delta"] as? String }.joined() == "Hello world")
        #expect(deltas.allSatisfy { ($0.payload["logprobs"] as? [Any])?.isEmpty == true })
        let created = try #require(events.first?.payload["response"] as? [String: Any])
        #expect(created["status"] as? String == "in_progress")
        #expect(created["usage"] is NSNull)
        let done = try terminalResponse(events)
        #expect(done["status"] as? String == "completed")
        #expect(done["id"] as? String == created["id"] as? String)
        #expect((done["usage"] as? [String: Any])?["total_tokens"] as? Int == 7)
        #expect(outputTypes(done) == ["message"])
        // The convenience field is sent in the non-streaming body only.
        #expect(done["output_text"] == nil)
    }

    // MARK: Reasoning

    @Test
    func streamingReasoningFromAnOpenThinkPrompt() async throws {
        let (server, port) = try await start(ScriptedEngine(chunks: [
            GenerateChunk(text: "a"),
            GenerateChunk(text: "</think>"),
            GenerateChunk(text: "\n\n"),
            GenerateChunk(text: "b", finishReason: .stop, usage: TokenUsage(promptTokens: 1, completionTokens: 3)),
        ], opensThink: true), port: 21_160)
        let (data, _) = try await post(port: port, body: streamRequest())
        await server.stop()

        let events = try events(data)
        #expect(events.map(\.type) == [
            "response.created", "response.in_progress",
            "response.output_item.added", "response.reasoning_summary_part.added", "response.content_part.added",
            "response.reasoning_summary_text.delta", "response.reasoning_text.delta",
            "response.reasoning_summary_text.done", "response.reasoning_summary_part.done",
            "response.reasoning_text.done", "response.content_part.done", "response.output_item.done",
            "response.output_item.added", "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done", "response.content_part.done", "response.output_item.done",
            "response.completed",
        ])
        let reasoningDelta = try #require(events.first { $0.type == "response.reasoning_text.delta" })
        #expect(reasoningDelta.payload["delta"] as? String == "a")
        #expect(reasoningDelta.payload["output_index"] as? Int == 0)
        let summaryDelta = try #require(events.first { $0.type == "response.reasoning_summary_text.delta" })
        #expect(summaryDelta.payload["delta"] as? String == "a")
        #expect(summaryDelta.payload["summary_index"] as? Int == 0)
        #expect(summaryDelta.payload["item_id"] as? String == reasoningDelta.payload["item_id"] as? String)
        let textDelta = try #require(events.first { $0.type == "response.output_text.delta" })
        #expect(textDelta.payload["delta"] as? String == "b")
        #expect(textDelta.payload["output_index"] as? Int == 1)
        let done = try terminalResponse(events)
        #expect(outputTypes(done) == ["reasoning", "message"])
        let reasoning = try #require((done["output"] as? [[String: Any]])?.first)
        #expect(((reasoning["content"] as? [[String: Any]])?.first)?["text"] as? String == "a")
        #expect(((reasoning["summary"] as? [[String: Any]])?.first)?["text"] as? String == "a")
    }

    @Test
    func nonStreamingReasoningFromThinkTags() async throws {
        let (server, port) = try await start(ScriptedEngine(chunks: [
            GenerateChunk(text: "<think>a</think>b", finishReason: .stop, usage: TokenUsage(promptTokens: 1, completionTokens: 3)),
        ]), port: 21_190)
        let (data, _) = try await post(port: port, body: ["model": Self.model, "input": "hi"])
        await server.stop()

        let body = try json(data)
        #expect(outputTypes(body) == ["reasoning", "message"])
        let output = try #require(body["output"] as? [[String: Any]])
        #expect(((output[0]["content"] as? [[String: Any]])?.first)?["text"] as? String == "a")
        #expect(body["output_text"] as? String == "b")
    }

    // MARK: Tool calls

    @Test
    func toolCallNonStreaming() async throws {
        let (server, port) = try await start(ScriptedEngine(chunks: [toolChunk([weatherCall])]), port: 21_220)
        let (data, _) = try await post(port: port, body: ["model": Self.model, "input": "weather?", "tools": [weatherTool]])
        await server.stop()

        let body = try json(data)
        #expect(body["status"] as? String == "completed")
        // A tool-only turn has no empty message item.
        #expect(outputTypes(body) == ["function_call"])
        let call = try #require((body["output"] as? [[String: Any]])?.first)
        #expect(call["call_id"] as? String == "call_abc123")
        #expect(call["name"] as? String == "get_weather")
        #expect(call["arguments"] as? String == "{\"city\":\"Paris\"}")
        #expect(call["status"] as? String == "completed")
        #expect((call["id"] as? String)?.hasPrefix("fc_") == true)
        #expect(call["namespace"] == nil)
    }

    @Test
    func toolCallStreaming() async throws {
        let (server, port) = try await start(ScriptedEngine(chunks: [toolChunk([weatherCall])]), port: 21_250)
        let (data, _) = try await post(port: port, body: streamRequest(["tools": [weatherTool]]))
        await server.stop()

        let events = try events(data)
        #expect(events.map(\.type) == [
            "response.created", "response.in_progress",
            "response.output_item.added",
            "response.function_call_arguments.delta", "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed",
        ])
        let added = try #require(events[2].payload["item"] as? [String: Any])
        #expect(added["arguments"] as? String == "")
        #expect(added["status"] as? String == "in_progress")
        #expect(events[3].payload["delta"] as? String == "{\"city\":\"Paris\"}")
        #expect(events[4].payload["arguments"] as? String == "{\"city\":\"Paris\"}")
        #expect(events[3].payload["item_id"] as? String == added["id"] as? String)
        let item = try #require(events[5].payload["item"] as? [String: Any])
        #expect(item["type"] as? String == "function_call")
        #expect(item["call_id"] as? String == "call_abc123")
        #expect(item["arguments"] as? String == "{\"city\":\"Paris\"}")
        #expect(outputTypes(try terminalResponse(events)) == ["function_call"])
    }

    /// A thinking model ends its reasoning with blank lines before the call;
    /// they do not make a message item.
    @Test
    func whitespaceBeforeAToolCallMakesNoMessageItem() async throws {
        let (server, port) = try await start(ScriptedEngine(chunks: [
            GenerateChunk(text: "plan</think>"), GenerateChunk(text: "\n\n"), toolChunk([weatherCall]),
        ], opensThink: true), port: 21_580)
        let (streamData, _) = try await post(port: port, body: streamRequest(["tools": [weatherTool]]))
        let (bodyData, _) = try await post(port: port, body: ["model": Self.model, "input": "hi", "tools": [weatherTool]])
        await server.stop()

        #expect(outputTypes(try terminalResponse(try events(streamData))) == ["reasoning", "function_call"])
        let body = try json(bodyData)
        #expect(outputTypes(body) == ["reasoning", "function_call"])
        #expect(body["output_text"] as? String == "")
    }

    @Test
    func textThenToolCallStreaming() async throws {
        let (server, port) = try await start(ScriptedEngine(chunks: [
            GenerateChunk(text: "Let me check."), toolChunk([weatherCall]),
        ]), port: 21_280)
        let (data, _) = try await post(port: port, body: streamRequest(["tools": [weatherTool]]))
        await server.stop()

        let events = try events(data)
        let itemsDone = events.filter { $0.type == "response.output_item.done" }
        #expect(itemsDone.compactMap { ($0.payload["item"] as? [String: Any])?["type"] as? String }
            == ["message", "function_call"])
        #expect(itemsDone.compactMap { $0.payload["output_index"] as? Int } == [0, 1])
        let argumentsDelta = try #require(events.first { $0.type == "response.function_call_arguments.delta" })
        #expect(argumentsDelta.payload["output_index"] as? Int == 1)
        #expect(outputTypes(try terminalResponse(events)) == ["message", "function_call"])
    }

    @Test
    func namespacedToolCallCarriesItsNamespace() async throws {
        let call = ToolCallRequest(id: "call_w", name: "wait_agent", arguments: ["id": .string("a1")])
        let (server, port) = try await start(ScriptedEngine(chunks: [toolChunk([call])]), port: 21_310)
        let tools: [[String: Any]] = [
            weatherTool,
            ["type": "namespace", "name": "multi_agent_v1", "tools": [
                ["type": "function", "name": "wait_agent", "parameters": ["type": "object"]],
            ]],
            ["type": "web_search", "external_web_access": false],
        ]
        let (streamData, _) = try await post(port: port, body: streamRequest(["tools": tools]))
        let (bodyData, _) = try await post(port: port, body: ["model": Self.model, "input": "hi", "tools": tools])
        await server.stop()

        let streamed = try #require(try events(streamData).first { $0.type == "response.output_item.done" })
        let streamedItem = try #require(streamed.payload["item"] as? [String: Any])
        #expect(streamedItem["name"] as? String == "wait_agent")
        #expect(streamedItem["namespace"] as? String == "multi_agent_v1")
        let item = try #require((try json(bodyData)["output"] as? [[String: Any]])?.first)
        #expect(item["name"] as? String == "wait_agent")
        #expect(item["namespace"] as? String == "multi_agent_v1")
    }

    @Test
    func customToolCallStreaming() async throws {
        let call = ToolCallRequest(id: "call_p", name: "apply_patch", arguments: ["input": .string("*** Begin Patch")])
        let (server, port) = try await start(ScriptedEngine(chunks: [toolChunk([call])]), port: 21_340)
        let (data, _) = try await post(port: port, body: streamRequest([
            "tools": [["type": "custom", "name": "apply_patch", "description": "Patch."]],
        ]))
        await server.stop()

        let events = try events(data)
        #expect(events.map(\.type).contains("response.custom_tool_call_input.delta"))
        let inputDone = try #require(events.first { $0.type == "response.custom_tool_call_input.done" })
        #expect(inputDone.payload["input"] as? String == "*** Begin Patch")
        let item = try #require(events.first { $0.type == "response.output_item.done" }?.payload["item"] as? [String: Any])
        #expect(item["type"] as? String == "custom_tool_call")
        #expect((item["id"] as? String)?.hasPrefix("ctc_") == true)
        #expect(item["input"] as? String == "*** Begin Patch")
    }

    // MARK: History

    @Test
    func historyReachesTheEngineAsChatMessages() async throws {
        let engine = ScriptedEngine(chunks: [GenerateChunk(text: "It is sunny.", finishReason: .stop)])
        let (server, port) = try await start(engine, port: 21_370)
        let (_, response) = try await post(port: port, body: [
            "model": Self.model,
            "instructions": "be brief",
            "input": [
                ["type": "message", "role": "developer", "content": [["type": "input_text", "text": "dev"]]],
                ["type": "message", "role": "user", "content": [["type": "input_text", "text": "weather?"]]],
                ["type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "x"],
                ["type": "function_call", "id": "fc_1", "name": "get_weather",
                 "arguments": "{\"city\":\"Paris\"}", "call_id": "call_1"],
                ["type": "function_call_output", "call_id": "call_1", "output": "22C"],
            ],
            "tools": [weatherTool],
        ])
        await server.stop()

        #expect(response.statusCode == 200)
        let captured = try #require(await engine.capturedRequest)
        #expect(captured.systemPrompt == "be brief\n\ndev")
        #expect(captured.messages.map(\.role) == [.user, .assistant, .tool])
        #expect(captured.messages[1].toolCalls == [
            ToolCallRequest(id: "call_1", name: "get_weather", arguments: ["city": .string("Paris")]),
        ])
        #expect(captured.messages[2].toolCallID == "call_1")
        #expect(captured.messages[2].content == "22C")
        #expect(captured.tools?.count == 1)
        // No `max_output_tokens`, as Codex sends: the Responses default.
        #expect(captured.parameters.maxTokens == 32_768)
    }

    // MARK: Statuses and errors

    @Test
    func lengthCapIsIncomplete() async throws {
        let chunks = [GenerateChunk(text: "trunc", finishReason: .length, usage: TokenUsage(promptTokens: 1, completionTokens: 1))]
        let (server, port) = try await start(ScriptedEngine(chunks: chunks), port: 21_400)
        let (streamData, _) = try await post(port: port, body: streamRequest(["max_output_tokens": 1]))
        let (bodyData, _) = try await post(port: port, body: ["model": Self.model, "input": "hi", "max_output_tokens": 1])
        await server.stop()

        let events = try events(streamData)
        #expect(events.last?.type == "response.incomplete")
        let terminal = try terminalResponse(events)
        #expect(terminal["status"] as? String == "incomplete")
        #expect((terminal["incomplete_details"] as? [String: Any])?["reason"] as? String == "max_output_tokens")
        #expect(terminal["max_output_tokens"] as? Int == 1)
        // The cut-off message is itself incomplete, streamed and not.
        #expect((terminal["output"] as? [[String: Any]])?.first?["status"] as? String == "incomplete")
        let streamedDone = try #require(events.first { $0.type == "response.output_item.done" })
        #expect((streamedDone.payload["item"] as? [String: Any])?["status"] as? String == "incomplete")
        let body = try json(bodyData)
        #expect(body["status"] as? String == "incomplete")
        #expect((body["incomplete_details"] as? [String: Any])?["reason"] as? String == "max_output_tokens")
        #expect((body["output"] as? [[String: Any]])?.first?["status"] as? String == "incomplete")
    }

    @Test
    func unknownModelIs404BeforeTheStream() async throws {
        let (server, port) = try await start(ScriptedEngine(chunks: []), port: 21_430)
        let (streamData, streamResponse) = try await post(port: port, body: ["model": "nope", "input": "hi", "stream": true])
        let (bodyData, bodyResponse) = try await post(port: port, body: ["model": "nope", "input": "hi"])
        await server.stop()

        #expect(streamResponse.statusCode == 404)
        #expect(streamResponse.value(forHTTPHeaderField: "Content-Type")?.hasPrefix("application/json") == true)
        #expect((try json(streamData)["error"] as? [String: Any])?["code"] as? String == "model_not_found")
        #expect(bodyResponse.statusCode == 404)
        #expect((try json(bodyData)["error"] as? [String: Any])?["code"] as? String == "model_not_found")
    }

    @Test
    func requestErrorsAre400() async throws {
        let (server, port) = try await start(ScriptedEngine(chunks: []), port: 21_460)
        let (badJSONData, badJSON) = try await postRaw(port: port, data: Data("{not json".utf8))
        let (previousData, previous) = try await post(port: port, body: [
            "model": Self.model, "input": "hi", "previous_response_id": "resp_1", "stream": true,
        ])
        let (_, missingInput) = try await post(port: port, body: ["model": Self.model])
        await server.stop()

        #expect(badJSON.statusCode == 400)
        #expect((try json(badJSONData)["error"] as? [String: Any])?["type"] as? String == "invalid_request_error")
        #expect(previous.statusCode == 400)
        let error = try #require(try json(previousData)["error"] as? [String: Any])
        #expect(error["type"] as? String == "invalid_request_error")
        #expect((error["message"] as? String)?.contains("previous_response_id") == true)
        #expect(missingInput.statusCode == 400)
    }

    @Test
    func batchSeamIsNeverConsulted() async throws {
        let engine = try await loadedEngine(ScriptedEngine(chunks: [GenerateChunk(text: "single", finishReason: .stop)]))
        let seam = RecordingBatchServing()
        let server = HummingbirdServer(engineProvider: { engine }, batchServing: seam)
        let port = try await server.start(preferredPort: 21_490)
        let (data, _) = try await post(port: port, body: ["model": Self.model, "input": "no tools here"])
        await server.stop()

        #expect(try json(data)["output_text"] as? String == "single")
        #expect(await seam.submitted.isEmpty)
    }

    @Test
    func engineErrorIsResponseFailed() async throws {
        let (server, port) = try await start(
            ScriptedEngine(chunks: [GenerateChunk(text: "partial")], failsAfterScript: true), port: 21_520)
        let (streamData, streamResponse) = try await post(port: port, body: streamRequest())
        let (bodyData, _) = try await post(port: port, body: ["model": Self.model, "input": "hi"])
        await server.stop()

        #expect(streamResponse.statusCode == 200)
        let events = try events(streamData)
        #expect(events.last?.type == "response.failed")
        // The partial message is closed before the terminal event.
        #expect(events.map(\.type).contains("response.output_item.done"))
        let terminal = try terminalResponse(events)
        #expect(terminal["status"] as? String == "failed")
        let error = try #require(terminal["error"] as? [String: Any])
        #expect(error["code"] as? String == "server_error")
        #expect(error["message"] as? String == "stub engine failed")
        #expect((terminal["output"] as? [[String: Any]])?.first?["status"] as? String == "incomplete")
        let body = try json(bodyData)
        #expect(body["status"] as? String == "failed")
        #expect((body["output"] as? [[String: Any]])?.first?["status"] as? String == "incomplete")
    }

    @Test
    func stallIsResponseFailed() async throws {
        let engine = try await loadedEngine(ScriptedEngine(chunks: [], hangsAfterScript: true))
        let server = HummingbirdServer(engineProvider: { engine }, stallTimeoutSeconds: 0.5)
        let port = try await server.start(preferredPort: 21_550)
        let (data, _) = try await post(port: port, body: streamRequest())
        let released = try await lockIsReleased(server)
        await server.stop()

        #expect(released)
        let events = try events(data)
        #expect(events.last?.type == "response.failed")
        let error = try #require(try terminalResponse(events)["error"] as? [String: Any])
        #expect((error["message"] as? String)?.contains("Generation stalled") == true)
        #expect(error["code"] as? String == "server_error")
        // A stall with no output still leaves no empty message item behind.
        #expect(outputTypes(try terminalResponse(events)).isEmpty)
    }

    // MARK: Budget, lock and request limits

    @Test
    func maxOutputTokensDefaultsTo32768() async throws {
        let engine = ScriptedEngine(chunks: [GenerateChunk(text: "ok", finishReason: .stop)])
        let (server, port) = try await start(engine, port: 21_640)
        _ = try await post(port: port, body: ["model": Self.model, "input": "hi", "stream": true])
        let defaulted = await engine.capturedRequest?.parameters.maxTokens
        _ = try await post(port: port, body: ["model": Self.model, "input": "hi", "max_output_tokens": 100])
        let explicit = await engine.capturedRequest?.parameters.maxTokens
        await server.stop()

        #expect(defaulted == 32_768)
        #expect(explicit == 100)
    }

    @Test
    func lockIsFreedAfterAClientDisconnect() async throws {
        let (server, port) = try await start(ScriptedEngine(chunks: [], dripsForever: true), port: 21_670)
        let request = try urlRequest(port: port, data: JSONSerialization.data(withJSONObject: streamRequest()))
        let (bytes, _) = try await URLSession.shared.bytes(for: request)
        var sawDelta = false
        for try await line in bytes.lines where line.contains("response.output_text.delta") {
            sawDelta = true
            break
        }
        bytes.task.cancel()
        let released = try await lockIsReleased(server)
        await server.stop()

        #expect(sawDelta)
        #expect(released)
    }

    @Test
    func coldSwapFailureAfterHeadersIsResponseFailed() async throws {
        let engine = try await loadedEngine(ScriptedEngine(chunks: [GenerateChunk(text: "ok", finishReason: .stop)]))
        let broken = localModel(Self.brokenModel)
        let server = HummingbirdServer(engine: engine, modelResolver: { $0 == Self.brokenModel ? broken : nil })
        let port = try await server.start(preferredPort: 21_730)
        let (data, response) = try await post(port: port, body: [
            "model": Self.brokenModel, "input": "hi", "stream": true,
        ])
        let released = try await lockIsReleased(server)
        await server.stop()

        // The model resolves, so headers go out; the load then fails in-band.
        #expect(response.statusCode == 200)
        let events = try events(data)
        #expect(events.map(\.type) == ["response.created", "response.in_progress", "response.failed"])
        let error = try #require(try terminalResponse(events)["error"] as? [String: Any])
        #expect(error["code"] as? String == "load_failed")
        #expect((error["message"] as? String)?.hasPrefix("Failed to load \(Self.brokenModel)") == true)
        #expect(released)
    }

    @Test
    func reasoningEffortNoneTurnsThinkingOff() async throws {
        let engine = ScriptedEngine(chunks: [GenerateChunk(text: "ok", finishReason: .stop)])
        let (server, port) = try await start(engine, port: 21_760)
        _ = try await post(port: port, body: ["model": Self.model, "input": "hi", "reasoning": ["effort": "none"]])
        let off = await engine.capturedRequest?.templateKwargs?["enable_thinking"]
        _ = try await post(port: port, body: ["model": Self.model, "input": "hi", "reasoning": ["effort": "high"]])
        let untouched = await engine.capturedRequest?.templateKwargs?["enable_thinking"]
        await server.stop()

        #expect(off == .bool(false))
        #expect(untouched == nil)
    }

    @Test
    func originHeaderHidesCachedTokens() async throws {
        let (server, port) = try await start(ScriptedEngine(chunks: [
            GenerateChunk(text: "ok", finishReason: .stop,
                          usage: TokenUsage(promptTokens: 5, completionTokens: 1, cachedPromptTokens: 4)),
        ]), port: 21_820)
        let body = ["model": Self.model, "input": "hi"]
        let (native, _) = try await post(port: port, body: body)
        let (browser, _) = try await post(port: port, body: body, headers: ["Origin": "https://example.com"])
        await server.stop()

        func cached(_ data: Data) throws -> Int? {
            ((try json(data)["usage"] as? [String: Any])?["input_tokens_details"] as? [String: Any])?["cached_tokens"] as? Int
        }
        #expect(try cached(native) == 4)
        #expect(try cached(browser) == 0)
    }

    @Test
    func bodyOverTwoMegabytesIsAccepted() async throws {
        let (server, port) = try await start(ScriptedEngine(chunks: [GenerateChunk(text: "ok", finishReason: .stop)]), port: 21_850)
        let big = String(repeating: "a", count: 2 * 1024 * 1024 + 4096)
        let (data, response) = try await post(port: port, body: ["model": Self.model, "input": big])
        await server.stop()

        #expect(response.statusCode == 200)
        #expect(try json(data)["status"] as? String == "completed")
    }

    @Test
    func badImageOnADeveloperMessageIsAPlaceholder() async throws {
        let engine = ScriptedEngine(chunks: [GenerateChunk(text: "ok", finishReason: .stop)])
        let (server, port) = try await start(engine, port: 21_880)
        let (_, response) = try await post(port: port, body: [
            "model": Self.model,
            "input": [
                ["type": "message", "role": "developer", "content": [
                    ["type": "input_text", "text": "dev"],
                    ["type": "input_image", "image_url": "data:application/pdf;base64,AAAA"],
                ]],
                ["type": "message", "role": "user", "content": "hi"],
            ],
        ])
        let systemPrompt = await engine.capturedRequest?.systemPrompt
        await server.stop()

        #expect(response.statusCode == 200)
        #expect(systemPrompt == "dev\n[image omitted from a system message]")
    }
}
