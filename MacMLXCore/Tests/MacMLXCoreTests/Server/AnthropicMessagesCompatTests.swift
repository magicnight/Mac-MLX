import Foundation
import Testing

@testable import MacMLXCore

// MARK: - /v1/messages Claude Code compatibility tests
//
// A real HummingbirdServer over HTTP, backed by a scripted stub engine — no
// model, no Metal. Covers what Claude Code sends and expects beyond the basic
// Messages wire format: `role:"system"` turns (its reminders) reaching the
// engine, `thinking` in and out, `tool_choice` modes, the fields it sends that
// macMLX ignores, and swap failures in Anthropic's error envelope. The bodies
// are synthetic, shaped like Claude Code 2.1's.
//
// Port assignments (22_000 range, spaced by 30 for port-retry headroom):
//   claudeCodeFirstTurnSystemTurnReachesTheEngine     : 22_000
//   claudeCodeSecondTurnShapeReachesTheEngine         : 22_030
//   ignoredFieldsDoNotFailTheRequest                  : 22_060
//   thinkingDisabledTurnsTheTemplateOff               : 22_090
//   adaptiveThinkingStreamsAThinkingBlockFirst        : 22_120
//   adaptiveThinkingNonStreamingHasAThinkingBlock     : 22_150
//   omittedThinkingDropsReasoningAndBlankText         : 22_180
//   omittedThinkingNonStreaming                       : 22_210
//   thinkingAbsentReturnsNoReasoning                  : 22_240
//   toolChoiceModes                                   : 22_270
//   unknownModelIsAnAnthropic404                      : 22_300
//   unknownModelStreamingIsAnAnthropic404             : 22_330
//   loadFailureIsAnAnthropic500                       : 22_360
//   missingMaxTokensIs400                             : 22_390

@Suite("AnthropicMessagesCompat")
struct AnthropicMessagesCompatTests {

    // MARK: Stubs

    private struct StubFailure: Error, LocalizedError {
        var errorDescription: String? { "stub engine failed" }
    }

    /// Plays back a fixed chunk script, reports whether the prompt opens a
    /// think block, refuses to load `brokenModel`, and records the request it
    /// was asked to generate.
    private actor ScriptedEngine: InferenceEngine {
        nonisolated let engineID: EngineID = .mlxSwift
        private(set) var status: EngineStatus = .idle
        private(set) var loadedModel: LocalModel?
        let version = "anthropic-stub-1"
        private(set) var capturedRequest: GenerateRequest?

        private let chunks: [GenerateChunk]
        private let opensThink: Bool

        init(chunks: [GenerateChunk], opensThink: Bool = false) {
            self.chunks = chunks
            self.opensThink = opensThink
        }

        func load(_ model: LocalModel) async throws {
            if model.id == AnthropicMessagesCompatTests.brokenModel { throw StubFailure() }
            loadedModel = model
            status = .ready(model: model.id)
        }

        func unload() async throws {
            loadedModel = nil
            status = .idle
        }

        nonisolated func generate(_ request: GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error> {
            let chunks = self.chunks
            return AsyncThrowingStream { continuation in
                Task {
                    await self.capture(request)
                    for chunk in chunks {
                        continuation.yield(chunk)
                    }
                    continuation.finish()
                }
            }
        }

        func promptOpensThinkBlock(_ request: GenerateRequest) async -> Bool { opensThink }

        private func capture(_ request: GenerateRequest) { capturedRequest = request }

        func healthCheck() async -> Bool { true }
    }

    // MARK: Helpers

    private static let model = "stub-model"
    fileprivate static let brokenModel = "broken-model"

    private func localModel(_ id: String) -> LocalModel {
        LocalModel(
            id: id, displayName: id, directory: URL(filePath: "/tmp"), sizeBytes: 0,
            format: .mlx, quantization: nil, parameterCount: nil, architecture: nil)
    }

    private func start(
        _ engine: ScriptedEngine,
        port: Int,
        resolver: HummingbirdServer.ModelResolver? = nil
    ) async throws -> (HummingbirdServer, Int) {
        try await engine.load(localModel(Self.model))
        let server = resolver.map { HummingbirdServer(engine: engine, modelResolver: $0) }
            ?? HummingbirdServer(engine: engine)
        return (server, try await server.start(preferredPort: port))
    }

    /// POSTs to `/v1/messages?beta=true`, the URL Claude Code uses, with its
    /// version and beta headers.
    private func post(port: Int, body: [String: Any]) async throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: try #require(URL(string: "http://127.0.0.1:\(port)/v1/messages?beta=true")))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
        request.setValue(
            "mid-conversation-system-2026-04-07,interleaved-thinking-2025-05-14",
            forHTTPHeaderField: "anthropic-beta")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await URLSession.shared.data(for: request)
        return (data, try #require(response as? HTTPURLResponse))
    }

    private func json(_ data: Data) throws -> [String: Any] {
        try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    /// The SSE body as (event name, payload) pairs, checking each payload's
    /// `type` matches its event line.
    private func events(_ data: Data) throws -> [(event: String, payload: [String: Any])] {
        let text = String(decoding: data, as: UTF8.self)
        return try text.components(separatedBy: "\n\n").filter { !$0.isEmpty }.map { frame in
            let lines = frame.components(separatedBy: "\n")
            #expect(lines.count == 2)
            let event = String(try #require(lines.first).dropFirst("event: ".count))
            let payload = try json(Data(try #require(lines.last).dropFirst("data: ".count).utf8))
            #expect(payload["type"] as? String == event)
            return (event, payload)
        }
    }

    /// Each event as a short label: the event name, plus the block type for a
    /// block start and the delta type for a block delta, with the index.
    private func labels(_ events: [(event: String, payload: [String: Any])]) -> [String] {
        events.map { event, payload in
            let index = (payload["index"] as? Int).map { "@\($0)" } ?? ""
            switch event {
            case "content_block_start":
                return "start:\((payload["content_block"] as? [String: Any])?["type"] as? String ?? "?")\(index)"
            case "content_block_delta":
                return "delta:\((payload["delta"] as? [String: Any])?["type"] as? String ?? "?")\(index)"
            case "content_block_stop":
                return "stop\(index)"
            default:
                return event
            }
        }
    }

    private func deltaTexts(_ events: [(event: String, payload: [String: Any])], key: String) -> String {
        events.compactMap { ($0.payload["delta"] as? [String: Any])?[key] as? String }.joined()
    }

    private let weatherCall = ToolCallRequest(id: "toolu_01", name: "get_weather", arguments: ["city": .string("Paris")])

    private let weatherTool: [String: Any] = [
        "name": "get_weather", "description": "Weather.",
        "input_schema": ["type": "object", "properties": ["city": ["type": "string"]]],
    ]

    /// A reasoning model's turn: reasoning (the prompt opened `<think>`), the
    /// blank lines after `</think>`, "Hello", then a tool call.
    private var thinkThenTextThenTool: [GenerateChunk] {
        [
            GenerateChunk(text: "Let me "),
            GenerateChunk(text: "think.</think>\n\n"),
            GenerateChunk(text: "Hello"),
            GenerateChunk(text: "", finishReason: .toolCalls,
                          usage: TokenUsage(promptTokens: 9, completionTokens: 4), toolCalls: [weatherCall]),
        ]
    }

    /// The fields Claude Code sends on every request besides the turns.
    private var claudeCodeFields: [String: Any] {
        [
            "model": Self.model,
            "max_tokens": 32_000,
            "system": [
                ["type": "text", "text": "x-anthropic-billing-header: cc_version=2.1"],
                ["type": "text", "text": "You are Claude Code.", "cache_control": ["type": "ephemeral"]],
                ["type": "text", "text": "Tool use rules.", "cache_control": ["type": "ephemeral"]],
            ],
            "tools": [weatherTool],
            "metadata": ["user_id": "user_abc"],
            "thinking": ["type": "adaptive", "display": "omitted"],
            "context_management": ["edits": [["type": "clear_thinking_20251015", "keep": "all"]]],
            "safeguards": ["mode": "default"],
            "output_config": ["effort": "high"],
        ]
    }

    // MARK: System turns

    @Test
    func claudeCodeFirstTurnSystemTurnReachesTheEngine() async throws {
        let engine = ScriptedEngine(chunks: [GenerateChunk(text: "ok", finishReason: .stop)])
        let (server, port) = try await start(engine, port: 22_000)
        var body = claudeCodeFields
        body["stream"] = true
        body["messages"] = [
            ["role": "user", "content": [
                ["type": "text", "text": "<system-reminder>skills</system-reminder>"],
                ["type": "text", "text": "<system-reminder>context</system-reminder>"],
                ["type": "text", "text": "Run echo hi", "cache_control": ["type": "ephemeral"]],
            ]],
            ["role": "system", "content": [
                ["type": "text", "text": "<system-reminder>CLAUDE.md and memory</system-reminder>",
                 "cache_control": ["type": "ephemeral"]],
            ]],
        ]
        let (_, response) = try await post(port: port, body: body)
        let captured = await engine.capturedRequest
        await server.stop()

        #expect(response.statusCode == 200)
        let request = try #require(captured)
        #expect(request.systemPrompt
            == "x-anthropic-billing-header: cc_version=2.1\nYou are Claude Code.\nTool use rules.")
        try #require(request.messages.map(\.role) == [.user])
        #expect(request.messages[0].content == "<system-reminder>skills</system-reminder>\n"
            + "<system-reminder>context</system-reminder>\nRun echo hi\n\n"
            + "<system-reminder>CLAUDE.md and memory</system-reminder>")
        #expect(request.tools?.count == 1)
    }

    @Test
    func claudeCodeSecondTurnShapeReachesTheEngine() async throws {
        let engine = ScriptedEngine(chunks: [GenerateChunk(text: "Done.", finishReason: .stop)])
        let (server, port) = try await start(engine, port: 22_030)
        var body = claudeCodeFields
        body["messages"] = [
            ["role": "user", "content": [["type": "text", "text": "Run echo hi"]]],
            ["role": "system", "content": [["type": "text", "text": "<system-reminder>memory</system-reminder>"]]],
            ["role": "assistant", "content": [
                ["type": "thinking", "thinking": "I should run it.", "signature": "macmlx"],
                ["type": "tool_use", "id": "toolu_01", "name": "Bash", "input": ["command": "echo hi"]],
            ]],
            ["role": "user", "content": [
                ["type": "tool_result", "tool_use_id": "toolu_01", "content": "hi", "is_error": false],
            ]],
            ["role": "system", "content": [["type": "text", "text": "<total_tokens>9000 tokens left</total_tokens>"]]],
        ]
        let (_, response) = try await post(port: port, body: body)
        let captured = await engine.capturedRequest
        await server.stop()

        #expect(response.statusCode == 200)
        let request = try #require(captured)
        try #require(request.messages.map(\.role) == [.user, .assistant, .tool, .user])
        // The first turn renders as it did on turn one: the reminder stays put.
        #expect(request.messages[0].content == "Run echo hi\n\n<system-reminder>memory</system-reminder>")
        // Replayed thinking is ignored; the call and its result pair up.
        #expect(request.messages[1].content == "")
        #expect(request.messages[1].toolCalls?.map(\.id) == ["toolu_01"])
        #expect(request.messages[2].toolCallID == "toolu_01")
        #expect(request.messages[2].content == "hi")
        #expect(request.messages[3].content == "<total_tokens>9000 tokens left</total_tokens>")
    }

    @Test
    func ignoredFieldsDoNotFailTheRequest() async throws {
        let engine = ScriptedEngine(chunks: [GenerateChunk(text: "ok", finishReason: .stop)])
        let (server, port) = try await start(engine, port: 22_060)
        var body = claudeCodeFields
        body["messages"] = [
            ["role": "user", "content": [["type": "text", "text": "hi", "cache_control": ["type": "ephemeral"]]]],
            ["role": "system", "content": "a string system turn"],
            ["role": "system", "content": [
                ["type": "image", "source": ["type": "base64", "media_type": "image/png", "data": "AAAA"]],
            ]],
        ]
        body["top_k"] = 40
        body["stop_sequences"] = ["\n\nHuman:"]
        body["service_tier"] = "auto"
        body["tool_choice"] = ["type": "auto", "disable_parallel_tool_use": true]
        body["thinking"] = ["type": "enabled", "budget_tokens": 1024]
        let (data, response) = try await post(port: port, body: body)
        let captured = await engine.capturedRequest
        await server.stop()

        #expect(response.statusCode == 200, "\(String(decoding: data, as: UTF8.self))")
        // A system turn's image is a placeholder, never decoded.
        #expect(captured?.messages.first?.content
            == "hi\n\na string system turn\n\n[image omitted from a system message]")
        #expect(captured?.messages.first?.images.isEmpty == true)
    }

    // MARK: Thinking

    @Test
    func thinkingDisabledTurnsTheTemplateOff() async throws {
        let engine = ScriptedEngine(chunks: [GenerateChunk(text: "ok", finishReason: .stop)])
        let (server, port) = try await start(engine, port: 22_090)
        let base: [String: Any] = [
            "model": Self.model, "max_tokens": 64, "messages": [["role": "user", "content": "hi"]],
        ]
        _ = try await post(port: port, body: base.merging(["thinking": ["type": "disabled"]]) { $1 })
        let off = await engine.capturedRequest?.templateKwargs?["enable_thinking"]
        _ = try await post(port: port, body: base.merging(["thinking": ["type": "adaptive"]]) { $1 })
        let adaptive = await engine.capturedRequest?.templateKwargs?["enable_thinking"]
        _ = try await post(port: port, body: base)
        let absent = await engine.capturedRequest?.templateKwargs?["enable_thinking"]
        await server.stop()

        #expect(off == .bool(false))
        #expect(adaptive == nil)
        #expect(absent == nil)
    }

    @Test
    func adaptiveThinkingStreamsAThinkingBlockFirst() async throws {
        let engine = ScriptedEngine(chunks: thinkThenTextThenTool, opensThink: true)
        let (server, port) = try await start(engine, port: 22_120)
        let (data, response) = try await post(port: port, body: [
            "model": Self.model, "max_tokens": 64, "stream": true, "tools": [weatherTool],
            "thinking": ["type": "adaptive"],
            "messages": [["role": "user", "content": "Weather in Paris?"]],
        ])
        await server.stop()

        #expect(response.statusCode == 200)
        let events = try events(data)
        let labels = labels(events)
        // Collapse runs of the same delta for the sequence check.
        let sequence = labels.enumerated().filter { $0.offset == 0 || labels[$0.offset - 1] != $0.element }.map(\.element)
        #expect(sequence == [
            "message_start",
            "start:thinking@0", "delta:thinking_delta@0", "delta:signature_delta@0", "stop@0",
            "start:text@1", "delta:text_delta@1", "stop@1",
            "start:tool_use@2", "delta:input_json_delta@2", "stop@2",
            "message_delta", "message_stop",
        ])
        let thinkingStart = try #require(events.first { $0.event == "content_block_start" }?.payload["content_block"]
            as? [String: Any])
        #expect(thinkingStart["thinking"] as? String == "")
        #expect(thinkingStart["signature"] as? String == "")
        #expect(deltaTexts(events, key: "thinking") == "Let me think.")
        #expect((deltaTexts(events, key: "signature")).isEmpty == false)
        // The answer starts at its first non-whitespace text.
        #expect(deltaTexts(events, key: "text") == "Hello")
        let messageDelta = try #require(events.first { $0.event == "message_delta" }?.payload)
        #expect((messageDelta["delta"] as? [String: Any])?["stop_reason"] as? String == "tool_use")
    }

    @Test
    func adaptiveThinkingNonStreamingHasAThinkingBlock() async throws {
        let engine = ScriptedEngine(chunks: thinkThenTextThenTool, opensThink: true)
        let (server, port) = try await start(engine, port: 22_150)
        let (data, response) = try await post(port: port, body: [
            "model": Self.model, "max_tokens": 64, "tools": [weatherTool],
            "thinking": ["type": "enabled", "budget_tokens": 2048],
            "messages": [["role": "user", "content": "Weather in Paris?"]],
        ])
        await server.stop()

        #expect(response.statusCode == 200)
        let content = try #require(try json(data)["content"] as? [[String: Any]])
        #expect(content.compactMap { $0["type"] as? String } == ["thinking", "text", "tool_use"])
        #expect(content[0]["thinking"] as? String == "Let me think.")
        #expect((content[0]["signature"] as? String)?.isEmpty == false)
        #expect(content[1]["text"] as? String == "Hello")
        #expect(content[2]["id"] as? String == "toolu_01")
    }

    @Test
    func omittedThinkingDropsReasoningAndBlankText() async throws {
        // Claude Code's setting. A tool-only reply: the blank lines after
        // `</think>` must not open a text block before the tool call.
        let engine = ScriptedEngine(chunks: [
            GenerateChunk(text: "Plan the call.</think>"),
            GenerateChunk(text: "\n\n"),
            GenerateChunk(text: "", finishReason: .toolCalls,
                          usage: TokenUsage(promptTokens: 9, completionTokens: 4), toolCalls: [weatherCall]),
        ], opensThink: true)
        let (server, port) = try await start(engine, port: 22_180)
        let (data, response) = try await post(port: port, body: [
            "model": Self.model, "max_tokens": 64, "stream": true, "tools": [weatherTool],
            "thinking": ["type": "adaptive", "display": "omitted"],
            "messages": [["role": "user", "content": "Weather in Paris?"]],
        ])
        await server.stop()

        #expect(response.statusCode == 200)
        let events = try events(data)
        #expect(labels(events) == [
            "message_start",
            "start:tool_use@0", "delta:input_json_delta@0", "stop@0",
            "message_delta", "message_stop",
        ])
        #expect(!String(decoding: data, as: UTF8.self).contains("Plan the call"))
    }

    @Test
    func omittedThinkingNonStreaming() async throws {
        let engine = ScriptedEngine(chunks: thinkThenTextThenTool, opensThink: true)
        let (server, port) = try await start(engine, port: 22_210)
        let (data, response) = try await post(port: port, body: [
            "model": Self.model, "max_tokens": 64, "tools": [weatherTool],
            "thinking": ["type": "adaptive", "display": "omitted"],
            "messages": [["role": "user", "content": "Weather in Paris?"]],
        ])
        await server.stop()

        #expect(response.statusCode == 200)
        let content = try #require(try json(data)["content"] as? [[String: Any]])
        #expect(content.compactMap { $0["type"] as? String } == ["text", "tool_use"])
        #expect(content[0]["text"] as? String == "Hello")
    }

    @Test
    func thinkingAbsentReturnsNoReasoning() async throws {
        let engine = ScriptedEngine(chunks: [
            GenerateChunk(text: "Let me think.</think>\n\nHello", finishReason: .stop,
                          usage: TokenUsage(promptTokens: 9, completionTokens: 4)),
        ], opensThink: true)
        let (server, port) = try await start(engine, port: 22_240)
        let body: [String: Any] = [
            "model": Self.model, "max_tokens": 64, "messages": [["role": "user", "content": "hi"]],
        ]
        let (data, response) = try await post(port: port, body: body)
        let (streamData, _) = try await post(port: port, body: body.merging(["stream": true]) { $1 })
        await server.stop()

        #expect(response.statusCode == 200)
        let content = try #require(try json(data)["content"] as? [[String: Any]])
        #expect(content.count == 1)
        #expect(content[0]["type"] as? String == "text")
        #expect(content[0]["text"] as? String == "Hello")
        let events = try events(streamData)
        #expect(labels(events) == [
            "message_start", "start:text@0", "delta:text_delta@0", "stop@0", "message_delta", "message_stop",
        ])
        #expect(deltaTexts(events, key: "text") == "Hello")
    }

    // MARK: tool_choice

    @Test
    func toolChoiceModes() async throws {
        let engine = ScriptedEngine(chunks: [GenerateChunk(text: "ok", finishReason: .stop)])
        let (server, port) = try await start(engine, port: 22_270)
        let base: [String: Any] = [
            "model": Self.model, "max_tokens": 64, "tools": [weatherTool],
            "messages": [["role": "user", "content": "Weather in Paris?"]],
        ]
        func offered(_ choice: [String: Any]) async throws -> (status: Int, tools: Int?) {
            let (_, response) = try await post(port: port, body: base.merging(["tool_choice": choice]) { $1 })
            return (response.statusCode, await engine.capturedRequest?.tools?.count)
        }
        let none = try await offered(["type": "none"])
        let any = try await offered(["type": "any"])
        let tool = try await offered(["type": "tool", "name": "get_weather"])
        let auto = try await offered(["type": "auto"])
        let (namelessData, nameless) = try await post(
            port: port, body: base.merging(["tool_choice": ["type": "tool"]]) { $1 })
        let (_, unknown) = try await post(port: port, body: base.merging(["tool_choice": ["type": "sometimes"]]) { $1 })
        await server.stop()

        #expect(none.status == 200 && none.tools == nil)
        #expect(any.status == 200 && any.tools == 1)
        #expect(tool.status == 200 && tool.tools == 1)
        #expect(auto.status == 200 && auto.tools == 1)
        #expect(nameless.statusCode == 400)
        #expect(unknown.statusCode == 400)
        let error = try #require(try json(namelessData)["error"] as? [String: Any])
        #expect(error["type"] as? String == "invalid_request_error")
    }

    // MARK: Errors

    @Test
    func unknownModelIsAnAnthropic404() async throws {
        let engine = ScriptedEngine(chunks: [])
        let (server, port) = try await start(engine, port: 22_300, resolver: { _ in nil })
        let (data, response) = try await post(port: port, body: [
            "model": "never-existed", "max_tokens": 64, "messages": [["role": "user", "content": "hi"]],
        ])
        await server.stop()

        #expect(response.statusCode == 404)
        let body = try json(data)
        #expect(body["type"] as? String == "error")
        let error = try #require(body["error"] as? [String: Any])
        #expect(error["type"] as? String == "not_found_error")
        #expect((error["message"] as? String)?.hasPrefix("Model not found: never-existed.") == true)
    }

    @Test
    func unknownModelStreamingIsAnAnthropic404() async throws {
        let engine = ScriptedEngine(chunks: [])
        let (server, port) = try await start(engine, port: 22_330, resolver: { _ in nil })
        let (data, response) = try await post(port: port, body: [
            "model": "never-existed", "max_tokens": 64, "stream": true,
            "messages": [["role": "user", "content": "hi"]],
        ])
        await server.stop()

        #expect(response.statusCode == 404)
        let error = try #require(try json(data)["error"] as? [String: Any])
        #expect(error["type"] as? String == "not_found_error")
    }

    @Test
    func loadFailureIsAnAnthropic500() async throws {
        let engine = ScriptedEngine(chunks: [])
        let broken = localModel(Self.brokenModel)
        let (server, port) = try await start(engine, port: 22_360, resolver: { $0 == Self.brokenModel ? broken : nil })
        let (data, response) = try await post(port: port, body: [
            "model": Self.brokenModel, "max_tokens": 64, "messages": [["role": "user", "content": "hi"]],
        ])
        await server.stop()

        #expect(response.statusCode == 500)
        let body = try json(data)
        #expect(body["type"] as? String == "error")
        let error = try #require(body["error"] as? [String: Any])
        #expect(error["type"] as? String == "api_error")
        #expect((error["message"] as? String)?.hasPrefix("Failed to load \(Self.brokenModel)") == true)
    }

    @Test
    func missingMaxTokensIs400() async throws {
        let engine = ScriptedEngine(chunks: [GenerateChunk(text: "ok", finishReason: .stop)])
        let (server, port) = try await start(engine, port: 22_390)
        let (data, response) = try await post(port: port, body: [
            "model": Self.model, "messages": [["role": "user", "content": "hi"]],
        ])
        let captured = await engine.capturedRequest
        await server.stop()

        #expect(response.statusCode == 400)
        let error = try #require(try json(data)["error"] as? [String: Any])
        #expect(error["message"] as? String == "`max_tokens` is required.")
        #expect(captured == nil)
    }
}
