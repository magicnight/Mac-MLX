import Foundation
import Testing

@testable import MacMLXCore

// MARK: - Reasoning round trip over the three chat endpoints
//
// A real HummingbirdServer over HTTP, backed by a scripted stub engine — no
// model, no Metal. Covers the reasoning a client replays reaching the engine
// as `ChatMessage.reasoning` (chat `reasoning_content` / `reasoning`,
// Anthropic `thinking` blocks, Responses `reasoning` items), and the server
// returning its reasoning in the opaque fields clients replay verbatim
// (Responses `encrypted_content` when `include` asks, an Anthropic block's
// `signature`), each read back on the next turn.
//
// Port assignments (23_000 range, spaced by 30 for port-retry headroom):
//   chatReasoningContentReachesTheEngine            : 23_000
//   anthropicThinkingBlocksReachTheEngine           : 23_030
//   anthropicOmittedReplyRoundTrips                 : 23_060
//   anthropicStreamedReplyRoundTrips                : 23_090
//   responsesEncryptedContentNonStreaming           : 23_120
//   responsesEncryptedContentStreaming              : 23_150
//   responsesEncryptedOnlyReplayRoundTrips          : 23_180
//   anthropicNonStringThinkingFieldsAreIgnored      : 23_210

@Suite("ReasoningHistoryEndpoint")
struct ReasoningHistoryEndpointTests {

    // MARK: Stub

    /// Plays back a fixed chunk script, reports whether the prompt opens a
    /// think block, and records the request it was asked to generate.
    private actor ScriptedEngine: InferenceEngine {
        nonisolated let engineID: EngineID = .mlxSwift
        private(set) var status: EngineStatus = .idle
        private(set) var loadedModel: LocalModel?
        let version = "reasoning-stub-1"
        private(set) var capturedRequest: GenerateRequest?

        private let chunks: [GenerateChunk]
        private let opensThink: Bool

        init(chunks: [GenerateChunk], opensThink: Bool = true) {
            self.chunks = chunks
            self.opensThink = opensThink
        }

        func load(_ model: LocalModel) async throws {
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

    private func start(_ engine: ScriptedEngine, port: Int) async throws -> (HummingbirdServer, Int) {
        try await engine.load(LocalModel(
            id: Self.model, displayName: Self.model, directory: URL(filePath: "/tmp"), sizeBytes: 0,
            format: .mlx, quantization: nil, parameterCount: nil, architecture: nil))
        let server = HummingbirdServer(engine: engine)
        return (server, try await server.start(preferredPort: port))
    }

    private func post(port: Int, path: String, body: [String: Any]) async throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: try #require(URL(string: "http://127.0.0.1:\(port)\(path)")))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await URLSession.shared.data(for: request)
        return (data, try #require(response as? HTTPURLResponse))
    }

    private func json(_ data: Data) throws -> [String: Any] {
        try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    /// The payloads of an SSE body (`event:` + `data:` frames).
    private func payloads(_ data: Data) throws -> [[String: Any]] {
        try String(decoding: data, as: UTF8.self).components(separatedBy: "\n\n").filter { !$0.isEmpty }.map {
            let line = try #require($0.components(separatedBy: "\n").last)
            return try json(Data(line.dropFirst("data: ".count).utf8))
        }
    }

    private let call = ToolCallRequest(id: "call_1", name: "get_weather", arguments: ["city": .string("Paris")])

    /// A thinking model's tool-calling turn: reasoning, then the call.
    private var reasonThenCall: [GenerateChunk] {
        [
            GenerateChunk(text: "Look up "),
            GenerateChunk(text: "Paris.</think>\n\n"),
            GenerateChunk(text: "", finishReason: .toolCalls,
                          usage: TokenUsage(promptTokens: 9, completionTokens: 4), toolCalls: [call]),
        ]
    }

    private let responsesTool: [String: Any] = [
        "type": "function", "name": "get_weather", "description": "Weather.",
        "parameters": ["type": "object", "properties": ["city": ["type": "string"]]],
    ]

    private let anthropicTool: [String: Any] = [
        "name": "get_weather", "description": "Weather.",
        "input_schema": ["type": "object", "properties": ["city": ["type": "string"]]],
    ]

    private var toolUse: [String: Any] {
        ["type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": ["city": "Paris"]]
    }

    private var toolResult: [String: Any] {
        ["role": "user", "content": [["type": "tool_result", "tool_use_id": "toolu_1", "content": "Sunny"]]]
    }

    // MARK: /v1/chat/completions

    @Test
    func chatReasoningContentReachesTheEngine() async throws {
        let engine = ScriptedEngine(chunks: [GenerateChunk(text: "ok", finishReason: .stop)], opensThink: false)
        let (server, port) = try await start(engine, port: 23_000)
        func reasoning(_ messages: [[String: Any]]) async throws -> [String?] {
            let (data, response) = try await post(port: port, path: "/v1/chat/completions", body: [
                "model": Self.model, "messages": messages,
            ])
            #expect(response.statusCode == 200, "\(String(decoding: data, as: UTF8.self))")
            return await engine.capturedRequest?.messages.map(\.reasoning) ?? []
        }
        let toolCall: [String: Any] = [
            "id": "call_1", "type": "function",
            "function": ["name": "get_weather", "arguments": "{\"city\":\"Paris\"}"],
        ]
        let primary = try await reasoning([
            ["role": "user", "content": "Weather?", "reasoning_content": "a user's is ignored"],
            ["role": "assistant", "content": NSNull(), "tool_calls": [toolCall],
             "reasoning_content": "Look up Paris.", "reasoning": "not this one"],
            ["role": "tool", "tool_call_id": "call_1", "content": "Sunny"],
            ["role": "assistant", "content": "Sunny.", "reasoning": "OpenRouter's spelling."],
            ["role": "assistant", "content": "x", "reasoning": ["text": "not a string"], "reasoning_content": ""],
            ["role": "user", "content": "Thanks"],
        ])
        await server.stop()

        #expect(primary == [nil, "Look up Paris.", nil, "OpenRouter's spelling.", nil, nil])
    }

    // MARK: /v1/messages

    @Test
    func anthropicThinkingBlocksReachTheEngine() async throws {
        let engine = ScriptedEngine(chunks: [GenerateChunk(text: "ok", finishReason: .stop)], opensThink: false)
        let (server, port) = try await start(engine, port: 23_030)
        let (data, response) = try await post(port: port, path: "/v1/messages", body: [
            "model": Self.model, "max_tokens": 64,
            "messages": [
                ["role": "user", "content": "Weather?"],
                ["role": "assistant", "content": [
                    ["type": "thinking", "thinking": "First block.", "signature": "anthropic-sig"],
                    ["type": "redacted_thinking", "data": "opaque"],
                    ["type": "thinking", "thinking": "", "signature": ReasoningCarrier.encode("Second, from the signature.")],
                    toolUse,
                ]],
                toolResult,
                ["role": "assistant", "content": [
                    ["type": "thinking", "thinking": "", "signature": "a real Anthropic signature"],
                    ["type": "text", "text": "Sunny."],
                ]],
                ["role": "user", "content": "Thanks"],
            ],
        ])
        let captured = await engine.capturedRequest
        await server.stop()

        #expect(response.statusCode == 200, "\(String(decoding: data, as: UTF8.self))")
        let messages = try #require(captured?.messages)
        try #require(messages.map(\.role) == [.user, .assistant, .tool, .assistant, .user])
        #expect(messages[1].reasoning == "First block.\n\nSecond, from the signature.")
        // A foreign signature carries nothing macMLX can read.
        #expect(messages.map(\.reasoning) == [nil, "First block.\n\nSecond, from the signature.", nil, nil, nil])
    }

    /// A client that sends `thinking` or `signature` as something other than
    /// a string is served as before these fields were read: they are ignored.
    @Test
    func anthropicNonStringThinkingFieldsAreIgnored() async throws {
        let engine = ScriptedEngine(chunks: [GenerateChunk(text: "ok", finishReason: .stop)], opensThink: false)
        let (server, port) = try await start(engine, port: 23_210)
        let (data, response) = try await post(port: port, path: "/v1/messages", body: [
            "model": Self.model, "max_tokens": 64,
            "messages": [
                ["role": "user", "content": "Weather?"],
                ["role": "assistant", "content": [
                    ["type": "thinking", "thinking": 42, "signature": ["opaque": true]],
                    ["type": "thinking", "thinking": NSNull(), "signature": ReasoningCarrier.encode("Still read.")],
                    toolUse,
                ]],
                toolResult,
            ],
        ])
        let captured = await engine.capturedRequest
        await server.stop()

        #expect(response.statusCode == 200, "\(String(decoding: data, as: UTF8.self))")
        #expect(captured?.messages.map(\.reasoning) == [nil, "Still read.", nil])
    }

    /// Claude Code's setting: the reply's text-less thinking block, replayed
    /// as Claude Code replays it, gives the reasoning back.
    @Test
    func anthropicOmittedReplyRoundTrips() async throws {
        let engine = ScriptedEngine(chunks: reasonThenCall)
        let (server, port) = try await start(engine, port: 23_060)
        let base: [String: Any] = [
            "model": Self.model, "max_tokens": 64, "tools": [anthropicTool],
            "thinking": ["type": "adaptive", "display": "omitted"],
        ]
        let question: [String: Any] = ["role": "user", "content": "Weather?"]
        let (data, _) = try await post(
            port: port, path: "/v1/messages", body: base.merging(["messages": [question]]) { $1 })
        let content = try #require(try json(data)["content"] as? [[String: Any]])
        let useID = try #require(content.last?["id"] as? String)
        let (replayData, replay) = try await post(port: port, path: "/v1/messages", body: base.merging([
            "messages": [
                question, ["role": "assistant", "content": content],
                ["role": "user", "content": [["type": "tool_result", "tool_use_id": useID, "content": "Sunny"]]],
            ],
        ]) { $1 })
        #expect(replay.statusCode == 200, "\(String(decoding: replayData, as: UTF8.self))")
        let replayed = await engine.capturedRequest
        await server.stop()

        #expect(content.compactMap { $0["type"] as? String } == ["thinking", "tool_use"])
        #expect(content[0]["thinking"] as? String == "")
        #expect(replayed?.messages.map(\.reasoning) == [nil, "Look up Paris.", nil])
    }

    @Test
    func anthropicStreamedReplyRoundTrips() async throws {
        let engine = ScriptedEngine(chunks: reasonThenCall)
        let (server, port) = try await start(engine, port: 23_090)
        var signatures: [String] = []
        for display in ["omitted", "summarized"] {
            let (data, _) = try await post(port: port, path: "/v1/messages", body: [
                "model": Self.model, "max_tokens": 64, "stream": true, "tools": [anthropicTool],
                "thinking": ["type": "enabled", "budget_tokens": 1024, "display": display],
                "messages": [["role": "user", "content": "Weather?"]],
            ])
            signatures.append(try payloads(data).compactMap {
                (($0["delta"] as? [String: Any])?["signature"]) as? String
            }.joined())
        }
        await server.stop()

        #expect(signatures.map { ReasoningCarrier.decode($0) } == ["Look up Paris.", "Look up Paris."])
    }

    // MARK: /v1/responses

    private func reasoningItem(_ output: [[String: Any]]) throws -> [String: Any] {
        try #require(output.first { $0["type"] as? String == "reasoning" })
    }

    @Test
    func responsesEncryptedContentNonStreaming() async throws {
        let engine = ScriptedEngine(chunks: reasonThenCall)
        let (server, port) = try await start(engine, port: 23_120)
        let base: [String: Any] = ["model": Self.model, "input": "Weather?", "tools": [responsesTool]]
        let (included, _) = try await post(port: port, path: "/v1/responses", body: base.merging([
            "include": ["reasoning.encrypted_content"],
        ]) { $1 })
        let (plain, _) = try await post(port: port, path: "/v1/responses", body: base)
        // An `include` of an unexpected shape is ignored, not a 400.
        let (_, odd) = try await post(port: port, path: "/v1/responses", body: base.merging([
            "include": "reasoning.encrypted_content",
        ]) { $1 })
        await server.stop()

        let item = try reasoningItem(try #require(try json(included)["output"] as? [[String: Any]]))
        #expect(ReasoningCarrier.decode(item["encrypted_content"] as? String) == "Look up Paris.")
        let plainItem = try reasoningItem(try #require(try json(plain)["output"] as? [[String: Any]]))
        #expect(plainItem["encrypted_content"] == nil)
        #expect(odd.statusCode == 200)
    }

    @Test
    func responsesEncryptedContentStreaming() async throws {
        let engine = ScriptedEngine(chunks: reasonThenCall)
        let (server, port) = try await start(engine, port: 23_150)
        let base: [String: Any] = ["model": Self.model, "input": "Weather?", "stream": true, "tools": [responsesTool]]
        let (included, _) = try await post(port: port, path: "/v1/responses", body: base.merging([
            "include": ["reasoning.encrypted_content"],
        ]) { $1 })
        let (plain, _) = try await post(port: port, path: "/v1/responses", body: base)
        await server.stop()

        /// The reasoning item as `output_item.done` and as the terminal
        /// response carry it, and as `output_item.added` announced it.
        func items(_ data: Data) throws -> (added: [String: Any], done: [String: Any], completed: [String: Any]) {
            let events = try payloads(data)
            let added = try #require(events.first {
                $0["type"] as? String == "response.output_item.added"
            }?["item"] as? [String: Any])
            let done = try #require(events.first {
                $0["type"] as? String == "response.output_item.done"
            }?["item"] as? [String: Any])
            let completed = try #require(events.last?["response"] as? [String: Any])
            return (added, done, try reasoningItem(try #require(completed["output"] as? [[String: Any]])))
        }
        let withInclude = try items(included)
        #expect(withInclude.done["type"] as? String == "reasoning")
        #expect(withInclude.added["encrypted_content"] == nil)
        #expect(ReasoningCarrier.decode(withInclude.done["encrypted_content"] as? String) == "Look up Paris.")
        #expect(ReasoningCarrier.decode(withInclude.completed["encrypted_content"] as? String) == "Look up Paris.")
        let without = try items(plain)
        #expect(without.done["encrypted_content"] == nil)
        #expect(without.completed["encrypted_content"] == nil)
    }

    /// A client that replays only `encrypted_content` (no `content`) still
    /// gives the reasoning back.
    @Test
    func responsesEncryptedOnlyReplayRoundTrips() async throws {
        let engine = ScriptedEngine(chunks: reasonThenCall)
        let (server, port) = try await start(engine, port: 23_180)
        let question: [String: Any] = ["type": "message", "role": "user", "content": "Weather?"]
        let base: [String: Any] = [
            "model": Self.model, "tools": [responsesTool], "include": ["reasoning.encrypted_content"],
        ]
        let (data, _) = try await post(
            port: port, path: "/v1/responses", body: base.merging(["input": [question]]) { $1 })
        let output = try #require(try json(data)["output"] as? [[String: Any]])
        var item = try reasoningItem(output)
        item["content"] = NSNull()
        item["summary"] = [] as [Any]
        let call = try #require(output.first { $0["type"] as? String == "function_call" })
        _ = try await post(port: port, path: "/v1/responses", body: base.merging([
            "input": [question, item, call,
                      ["type": "function_call_output", "call_id": call["call_id"] ?? "", "output": "Sunny"]],
        ]) { $1 })
        let replayed = await engine.capturedRequest
        await server.stop()

        #expect(replayed?.messages.map(\.reasoning) == [nil, "Look up Paris.", nil])
    }
}
