import Foundation
import Testing

@testable import MacMLXCore

// MARK: - /v1/messages/count_tokens endpoint tests
//
// A real HummingbirdServer over HTTP, backed by stub engines — no model, no
// Metal. Covers the endpoint's plumbing: the count the engine returns, the
// request it is asked to count (the one `/v1/messages` would generate from),
// and the errors, all in Anthropic's envelope. That the MLX engine's count
// equals generation's prompt length is the gated
// `AnthropicCountTokensE2ETests`.
//
// Port assignments (22_400 range, spaced by 30):
//   countIsReturned                       : 22_420
//   toolsAndSystemTurnsReachTheCounter    : 22_450
//   unknownModelIs404                     : 22_480
//   badJSONIs400                          : 22_510
//   engineWithoutCountingIs501            : 22_540

@Suite("AnthropicCountTokens")
struct AnthropicCountTokensTests {

    /// An engine that keeps `countPromptTokens`'s protocol default.
    private actor NonCountingEngine: InferenceEngine {
        nonisolated let engineID: EngineID = .mlxSwift
        private(set) var status: EngineStatus = .idle
        private(set) var loadedModel: LocalModel?
        let version = "non-counting-1"

        func load(_ model: LocalModel) async throws {
            loadedModel = model
            status = .ready(model: model.id)
        }

        func unload() async throws {
            loadedModel = nil
            status = .idle
        }

        nonisolated func generate(_ request: GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error> {
            AsyncThrowingStream { $0.finish() }
        }

        func healthCheck() async -> Bool { true }
    }

    private static let model = "stub-model"

    private func localModel(_ id: String) -> LocalModel {
        LocalModel(
            id: id, displayName: id, directory: URL(filePath: "/tmp"), sizeBytes: 0,
            format: .mlx, quantization: nil, parameterCount: nil, architecture: nil)
    }

    private func post(port: Int, data: Data) async throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: try #require(URL(string: "http://127.0.0.1:\(port)/v1/messages/count_tokens?beta=true")))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
        request.httpBody = data
        let (body, response) = try await URLSession.shared.data(for: request)
        return (body, try #require(response as? HTTPURLResponse))
    }

    private func post(port: Int, body: [String: Any]) async throws -> (Data, HTTPURLResponse) {
        try await post(port: port, data: JSONSerialization.data(withJSONObject: body))
    }

    private func json(_ data: Data) throws -> [String: Any] {
        try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    /// The Anthropic error object of an error response, checking the envelope.
    private func anthropicError(_ data: Data) throws -> [String: Any] {
        let body = try json(data)
        #expect(body["type"] as? String == "error")
        return try #require(body["error"] as? [String: Any])
    }

    private let weatherTool: [String: Any] = [
        "name": "get_weather", "description": "Weather.",
        "input_schema": ["type": "object", "properties": ["city": ["type": "string"]]],
    ]

    @Test
    func countIsReturned() async throws {
        let engine = StubInferenceEngine(engineID: .mlxSwift, countedPromptTokens: 1234)
        try await engine.load(localModel(Self.model))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 22_420)
        // No `max_tokens`: count_tokens takes no generation fields.
        let (data, response) = try await post(port: port, body: [
            "model": Self.model, "messages": [["role": "user", "content": "Hello"]],
        ])
        await server.stop()

        #expect(response.statusCode == 200)
        #expect(try json(data) as NSDictionary == ["input_tokens": 1234] as NSDictionary)
    }

    @Test
    func toolsAndSystemTurnsReachTheCounter() async throws {
        let engine = StubInferenceEngine(engineID: .mlxSwift, countedPromptTokens: 7)
        try await engine.load(localModel(Self.model))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 22_450)
        let (_, response) = try await post(port: port, body: [
            "model": Self.model, "max_tokens": 32_000, "stream": true,
            "system": [["type": "text", "text": "You are Claude Code."]],
            "tools": [weatherTool],
            "thinking": ["type": "disabled"],
            "messages": [
                ["role": "user", "content": "Weather in Paris?"],
                ["role": "system", "content": "<system-reminder>memory</system-reminder>"],
            ],
        ])
        let counted = await engine.countedRequest
        await server.stop()

        #expect(response.statusCode == 200)
        let request = try #require(counted)
        #expect(request.systemPrompt == "You are Claude Code.")
        #expect(request.messages.map(\.content) == ["Weather in Paris?\n\n<system-reminder>memory</system-reminder>"])
        #expect(request.tools?.count == 1)
        #expect(request.templateKwargs?["enable_thinking"] == .bool(false))
    }

    @Test
    func unknownModelIs404() async throws {
        let engine = StubInferenceEngine(engineID: .mlxSwift, countedPromptTokens: 7)
        let server = HummingbirdServer(engine: engine, modelResolver: { _ in nil })
        let port = try await server.start(preferredPort: 22_480)
        let (data, response) = try await post(port: port, body: [
            "model": "never-existed", "messages": [["role": "user", "content": "Hello"]],
        ])
        await server.stop()

        #expect(response.statusCode == 404)
        let error = try anthropicError(data)
        #expect(error["type"] as? String == "not_found_error")
        #expect((error["message"] as? String)?.hasPrefix("Model not found: never-existed.") == true)
    }

    @Test
    func badJSONIs400() async throws {
        let engine = StubInferenceEngine(engineID: .mlxSwift, countedPromptTokens: 7)
        try await engine.load(localModel(Self.model))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 22_510)
        let (data, response) = try await post(port: port, data: Data("{\"model\": ".utf8))
        let (_, missingMessages) = try await post(port: port, body: ["model": Self.model])
        let counted = await engine.countedRequest
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(try anthropicError(data)["type"] as? String == "invalid_request_error")
        #expect(missingMessages.statusCode == 400)
        #expect(counted == nil)
    }

    @Test
    func engineWithoutCountingIs501() async throws {
        let engine = NonCountingEngine()
        try await engine.load(localModel(Self.model))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 22_540)
        let (data, response) = try await post(port: port, body: [
            "model": Self.model, "messages": [["role": "user", "content": "Hello"]],
        ])
        let lockFreed = await pollUntil { await !server.generationLockIsHeld }
        await server.stop()

        #expect(response.statusCode == 501)
        let error = try anthropicError(data)
        #expect(error["type"] as? String == "api_error")
        #expect(error["message"] as? String == "token counting is not supported by this engine")
        #expect(lockFreed)
    }

    /// Polls `condition` every 50 ms for at most 60 s (the lock's release
    /// runs in a detached task, so it lands shortly after the response).
    private func pollUntil(_ condition: () async -> Bool) async -> Bool {
        for _ in 0..<1_200 {
            if await condition() { return true }
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        return false
    }
}
