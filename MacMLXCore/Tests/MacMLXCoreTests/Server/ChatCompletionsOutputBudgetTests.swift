import Foundation
import Testing
@testable import MacMLXCore

/// The output budget of a chat completion that names none. Xcode 27's
/// locally hosted provider sends no `max_tokens`, and at the old 2048 default
/// Qwen3.6 spent all of it reasoning, so Xcode showed no answer. With no
/// limit given the budget is `defaultMaxOutputTokens`, as on `/v1/responses`.
///
/// Port assignments (spaced by 10):
///   requestWithoutALimitGetsTheDefaultBudget : 23_400
///   explicitLimitsAreHonored                 : 23_410
@Suite("ChatCompletionsOutputBudget")
struct ChatCompletionsOutputBudgetTests {

    /// Records the parameters of each request it is given.
    private actor CapturingStubEngine: InferenceEngine {
        nonisolated let engineID: EngineID = .mlxSwift
        private(set) var status: EngineStatus = .idle
        private(set) var loadedModel: LocalModel?
        let version = "output-budget-1"
        private(set) var maxTokens: [Int] = []

        func load(_ model: LocalModel) async throws {
            status = .loading(model: model.id)
            loadedModel = model
            status = .ready(model: model.id)
        }

        func unload() async throws {
            loadedModel = nil
            status = .idle
        }

        nonisolated func generate(_ request: GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error> {
            AsyncThrowingStream { continuation in
                Task { [weak self] in
                    await self?.record(request.parameters.maxTokens)
                    continuation.yield(GenerateChunk(
                        text: "ok", finishReason: .stop, usage: TokenUsage(promptTokens: 1, completionTokens: 1)))
                    continuation.finish()
                }
            }
        }

        private func record(_ value: Int) { maxTokens.append(value) }

        func healthCheck() async -> Bool { true }
    }

    private func post(_ url: URL, _ extra: [String: Any], stream: Bool) async throws {
        var body: [String: Any] = [
            "model": "stub-model",
            "messages": [["role": "user", "content": "Hi"]],
            "stream": stream,
        ]
        body.merge(extra) { _, new in new }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (_, response) = try await URLSession.shared.data(for: req)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
    }

    private func serve(_ engine: CapturingStubEngine, port: Int) async throws -> (HummingbirdServer, URL) {
        try await engine.load(LocalModel(
            id: "stub-model", displayName: "stub-model", directory: URL(filePath: "/tmp"), sizeBytes: 0,
            format: .mlx, quantization: nil, parameterCount: nil, architecture: nil))
        let server = HummingbirdServer(engineProvider: { engine })
        let bound = try await server.start(preferredPort: port)
        return (server, try #require(URL(string: "http://127.0.0.1:\(bound)/v1/chat/completions")))
    }

    @Test
    func requestWithoutALimitGetsTheDefaultBudget() async throws {
        let engine = CapturingStubEngine()
        let (server, url) = try await serve(engine, port: 23_400)
        try await post(url, [:], stream: true)
        try await post(url, [:], stream: false)
        await server.stop()

        #expect(HummingbirdServer.defaultMaxOutputTokens == 32_768)
        #expect(await engine.maxTokens == [32_768, 32_768])
    }

    @Test
    func explicitLimitsAreHonored() async throws {
        let engine = CapturingStubEngine()
        let (server, url) = try await serve(engine, port: 23_410)
        try await post(url, ["max_tokens": 300], stream: true)
        try await post(url, ["max_completion_tokens": 200, "max_tokens": 300], stream: false)
        await server.stop()

        #expect(await engine.maxTokens == [300, 200])
    }
}
