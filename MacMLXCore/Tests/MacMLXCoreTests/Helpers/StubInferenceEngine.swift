import Foundation
@testable import MacMLXCore

/// Minimal in-process `InferenceEngine` used to verify the protocol shape and
/// to stand in for real engines in higher-layer tests.
public actor StubInferenceEngine: InferenceEngine {
    public let engineID: EngineID
    public private(set) var status: EngineStatus = .idle
    public private(set) var loadedModel: LocalModel?
    public let version = "stub-1"

    public init(engineID: EngineID) {
        self.engineID = engineID
    }

    public func load(_ model: LocalModel) async throws {
        status = .loading(model: model.id)
        loadedModel = model
        status = .ready(model: model.id)
    }

    public func unload() async throws {
        loadedModel = nil
        status = .idle
    }

    public nonisolated func generate(
        _ request: GenerateRequest
    ) -> AsyncThrowingStream<GenerateChunk, Error> {
        AsyncThrowingStream { continuation in
            continuation.yield(GenerateChunk(text: "stub-"))
            continuation.yield(GenerateChunk(
                text: "response",
                finishReason: .stop,
                usage: TokenUsage(promptTokens: 1, completionTokens: 2)
            ))
            continuation.finish()
        }
    }

    public func healthCheck() async -> Bool { true }
}
