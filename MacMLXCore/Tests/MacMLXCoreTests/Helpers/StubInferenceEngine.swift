import Foundation
@testable import MacMLXCore

/// Minimal in-process `InferenceEngine` used to verify the protocol shape and
/// to stand in for real engines in higher-layer tests.
public actor StubInferenceEngine: InferenceEngine {
    public let engineID: EngineID
    public private(set) var status: EngineStatus = .idle
    public private(set) var loadedModel: LocalModel?
    public let version = "stub-1"
    /// What the terminal chunk reports as `TokenUsage.promptTokens` /
    /// `.cachedPromptTokens`; a nil cache figure (the default) models an engine
    /// that reported none.
    private let promptTokens: Int
    private let cachedPromptTokens: Int?

    public init(engineID: EngineID, promptTokens: Int = 1, cachedPromptTokens: Int? = nil) {
        self.engineID = engineID
        self.promptTokens = promptTokens
        self.cachedPromptTokens = cachedPromptTokens
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
        let promptTokens = self.promptTokens
        let cachedPromptTokens = self.cachedPromptTokens
        return AsyncThrowingStream { continuation in
            continuation.yield(GenerateChunk(text: "stub-"))
            continuation.yield(GenerateChunk(
                text: "response",
                finishReason: .stop,
                usage: TokenUsage(
                    promptTokens: promptTokens, completionTokens: 2,
                    cachedPromptTokens: cachedPromptTokens)
            ))
            continuation.finish()
        }
    }

    public func healthCheck() async -> Bool { true }
}
