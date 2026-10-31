// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MacMLXCore

/// A full-path VLM's processor hands the engine `[1, N]` tokens and a mask
/// even for text, while the text path's cache hit, history boundary and
/// usage all slice `[N]`. These drive a tiny random model behind a processor
/// shaped like the VLM ones through the hit and the boundary, on each family's
/// seam; without the normalization the hit slices away the whole prompt.
@Suite(
    "MLXSwiftEngine full-path VLM input",
    .serialized, .enabled(if: mlxMetallibIsAvailable, "needs the MLX Metal library"))
struct FullPathVLMEngineTests {

    private static let noColdTier = PromptCacheConfig(coldEnabled: false)

    private struct Collected {
        var text = ""
        var promptTokens: Int?
        var cachedPromptTokens: Int?
    }

    private func collect(_ engine: MLXSwiftEngine, _ request: GenerateRequest) async throws -> Collected {
        var out = Collected()
        for try await chunk in engine.generate(request) {
            out.text += chunk.text
            if let usage = chunk.usage {
                out.promptTokens = usage.promptTokens
                out.cachedPromptTokens = usage.cachedPromptTokens
            }
        }
        return out
    }

    private func engine(family: FullPathVLM.Family) async throws -> MLXSwiftEngine {
        let model = LocalModel(
            id: "tiny-vlm", displayName: "tiny-vlm",
            directory: URL(fileURLWithPath: "/nonexistent/tiny-vlm", isDirectory: true),
            sizeBytes: 0, format: .mlxVLM, quantization: nil, parameterCount: nil, architecture: nil)
        let container = ModelContainer(
            context: ModelContext(
                configuration: ModelConfiguration(directory: model.directory),
                model: try TinyRandomLanguageModel.qwen3.make(),
                processor: BatchShapedProcessor(),
                tokenizer: LetterTokenizer()))
        let engine = MLXSwiftEngine(promptCache: Self.noColdTier)
        await engine.installForTesting(container: container, model: model, fullPath: FullPathVLM(family: family))
        return engine
    }

    private func request(turns: Int) -> GenerateRequest {
        let messages = (0..<turns).map { ChatMessage(role: $0.isMultiple(of: 2) ? .user : .assistant, content: "turn \($0)") }
        return GenerateRequest(
            model: "tiny-vlm", messages: messages,
            parameters: GenerationParameters(temperature: 0, topP: 1.0, maxTokens: 3, stream: true))
    }

    @Test(arguments: FullPathVLM.Family.allCases)
    func aLongerPromptHitsTheCacheItsPrefixLeft(_ family: FullPathVLM.Family) async throws {
        let engine = try await engine(family: family)
        let first = try await collect(engine, request(turns: 1))
        #expect(first.promptTokens == BatchShapedProcessor.tokenCount(messages: 1))
        #expect(first.cachedPromptTokens ?? 0 == 0)

        let second = try await collect(engine, request(turns: 3))
        #expect(second.promptTokens == BatchShapedProcessor.tokenCount(messages: 3), "usage counts the whole prompt")
        let reused = try #require(second.cachedPromptTokens)
        #expect(reused >= BatchShapedProcessor.tokenCount(messages: 1), "the first prompt's tokens are served from the cache")
        let record = try #require(await engine.lastPromptCacheRecord)
        #expect(record.recorded)
    }

    @Test(arguments: FullPathVLM.Family.allCases)
    func theHistoryBoundarySplitsAFlatPrompt(_ family: FullPathVLM.Family) async throws {
        let engine = try await engine(family: family)
        await engine.setForcedHistoryBoundary(BatchShapedProcessor.tokenCount(messages: 1))
        let out = try await collect(engine, request(turns: 3))
        #expect(out.promptTokens == BatchShapedProcessor.tokenCount(messages: 3), "the boundary prefill is counted")
        #expect(out.text.count == 3)
    }
}

/// Answers like the VLM processors do for text: `[1, N]` tokens and an
/// all-ones int8 mask. Each message adds a fixed run of tokens, so a longer
/// conversation's prompt extends a shorter one's.
private struct BatchShapedProcessor: UserInputProcessor {
    static let tokensPerMessage = 12

    static func tokenCount(messages: Int) -> Int { 4 + messages * tokensPerMessage }

    static func tokens(messages: Int) -> [Int32] {
        (0..<tokenCount(messages: messages)).map { Int32($0 % 50 + 1) }
    }

    func prepare(input: UserInput) async throws -> LMInput {
        guard case .chat(let messages) = input.prompt else {
            throw CocoaError(.featureUnsupported)
        }
        let ids = Self.tokens(messages: messages.count)
        return LMInput(text: .init(
            tokens: MLXArray(ids).reshaped(1, ids.count),
            mask: MLXArray.ones([1, ids.count], dtype: .int8)))
    }
}

/// One letter per token, no special tokens, so generation runs to its token
/// budget and the text's length counts the tokens.
private struct LetterTokenizer: Tokenizer {
    let bosToken: String? = nil
    let eosToken: String? = nil
    let unknownToken: String? = nil

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(repeating: "a", count: tokenIds.count)
    }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { "a" }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}
