import Foundation
import MLXLMCommon
import Testing

@testable import MacMLXCore

/// `TokenizerBridge.encode(text:addSpecialTokens:)` must honour `false` for
/// every tokenizer, including RoBERTa's. swift-transformers' `RobertaProcessing`
/// ignores the flag and always wraps the tokens in `<s> … </s>`; mlx-swift-lm's
/// pair encoders rely on the flag (they add the special tokens themselves), so
/// the leak doubled the wrapping and silently changed every RoBERTa-tokenizer
/// reranker's scores. The fixture is a three-word BPE tokenizer with a
/// `RobertaProcessing` post-processor; no model, no Metal.
@Suite("TokenizerBridge special tokens")
struct TokenizerBridgeSpecialTokensTests {

    private func load() async throws -> any MLXLMCommon.Tokenizer {
        let dir = try #require(
            Bundle.module.url(forResource: "roberta_tiny_tokenizer", withExtension: nil, subdirectory: "Fixtures"))
        return try await HuggingFaceTokenizerLoader().load(from: dir)
    }

    @Test
    func addSpecialTokensTrueWrapsInBosAndEos() async throws {
        let tokenizer = try await load()
        #expect(tokenizer.encode(text: "hi", addSpecialTokens: true) == [0, 6, 2])
    }

    /// The bug: before the bridge bypassed the post-processor, this returned
    /// `[0, 6, 2]` as well.
    @Test
    func addSpecialTokensFalseReturnsOnlyTheContentTokens() async throws {
        let tokenizer = try await load()
        #expect(tokenizer.encode(text: "hi", addSpecialTokens: false) == [6])
        #expect(tokenizer.encode(text: "hi hi", addSpecialTokens: false) == [6, 8])
    }
}
