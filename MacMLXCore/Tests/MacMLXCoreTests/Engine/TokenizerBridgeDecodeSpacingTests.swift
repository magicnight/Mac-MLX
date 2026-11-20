import Foundation
import MLXLMCommon
import Testing

@testable import MacMLXCore

/// `TokenizerBridge.decode` must return the spaces the tokens carry. A
/// tokenizer config that does not set `clean_up_tokenization_spaces` (Gemma 4's
/// does not) was read as "clean up" by swift-transformers up to 1.3.5, and the
/// cleanup deleted the space before `,` and before `n't`, so text the model
/// wrote as `hi , hi n't` reached the client as `hi, hin't`. 1.4.0 leaves BPE
/// output alone. The fixture is a tiny byte-level BPE tokenizer whose config
/// leaves the flag out; no model, no Metal.
@Suite("TokenizerBridge decode spacing")
struct TokenizerBridgeDecodeSpacingTests {

    private func load() async throws -> any MLXLMCommon.Tokenizer {
        let dir = try #require(
            Bundle.module.url(forResource: "bpe_no_cleanup_tokenizer", withExtension: nil, subdirectory: "Fixtures"))
        return try await HuggingFaceTokenizerLoader().load(from: dir)
    }

    @Test
    func decodeKeepsTheSpaceBeforePunctuationAndContractions() async throws {
        let tokenizer = try await load()
        // `hi`, `Ġ,`, `Ġhi`, `Ġn't`: the ids a model would emit, decoded as-is.
        #expect(tokenizer.decode(tokenIds: [6, 10, 8, 16], skipSpecialTokens: true) == "hi , hi n't")
    }
}
