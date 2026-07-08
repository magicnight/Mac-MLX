import Foundation
import MLXLMCommon
@preconcurrency import Tokenizers

// MARK: - Tokenizer loader

/// Concrete `TokenizerLoader` that uses the HuggingFace swift-transformers library.
///
/// This is the manual equivalent of the `#huggingFaceTokenizerLoader()` macro
/// from MLXHuggingFace, inlined here so MacMLXCore does not need the macro-only
/// MLXHuggingFace product. Shared by both the generation engine (`MLXSwiftEngine`)
/// and the embedding engine (`EmbeddingEngine`), which each hand it to their
/// respective `loadContainer(from:using:)` factory call.
struct HuggingFaceTokenizerLoader: TokenizerLoader, @unchecked Sendable {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let upstream = try await AutoTokenizer.from(modelFolder: directory)
        return TokenizerBridge(upstream)
    }
}

/// Bridge between `Tokenizers.Tokenizer` (swift-transformers) and `MLXLMCommon.Tokenizer`.
struct TokenizerBridge: MLXLMCommon.Tokenizer, @unchecked Sendable {
    private let upstream: any Tokenizers.Tokenizer

    init(_ upstream: any Tokenizers.Tokenizer) {
        self.upstream = upstream
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        upstream.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        upstream.convertIdToToken(id)
    }

    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        do {
            return try upstream.applyChatTemplate(
                messages: messages, tools: tools, additionalContext: additionalContext)
        } catch let tokenizerError as Tokenizers.TokenizerError {
            if case .chatTemplate = tokenizerError {
                throw MLXLMCommon.TokenizerError.missingChatTemplate
            }
            throw tokenizerError
        }
    }
}
