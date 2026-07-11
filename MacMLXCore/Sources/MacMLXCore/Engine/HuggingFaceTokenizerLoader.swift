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
        // Resolve a per-model chat-template override (user file → built-in by
        // model_type → checkpoint default) BEFORE any template compilation. A
        // model with no override yields `resolved == nil` here and is
        // byte-for-byte unchanged. Log both outcomes worth knowing about — an
        // applied override, AND a user override file that was present but
        // unusable (empty / unreadable / not valid UTF-8) — never silent either
        // way.
        let resolution = ChatTemplateOverride.resolveDetailed(modelDirectory: directory)
        if let skippedReason = resolution.skippedUserFileReason {
            await LogManager.shared.warning(
                "Chat-template user override \(ChatTemplateOverride.userOverrideFilename) "
                    + "found but \(skippedReason) — ignoring, falling back to "
                    + (resolution.resolved?.source ?? "the checkpoint's own template")
                    + " for model at \(directory.lastPathComponent)",
                category: .inference
            )
        }
        if let override = resolution.resolved {
            await LogManager.shared.info(
                "Chat-template override applied (\(override.source)) "
                    + "for model at \(directory.lastPathComponent)",
                category: .inference
            )
        }
        return TokenizerBridge(upstream, chatTemplateOverride: resolution.resolved?.template)
    }
}

/// Bridge between `Tokenizers.Tokenizer` (swift-transformers) and `MLXLMCommon.Tokenizer`.
struct TokenizerBridge: MLXLMCommon.Tokenizer, @unchecked Sendable {
    private let upstream: any Tokenizers.Tokenizer
    /// When non-nil, this Jinja template is applied INSTEAD of the checkpoint's
    /// own `chat_template` (see ``ChatTemplateOverride``). Passed to
    /// swift-transformers as a `.literal`, which the library prefers over the
    /// tokenizer-config template — the substitution therefore happens before any
    /// template compilation.
    private let chatTemplateOverride: String?

    init(_ upstream: any Tokenizers.Tokenizer, chatTemplateOverride: String? = nil) {
        self.upstream = upstream
        self.chatTemplateOverride = chatTemplateOverride
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
            if let chatTemplateOverride {
                // `.literal` wins over the config template in swift-transformers.
                // Called through the `any Tokenizer` existential, so every
                // parameter of the protocol requirement must be explicit (no
                // default args): `addGenerationPrompt: true`, `truncation: false`,
                // `maxLength: nil`, and the tools/additionalContext threading all
                // match the non-override call below exactly — only the template
                // SOURCE differs.
                return try upstream.applyChatTemplate(
                    messages: messages,
                    chatTemplate: .literal(chatTemplateOverride),
                    addGenerationPrompt: true,
                    truncation: false,
                    maxLength: nil,
                    tools: tools,
                    additionalContext: additionalContext)
            }
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
