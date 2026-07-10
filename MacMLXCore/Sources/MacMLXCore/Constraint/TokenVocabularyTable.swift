// Copyright © 2026 macMLX. English comments only.

/// A per-model map from token id to its ``TokenClassification`` — the bridge
/// between an LLM tokenizer's vocabulary and the byte-level JSON constraint.
///
/// Building this table is the one up-front cost of constrained decoding: it
/// decodes every token id in the vocabulary exactly once. The result is cached
/// per model (see ``TokenVocabularyCache``) so the cost is paid at most once per
/// resident model, not once per request.
///
/// The type is deliberately MLX- and tokenizer-agnostic: it is built from a
/// plain `(Int) -> String?` decode closure and a set of stop-token ids, so it is
/// unit-testable with a scripted vocabulary and no Metal runtime.
public struct TokenVocabularyTable: Sendable {

    /// `classifications[id]` is the classification of token `id`, for
    /// `id` in `0..<count`.
    public let classifications: [TokenClassification]

    /// The number of tokens — the model's logit vocabulary dimension.
    public var count: Int { classifications.count }

    /// The Unicode replacement scalar. A standalone decode that contains it did
    /// not resolve to a complete UTF-8 scalar, so the token's exact bytes are
    /// unrecoverable and it is classified ``TokenClassification/unusable``.
    private static let replacement: Character = "\u{FFFD}"

    /// Build the table.
    ///
    /// - Parameters:
    ///   - vocabularySize: the number of token ids to classify (`0..<size`),
    ///     authoritatively the model's logit vocabulary dimension.
    ///   - stopTokenIDs: every token id that terminates generation (the model
    ///     config's EOS ids, the tokenizer's EOS id, and any extra stop tokens).
    ///     These are classified ``TokenClassification/eos`` regardless of how
    ///     they decode.
    ///   - decode: standalone decode of one token id to its literal text
    ///     (`skipSpecialTokens: false`). Returning `nil` or an empty string marks
    ///     the token ``TokenClassification/unusable``.
    public init(
        vocabularySize: Int,
        stopTokenIDs: Set<Int>,
        decode: (Int) -> String?
    ) {
        var result = [TokenClassification]()
        result.reserveCapacity(Swift.max(0, vocabularySize))
        for id in 0..<Swift.max(0, vocabularySize) {
            if stopTokenIDs.contains(id) {
                result.append(.eos)
                continue
            }
            guard let text = decode(id), !text.isEmpty else {
                result.append(.unusable)
                continue
            }
            if text.contains(Self.replacement) {
                result.append(.unusable)
                continue
            }
            result.append(.bytes(Array(text.utf8)))
        }
        self.classifications = result
    }

    /// Direct-injection initializer for tests and cache reconstruction.
    public init(classifications: [TokenClassification]) {
        self.classifications = classifications
    }

    /// The classification of token `id`, or ``TokenClassification/unusable`` for
    /// an out-of-range id (defensive: an id the model can't actually emit).
    @inlinable
    public func classification(of id: Int) -> TokenClassification {
        guard id >= 0, id < classifications.count else { return .unusable }
        return classifications[id]
    }
}
