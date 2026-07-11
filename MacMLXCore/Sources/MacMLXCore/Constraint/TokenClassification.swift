// Copyright © 2026 macMLX. English comments only.

/// How a single vocabulary token participates in the JSON constraint.
///
/// Computed once per model by ``TokenVocabularyTable`` and consulted for every
/// candidate token at every decode step by ``JSONConstraintProcessor``.
public enum TokenClassification: Equatable, Sendable {
    /// A normal token whose exact byte contribution is known. These bytes are
    /// fed through ``JSONGrammarState`` (or the schema automaton) to decide
    /// legality.
    case bytes([UInt8])

    /// An end-of-sequence / stop token. Legal only when the constraint is in an
    /// accepting (complete) state; masked otherwise, which is what forces the
    /// model to finish the JSON document before it may stop.
    case eos

    /// A token that must never be emitted under constraint: a token that decodes
    /// to nothing, or whose exact bytes cannot be recovered from a standalone
    /// decode (a byte-fragment token that is only half of a UTF-8 scalar). Always
    /// masked.
    ///
    /// Masking these is the conservative, correctness-first choice: emitting a
    /// token whose bytes we cannot validate could break well-formedness. The only
    /// cost is that multi-byte Unicode inside string values must be spellable via
    /// complete-scalar tokens — always true for ASCII JSON, and true in practice
    /// for the common-scalar tokens of production tokenizers.
    case unusable
}
