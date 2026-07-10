// Copyright © 2026 macMLX. English comments only.

import Foundation

/// Raised by the batched-decode core (``BatchDecodeRunner``) when a cohort
/// cannot be served on the batched path and MUST fall back to sequential
/// (`B == 1`) decode instead.
///
/// A2a never performs the sequential fallback itself — it refuses loudly so the
/// caller (the A2c scheduler / A2d server) can route the request through the
/// existing single-generation path. This mirrors ``batchPositioned(_:batch:)``'s
/// "fail early, never silently produce garbage" contract.
enum BatchUnsupportedError: Error, Equatable, CustomStringConvertible {
    /// `batchPositioned(_:batch:)` returned `nil`: the model's KV caches are not
    /// a safely batch-positionable dense type (`CacheList`, `QuantizedKVCache`,
    /// …). See ``BatchPositionedCacheWrapper``.
    case cacheNotBatchable

    /// The cohort was empty. A batched decode needs at least one row.
    case emptyCohort

    /// A2a is the equal-length (v1-min) increment: every prompt in a cohort must
    /// tokenize to the SAME length. Ragged cohorts require the A2b `BatchKVCache`
    /// port and are rejected here. Associated value = the offending lengths.
    case unequalPromptLengths([Int])

    /// The injected ``BatchStepEvaluator`` violated its contract: `prefill(_:)`/
    /// `step(_:)` must return exactly one token per row (`slots.count`).
    /// Associated values = expected count and the count actually returned.
    case evaluatorContractViolation(expected: Int, got: Int)

    var description: String {
        switch self {
        case .cacheNotBatchable:
            return
                "batched decode refused: model caches are not batch-positionable "
                + "(not a plain KVCacheSimple/RotatingKVCache) — fall back to sequential decode"
        case .emptyCohort:
            return "batched decode refused: empty cohort (need at least one row)"
        case .unequalPromptLengths(let lengths):
            return
                "batched decode refused: A2a requires equal-length prompts, got lengths \(lengths) "
                + "(ragged cohorts need the A2b BatchKVCache port)"
        case .evaluatorContractViolation(let expected, let got):
            return
                "batched decode aborted: evaluator returned \(got) token(s), expected \(expected) "
                + "(BatchStepEvaluator contract violation)"
        }
    }
}
