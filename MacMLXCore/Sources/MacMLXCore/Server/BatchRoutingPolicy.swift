// Copyright © 2026 macMLX. English comments only.

/// Request-level routing gate for the A2d continuous-batching server path.
///
/// Decides, from facts visible at the HTTP layer ALONE, whether a
/// `/v1/chat/completions` (or its legacy `/v1/completions` alias) request is
/// ELIGIBLE to attempt the batched path. Model-level eligibility (text-vs-VLM,
/// verified-`ropeOffset` architecture, dense cache) is deliberately NOT decided
/// here — it belongs to the resident model and is enforced one layer down by
/// ``BatchGenerationServing/submit(_:)`` returning `nil` (fall back to the
/// legacy single-stream path). Splitting the gate this way keeps this predicate
/// pure and MLX-free — unit-tested under a plain `swift test` — and matches the
/// design's "the coverage gate decides per resident model".
///
/// Only OpenAI chat/completions requests are ever routed through here; VLM,
/// speculative, embeddings, Anthropic, and Ollama paths are untouched and always
/// take the existing single-stream path (this predicate is never consulted for
/// them).
public enum BatchRoutingPolicy {
    /// Whether to ATTEMPT the batched path for a request.
    ///
    /// A `true` result still defers the final say to the seam
    /// (``BatchGenerationServing/submit(_:)`` returns `nil` for an uncoverable
    /// resident model, which routes the request to the legacy path). A `false`
    /// result routes straight to the legacy single-stream path with ZERO batched
    /// work performed — so a "no" here can never double-bill tokens or
    /// double-count in-flight work.
    ///
    /// - Parameters:
    ///   - batchingEnabled: the server has a batch-serving seam installed. This
    ///     is the default-off switch: with no seam the result is always `false`,
    ///     so every request takes the legacy path byte-for-byte (zero
    ///     regression).
    ///   - hasDraftModel: the request set `draft_model` (speculative decoding).
    ///     Batching × speculative decoding is mutually exclusive in v1 — mirrors
    ///     mlx-lm's server `is_batchable`, where a resident draft model forces
    ///     the sequential path.
    ///   - hasResponseFormat: the request set `response_format` (Track C
    ///     structured output). The batched step evaluator has no constraint
    ///     support in v1, so a constrained request must take the single-stream
    ///     path where the constraint logit mask is installed. Same
    ///     `is_batchable` spirit as `hasDraftModel`.
    ///
    /// - Note: mlx-lm's `is_batchable` also excludes a per-request `seed`. macMLX
    ///   has no per-request `seed` request field yet (see
    ///   `ChatCompletionRequest`), so there is nothing to gate on today. Add a
    ///   `hasPerRequestSeed` parameter here the moment such a field is
    ///   introduced, rather than inventing a dead argument now.
    public static func shouldAttemptBatch(
        batchingEnabled: Bool,
        hasDraftModel: Bool,
        hasResponseFormat: Bool
    ) -> Bool {
        batchingEnabled && !hasDraftModel && !hasResponseFormat
    }
}
