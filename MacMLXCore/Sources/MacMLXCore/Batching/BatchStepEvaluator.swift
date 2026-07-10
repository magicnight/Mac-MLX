// Copyright © 2026 macMLX. English comments only.

/// The MLX seam of batched decode: turns tokens into next-token IDs for a whole
/// cohort in lockstep.
///
/// Splitting the model forward + sampling behind this protocol lets the
/// ``BatchDecodeRunner`` scheduling logic (per-row stop / EOS / max-tokens /
/// finished-row masking / fan-out) run in CI against a scripted stub — no model,
/// no Metal — while the production ``ModelBatchStepEvaluator`` carries all the
/// non-`Sendable` MLX state.
///
/// ## Contract
///  - The cohort is `B` rows, fixed for the evaluator's lifetime.
///  - ``prefill(_:)`` is called exactly once with `B` equal-length prompt token
///    arrays; it runs the batched prompt forward and returns the `B` first
///    sampled tokens (one per row).
///  - ``step(_:)`` is called once per decode step with a `B`-length fed-back
///    token array (finished rows carry a pad token — see ``BatchDecodeSlot``),
///    runs one batched `[B, 1]` forward, and returns the `B` next tokens.
///  - Returned arrays are always length `B`, row-aligned with the input.
///
/// ## Isolation
/// Implementations may hold non-`Sendable` MLX state (`LanguageModel`,
/// `[KVCache]`) and MUST be created and driven within a single isolation domain
/// (inside `ModelContainer.perform` today; the A2c `BatchScheduler` actor later).
/// The protocol is intentionally NOT `Sendable`.
protocol BatchStepEvaluator {
    /// Run the one-shot batched prefill over `promptRows` (`B` equal-length
    /// prompts) and return the `B` first sampled tokens.
    func prefill(_ promptRows: [[Int]]) throws -> [Int]

    /// Run one batched decode step feeding back `fed` (`B` tokens) and return the
    /// `B` next sampled tokens.
    func step(_ fed: [Int]) throws -> [Int]
}
