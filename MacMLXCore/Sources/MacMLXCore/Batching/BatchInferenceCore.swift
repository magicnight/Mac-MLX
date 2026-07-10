// Copyright © 2026 macMLX. English comments only.

/// The MLX seam of the continuous-batching scheduler: the three stateful
/// operations ``BatchScheduler`` drives against a live ragged cohort — admit new
/// rows, decode one lockstep step, evict finished rows.
///
/// Splitting the model forward + cache surgery behind this protocol lets the
/// ``BatchScheduler`` scheduling logic (admission timing, batch-full gating,
/// prefill→decode migration, row-exit shrink, per-slot cancellation, fan-out)
/// run in CI against a scripted stub — no model, no Metal — while the production
/// ``ModelBatchInferenceCore`` carries all the non-`Sendable` MLX state (model,
/// running `[BatchKVCache]`, per-row samplers/processors).
///
/// ## Continuous-batching contract (mirrors mlx-lm `BatchGenerator`)
/// The running batch is an ORDERED list of rows; `rowCount` is its current
/// width. All three operations keep that order stable:
///  - ``admit(_:)`` prefills each new request B=1 (the correct scalar-offset
///    path, sidestepping the batched-prefill RoPE bug), merges it onto the END
///    of the running batch, and returns each new row's first sampled token,
///    row-aligned with `rows`. After it, `rowCount` grew by `rows.count`.
///  - ``decode(_:)`` runs ONE batched `[rowCount, 1]` forward feeding `feedback`
///    (one token per current row) and returns the `rowCount` next tokens.
///  - ``evict(keeping:)`` filters the running batch down to `keepRows` (indices
///    into the CURRENT row order, ascending), reclaiming the evicted rows'
///    cache. `keepRows == 0..<rowCount` is a no-op; an empty `keepRows` empties
///    the batch.
///
/// This is the two-phase prompt/generation split of Python's `BatchGenerator`
/// (`PromptProcessingBatch` → `GenerationBatch`), specialised to macMLX's
/// constraint that only the DECODE phase may be batched.
///
/// ## Isolation
/// Implementations may hold non-`Sendable` MLX state and MUST be created and
/// driven within a single isolation domain (the ``BatchScheduler`` actor). The
/// protocol is intentionally NOT `Sendable`.
protocol BatchInferenceCore {
    /// The current running-batch width (number of live rows).
    var rowCount: Int { get }

    /// Prefill each request B=1, merge onto the running batch, and return each
    /// new row's first sampled token (row-aligned with `rows`). Grows the batch
    /// by `rows.count`.
    ///
    /// - Throws: ``BatchUnsupportedError/cacheNotBatchable`` when the model's
    ///   caches cannot be safely batch-positioned (checked on first admit).
    func admit(_ rows: [BatchSlotConfig]) throws -> [Int]

    /// Run one batched decode step feeding `feedback` (one token per current
    /// row) and return the `rowCount` next sampled tokens.
    func decode(_ feedback: [Int]) throws -> [Int]

    /// Filter the running batch to `keepRows` (ascending indices into the
    /// current row order). No-op when every row is kept; empties the batch when
    /// `keepRows` is empty.
    func evict(keeping keepRows: [Int])
}
