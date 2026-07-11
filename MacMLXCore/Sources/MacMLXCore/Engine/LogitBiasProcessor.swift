// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLX
import MLXLMCommon

/// Additive per-token logit bias (OpenAI `logit_bias`), mirroring mlx-lm's
/// `logit_bias` entry in `make_logits_processors`: each configured token id has
/// a fixed bias added to its logit at every decode step, before sampling.
///
/// ## Composition
/// This runs as an INNER processor — before any structured-output constraint
/// mask (bias first, constraint last, the constraint's last-applied invariant)
/// and, matching mlx-lm's processor ordering, before the repetition/penalty
/// processor. It never removes a token; a large negative bias only lowers a
/// token's logit (a `-100` bias is enough to make it effectively unreachable at
/// any realistic temperature, but the constraint mask — not this — is what
/// hard-forbids tokens).
///
/// The bias columns are precomputed as device tensors once, so every step is a
/// single GPU scatter-add with no host round-trip.
public struct LogitBiasProcessor: LogitProcessor {

    /// Token ids to bias (Int32, non-negative, sorted). Parallel to ``values``.
    private let indices: MLXArray
    /// Bias amounts to add at ``indices``.
    private let values: MLXArray

    /// - Returns: nil when `bias` is empty (or contains only negative ids, which
    ///   are dropped) — the caller then installs no processor at all, so an
    ///   empty/absent `logit_bias` costs nothing.
    public init?(bias: [Int: Float]) {
        // Drop negative ids here (host-side) — a token id is a non-negative
        // vocabulary index; out-of-range-high ids are masked at `process` time
        // (vocabulary size is only known then). Sorted for deterministic tensors.
        let sorted = bias.filter { $0.key >= 0 }.sorted { $0.key < $1.key }
        guard !sorted.isEmpty else { return nil }
        self.indices = MLXArray(sorted.map { Int32($0.key) })
        self.values = MLXArray(sorted.map { $0.value })
    }

    public mutating func prompt(_ prompt: MLXArray) {
        // logit_bias is a per-step additive constant; the prompt does not affect it.
    }

    public func process(logits: MLXArray) -> MLXArray {
        // `logits` is [1, vocab]. Scatter-add each bias into its column.
        let vocab = logits.dim(-1)
        let flat = logits.reshaped([vocab])
        // Defensively mask any id >= vocab: redirect it to slot 0 and zero its
        // added value, so an out-of-range client id is a no-op instead of an
        // out-of-bounds scatter.
        let inRange = indices .< MLXArray(Int32(vocab))
        let safeIndices = MLX.where(inRange, indices, MLXArray(Int32(0)))
        let safeValues = MLX.where(inRange, values, MLXArray(Float(0))).asType(flat.dtype)
        let updated = flat.at[safeIndices].add(safeValues)
        return updated.reshaped([1, vocab])
    }

    public mutating func didSample(token: MLXArray) {
        // Stateless — the same bias applies at every step.
    }
}
