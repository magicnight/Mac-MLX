// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLX
import MLXNN
import MLXLMCommon

/// XTC ("Exclude Top Choices") sampler, a faithful port of mlx-lm's `apply_xtc`
/// (`mlx_lm/sample_utils.py`). With probability `xtc_probability`, every token
/// whose probability exceeds `xtc_threshold` is removed EXCEPT the least-probable
/// such token — excluding the "obvious" continuations to raise diversity while
/// keeping at least one high-confidence option and the whole low-probability tail.
///
/// ## Exact algorithm (matches `apply_xtc`)
/// ```
/// probs = softmax(logits)
/// mask  = probs > min(where(probs > threshold, probs, +inf))
/// out   = where(uniform(0,1) > probability, logits, where(mask, -inf, logits))
/// ```
/// `min(where(probs > threshold, probs, +inf))` is the smallest probability
/// among tokens above the threshold (or `+inf` when none is), so `mask` marks
/// exactly the tokens strictly more probable than that least-above-threshold
/// token. With fewer than two tokens above the threshold nothing is masked.
///
/// ## Composition
/// This wraps a base ``LogitSampler`` (the one ``GenerateParameters/sampler()``
/// would build from temperature/top_p/top_k/min_p) and delegates to it after
/// applying the XTC mask. mlx-lm slots XTC between `min_p` and `top_k` within one
/// filter chain; here XTC is applied to the logits BEFORE the base sampler's own
/// top_p filtering. This IS a behavioral difference, not a rounding one: XTC's
/// candidate set is the softmax of whatever it receives, so pre-filtering (as
/// mlx-lm does) renormalizes survivors and can change which tokens exceed
/// `xtc_threshold` — the two orderings can produce different sampling supports.
/// Applying XTC to the TRUE model distribution (as here) is arguably more
/// faithful to XTC's published intent; we keep this order deliberately and
/// document the divergence honestly. Blast radius is bounded today because
/// macMLX exposes only `top_p` from the base filter set (no `top_k`/`min_p`).
/// XTC is meaningful only at `temperature > 0` (mlx-lm disables it under
/// argmax); the engine only installs it then.
public struct XTCSampler: LogitSampler {

    private let base: any LogitSampler
    private let probability: Float
    private let threshold: Float
    private let randomState: MLXRandom.RandomState

    /// - Parameters:
    ///   - base: the underlying sampler applied after the XTC mask.
    ///   - probability: `xtc_probability` in `[0, 1]` — per-step chance of masking.
    ///   - threshold: `xtc_threshold` in `[0, 0.5]` — probability above which a
    ///     token is an exclusion candidate.
    ///   - seed: optional seed for the per-step probability draw, so a seeded
    ///     request is reproducible. Offset from the base sampler's seed so the two
    ///     RNG streams are independent.
    ///
    /// - Note: mlx-lm's `xtc_special_tokens` (ids never excluded) is intentionally
    ///   NOT exposed in v1 — no request field feeds it — so XTC here never spares
    ///   special tokens. Add a parameter here when a wire field is introduced.
    public init(
        base: any LogitSampler,
        probability: Float,
        threshold: Float,
        seed: UInt64? = nil
    ) {
        self.base = base
        self.probability = probability
        self.threshold = threshold
        self.randomState = seed.map { MLXRandom.RandomState(seed: $0 &+ 0x5854_4300) }
            ?? MLXRandom.RandomState()
    }

    public func sample(logits: MLXArray) -> MLXArray {
        let probs = softmax(logits, axis: -1)
        // Smallest probability among tokens strictly above the threshold, per row
        // (+inf when none qualifies, which yields an all-false mask below).
        let aboveThreshold = MLX.where(probs .> threshold, probs, MLXArray(Float.infinity))
        let thresholdMin = aboveThreshold.min(axis: -1, keepDims: true)
        let mask = probs .> thresholdMin
        let negInf = MLXArray(-Float.infinity).asType(logits.dtype)
        let masked = MLX.where(mask, negInf, logits)
        let gated: MLXArray = withRandomState(randomState) {
            let u = MLXRandom.uniform(low: Float(0), high: Float(1), [1])
            return MLX.where(u .> probability, logits, masked)
        }
        return base.sample(logits: gated)
    }
}
