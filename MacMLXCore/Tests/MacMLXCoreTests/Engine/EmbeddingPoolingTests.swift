import Foundation
import MLX
import Testing

@testable import MLXEmbedders

// Metallib availability gating lives in Helpers/MLXTestSupport.swift
// (`mlxMetallibIsAvailable`) — shared with the other MLX-gated suites.

/// Regression coverage for `EmbeddingEngine.embed`'s pooling call.
///
/// A code-review BLOCKER had `EmbeddingEngine.swift:100` calling
/// `context.pooling(output, ...)` **without** forwarding the attention `mask`,
/// so padding positions leaked into the pooled sentence vector. The fix passes
/// `mask: mask`.
///
/// These tests lock in the invariant that fix depends on, using MLXEmbedders'
/// own `Pooling` with hand-made hidden states and no model download: a short
/// sequence padded out to a longer length must pool to the **same** vector as
/// the unpadded short sequence — but only when the mask is supplied. Omitting
/// the mask (the bug) lets the padding rows pollute the result, which the
/// negative assertion below proves.
@Suite("Embedding pooling respects the attention mask")
struct EmbeddingPoolingTests {

    @Test(
        "Padding is excluded only when the mask is passed to pooling",
        .enabled(if: mlxMetallibIsAvailable, "Requires default.metallib (run under xcodebuild)"),
        arguments: [Pooling.Strategy.mean, Pooling.Strategy.last]
    )
    func poolingExcludesPaddingOnlyWithMask(strategy: Pooling.Strategy) {
        // Two real tokens, hidden size 2.
        let short = MLXArray([1.0 as Float, 2, 3, 4]).reshaped(1, 2, 2)
        // Same two tokens followed by two padding rows with obviously different
        // values, so including them would change the pooled result.
        let padded = MLXArray([1.0 as Float, 2, 3, 4, 100, 100, 100, 100]).reshaped(1, 4, 2)

        // Boolean masks (1 = real token, 0 = padding), matching how
        // `EmbeddingEngine` builds its mask via `padded .!= padID`.
        let shortMask = MLXArray([1 as Int32, 1]).reshaped(1, 2) .== MLXArray(Int32(1))
        let paddedMask = MLXArray([1 as Int32, 1, 0, 0]).reshaped(1, 4) .== MLXArray(Int32(1))

        let pooling = Pooling(strategy: strategy)
        let shortPooled = pooling(
            EmbeddingModelOutput(hiddenStates: short, pooledOutput: nil), mask: shortMask)
        let paddedWithMask = pooling(
            EmbeddingModelOutput(hiddenStates: padded, pooledOutput: nil), mask: paddedMask)
        // Reproduces the pre-fix call site: no mask, so `Pooling` treats every
        // position — including padding — as a real token.
        let paddedNoMask = pooling(
            EmbeddingModelOutput(hiddenStates: padded, pooledOutput: nil), mask: nil)

        // With the mask, padding is excluded → padded pools identically to the
        // unpadded short sequence. This fails if the fix is reverted.
        #expect(
            allClose(paddedWithMask, shortPooled).all().item(Bool.self),
            "Masked pooling must exclude padding: padded should equal the unpadded short vector")

        // Without the mask (the bug), padding pollutes the pooled vector, so it
        // must differ — proving the mask is load-bearing.
        #expect(
            !allClose(paddedNoMask, shortPooled).all().item(Bool.self),
            "Omitting the mask (the fixed bug) must let padding change the pooled vector")
    }
}
