// Copyright © 2026 macMLX. English comments only.

import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

@testable import MacMLXCore

/// ``Gemma2OverlayModel`` against upstream's `Gemma2Model`, on tiny
/// random-weight instances built from one configuration
/// (``TinyRandomLanguageModel``), so no checkpoint is needed.
///
/// Upstream ADDS a boolean causal mask to the soft-capped scores, so its
/// multi-token prefill is not causal; single-token decode (mask nil) is fine.
/// The overlay is meant to differ from upstream in that mask and nowhere else,
/// which these tests pin from both sides: the overlay's prefill equals its own
/// token-by-token decode, its decode equals upstream's decode on the same
/// weights, and its parameter keys equal upstream's so real checkpoints load
/// unchanged.
///
/// ``testUpstreamGemma2PrefillIsNotCausal`` is a canary rather than a
/// documented expectation: it asserts upstream's prefill still diverges, so the
/// day an mlx-swift-lm bump fixes it this suite fails and the overlay (with its
/// `ModelOverlay` registration) can be retired instead of shadowing a fixed
/// upstream forever.
final class Gemma2OverlayTests: XCTestCase {

    private let prompt = [9, 1, 22, 40, 7, 13, 2, 31]

    /// MLX runs float32 matmuls as TF32 on M5 by default (`MLX_ENABLE_TF32`),
    /// and a whole-prompt prefill and a token-at-a-time decode do not lose the
    /// same mantissa bits. The gate is the project's 1e-4 with TF32 off; with
    /// TF32 on it is loose enough for that rounding and still two orders of
    /// magnitude under upstream's divergence. For the tight gate:
    ///
    ///   TEST_RUNNER_MLX_ENABLE_TF32=0 xcodebuild test -scheme MacMLXCore \
    ///     -destination 'platform=macOS' -skipPackagePluginValidation \
    ///     -only-testing:MacMLXCoreTests/Gemma2OverlayTests
    private let tolerance: Float =
        ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "0" ? 1e-4 : 0.05

    // MARK: - Causality

    func testOverlayPrefillMatchesTokenByTokenDecode() throws {
        try requireTrustworthyMetalOrSkip()
        let errors = try prefillVersusIncrementalErrors(
            try TinyRandomLanguageModel.gemma2Overlay.make())
        print("[gemma2-overlay] overlay prefill vs incremental, per position: \(errors)")
        let error = errors.max() ?? .infinity
        XCTAssertLessThan(
            error, tolerance,
            "a causal prefill must reproduce the token-by-token logits at every position")
    }

    /// Canary: upstream's prefill is not causal. When this fails, mlx-swift-lm
    /// has fixed `Gemma2Model`'s mask and the overlay can go.
    func testUpstreamGemma2PrefillIsNotCausal() throws {
        try requireTrustworthyMetalOrSkip()
        let errors = try prefillVersusIncrementalErrors(try TinyRandomLanguageModel.gemma2.make())
        print("[gemma2-overlay] upstream prefill vs incremental, per position: \(errors)")
        let error = errors.max() ?? 0
        XCTAssertGreaterThan(
            error, 0.5,
            "upstream Gemma2Model's prefill now matches its decode: if mlx-swift-lm takes "
                + "the mask from the cache and applies a boolean mask with `which`, retire "
                + "Gemma2OverlayModel and its ModelOverlay registration")
    }

    /// The prompt run with no cache at all (the mask then comes from
    /// `makeAttentionMask`'s cache-less branch) matches the cached prefill.
    func testOverlayUncachedForwardMatchesCachedPrefill() throws {
        try requireTrustworthyMetalOrSkip()
        let model = try TinyRandomLanguageModel.gemma2Overlay.make()
        let input = MLXArray(prompt.map { Int32($0) }, [1, prompt.count])
        let uncached = model(input, cache: nil)
        let cached = model(input, cache: try model.newCache(parameters: nil))
        let error = abs(uncached - cached).max().item(Float.self)
        XCTAssertLessThan(error, 1e-6, "the cache must not change the prefill (\(error))")
    }

    /// A symbolic `.causal` mask and the equivalent boolean array give the same
    /// attention output. No cache in macMLX hands Gemma 2 `.causal` today (the
    /// model asks for an array), but the branch builds its own mask and must
    /// stay correct.
    func testCausalMaskModeMatchesBooleanArrayMask() throws {
        try requireMLXRuntimeOrSkip()
        let model = try XCTUnwrap(
            try TinyRandomLanguageModel.gemma2Overlay.make() as? Gemma2OverlayModel)
        let attention = try XCTUnwrap(model.model.layers.first).attention
        MLXRandom.seed(11)
        let x = MLXRandom.normal([1, 5, 32])
        let symbolic = attention(x, mask: .causal, cache: nil)
        let array = attention(x, mask: .array(createCausalMask(n: 5, offset: 0)), cache: nil)
        XCTAssertEqual(abs(symbolic - array).max().item(Float.self), 0)
    }

    // MARK: - Equivalence with upstream outside the mask

    /// Single-token decode is the path upstream gets right (its mask is nil at
    /// one token), so on the same weights the two must agree there exactly:
    /// the overlay changed the mask and nothing else.
    func testOverlayDecodeMatchesUpstreamDecodeOnSharedWeights() throws {
        try requireMLXRuntimeOrSkip()
        let upstream = try TinyRandomLanguageModel.gemma2.make()
        let overlay = try TinyRandomLanguageModel.gemma2Overlay.make()
        let upstreamModule = try XCTUnwrap(upstream as? Module)
        let overlayModule = try XCTUnwrap(overlay as? Module)
        try overlayModule.update(parameters: upstreamModule.parameters(), verify: .all)

        let upstreamLogits = try incrementalLogits(upstream)
        let overlayLogits = try incrementalLogits(overlay)
        for (step, (lhs, rhs)) in zip(upstreamLogits, overlayLogits).enumerated() {
            let error = abs(lhs - rhs).max().item(Float.self)
            XCTAssertLessThan(error, 1e-6, "decode step \(step) differs from upstream by \(error)")
        }
    }

    /// The overlay's flattened parameter keys and shapes equal upstream's for the
    /// same configuration, so an mlx-community Gemma 2 checkpoint (no
    /// `sanitize` on either side) loads into it unchanged.
    func testParameterKeysAndShapesMatchUpstream() throws {
        try requireMLXRuntimeOrSkip()
        let upstream = try XCTUnwrap(try TinyRandomLanguageModel.gemma2.make() as? Module)
        let overlay = try XCTUnwrap(try TinyRandomLanguageModel.gemma2Overlay.make() as? Module)
        let upstreamShapes = Dictionary(
            uniqueKeysWithValues: upstream.parameters().flattened().map { ($0.0, $0.1.shape) })
        let overlayShapes = Dictionary(
            uniqueKeysWithValues: overlay.parameters().flattened().map { ($0.0, $0.1.shape) })
        XCTAssertFalse(upstreamShapes.isEmpty)
        XCTAssertEqual(overlayShapes, upstreamShapes)
    }

    // MARK: - Helpers

    /// Per prompt position, the max absolute difference between the logits of
    /// one whole-prompt prefill and those of feeding the prompt one token at a
    /// time into a fresh cache. Position 0 sees only itself either way.
    private func prefillVersusIncrementalErrors(_ model: any LanguageModel) throws -> [Float] {
        let input = MLXArray(prompt.map { Int32($0) }, [1, prompt.count])
        let prefill = model(input, cache: try model.newCache(parameters: nil))[0]
        let incremental = try incrementalLogits(model)
        return incremental.enumerated().map { position, logits in
            abs(prefill[position] - logits).max().item(Float.self)
        }
    }

    /// Logits `[vocab]` after each prompt token, fed one at a time.
    private func incrementalLogits(_ model: any LanguageModel) throws -> [MLXArray] {
        let cache = try model.newCache(parameters: nil)
        return prompt.map { token in
            model(MLXArray([Int32(token)], [1, 1]), cache: cache)[0, -1]
        }
    }
}
