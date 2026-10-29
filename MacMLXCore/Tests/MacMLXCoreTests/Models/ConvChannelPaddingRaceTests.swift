import Testing
import Foundation
import MLX
@testable import MacMLXCore

/// Guards the channel-padding convolutions against `ml-explore/mlx#4660`.
///
/// When a convolution's channels are not a multiple of 16, the Metal backend
/// copies the input (and, for 3D, the weight) into a padded buffer, filling
/// the extra channels from a scalar `zero` that `pad_gpu` reads on the GPU.
/// `pad_and_slice_conv_3D_gpu` and `pad_in_channels_conv_2D_gpu` (host C++,
/// `mlx/backend/metal/conv.cpp`) let that scalar go out of scope as soon as
/// the fill was encoded, without `add_temporary`, so its buffer went back to
/// the allocator before the command buffer ran. A small allocation made in
/// that window — any host-side scalar — gets the same buffer, writes its own
/// value into it, and the GPU pads with that value instead of zero. Upstream's
/// repro: a conv3d of ones over 3 channels under `async_eval`, followed by
/// `mx.array(1.0)`, returned 128 (all 16 padded channels counted) instead of
/// 24. Host C++, so the submodule bump alone carries the fix.
///
/// The 3D path is the one macMLX reaches: the vision patch embeds of the
/// Qwen2-VL / Qwen2.5-VL family (`QwenVL.swift`), Qwen3-VL and GLM-OCR in
/// mlx-swift-lm 3.32.3 are `Conv3d(3 → hidden)`, and 3 is not a multiple of 16.
/// The model forward is lazy and evaluated asynchronously by the generation
/// loop, so a scalar created between encoding and execution is ordinary.
///
/// Reaching the defect takes (`dispatch_conv_3D_gpu`): input dilation 1,
/// `groups == 1`, and `C % 16 != 0` or `O` above 16 and not a multiple of 16.
/// The 2D path (`dispatch_conv_2D_gpu`) additionally wants stride 1, at least
/// 256 output positions, a kernel of at least 9 taps, `C` neither `<= 4` nor
/// a multiple of 16, and `O` at most 16 or a multiple of 16.
///
/// The failure is timing-dependent, so each test repeats the async evaluation
/// with scalar allocations right behind it, bounded by an iteration count and
/// a wall-clock budget, and counts the runs that came back wrong. All-ones
/// operands make the right answer an exact integer, so no reference
/// computation is needed: every output element is the kernel's tap count times
/// the real channel count. Against the fork's unpatched base (1026d23) the
/// conv3d came back 128 everywhere in 297 and 299 of 300 runs (two runs) and the conv2d 144 in
/// 300 of 300: upstream's fingerprint, every padded channel read as 1.
@Suite(
    "Conv channel-padding scalar lifetime",
    .serialized,
    .enabled(if: mlxMetallibIsAvailable, "Requires default.metallib (run under xcodebuild)"))
struct ConvChannelPaddingRaceTests {

    private static let maxIterations = 300
    private static let timeBudget: TimeInterval = 15
    /// Scalars allocated right after each async evaluation. One lands on a
    /// freed `zero` when the allocator hands it back; a few cover both of the
    /// 3D path's freed scalars and whatever else the encode released.
    private static let decoysPerIteration = 8

    private struct RaceOutcome {
        var ran = 0
        var corrupted = 0
        var worstLow: Float = .infinity
        var worstHigh: Float = -.infinity
    }

    /// Repeats `convolve` on fresh all-ones operands under `asyncEval`,
    /// allocating float32 scalars of value 1 straight after each dispatch, and
    /// counts the runs whose output is not uniformly `expected`.
    private func runRace(
        inputShape: [Int], weightShape: [Int], expected: Float,
        convolve: (MLXArray, MLXArray) -> MLXArray
    ) -> RaceOutcome {
        var outcome = RaceOutcome()
        let deadline = Date().addingTimeInterval(Self.timeBudget)
        while outcome.ran < Self.maxIterations, Date() < deadline {
            // The buffer cache hands back the OLDEST free buffer of a size
            // first (`BufferCache::reuse_from_cache` takes `lower_bound` of a
            // multimap), and earlier iterations leave dozens of 4-byte
            // buffers in it, so without this the decoys never reach the
            // `zero` freed a moment ago: 0 of 300 runs failed on the unpatched
            // base. Emptying it first makes the freed scalars the only
            // candidates: 297 and 299 of 300 conv3d runs, 300 of 300 conv2d runs.
            Memory.clearCache()
            let x = MLXArray.ones(inputShape)
            let w = MLXArray.ones(weightShape)
            let y = convolve(x, w)
            asyncEval(y)

            // Same dtype and size as the freed `zero` (float32, 4 bytes).
            // Kept alive until the result is read so none of them is recycled
            // in turn.
            var decoys: [MLXArray] = []
            decoys.reserveCapacity(Self.decoysPerIteration)
            for _ in 0 ..< Self.decoysPerIteration {
                decoys.append(MLXArray(Float(1)))
            }

            let low = y.min().item(Float.self)
            let high = y.max().item(Float.self)
            withExtendedLifetime(decoys) {}

            outcome.ran += 1
            if low != expected || high != expected {
                outcome.corrupted += 1
                outcome.worstLow = min(outcome.worstLow, low)
                outcome.worstHigh = max(outcome.worstHigh, high)
            }
        }
        // A pass says how hard the race was pushed; keep that in the log.
        print("[conv-race] input \(inputShape): \(outcome.corrupted) of \(outcome.ran) runs wrong")
        return outcome
    }

    /// Upstream's repro shape: 3 input channels padded to 16, 4 output
    /// channels padded to 16, a 2×2×2 kernel. Each output is 2·2·2·3 = 24;
    /// with every padded channel read as 1 it is 2·2·2·16 = 128.
    @Test("a 3-channel conv3d pads with zero under async evaluation")
    func conv3dPadsWithZeroUnderAsyncEval() {
        let outcome = runRace(
            inputShape: [1, 4, 8, 8, 3], weightShape: [4, 2, 2, 2, 3], expected: 24
        ) { x, w in
            MLX.conv3d(x, w)
        }
        #expect(outcome.ran > 0)
        #expect(
            outcome.corrupted == 0,
            """
            \(outcome.corrupted) of \(outcome.ran) async conv3d runs over 3 \
            channels did not return 24 everywhere (range seen \
            \(outcome.worstLow)…\(outcome.worstHigh); 128 means every padded \
            channel was read as 1). This is ml-explore/mlx#4660 — \
            pad_and_slice_conv_3D_gpu frees its zero scalar before the GPU \
            reads it.
            """)
    }

    /// The 2D sibling: 8 input channels padded to 16, 16 output channels, a
    /// 3×3 kernel over an 18×18 image (256 output positions, the `out_large`
    /// floor). Each output is 3·3·8 = 72; with the padding read as 1, 144.
    @Test("an 8-channel conv2d pads with zero under async evaluation")
    func conv2dPadsWithZeroUnderAsyncEval() {
        let outcome = runRace(
            inputShape: [1, 18, 18, 8], weightShape: [16, 3, 3, 8], expected: 72
        ) { x, w in
            MLX.conv2d(x, w)
        }
        #expect(outcome.ran > 0)
        #expect(
            outcome.corrupted == 0,
            """
            \(outcome.corrupted) of \(outcome.ran) async conv2d runs over 8 \
            channels did not return 72 everywhere (range seen \
            \(outcome.worstLow)…\(outcome.worstHigh); 144 means every padded \
            channel was read as 1). This is ml-explore/mlx#4660 — \
            pad_in_channels_conv_2D_gpu frees its zero scalar before the GPU \
            reads it.
            """)
    }
}
