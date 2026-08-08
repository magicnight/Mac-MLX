import Testing
import Foundation
import MLX
@testable import MacMLXCore

/// Guards the NAX split-K GEMM path against `ml-explore/mlx#3797`: the host
/// dispatch instantiated the JIT kernel template with the *output* dtype where
/// the kernel needs the *input* dtype, so a bf16 matmul had its inputs read as
/// float — out-of-bounds loads and silent garbage, with no crash and no NaN
/// guard. Fixed upstream by `ml-explore/mlx#3810` (shipped in mlx v0.32.0).
///
/// Reaching that dispatch needs every one of these to hold (see
/// `mlx/backend/metal/matmul.cpp`, "Case 2: Large K with sufficient M, N"):
///
/// - NAX hardware and OS (`metal::is_nax_available()`: macOS 26.2+, GPU
///   architecture generation ≥ 17)
/// - a non-float32 input dtype
/// - `batch_size_out == 1`
/// - `M * N >= 2048 * 2048`
/// - `K >= 10240`
/// - `K >= 3 * max(M, N)`
///
/// The shapes below sit exactly on those thresholds. The bug is also
/// JIT-specific — mlx-swift builds with `MLX_METAL_JIT=ON`
/// (`tools/update-mlx.sh`), so it reaches mlx-swift consumers while the
/// AOT-compiled Python wheel is unaffected.
///
/// On hardware without NAX the dispatch never happens and these are simply
/// ordinary matmul parity checks — they pass either way, so the suite is safe
/// to run everywhere. It is the M5-class machines that this pins.
@Suite(
    "NAX split-K GEMM parity",
    .enabled(if: mlxMetallibIsAvailable, "Requires default.metallib (run under xcodebuild)"))
struct NAXSplitKGemmParityTests {

    /// Cosine similarity between two flattened arrays, computed in float32.
    /// Cosine rather than an absolute tolerance because the failure mode is
    /// wholesale garbage (or NaN), not a rounding-scale drift — and bf16 vs
    /// float32 legitimately differs in the third decimal.
    private func cosineSimilarity(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        let a = lhs.asType(.float32).flattened()
        let b = rhs.asType(.float32).flattened()
        let dot = MLX.sum(a * b).item(Float.self)
        let normA = MLX.sqrt(MLX.sum(a * a)).item(Float.self)
        let normB = MLX.sqrt(MLX.sum(b * b)).item(Float.self)
        guard normA > 0, normB > 0 else { return 0 }
        return dot / (normA * normB)
    }

    /// bf16 matmul at the exact NAX split-K dispatch corner, against a float32
    /// reference of the same operands.
    ///
    /// `M = N = 2048` puts `M * N` exactly at the `2048 * 2048` threshold, and
    /// `K = 12288` clears both `K >= 10240` and `K >= 3 * max(M, N)` (6144).
    @Test("bf16 matmul at the split-K dispatch corner matches a float32 reference")
    func bf16SplitKCornerMatchesFloat32Reference() {
        let m = 2048, n = 2048, k = 12288

        // Deterministic operands: a fixed seed keeps a failure reproducible,
        // and small magnitudes keep the float32 reference well inside range.
        MLXRandom.seed(0x5EED)
        let aF32 = MLXRandom.normal([m, k], scale: 0.02)
        let bF32 = MLXRandom.normal([k, n], scale: 0.02)

        let reference = MLX.matmul(aF32, bF32)
        let underTest = MLX.matmul(aF32.asType(.bfloat16), bF32.asType(.bfloat16))
        MLX.eval(reference, underTest)

        let similarity = cosineSimilarity(underTest, reference)

        // A correct bf16 matmul of this size lands well above 0.999 against the
        // float32 reference. The #3797 failure produced NaN and ~1e35 values at
        // every size tried, so any real regression falls far below this bar
        // rather than nibbling at it.
        #expect(
            similarity > 0.99,
            """
            bf16 matmul diverged from the float32 reference at the NAX split-K \
            dispatch corner (M=\(m), N=\(n), K=\(k)): cosine \(similarity). \
            On NAX hardware this is ml-explore/mlx#3797 — the JIT kernel is \
            being instantiated with the output dtype instead of the input \
            dtype. Fix: ml-explore/mlx#3810.
            """)

        // A garbage kernel produced non-finite output, so pin that separately:
        // it is the symptom a cosine check could in principle average over.
        let allFinite = MLX.all(MLX.isFinite(underTest)).item(Bool.self)
        #expect(allFinite, "bf16 matmul produced non-finite values at the split-K corner")
    }

    /// The same operands one step *outside* the dispatch window (`K` below the
    /// 10240 floor) take the ordinary GEMM path. If this passes while the test
    /// above fails, the divergence is specific to the split-K dispatch rather
    /// than to bf16 matmul in general.
    @Test("bf16 matmul just below the split-K K threshold matches the reference")
    func bf16BelowSplitKThresholdMatchesFloat32Reference() {
        let m = 2048, n = 2048, k = 8192  // K < 10240 → ordinary path

        MLXRandom.seed(0x5EED)
        let aF32 = MLXRandom.normal([m, k], scale: 0.02)
        let bF32 = MLXRandom.normal([k, n], scale: 0.02)

        let reference = MLX.matmul(aF32, bF32)
        let underTest = MLX.matmul(aF32.asType(.bfloat16), bF32.asType(.bfloat16))
        MLX.eval(reference, underTest)

        let similarity = cosineSimilarity(underTest, reference)
        #expect(
            similarity > 0.99,
            "bf16 matmul diverged from the float32 reference off the split-K path: cosine \(similarity)")
    }
}
