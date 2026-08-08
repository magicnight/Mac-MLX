import Testing
import Foundation
import MLX
@testable import MacMLXCore

/// Triage probe for `ml-explore/mlx#3560`, which our pinned fork predates.
///
/// The upstream change is one character, and its description is "Fixed typo :)",
/// but the line is `BaseMMAFrag<T, 8, 8>`'s bounds-checked load in
/// `steel/gemm/mma.h`:
///
///     - src[(off_x + i) * str_x + (off_x + j) * str_y]
///     + src[(off_x + i) * str_x + (off_y + j) * str_y]
///
/// `off_x` where `off_y` belongs. That is the *safe* load — the one steel GEMM
/// uses for a tile that hangs off the edge of the matrix — so a wrong column
/// can be read into an edge fragment and no bounds check fires. It only
/// diverges when `off_x != off_y`, so square-aligned tiles hide it.
///
/// This probe exists to answer whether that reaches us, not to assert a fix.
/// Shapes are deliberately awkward: dimensions that are not multiples of the
/// 32/64-element tiling, so partial tiles are unavoidable, across several
/// aspect ratios so the fragment offsets differ rather than coincide.
@Suite(
    "Steel GEMM safe-load parity",
    .enabled(if: mlxMetallibIsAvailable, "Requires default.metallib (run under xcodebuild)"))
struct SteelGemmSafeLoadParityTests {

    private func maxAbsDiff(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        MLX.max(MLX.abs(lhs.asType(.float32) - rhs.asType(.float32))).item(Float.self)
    }

    /// bf16 matmuls at ragged shapes, each against a float32 reference of the
    /// same operands. A correct GEMM stays at bf16 rounding scale here; the
    /// edge-fragment defect reads unrelated matrix entries, which lands orders
    /// of magnitude away.
    @Test(
        "ragged-shape bf16 matmul matches a float32 reference",
        arguments: [
            (65, 65, 65),        // one element past a 64-tile in every dimension
            (129, 97, 257),      // all three ragged, different remainders
            (33, 1025, 129),     // wide N, ragged M and K
            (1025, 33, 129),     // wide M, ragged N and K
            (2049, 2049, 129),   // large and ragged in both output dimensions
        ])
    func raggedShapeMatmulMatchesFloat32Reference(m: Int, n: Int, k: Int) {
        MLXRandom.seed(0xA11CE)
        let a = MLXRandom.normal([m, k], scale: 0.05)
        let b = MLXRandom.normal([k, n], scale: 0.05)

        let reference = MLX.matmul(a, b)
        let underTest = MLX.matmul(a.asType(.bfloat16), b.asType(.bfloat16))
        MLX.eval(reference, underTest)

        let difference = maxAbsDiff(underTest, reference)

        // bf16 has ~3 decimal digits; at K <= 2049 with 0.05-scale operands the
        // honest error stays well under 0.05. A fragment reading the wrong
        // column produces a difference on the order of the values themselves.
        #expect(
            difference < 0.05,
            """
            bf16 matmul diverged from its float32 reference at a ragged shape \
            (M=\(m), N=\(n), K=\(k)): max |diff| \(difference). If this fails, \
            ml-explore/mlx#3560 — the off_x/off_y mix-up in the steel GEMM \
            safe load — reaches us and should be carried onto the fork.
            """)
        #expect(
            MLX.all(MLX.isFinite(underTest)).item(Bool.self),
            "ragged-shape bf16 matmul produced non-finite values")
    }
}
