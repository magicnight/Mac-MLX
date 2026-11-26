import Testing
import Foundation
import MLX
@testable import MacMLXCore

/// Guards the split-D NAX attention kernel's runtime build against
/// `ml-explore/mlx#4832`.
///
/// `get_steel_attention_nax_kernel` (`mlx/backend/metal/jit_kernels.cpp`)
/// instantiates one of two templates and, from core v0.32.3 on, handed both
/// the same argument list. `attention_nax` takes a value head dimension `BV`;
/// `attention_nax_dsplit`, the kernel for head dimensions 256 and 512, does
/// not, so its instantiation had one argument too many and Metal refused the
/// source: "Unable to build metal library from source", a fatal error.
/// mlx-swift builds its kernels at run time (`MLX_METAL_JIT`), where the
/// Python wheel ships them compiled, so only a Swift build sees it. The fork
/// carries the fix (upstream PR #4903, merged after core v0.32.4); it is host
/// C++, so the submodule pointer is the whole change.
///
/// The kernel is selected on a NAX GPU (M5) for fused full attention at head
/// dimension 256 or 512 (`ScaledDotProductAttention::use_fallback` and the
/// NAX dispatch in `scaled_dot_product_attention.cpp`), which needs:
///
/// - inputs that are not float32, or TF32 left on (`MLX_ENABLE_TF32`); the
///   inputs here are bfloat16, what a Gemma 4 checkpoint runs in, so the
///   variable does not decide whether the test reaches the kernel;
/// - at head dimension 256, a query of at least 1,024 tokens;
/// - at head dimension 512, at least 1,024 query blocks of 32 rows over
///   batch and heads, hence 32 heads at 1,024 tokens.
///
/// Gemma 4's head dimension is 256, and `FullPathVLMImageGateSmokeTests`
/// died here on mlx-swift 0.33.0 before the fix. On other GPUs these shapes
/// take another kernel and the test only checks the numbers.
///
/// A failure of the build is a crash of the test process, not a failed
/// expectation. The reference is plain softmax attention in float32 over the
/// same bfloat16 values. It runs on the CPU stream from end to end and is
/// evaluated on its own, so no GPU work waits on the CPU: such a wait can end
/// the process with a GPU timeout when the CPU work is long (mlx#2827). The
/// first version of this test scaled the scores and compared the results on
/// the GPU, and the CI runner's Metal job ended with that timeout.
@Suite(
    "SDPA split-D kernel builds at run time",
    .enabled(if: mlxMetallibIsAvailable, "Requires default.metallib (run under xcodebuild)"))
struct SDPASplitDKernelJITTests {

    @Test("causal attention over 1,024 tokens", arguments: [(256, 2), (512, 32)])
    func longCausalAttentionMatchesReference(headDim: Int, heads: Int) {
        let length = 1024
        let scale = 1 / Float(headDim).squareRoot()

        let keys = MLXRandom.split(key: MLXRandom.key(0x4832 + UInt64(headDim)), into: 3)
        let shape = [1, heads, length, headDim]
        let q = MLXRandom.normal(shape, scale: 0.5, key: keys[0]).asType(.bfloat16)
        let k = MLXRandom.normal(shape, scale: 0.5, key: keys[1]).asType(.bfloat16)
        let v = MLXRandom.normal(shape, scale: 0.5, key: keys[2]).asType(.bfloat16)
        MLX.eval(q, k, v)

        let underTest = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .causal
        ).asType(.float32)
        MLX.eval(underTest)

        let cpu = StreamOrDevice.cpu
        let q32 = q.asType(.float32, stream: cpu)
        let k32 = k.asType(.float32, stream: cpu)
        let v32 = v.asType(.float32, stream: cpu)
        let positions = MLXArray(Int32(0) ..< Int32(length))
        let visible = MLX.greaterEqual(
            positions.expandedDimensions(axis: 1, stream: cpu),
            positions.expandedDimensions(axis: 0, stream: cpu), stream: cpu)
        var scores = MLX.multiply(
            MLX.matmul(q32, k32.transposed(0, 1, 3, 2, stream: cpu), stream: cpu), scale,
            stream: cpu)
        scores = MLX.which(visible, scores, MLXArray(-Float.infinity), stream: cpu)
        let reference = MLX.matmul(
            MLX.softmax(scores, axis: -1, precise: true, stream: cpu), v32, stream: cpu)
        MLX.eval(reference)

        // Per query row, against that row's own largest output: a late row
        // averages over a thousand values and is far smaller than the first,
        // so one absolute bound would let a wrong late row through.
        let rowError = MLX.abs(MLX.subtract(underTest, reference, stream: cpu), stream: cpu)
            .max(axis: -1, stream: cpu)
        let rowScale = MLX.abs(reference, stream: cpu).max(axis: -1, stream: cpu)
        let relative = MLX.max(MLX.divide(rowError, rowScale, stream: cpu), stream: cpu)
            .item(Float.self)
        // The output is rounded to bfloat16 (8 significand bits, 0.4% of an
        // element) and the kernel sums in another order than the reference.
        // A wrong half of the head dimension or a wrong row moves a row by
        // its own size.
        #expect(
            relative < 0.03,
            "causal attention at head dimension \(headDim) diverged from the reference: worst row off by \(relative) of its largest output")
    }
}
