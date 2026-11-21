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
/// carries the fix (upstream PR #4903, open when it was picked); it is host
/// C++, so the submodule pointer is the whole change.
///
/// The kernel is selected on a NAX GPU (M5) for full attention at head
/// dimension 256 or 512 once the query is long enough for the steel path;
/// Gemma 4's head dimension is 256, and `FullPathVLMImageGateSmokeTests` died
/// here on mlx-swift 0.33.0 before the fix. On other GPUs these shapes take
/// another kernel and the test only checks the numbers.
///
/// A failure of the build is a crash of the test process, not a failed
/// expectation. The reference is plain softmax attention on the CPU stream.
@Suite(
    "SDPA split-D kernel builds at run time",
    .enabled(if: mlxMetallibIsAvailable, "Requires default.metallib (run under xcodebuild)"))
struct SDPASplitDKernelJITTests {

    @Test("causal attention at head dimension 256 and 512 over 1,024 tokens", arguments: [256, 512])
    func longCausalAttentionMatchesReference(headDim: Int) {
        let heads = 2, length = 1024
        let scale = 1 / Float(headDim).squareRoot()

        let keys = MLXRandom.split(key: MLXRandom.key(0x4832 + UInt64(headDim)), into: 3)
        let q = MLXRandom.normal([1, heads, length, headDim], scale: 0.5, key: keys[0])
        let k = MLXRandom.normal([1, heads, length, headDim], scale: 0.5, key: keys[1])
        let v = MLXRandom.normal([1, heads, length, headDim], scale: 0.5, key: keys[2])
        MLX.eval(q, k, v)

        let underTest = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .causal)

        let positions = MLXArray(Int32(0) ..< Int32(length))
        let visible = positions.expandedDimensions(axis: 1) .>= positions.expandedDimensions(axis: 0)
        var scores = MLX.matmul(q, k.transposed(0, 1, 3, 2), stream: .cpu) * scale
        scores = MLX.which(visible, scores, MLXArray(-Float.infinity), stream: .cpu)
        let reference = MLX.matmul(
            MLX.softmax(scores, axis: -1, precise: true, stream: .cpu), v, stream: .cpu)
        MLX.eval(underTest, reference)

        let difference = MLX.max(MLX.abs(underTest - reference)).item(Float.self)
        // Float32 on both sides; M5 runs the GPU matmuls as TF32 by default,
        // which costs these outputs a few 1e-3 at most.
        #expect(
            difference < 2e-2,
            "causal attention at head dimension \(headDim) diverged from the reference: max |diff| \(difference)")
    }
}
