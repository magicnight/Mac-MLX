import Testing
import Foundation
import MLX
@testable import MacMLXCore

/// Guards the long-context GQA decode attention kernel against
/// `ml-explore/mlx#4431`.
///
/// `sdpa_vector_2pass_1_gqa` (`mlx/backend/metal/kernels/sdpa_vector.h`)
/// offset its key and value pointers by the KV head alone, never by the batch
/// index, so every batch element attended over batch 0's keys and values.
/// Batch 0 came out right and every other row was silently wrong. The kernel
/// is compiled ahead of time into `default.metallib`
/// (`scaled_dot_product_attention.metal` includes the header), so the fix is
/// the patched `mlx-generated/metal/sdpa_vector.h`, not the submodule.
///
/// The kernel is selected only when all of these hold
/// (`ScaledDotProductAttention::eval_gpu` → `sdpa_vector_2pass`):
///
/// - a single query (`q_len == 1`; vector mode is `q_len <= 8`)
/// - no array mask and no sinks
/// - GQA factor exactly 8, with matching head dims of 64 or 128
/// - at least 8192 keys (the GQA two-pass route itself starts at 4096)
///
/// and the defect needs a batch above 1 on top. No production path reaches it
/// today: single-stream decode has batch 1, and batched decode always passes
/// the array mask that `BatchKVCache.makeMask` builds, which takes the kernel
/// off the `_gqa` variant. A batched path that dropped the mask when no row
/// is padded would land on it at 8K context with any 8:1 GQA model.
///
/// The reference is plain softmax attention over the keys and values repeated
/// across the GQA groups, on the CPU stream: an independent implementation,
/// and float32 matmuls there are not subject to M5's TF32 default. Against the
/// fork's unpatched base (1026d23) row 0 agreed and row 1 was off by 0.61
/// (16/2 heads, dim 128) and 0.32 (64/8 heads, dim 64).
@Suite(
    "SDPA vector GQA batch parity",
    .enabled(if: mlxMetallibIsAvailable, "Requires default.metallib (run under xcodebuild)"))
struct SDPAVectorGQABatchParityTests {

    @Test(
        "batched single-query GQA attention at 8K keys reads each row's own keys",
        arguments: [(16, 2, 128), (64, 8, 64)])
    func batchedGQADecodeMatchesReference(queryHeads: Int, kvHeads: Int, headDim: Int) {
        let batch = 2, keyLength = 8192
        let repeats = queryHeads / kvHeads  // 8: the `_gqa` kernel's only factor
        let scale: Float = 1

        // Explicit keys rather than the global seed: the argument cases run in
        // parallel and would otherwise interleave draws from one generator.
        let keys = MLXRandom.split(key: MLXRandom.key(0x4431), into: 3)
        let q = MLXRandom.normal([batch, queryHeads, 1, headDim], scale: 0.5, key: keys[0])
        let k = MLXRandom.normal([batch, kvHeads, keyLength, headDim], scale: 0.5, key: keys[1])
        let v = MLXRandom.normal([batch, kvHeads, keyLength, headDim], scale: 0.5, key: keys[2])
        MLX.eval(q, k, v)

        let underTest = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .none)

        let kr = MLX.repeated(k, count: repeats, axis: 1, stream: .cpu)
        let vr = MLX.repeated(v, count: repeats, axis: 1, stream: .cpu)
        let scores = MLX.matmul(q, kr.transposed(0, 1, 3, 2), stream: .cpu) * scale
        let reference = MLX.matmul(
            MLX.softmax(scores, axis: -1, precise: true, stream: .cpu), vr, stream: .cpu)
        MLX.eval(underTest, reference)

        for row in 0 ..< batch {
            let difference = MLX.max(MLX.abs(underTest[row] - reference[row])).item(Float.self)
            // The kernel accumulates in float32 against a float32 reference, so
            // honest error sits near 1e-6. Another row's keys and values move
            // outputs of this scale by tenths.
            #expect(
                difference < 1e-3,
                """
                Batch row \(row) of single-query GQA attention diverged from the \
                reference (heads \(queryHeads)/\(kvHeads), head dim \(headDim), \
                \(keyLength) keys): max |diff| \(difference). If row 0 agrees \
                and the others do not, this is ml-explore/mlx#4431 — \
                sdpa_vector_2pass_1_gqa reads batch 0's keys and values for \
                every row.
                """)
        }
    }
}
