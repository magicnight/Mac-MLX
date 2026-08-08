import Testing
import Foundation
import MLX
@testable import MacMLXCore

/// Two upstream mlx quantized-matmul defects that our pinned fork predates.
/// Both produce silently wrong numbers — no crash, no NaN guard — which is
/// exactly the failure mode our fork's release scan
/// (`scripts/scan-upstream-mlx-fixes.sh`) exists to stop shipping.
///
/// - `ml-explore/mlx#3497` — `qvm` mishandles a stride-0 batch dimension, the
///   shape GQA produces when a quantized KV cache is broadcast across repeat
///   groups. Not hardware-gated: upstream reproduced it on an M4 Pro.
/// - `ml-explore/mlx#3631` — the NAX `qmm` edge-tile bounds are computed in
///   `short`, which wraps past 32767. Reachable through an lm_head whose vocab
///   is unaligned and larger than 2^15 — MiniCPM3-4B (73448) is one, and one
///   we list as supported.
///
/// Both suites are ordinary matmul parity checks on hardware that does not take
/// the affected path, so they are safe to run anywhere; it is Apple silicon
/// with a quantized KV cache (#3497) and M5-class parts (#3631) that they pin.
@Suite(
    "Quantized matmul parity",
    .enabled(if: mlxMetallibIsAvailable, "Requires default.metallib (run under xcodebuild)"))
struct QuantizedMatmulParityTests {

    /// Largest absolute elementwise difference, computed in float32.
    private func maxAbsDiff(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        MLX.max(MLX.abs(lhs.asType(.float32) - rhs.asType(.float32))).item(Float.self)
    }

    // MARK: - mlx#3497 — stride-0 batch dimension (GQA + quantized KV cache)

    /// `quantizedScaledDotProductAttention` (mlx-swift-lm `KVCache.swift`)
    /// broadcasts a quantized KV cache across GQA repeat groups with
    /// `expandedDimensions(_:axis: -3)`, which leaves a **stride-0** batch
    /// dimension on the weights. `ensure_row_contiguous_matrix` only checks the
    /// last two dimensions, so that zero stride reaches the kernel verbatim;
    /// `qvm` — chosen when M is below the vector limit — reads the wrong rows.
    ///
    /// Upstream's numbers (M4 Pro): quantization noise is ~2e-4, while the
    /// broadcast path drifts to ~0.08–0.14 for `M >= 2` once N passes ~2048.
    /// `M == 1` is always correct, which is why plain single-token decode never
    /// showed it — but speculative decoding verifies several tokens per step,
    /// landing squarely in the broken range.
    @Test(
        "a GQA-broadcast quantized matmul matches its dequantized reference",
        arguments: [(2048, 1), (2048, 2), (4096, 2)])
    func gqaBroadcastMatchesDequantizedReference(n: Int, m: Int) {
        let batch = 1, kvHeads = 4, repeats = 4, headDim = 256
        let queryHeads = kvHeads * repeats
        let groupSize = 64, keyBits = 8

        MLXRandom.seed(42)
        let keys = MLXRandom.normal([batch, kvHeads, n, headDim]).asType(.float16)
        let (keysQ, keyScales, keyBiases) = MLX.quantized(
            keys, groupSize: groupSize, bits: keyBits)

        let queries = (MLXRandom.normal([batch, queryHeads, m, headDim])
            * pow(Float(headDim), -0.5)).asType(.float16)
        let grouped = queries.reshaped([batch, kvHeads, repeats, m, headDim])

        // The values half is the one that matters. `QuantizedMatmul::eval_gpu`
        // routes a transposed matmul to qmm/qmv and only reaches `qvm` /
        // `qvm_split_k` — the kernels #3497 fixes — when `transpose` is false,
        // which is the scores @ values product. Testing the keys product alone
        // never touches the defect.
        let valueBits = 4
        let values = MLXRandom.normal([batch, kvHeads, n, headDim]).asType(.float16)
        let (valuesQ, valueScales, valueBiases) = MLX.quantized(
            values, groupSize: groupSize, bits: valueBits)

        let keysDequantized = MLX.dequantized(
            keysQ, scales: keyScales, biases: keyBiases,
            groupSize: groupSize, bits: keyBits)
        let valuesDequantized = MLX.dequantized(
            valuesQ, scales: valueScales, biases: valueBiases,
            groupSize: groupSize, bits: valueBits)

        // Reference: dequantize, then plain float matmuls of the same operands.
        let referenceScores = MLX.softmax(
            MLX.matmul(
                grouped.asType(.float32),
                expandedDimensions(keysDequantized, axis: -3).asType(.float32)
                    .transposed(0, 1, 2, 4, 3)),
            axis: -1)
        let reference = MLX.matmul(
            referenceScores,
            expandedDimensions(valuesDequantized, axis: -3).asType(.float32))

        // Under test: the broadcast the GQA attention path actually performs —
        // both products, against weights carrying a stride-0 batch dimension.
        let scores = MLX.softmax(
            MLX.quantizedMatmul(
                grouped,
                expandedDimensions(keysQ, axis: -3),
                scales: expandedDimensions(keyScales, axis: -3),
                biases: keyBiases.map { expandedDimensions($0, axis: -3) },
                transpose: true, groupSize: groupSize, bits: keyBits),
            axis: -1)
        let underTest = MLX.quantizedMatmul(
            scores,
            expandedDimensions(valuesQ, axis: -3),
            scales: expandedDimensions(valueScales, axis: -3),
            biases: valueBiases.map { expandedDimensions($0, axis: -3) },
            transpose: false, groupSize: groupSize, bits: valueBits)
        MLX.eval(reference, underTest)

        let difference = maxAbsDiff(underTest, reference)

        // Quantization alone lands around 2e-4 here. 0.01 sits far above that
        // and far below the ~0.08 upstream measured when the bug is live, so
        // this neither trips on rounding nor passes while the rows are wrong.
        #expect(
            difference < 0.01,
            """
            GQA-broadcast quantized matmul diverged from its dequantized \
            reference (N=\(n), M=\(m)): max |diff| \(difference). This is \
            ml-explore/mlx#3497 — qvm reading a stride-0 batch dimension. \
            Fix: carry the cherry-pick on the mlx fork.
            """)
    }

    // MARK: - mlx#3631 — int16 wrap in NAX qmm edge-tile bounds

    /// `qmm_t_nax_tgp_impl` computes per-simdgroup edge sizes with
    /// `short(N - (y_col + tn))`, which wraps past 32767. With an unaligned N
    /// above 2^15 a column band is left unwritten or corrupted whenever the
    /// M-tile is partial.
    ///
    /// The shape is upstream's own repro, and its provenance matters: N =
    /// 73448 is MiniCPM3-4B's vocab (`% 64 == 40`), M = 16 is an ordinary short
    /// prompt, and upstream reports wrong logits — hence wrong generations — on
    /// an M5 for exactly this. Every other vocab we support is a multiple of
    /// 64 and takes the aligned branch.
    @Test("an unaligned large-N quantized matmul matches its dequantized reference")
    func unalignedLargeNMatchesDequantizedReference() {
        let k = 2560, n = 73448, m = 16  // n % 64 == 40, n > 2^15, m % 32 != 0
        let groupSize = 64, bits = 4

        MLXRandom.seed(0)
        let weights = MLXRandom.normal([n, k]).asType(.float16)
        let (weightsQ, scales, biases) = MLX.quantized(
            weights, groupSize: groupSize, bits: bits)
        let dequantized = MLX.dequantized(
            weightsQ, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits).asType(.float32)

        let x = MLXRandom.normal([m, k]).asType(.bfloat16)
        let underTest = MLX.quantizedMatmul(
            x, weightsQ, scales: scales, biases: biases,
            transpose: true, groupSize: groupSize, bits: bits)
        let reference = MLX.matmul(x.asType(.float32), dequantized.transposed())
        MLX.eval(underTest, reference)

        let difference = maxAbsDiff(underTest, reference)

        // Upstream measures ~13 on the broken path against quantization-level
        // error otherwise. bf16 inputs at K=2560 leave real headroom, so the
        // bar sits well above honest error and an order of magnitude below the
        // failure.
        #expect(
            difference < 1.0,
            """
            Unaligned large-N quantized matmul diverged from its dequantized \
            reference (N=\(n), K=\(k), M=\(m)): max |diff| \(difference). On \
            NAX hardware this is ml-explore/mlx#3631 — edge-tile bounds \
            computed in short() wrapping past 32767.
            """)
    }
}
