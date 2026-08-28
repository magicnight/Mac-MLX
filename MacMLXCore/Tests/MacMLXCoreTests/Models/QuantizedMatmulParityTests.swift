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

    // MARK: - mlx#4251 — qvm split-K dropped the tail columns

    /// `qvm_split_k` sized its dispatch grid with `N / bn`, a floor division, so
    /// any output width that is not a multiple of the tile left its final
    /// columns without a threadgroup — and therefore unwritten. Its own
    /// non-split-K sibling, ten lines away in the same file, already used the
    /// ceiling form; that internal disagreement is the clearest evidence the
    /// floor was wrong.
    ///
    /// Reaching it takes the whole dispatch chain: `M` below the vector limit,
    /// `transpose == false`, and `K >= 1024`. The tile is
    /// `min(group_size, 32) * 2`, i.e. 64 for every group size, while a
    /// non-transposed weight is quantized along `N` and so only has to be a
    /// multiple of `group_size`. At group size 64 or 128 that makes `N` a
    /// multiple of 64 and the floor is exact — it is **group size 32** that
    /// admits an `N` of 96, 160, 224 and leaves 32 columns behind.
    @Test("a group-32 quantized vector-matrix product writes its tail columns")
    func groupSize32QvmWritesTailColumns() {
        let k = 2048, n = 96, m = 1  // K >= 1024 → split-K; n % 64 == 32 → tail
        let groupSize = 32, bits = 4

        MLXRandom.seed(3)
        let weights = MLXRandom.normal([k, n]).asType(.float16)
        let (weightsQ, scales, biases) = MLX.quantized(
            weights, groupSize: groupSize, bits: bits)
        let dequantized = MLX.dequantized(
            weightsQ, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits).asType(.float32)

        let x = MLXRandom.normal([m, k]).asType(.bfloat16)
        let underTest = MLX.quantizedMatmul(
            x, weightsQ, scales: scales, biases: biases,
            transpose: false, groupSize: groupSize, bits: bits)
        let reference = MLX.matmul(x.asType(.float32), dequantized)
        MLX.eval(underTest, reference)

        let difference = maxAbsDiff(underTest, reference)
        #expect(
            difference < 1.0,
            """
            Group-32 quantized vector-matrix product diverged from its \
            dequantized reference (N=\(n), K=\(k), M=\(m)): max |diff| \
            \(difference). This is ml-explore/mlx#4251 — qvm_split_k floors \
            N / 64, so the last \(n % 64) columns are never dispatched.
            """)
    }

    // MARK: - mlx#3922 — sorted gather_qmm above 32K rows

    /// The sorted-RHS affine NAX kernel narrowed its remaining-row count to
    /// `short` before clamping it to the tile, so once the gathered activation
    /// matrix passes 32768 rows the intermediate wraps negative and early tiles
    /// leave output rows unwritten. Same shape as `#3631`, which this fork
    /// already carries, and it lands on the same place: quantized MoE.
    ///
    /// 32768 rows is not exotic for us. A sorted gather_qmm sees one row per
    /// (token, expert) pair, so an 8K-token prompt through top-4 routing is
    /// already at the seam, and the tiered SSD KV cache exists to serve prompts
    /// far longer than that.
    ///
    /// The check is deliberately coarse — did every row get written — rather
    /// than a tight numeric bound, because the failure is unwritten memory
    /// rather than drift.
    @Test("a sorted gather_qmm writes every row past the 32K seam")
    func sortedGatherQMMWritesEveryRowPast32K() {
        let m = 32769, k = 256, n = 64, experts = 2  // m > 32768 is the seam
        let groupSize = 64, bits = 4

        MLXRandom.seed(5)
        let weights = MLXRandom.normal([experts, n, k]).asType(.float16)
        let (weightsQ, scales, biases) = MLX.quantized(
            weights, groupSize: groupSize, bits: bits)

        let x = MLXRandom.normal([m, k], scale: 0.5).asType(.bfloat16)
        // Sorted: every row of the first half hits expert 0, the rest expert 1.
        let rhsIndices = MLX.concatenated([
            MLXArray.zeros([m / 2], type: Int32.self),
            MLXArray.ones([m - m / 2], type: Int32.self),
        ])

        let underTest = MLX.gatherQuantizedMM(
            x.expandedDimensions(axis: 1), weightsQ, scales: scales, biases: biases,
            rhsIndices: rhsIndices, transpose: true,
            groupSize: groupSize, bits: bits, sortedIndices: true)
        MLX.eval(underTest)

        // An unwritten row is left at whatever the buffer held; with inputs this
        // far from zero, a row that is exactly zero across all N did not get
        // written. Checking the row sums keeps this O(M) rather than O(M*N).
        let rowMagnitude = MLX.sum(MLX.abs(underTest.asType(.float32)), axis: -1)
            .reshaped([m])
        let emptyRows = MLX.sum((rowMagnitude .== 0).asType(.int32)).item(Int32.self)
        #expect(
            emptyRows == 0,
            """
            \(emptyRows) of \(m) rows came back all-zero from a sorted \
            gather_qmm. On NAX hardware this is ml-explore/mlx#3922 — the \
            remaining-row count narrows to short before the clamp, so past \
            32768 rows the early tiles never write.
            """)
    }
}
