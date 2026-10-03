import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest
@testable import MacMLXCore

/// Regression guard for batched single-token RoPE.
///
/// mlx-core before 0.32.0 corrupted `mx.fast.rope` for a `[B > 1, H, L = 1, D]`
/// input with a SCALAR offset (ml-explore/mlx#3494 / #3496, fixed by #3498):
/// bit-identical rows came out different. The fork carried that fix from
/// 2026-07-10 behind `BatchPositionedCacheWrapper`, a shim that routed every
/// batched decode through the per-row array-offset kernel instead. With the
/// base at core v0.32.2 the scalar path is correct and the shim is gone; this
/// test is what its own tests left behind.
///
/// Both offset forms must keep bit-identical rows bit-identical. The scalar
/// assertion is the one that matters: a failure there means the dependency
/// moved to a core without #3498 — do not ship that. The `.batch` form is what
/// ragged batching (A2b) positions rows with, so it is held too.
///
/// The MLX-touching assertions are gated with `requireMLXRuntimeOrSkip()` so a
/// bare `swift test` (no metallib) skips cleanly; they run under `xcodebuild`.
final class BatchedScalarRoPETests: XCTestCase {

    /// `max |row_r − row_0|` over a `[B, …]` tensor. 0.0 ⇒ rows are identical.
    private func crossRow(_ a: MLXArray) -> Float {
        abs(a - a[0]).max().item(Float.self)
    }

    /// Build `[B, 1, 1, D]` whose rows are bit-identical copies of one random
    /// row. Materialized so it is a genuine contiguous array, exactly like the
    /// real harness's batched decode input.
    private func identicalRows(dim: Int, batch: Int) -> MLXArray {
        let row = MLXRandom.normal([1, 1, 1, dim]).asType(.float16)
        let stacked = concatenated(Array(repeating: row, count: batch), axis: 0)
        stacked.eval()
        return stacked
    }

    func testBatchedSingleTokenRoPEKeepsIdenticalRowsIdentical() throws {
        try requireMLXRuntimeOrSkip()

        let batch = 2
        let dim = 256
        let position = 5
        let rope = RoPE(dimensions: dim, traditional: false, base: 10000, scale: 1)
        let x = identicalRows(dim: dim, batch: batch)

        // Scalar offset over a B > 1 single-token decode: the path #3498 fixed.
        let scalar = rope(x, offset: position)
        let scalarCrossRow = crossRow(scalar)
        XCTAssertFalse(scalarCrossRow.isNaN, "scalar batched-decode RoPE must not produce NaN")
        XCTAssertLessThan(
            scalarCrossRow, 1e-5,
            "scalar batched-decode RoPE must keep identical rows bit-identical — the "
                + "vendored core carries ml-explore/mlx#3498; a failure here means the "
                + "dependency lost it (observed cross-row = \(scalarCrossRow))")

        // Per-row array offset, as ragged batching positions rows.
        let batched = rope(x, offset: MLXArray(Array(repeating: Int32(position), count: batch)))
        let batchedCrossRow = crossRow(batched)
        XCTAssertLessThan(
            batchedCrossRow, 1e-5,
            "the .batch offset path must keep identical rows bit-identical "
                + "(observed cross-row = \(batchedCrossRow))")
    }
}
