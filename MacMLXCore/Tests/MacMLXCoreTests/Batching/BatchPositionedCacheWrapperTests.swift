// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MacMLXCore

/// Permanent regression guard for `BatchPositionedCacheWrapper` and
/// `batchPositioned(_:batch:)` — the continuous-batching RoPE correctness shim.
///
/// Three things are locked down:
///
///  1. **Wrapper contract** — a wrapped stock cache exposes `.batch` per-row RoPE
///     offsets that track the inner cache's live `offset`, and the factory wraps
///     every cache it is handed when that cache is a safe dense type.
///
///  2. **The decisive primitive (bug vs. fix)** — on `MLXNN.RoPE` fed
///     `[B = 2, H = 1, L = 1, D]` identical rows, a SCALAR offset diverges /
///     NaNs across rows (the mlx-swift 0.31.6 batched single-token decode bug,
///     ml-explore/mlx#3494 · #3496, mlx-swift#441), while a `.batch([2])` offset
///     yields bit-identical rows (cross-row 0.0). This is exactly the routing the
///     wrapper performs, reduced to the one primitive that matters.
///
///  3. **Refusal safety** — `batchPositioned(_:batch:)` returns `nil` (refuses
///     the whole batch) rather than wrapping a `CacheList` or `QuantizedKVCache`,
///     both of which would otherwise crash later: `CacheList.update` and
///     `QuantizedKVCache.update` are both `fatalError` (hybrid/quantized models
///     reach their real path via a concrete-type cast or capability probe this
///     wrapper cannot satisfy). A correctness shim must fail loudly and early,
///     never silently no-op or crash downstream.
///
/// ## Deletion tripwire
/// The "scalar path is broken" assertion is intentionally a canary. Once
/// mlx-swift vendors mlx-core ≥ 0.32.0 the SCALAR path becomes correct, that
/// assertion starts failing, and this test is the signal that
/// `BatchPositionedCacheWrapper` / `batchPositioned(_:batch:)` can be deleted.
///
/// The MLX-touching assertions are gated with `requireMLXRuntimeOrSkip()` so a
/// bare `swift test` (no metallib) skips cleanly; they run under `xcodebuild`.
final class BatchPositionedCacheWrapperTests: XCTestCase {

    // MARK: - Helpers

    /// `max |row_r − row_0|` over a `[B, …]` tensor. 0.0 ⇒ rows are identical.
    private func crossRow(_ a: MLXArray) -> Float {
        abs(a - a[0]).max().item(Float.self)
    }

    /// The `.batch` RoPE offsets of a cache as host `Int32`s, or `nil` if the
    /// cache did not surface a `.batch` offset.
    private func batchOffsetValues(_ cache: any KVCache) -> [Int32]? {
        guard case .batch(let offsets) = cache.ropeOffset else { return nil }
        return offsets.asArray(Int32.self)
    }

    /// Build `[B, 1, 1, D]` whose rows are bit-identical copies of one random
    /// row. Materialized so it is a genuine contiguous array, exactly like the
    /// real harness's batched decode input.
    private func identicalRows(dim: Int, batch: Int) -> MLXArray {
        let row = MLXRandom.normal([1, 1, 1, dim]).asType(.float16)  // [1, 1, 1, D]
        let stacked = concatenated(Array(repeating: row, count: batch), axis: 0)  // [B, 1, 1, D]
        stacked.eval()
        return stacked
    }

    // MARK: - Wrapper contract

    func testWrappedCacheExposesBatchRopeOffsetTrackingInnerOffset() throws {
        try requireMLXRuntimeOrSkip()

        let batch = 2
        let inner = KVCacheSimple()
        let wrapped = BatchPositionedCacheWrapper(wrapping: inner, batch: batch)

        // Default stock caches report `.scalar`; the wrapper must report `.batch`.
        if case .scalar = inner.ropeOffset {} else {
            XCTFail("precondition: stock KVCacheSimple should report a .scalar ropeOffset")
        }
        XCTAssertEqual(
            batchOffsetValues(wrapped), [0, 0],
            "fresh wrapped cache should expose a per-row .batch offset of zeros")

        // Advancing the inner cache by a 3-token prefill must be reflected per-row.
        let headDim = 8
        let kvHeads = 2
        let keys = MLXRandom.normal([batch, kvHeads, 3, headDim]).asType(.float16)
        let values = MLXRandom.normal([batch, kvHeads, 3, headDim]).asType(.float16)
        _ = wrapped.update(keys: keys, values: values)

        XCTAssertEqual(wrapped.offset, 3, "wrapper must forward the inner cache offset")
        XCTAssertEqual(
            batchOffsetValues(wrapped), [3, 3],
            "batch offset must track the live inner offset, not a captured value")
    }

    func testBatchPositionedFactoryWrapsEveryCache() throws {
        try requireMLXRuntimeOrSkip()

        let caches: [KVCache] = [KVCacheSimple(), KVCacheSimple(), KVCacheSimple()]
        guard let wrapped = batchPositioned(caches, batch: 4) else {
            XCTFail("plain KVCacheSimple caches must be safely wrappable")
            return
        }

        XCTAssertEqual(wrapped.count, caches.count)
        for cache in wrapped {
            XCTAssertTrue(
                cache is BatchPositionedCacheWrapper,
                "every cache should be wrapped by the factory")
            XCTAssertEqual(
                batchOffsetValues(cache), [0, 0, 0, 0],
                "each wrapped cache should expose a 4-row .batch offset")
        }
    }

    func testBatchPositionedFactoryWrapsRotatingKVCache() throws {
        try requireMLXRuntimeOrSkip()

        let caches: [KVCache] = [RotatingKVCache(maxSize: 512, keep: 0), KVCacheSimple()]
        guard let wrapped = batchPositioned(caches, batch: 2) else {
            XCTFail("RotatingKVCache + KVCacheSimple must both be safely wrappable")
            return
        }

        XCTAssertEqual(wrapped.count, caches.count)
        XCTAssertTrue(wrapped.allSatisfy { $0 is BatchPositionedCacheWrapper })
    }

    // MARK: - Refusal safety

    func testBatchPositionedRefusesCacheListRatherThanCrashingLater() throws {
        try requireMLXRuntimeOrSkip()

        // Hybrid models (FalconH1, BaichuanM1) assemble a `CacheList` per layer
        // and reach children via `cache?[0] as? MambaCache` subscripting.
        // Wrapping the CacheList itself would defeat that cast AND crash
        // (`CacheList.update` is a fatalError) — the factory must refuse instead.
        let caches: [KVCache] = [CacheList(MambaCache(), KVCacheSimple())]
        XCTAssertNil(
            batchPositioned(caches, batch: 2),
            "CacheList is not a safely-wrappable dense cache type — must refuse, not wrap")
    }

    func testBatchPositionedRefusesQuantizedKVCacheRatherThanCrashingLater() throws {
        try requireMLXRuntimeOrSkip()

        // Models reach the quantized path via `cache as? QuantizedKVCacheProtocol`.
        // A BatchPositionedCacheWrapper does not conform to that protocol, so
        // wrapping a real QuantizedKVCache defeats the probe and falls through to
        // `update`, which QuantizedKVCache implements as a fatalError.
        let caches: [KVCache] = [QuantizedKVCache()]
        XCTAssertNil(
            batchPositioned(caches, batch: 2),
            "QuantizedKVCache is not a safely-wrappable dense cache type — must refuse, not wrap")
    }

    func testBatchPositionedRefusesWholeBatchIfAnyCacheIsUnsafe() throws {
        try requireMLXRuntimeOrSkip()

        // Mixed dense + unsafe: refuse the whole call rather than wrapping the
        // dense ones and silently leaving the unsafe one on the buggy scalar path.
        let caches: [KVCache] = [KVCacheSimple(), CacheList(MambaCache(), KVCacheSimple())]
        XCTAssertNil(batchPositioned(caches, batch: 2))
    }

    // MARK: - Decisive primitive: bug vs. fix

    func testScalarRopeIsBrokenWhileBatchOffsetIsCorrect() throws {
        try requireMLXRuntimeOrSkip()

        MLXRandom.seed(0)
        let batch = 2
        let dim = 256
        let position = 5
        let rope = RoPE(dimensions: dim, traditional: false, base: 10000, scale: 1)

        // Identical rows must, under a correct kernel, produce identical outputs.
        let x = identicalRows(dim: dim, batch: batch)  // [B, 1, 1, D]

        // Fixed path — the per-row array offset the wrapper installs. This is the
        // permanent guard: it must stay bit-identical across rows.
        let batched = rope(x, offset: MLXArray(Array(repeating: Int32(position), count: batch)))
        let batchedCrossRow = crossRow(batched)
        XCTAssertLessThan(
            batchedCrossRow, 1e-5,
            "the .batch offset path must keep identical rows bit-identical (the fix)")

        // Buggy path — a SCALAR offset over a `B > 1` single-token decode. Lanes
        // 1..<B read uninitialized memory, so rows diverge (or NaN). This is the
        // deletion tripwire: if it ever passes cleanly, mlx-swift has vendored the
        // mlx-core ≥ 0.32 fix (mlx-swift#441) and BatchPositionedCacheWrapper /
        // batchPositioned(_:batch:) can be removed.
        let scalar = rope(x, offset: position)
        let scalarCrossRow = crossRow(scalar)
        XCTAssertTrue(
            scalarCrossRow.isNaN || scalarCrossRow > 1e-3,
            "scalar batched-decode RoPE is expected broken (cross-row NaN/divergent); "
                + "a clean pass here signals mlx-swift#441 landed and the wrapper is deletable "
                + "(observed cross-row = \(scalarCrossRow))")
    }
}
