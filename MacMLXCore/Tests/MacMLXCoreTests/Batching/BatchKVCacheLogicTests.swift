// Copyright © 2026 macMLX. English comments only.

import MLXLMCommon
import XCTest

@testable import MacMLXCore

/// MLX-free proof of the ragged-cohort prefill ARITHMETIC behind
/// ``BatchKVCache`` — the host-side left-padding / offset / filter-shift
/// book-keeping in ``BatchPrefillAssembly``.
///
/// These touch no `MLXArray`, so they run under a plain `swift test` in CI with
/// no Metal backend (the numeric array parity — that the cache actually places
/// keys, masks pad columns, and evicts rows correctly against the Python
/// reference — is the Metal-gated `BatchKVCacheParityTests`).
final class BatchKVCacheLogicTests: XCTestCase {

    // MARK: - leftPad

    func testLeftPadRaggedCohortRightJustifies() {
        // Real lengths [3, 5, 2] → Lmax 5, left padding [2, 0, 3] (the exact
        // cohort the parity fixture captures).
        let prompts = [[11, 12, 13], [21, 22, 23, 24, 25], [31, 32]]
        let (padded, leftPadding) = BatchPrefillAssembly.leftPad(prompts: prompts, padToken: 0)

        XCTAssertEqual(leftPadding, [2, 0, 3])
        XCTAssertEqual(padded[0], [0, 0, 11, 12, 13])
        XCTAssertEqual(padded[1], [21, 22, 23, 24, 25])
        XCTAssertEqual(padded[2], [0, 0, 0, 31, 32])
        // Every row is right-justified to the common width.
        XCTAssertTrue(padded.allSatisfy { $0.count == 5 })
        // The last column is always a REAL token — this is why the batched
        // forward samples every row from the single last position.
        XCTAssertEqual(padded.map { $0.last }, [13, 25, 32])
    }

    func testLeftPadEqualLengthCohortAddsNoPadding() {
        let prompts = [[1, 2, 3], [4, 5, 6]]
        let (padded, leftPadding) = BatchPrefillAssembly.leftPad(prompts: prompts, padToken: 99)
        XCTAssertEqual(leftPadding, [0, 0])
        XCTAssertEqual(padded, prompts)
    }

    func testLeftPadSinglePromptHasZeroPadding() {
        let (padded, leftPadding) = BatchPrefillAssembly.leftPad(prompts: [[7, 8, 9]], padToken: 0)
        XCTAssertEqual(leftPadding, [0])
        XCTAssertEqual(padded, [[7, 8, 9]])
    }

    func testLeftPadEmptyCohort() {
        let (padded, leftPadding) = BatchPrefillAssembly.leftPad(prompts: [], padToken: 0)
        XCTAssertTrue(padded.isEmpty)
        XCTAssertTrue(leftPadding.isEmpty)
    }

    func testLeftPadTokenValueIsPreserved() {
        let (padded, _) = BatchPrefillAssembly.leftPad(prompts: [[5], [6, 7]], padToken: 42)
        XCTAssertEqual(padded[0], [42, 5])
    }

    // MARK: - Derived geometry

    func testLeftPaddingFromLengthsMatchesLeftPad() {
        let lengths = [3, 5, 2]
        XCTAssertEqual(BatchPrefillAssembly.leftPadding(forLengths: lengths), [2, 0, 3])
        // Consistent with the full padder.
        let prompts = lengths.map { Array(repeating: 1, count: $0) }
        let (_, leftPadding) = BatchPrefillAssembly.leftPad(prompts: prompts, padToken: 0)
        XCTAssertEqual(BatchPrefillAssembly.leftPadding(forLengths: lengths), leftPadding)
    }

    func testInitialOffsetsNegateLeftPadding() {
        // A fresh BatchKVCache starts each row at RoPE offset -leftPadding[i]
        // so the first real token lands at position 0.
        XCTAssertEqual(BatchPrefillAssembly.initialOffsets(leftPadding: [2, 0, 3]), [-2, 0, -3])
    }

    func testFilterLeftShiftIsClampedMinimum() {
        // Fixture case: keeping rows with left padding [2, 3] → shift by 2,
        // leaving [0, 1] (the parity fixture's post-filter left padding).
        XCTAssertEqual(BatchPrefillAssembly.filterLeftShift(leftPadding: [2, 3]), 2)
        // A row already at zero padding blocks any shift.
        XCTAssertEqual(BatchPrefillAssembly.filterLeftShift(leftPadding: [0, 1]), 0)
        XCTAssertEqual(BatchPrefillAssembly.filterLeftShift(leftPadding: []), 0)
    }

    // MARK: - Converter refusal gate (MLX-free: returns nil before any MLXArray)

    func testConverterRefusesRotatingKVCache() {
        // Sliding-window models (BatchRotatingKVCache) are deferred — the
        // converter must refuse the whole cohort, not silently drop to stock.
        let caches: [KVCache] = [KVCacheSimple(), RotatingKVCache(maxSize: 512, keep: 0)]
        XCTAssertNil(BatchCacheConverter.makeBatchCaches(from: caches, leftPadding: [0, 0]))
    }

    func testConverterRefusesQuantizedAndHybridCaches() {
        XCTAssertNil(
            BatchCacheConverter.makeBatchCaches(from: [QuantizedKVCache()], leftPadding: [0]))
        XCTAssertNil(
            BatchCacheConverter.makeBatchCaches(
                from: [CacheList(MambaCache(), KVCacheSimple())], leftPadding: [0]))
        XCTAssertNil(
            BatchCacheConverter.makeBatchCaches(from: [ChunkedKVCache()], leftPadding: [0]))
    }

    func testConverterRefusesWholeCohortIfAnyCacheIsNonDense() {
        // One rotating layer poisons the whole model (mixed cache kinds would
        // desynchronise per-row offsets/masks).
        let caches: [KVCache] = [
            KVCacheSimple(), KVCacheSimple(), RotatingKVCache(maxSize: 256, keep: 0),
        ]
        XCTAssertNil(BatchCacheConverter.makeBatchCaches(from: caches, leftPadding: [1, 0]))
    }
}
