// Copyright © 2026 macMLX. English comments only.

import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MacMLXCore

/// Numerical parity: the Swift ``BatchKVCache`` must reproduce mlx-lm's
/// `BatchKVCache` (main) step-for-step on a RAGGED cohort — per-row offset
/// book-keeping, left-padding masks, buffer placement, and the
/// filter/extract eviction primitives.
///
/// Fixture `Fixtures/batch_kv_cache_fixture.safetensors` is captured offline by
/// `docs/reference/capture_batch_kv_cache.py` (uv venv, mlx 0.32.0 + mlx-lm
/// 0.31.3). The captured cohort has real lengths `[3, 5, 2]`, i.e.
/// `leftPadding = [2, 0, 3]`, left-padded to `Lmax = 5`, then two decode steps,
/// a `filter([0, 2])`, and an `extract(1)`.
///
/// ## Why the plain gate (not the strict-parity gate)
/// Every operation exercised here is EXACT — zero-fill, concatenate, slice,
/// integer compare, gather — with NO floating-point accumulation (the keys and
/// values are placed and sliced back, never multiplied). So unlike the DeepSeek
/// attention/model parity tests (matmul/softmax, which drift on paravirtualized
/// Metal), this reproduces bit-for-bit anywhere MLX runs, including CI Metal.
/// It therefore uses `requireMLXRuntimeOrSkip()` and runs wherever a metallib
/// is present.
///
/// ## RoPE-version safety
/// `BatchKVCache`'s update / mask / filter / extract are pure array ops and
/// touch NO RoPE, so the fixture is unaffected by the mlx 0.32.0 (capture) vs
/// mlx-swift 0.31.1 (Swift core) batched-RoPE delta.
final class BatchKVCacheParityTests: XCTestCase {

    private let leftPadding = [2, 0, 3]
    private let lmax = 5

    // MARK: - Fixture helpers

    private func loadFixture() throws -> [String: MLXArray] {
        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "batch_kv_cache_fixture", withExtension: "safetensors",
                subdirectory: "Fixtures"),
            "batch_kv_cache_fixture not found in test bundle")
        return try MLX.loadArrays(url: url)
    }

    private func assertClose(_ actual: MLXArray, _ expected: MLXArray, _ label: String) {
        XCTAssertEqual(actual.shape, expected.shape, "\(label): shape mismatch")
        XCTAssertTrue(
            allClose(actual, expected, rtol: 1e-4, atol: 1e-4).item(Bool.self),
            "\(label): diverges from the Python reference")
    }

    /// Exact integer-array equality (offsets, left-padding, idx).
    private func assertIntEqual(_ actual: MLXArray, _ expected: MLXArray, _ label: String) {
        XCTAssertEqual(actual.shape, expected.shape, "\(label): shape mismatch")
        let diff = abs(actual.asType(.int32) - expected.asType(.int32)).max().item(Int32.self)
        XCTAssertEqual(diff, 0, "\(label): integer arrays differ")
    }

    /// The mask must be an EXACT array mask matching the captured bool mask.
    private func assertMaskEqual(
        _ mode: MLXFast.ScaledDotProductAttentionMaskMode, _ expected: MLXArray, _ label: String
    ) {
        guard case .array(let mask) = mode else {
            XCTFail("\(label): expected an array mask, got a symbolic mode")
            return
        }
        assertIntEqual(mask.asType(.uint8), expected, label)
    }

    // MARK: - Full ragged lifecycle parity

    func testRaggedCohortMatchesPythonReferenceStepForStep() throws {
        try requireMLXRuntimeOrSkip()
        let fixture = try loadFixture()

        let cache = BatchKVCache(leftPadding: leftPadding)

        // --- construction: per-row RoPE offset starts at -leftPadding ---
        assertIntEqual(
            cache.batchOffset, try XCTUnwrap(fixture["init_offset"]), "init batchOffset")
        XCTAssertEqual(cache.offset, 0, "fresh cache idx must be 0")

        // --- prefill: mask BEFORE update (idx == 0), then update ---
        let prefillMask = cache.makeMask(n: lmax, windowSize: nil, returnArray: true)
        assertMaskEqual(prefillMask, try XCTUnwrap(fixture["prefill_mask"]), "prefill mask")

        let (prefillKeys, prefillValues) = cache.update(
            keys: try XCTUnwrap(fixture["prefill_keys"]),
            values: try XCTUnwrap(fixture["prefill_values"]))
        assertClose(prefillKeys, try XCTUnwrap(fixture["prefill_fetch_keys"]), "prefill fetch keys")
        assertClose(
            prefillValues, try XCTUnwrap(fixture["prefill_fetch_values"]), "prefill fetch values")
        assertIntEqual(
            cache.batchOffset, try XCTUnwrap(fixture["offset_after_prefill"]),
            "batchOffset after prefill")
        XCTAssertEqual(cache.offset, lmax, "idx after prefill must be Lmax")

        // --- decode step 1: mask at idx == 5, then update ---
        let decode1Mask = cache.makeMask(n: 1, windowSize: nil, returnArray: false)
        assertMaskEqual(decode1Mask, try XCTUnwrap(fixture["decode1_mask"]), "decode1 mask")
        let (decode1Keys, _) = cache.update(
            keys: try XCTUnwrap(fixture["decode1_keys"]),
            values: try XCTUnwrap(fixture["decode1_values"]))
        assertClose(decode1Keys, try XCTUnwrap(fixture["decode1_fetch_keys"]), "decode1 fetch keys")
        assertIntEqual(
            cache.batchOffset, try XCTUnwrap(fixture["offset_after_decode1"]),
            "batchOffset after decode1")

        // --- decode step 2: mask at idx == 6, then update ---
        let decode2Mask = cache.makeMask(n: 1, windowSize: nil, returnArray: false)
        assertMaskEqual(decode2Mask, try XCTUnwrap(fixture["decode2_mask"]), "decode2 mask")
        let (decode2Keys, _) = cache.update(
            keys: try XCTUnwrap(fixture["decode2_keys"]),
            values: try XCTUnwrap(fixture["decode2_values"]))
        assertClose(decode2Keys, try XCTUnwrap(fixture["decode2_fetch_keys"]), "decode2 fetch keys")
        assertIntEqual(
            cache.batchOffset, try XCTUnwrap(fixture["offset_after_decode2"]),
            "batchOffset after decode2")

        // --- state snapshot (idx == 7): [keys, values, offset, leftPadding] ---
        let state = cache.state
        XCTAssertEqual(state.count, 4, "BatchKVCache state must be a 4-array tuple")
        assertClose(state[0], try XCTUnwrap(fixture["state_keys"]), "state keys")
        assertClose(state[1], try XCTUnwrap(fixture["state_values"]), "state values")
        assertIntEqual(state[2], try XCTUnwrap(fixture["state_offset"]), "state offset")
        assertIntEqual(state[3], try XCTUnwrap(fixture["state_left_padding"]), "state leftPadding")

        // --- filter: keep rows [0, 2]; min-left-pad shift reclaims 2 columns ---
        cache.filter(batchIndices: MLXArray([Int32(0), Int32(2)]))
        XCTAssertEqual(
            cache.offset, try XCTUnwrap(fixture["filter_idx"]).item(Int.self),
            "idx after filter (post min-left-pad shift)")
        let filtered = cache.state
        assertClose(filtered[0], try XCTUnwrap(fixture["filter_keys"]), "filter keys")
        assertClose(filtered[1], try XCTUnwrap(fixture["filter_values"]), "filter values")
        assertIntEqual(filtered[2], try XCTUnwrap(fixture["filter_offset"]), "filter offset")
        assertIntEqual(
            filtered[3], try XCTUnwrap(fixture["filter_left_padding"]), "filter leftPadding")

        // --- extract row 1 of the filtered cache (original row 2) ---
        let extracted = cache.extract(1)
        XCTAssertEqual(
            extracted.offset, try XCTUnwrap(fixture["extract_offset"]).item(Int.self),
            "extracted cache offset == row real length")
        let extractedState = extracted.state
        assertClose(extractedState[0], try XCTUnwrap(fixture["extract_keys"]), "extract keys")
        assertClose(extractedState[1], try XCTUnwrap(fixture["extract_values"]), "extract values")
    }

    // MARK: - extend (cohort merge) parity

    /// `extend` right-justifies both caches to the larger `idx` and concatenates
    /// their rows — the A2c cohort-merge primitive. The two inputs are built via
    /// the `state` setter so their buffers are tightly packed (`buffer == idx`),
    /// isolating the merge math from the 256-step growth buffer.
    func testExtendMergesRaggedCachesLikePythonReference() throws {
        try requireMLXRuntimeOrSkip()
        let fixture = try loadFixture()

        let cacheA = BatchKVCache(leftPadding: [1, 0])
        cacheA.state = [
            try XCTUnwrap(fixture["extendA_keys"]), try XCTUnwrap(fixture["extendA_values"]),
            try XCTUnwrap(fixture["extendA_offset"]), try XCTUnwrap(fixture["extendA_left_padding"]),
        ]
        let cacheB = BatchKVCache(leftPadding: [2])
        cacheB.state = [
            try XCTUnwrap(fixture["extendB_keys"]), try XCTUnwrap(fixture["extendB_values"]),
            try XCTUnwrap(fixture["extendB_offset"]), try XCTUnwrap(fixture["extendB_left_padding"]),
        ]

        cacheA.extend(other: cacheB)

        XCTAssertEqual(
            cacheA.offset, try XCTUnwrap(fixture["extend_idx"]).item(Int.self),
            "extend idx == max(idxA, idxB)")
        let merged = cacheA.state
        XCTAssertEqual(merged.count, 4, "merged cache must expose a 4-array state")
        assertClose(merged[0], try XCTUnwrap(fixture["extend_keys"]), "extend keys")
        assertClose(merged[1], try XCTUnwrap(fixture["extend_values"]), "extend values")
        assertIntEqual(merged[2], try XCTUnwrap(fixture["extend_offset"]), "extend offset")
        assertIntEqual(
            merged[3], try XCTUnwrap(fixture["extend_left_padding"]), "extend leftPadding")
    }

    // MARK: - Converter accept path (constructs BatchKVCache → MLX-gated)

    /// The dense-only converter turns a cohort of plain `KVCacheSimple` layers
    /// into per-layer `BatchKVCache`s, each seeded with the cohort's per-row RoPE
    /// offset `-leftPadding`. (The refusal gate is covered MLX-free in
    /// `BatchKVCacheLogicTests`.)
    func testConverterBuildsBatchCachesForDenseCohort() throws {
        try requireMLXRuntimeOrSkip()

        let stock: [KVCache] = [KVCacheSimple(), KVCacheSimple(), KVCacheSimple()]
        let cohortLeftPadding = [2, 0, 3]
        let converted = try XCTUnwrap(
            BatchCacheConverter.makeBatchCaches(from: stock, leftPadding: cohortLeftPadding),
            "an all-dense cohort must convert")

        XCTAssertEqual(converted.count, stock.count, "one BatchKVCache per layer")
        for cache in converted {
            let batchCache = try XCTUnwrap(
                cache as? BatchKVCache, "each converted cache must be a BatchKVCache")
            XCTAssertEqual(batchCache.offset, 0, "a fresh converted cache is empty")
            // A fresh cache's per-row RoPE offset is -leftPadding.
            assertIntEqual(
                batchCache.batchOffset, MLXArray(cohortLeftPadding.map { Int32(-$0) }),
                "converted cache batchOffset")
        }
    }
}
