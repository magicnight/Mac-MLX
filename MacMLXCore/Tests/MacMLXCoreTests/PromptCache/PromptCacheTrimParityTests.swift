// Copyright © 2026 macMLX. English comments only.

import MLX
import MLXLMCommon
import XCTest

@testable import MacMLXCore

/// Metal-gated (model-free) parity: prove that reusing a longer cached prefix —
/// COPY it, TRIM it back to the shared length, then incrementally prefill the
/// remaining suffix — reconstructs bit-for-bit the same KV state as a cold
/// full-length prefill. This is the KV-level guarantee behind the engine's
/// incremental-prefill path (`fetchNearest` → `trimPromptCache` → suffix feed).
///
/// Uses position-distinct synthetic keys/values (no model, no tokenizer), so it
/// runs anywhere real MLX is available. Skips under bare `swift test`.
final class PromptCacheTrimParityTests: XCTestCase {

    private func assertStatesClose(
        _ lhs: [MLXArray], _ rhs: [MLXArray], file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(lhs.count, rhs.count, "state array count", file: file, line: line)
        for (a, b) in zip(lhs, rhs) {
            XCTAssertEqual(a.shape, b.shape, "state array shape", file: file, line: line)
            XCTAssertTrue(
                allClose(a, b, rtol: 1e-4, atol: 1e-4).item(Bool.self),
                "KV state diverged", file: file, line: line)
        }
    }

    /// Full prefill of N vs (prefill M ≥ N → trim to K → prefill suffix K..<N).
    func testTrimThenIncrementalAppendMatchesFullPrefill() throws {
        try requireMLXRuntimeOrSkip()

        let heads = 1, dim = 2, bigM = 8, prefixK = 3, targetN = 5
        let keysAll = MLXArray(0 ..< Int32(heads * bigM * dim))
            .reshaped(1, heads, bigM, dim).asType(.float32)
        let valuesAll = (MLXArray(0 ..< Int32(heads * bigM * dim)).reshaped(1, heads, bigM, dim)
            + 1000).asType(.float32)

        // Reference: cold prefill of positions 0..<N in one shot.
        let reference = KVCacheSimple()
        _ = reference.update(
            keys: keysAll[.ellipsis, ..<targetN, 0...],
            values: valuesAll[.ellipsis, ..<targetN, 0...])

        // Stored longer entry at offset M, then reuse: copy → trim to K → append.
        let stored = KVCacheSimple()
        _ = stored.update(keys: keysAll, values: valuesAll)
        let reused = stored.copy()
        MLXLMCommon.trimPromptCache([reused], numTokens: bigM - prefixK)
        XCTAssertEqual(reused.offset, prefixK)
        _ = reused.update(
            keys: keysAll[.ellipsis, prefixK..<targetN, 0...],
            values: valuesAll[.ellipsis, prefixK..<targetN, 0...])

        XCTAssertEqual(reference.offset, targetN)
        XCTAssertEqual(reused.offset, targetN)
        assertStatesClose(reused.state, reference.state)
    }

    /// End-to-end through the store: inserting a length-M snapshot and fetching
    /// a shorter prefix yields a trimmed copy whose KV matches a fresh prefill
    /// of that prefix.
    func testStoreFetchTrimmedStateMatchesFreshPrefill() async throws {
        try requireMLXRuntimeOrSkip()

        let heads = 1, dim = 2, bigM = 8
        let keysAll = MLXArray(0 ..< Int32(heads * bigM * dim))
            .reshaped(1, heads, bigM, dim).asType(.float32)
        let valuesAll = (MLXArray(0 ..< Int32(heads * bigM * dim)).reshaped(1, heads, bigM, dim)
            + 1000).asType(.float32)

        let stored = KVCacheSimple()
        _ = stored.update(keys: keysAll, values: valuesAll)

        let root = FileManager.default.temporaryDirectory
            .appending(path: "mlxkv-\(UUID().uuidString)", directoryHint: .isDirectory)
        let store = PromptCacheStore(root: root)
        await store.insert(
            modelID: "M", tokens: Array(0..<bigM), snapshot: PromptCacheSnapshot([stored]))

        // Query [0,1,2] is a strict prefix → reuse is capped at 2 tokens.
        let hit = await store.fetchNearest(modelID: "M", tokens: [0, 1, 2])
        XCTAssertEqual(hit?.reusedTokenCount, 2)

        let reference = KVCacheSimple()
        _ = reference.update(
            keys: keysAll[.ellipsis, ..<2, 0...], values: valuesAll[.ellipsis, ..<2, 0...])

        guard let cache = hit?.snapshot.caches.first else {
            return XCTFail("expected a cache in the hit")
        }
        XCTAssertEqual(cache.offset, 2)
        assertStatesClose(cache.state, reference.state)
    }
}
