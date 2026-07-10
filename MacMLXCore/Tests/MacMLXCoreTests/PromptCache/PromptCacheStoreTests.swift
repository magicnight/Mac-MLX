// Copyright © 2026 macMLX. English comments only.

import MLX
import MLXLMCommon
import XCTest

@testable import MacMLXCore

/// Behavioural tests for the trie-backed ``PromptCacheStore``: nearest-prefix
/// fetch, incremental-reuse accounting, dual (count + byte) budget eviction,
/// prefix collapsing, copy-on-fetch isolation, and the safetensors cold tier.
///
/// MLX-gated (real `KVCache`/`MLXArray`) — runs under xcodebuild, skips under
/// bare `swift test`.
final class PromptCacheStoreTests: XCTestCase {

    private let model = "M"

    /// A `KVCacheSimple` whose offset equals `tokenCount`, mimicking a cache
    /// that has prefilled exactly that many tokens. Head dim 4, one layer.
    private func makeSnapshot(tokenCount: Int) -> PromptCacheSnapshot {
        let keys = MLXArray.zeros([1, 1, tokenCount, 4])
        let values = MLXArray.ones([1, 1, tokenCount, 4])
        let layer = KVCacheSimple()
        _ = layer.update(keys: keys, values: values)
        return PromptCacheSnapshot([layer])
    }

    /// A hybrid/recurrent snapshot: a `CacheList` wrapping a non-trimmable
    /// `MambaCache` alongside a `KVCacheSimple` — the Falcon-H1 /
    /// GatedDeltaNet layer shape. Its outer `CacheList.offset` is structurally
    /// 0 no matter how many tokens it holds, and `canTrimPromptCache` is false,
    /// which is exactly the case an offset-derived trim would mishandle.
    private func makeHybridSnapshot(tokenCount: Int) -> PromptCacheSnapshot {
        let keys = MLXArray.zeros([1, 1, tokenCount, 4])
        let values = MLXArray.ones([1, 1, tokenCount, 4])
        let attention = KVCacheSimple()
        _ = attention.update(keys: keys, values: values)
        let recurrent = MambaCache()
        return PromptCacheSnapshot([CacheList(recurrent, attention)])
    }

    private func tmpRoot() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "mlxkv-\(UUID().uuidString)", directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func offset(_ hit: PromptCacheHit?) -> Int? {
        hit?.snapshot.caches.first?.offset
    }

    // MARK: - nearest fetch

    func testExactHitReusesAllButLastToken() async throws {
        try requireMLXRuntimeOrSkip()
        let store = PromptCacheStore(root: tmpRoot())
        await store.insert(modelID: model, tokens: [1, 2, 3], snapshot: makeSnapshot(tokenCount: 3))

        let hit = await store.fetchNearest(modelID: model, tokens: [1, 2, 3])
        XCTAssertNotNil(hit)
        // Exact hit leaves exactly one token to feed the iterator.
        XCTAssertEqual(hit?.reusedTokenCount, 2)
        XCTAssertEqual(offset(hit), 2)
    }

    func testLongerContinuationTrimsToSharedPrefix() async throws {
        try requireMLXRuntimeOrSkip()
        let store = PromptCacheStore(root: tmpRoot())
        await store.insert(
            modelID: model, tokens: [1, 2, 3, 4, 5], snapshot: makeSnapshot(tokenCount: 5))

        // Query is a strict prefix; reuse is capped at tokenCount - 1.
        let hit = await store.fetchNearest(modelID: model, tokens: [1, 2])
        XCTAssertNotNil(hit)
        XCTAssertEqual(hit?.reusedTokenCount, 1)
        XCTAssertEqual(offset(hit), 1)
    }

    /// The headline product case: a growing multi-turn conversation reuses the
    /// whole previous turn and prefills only the newly-appended tokens.
    func testIncrementalReuseAcrossGrowingTurns() async throws {
        try requireMLXRuntimeOrSkip()
        let store = PromptCacheStore(root: tmpRoot())

        // Turn 1 stored as prompt+generated of length 4.
        await store.insert(
            modelID: model, tokens: [1, 2, 3, 4], snapshot: makeSnapshot(tokenCount: 4))

        // Turn 2 = turn 1 + a 2-token delta.
        let turn2 = [1, 2, 3, 4, 5, 6]
        let hit = await store.fetchNearest(modelID: model, tokens: turn2)
        XCTAssertNotNil(hit)
        // Reused == previous turn length → incremental prefill is only the delta.
        XCTAssertEqual(hit?.reusedTokenCount, 4)
        XCTAssertEqual(offset(hit), 4)
        let incrementalPrefill = turn2.count - (hit?.reusedTokenCount ?? 0)
        XCTAssertEqual(incrementalPrefill, 2)
    }

    func testMissReturnsNil() async throws {
        try requireMLXRuntimeOrSkip()
        let store = PromptCacheStore(root: tmpRoot())
        await store.insert(modelID: model, tokens: [1, 2, 3], snapshot: makeSnapshot(tokenCount: 3))
        // Disjoint query, nothing on disk.
        let hit = await store.fetchNearest(modelID: model, tokens: [7, 8, 9])
        XCTAssertNil(hit)
    }

    func testDifferentModelDoesNotHit() async throws {
        try requireMLXRuntimeOrSkip()
        let store = PromptCacheStore(root: tmpRoot())
        await store.insert(modelID: "A", tokens: [1, 2, 3], snapshot: makeSnapshot(tokenCount: 3))
        let hit = await store.fetchNearest(modelID: "B", tokens: [1, 2, 3])
        XCTAssertNil(hit)
    }

    /// H1 regression: a hybrid/recurrent cache (structural `offset == 0`,
    /// non-trimmable) must NOT be handed back untrimmed on an exact hit. An
    /// exact hit needs a one-token trim it cannot perform, so the store returns
    /// `nil` (cold/full prefill) instead of a full cache masquerading as a
    /// prefix reuse — the RoPE-misalignment / double-advanced-state bug. A
    /// shorter strict-prefix hit needs no trim, so it is still served.
    func testHybridExactHitReturnsNilButShorterHitServes() async throws {
        try requireMLXRuntimeOrSkip()
        let store = PromptCacheStore(root: tmpRoot())
        await store.insert(
            modelID: model, tokens: [1, 2], snapshot: makeHybridSnapshot(tokenCount: 2))

        // Exact re-fetch → one-token trim required → non-trimmable → nil.
        let exact = await store.fetchNearest(modelID: model, tokens: [1, 2])
        XCTAssertNil(exact)

        // Shorter strict-prefix source ([1,2] serves the [1,2] prefix of
        // [1,2,3]): reuse == held length → toTrim == 0 → served wholesale.
        let shorter = await store.fetchNearest(modelID: model, tokens: [1, 2, 3])
        XCTAssertNotNil(shorter)
        XCTAssertEqual(shorter?.reusedTokenCount, 2)
    }

    // MARK: - copy-on-fetch isolation

    func testFetchedSnapshotIsIndependentCopy() async throws {
        try requireMLXRuntimeOrSkip()
        let store = PromptCacheStore(root: tmpRoot())
        await store.insert(modelID: model, tokens: [1, 2, 3], snapshot: makeSnapshot(tokenCount: 3))

        // Mutate the first fetched copy destructively.
        let first = await store.fetchNearest(modelID: model, tokens: [1, 2, 3])
        if let cache = first?.snapshot.caches.first {
            cache.trim(cache.offset)  // drop to offset 0
        }

        // The stored entry must be untouched: a second fetch trims cleanly to 2.
        let second = await store.fetchNearest(modelID: model, tokens: [1, 2, 3])
        XCTAssertEqual(offset(second), 2)
    }

    // MARK: - dual budget eviction

    func testCountBudgetEvictsOldest() async throws {
        try requireMLXRuntimeOrSkip()
        let store = PromptCacheStore(root: tmpRoot(), maxEntries: 2)
        await store.insert(modelID: model, tokens: [1], snapshot: makeSnapshot(tokenCount: 1))
        await store.insert(modelID: model, tokens: [2], snapshot: makeSnapshot(tokenCount: 1))
        await store.insert(modelID: model, tokens: [3], snapshot: makeSnapshot(tokenCount: 1))
        let count = await store.residentCount
        XCTAssertEqual(count, 2)
    }

    func testByteBudgetEvicts() async throws {
        try requireMLXRuntimeOrSkip()
        // Measure one entry's footprint, then cap the byte budget at exactly it.
        let probe = PromptCacheStore(root: tmpRoot())
        await probe.insert(modelID: model, tokens: [1, 2, 3], snapshot: makeSnapshot(tokenCount: 3))
        let oneEntryBytes = await probe.residentBytes
        XCTAssertGreaterThan(oneEntryBytes, 0)

        let store = PromptCacheStore(root: tmpRoot(), maxBytes: oneEntryBytes)
        await store.insert(modelID: model, tokens: [1, 2, 3], snapshot: makeSnapshot(tokenCount: 3))
        await store.insert(modelID: model, tokens: [4, 5, 6], snapshot: makeSnapshot(tokenCount: 3))
        let count = await store.residentCount
        let bytes = await store.residentBytes
        XCTAssertEqual(count, 1)
        XCTAssertLessThanOrEqual(bytes, oneEntryBytes)
    }

    // MARK: - prefix collapsing

    func testInsertingLongerCollapsesStrictPrefix() async throws {
        try requireMLXRuntimeOrSkip()
        let store = PromptCacheStore(root: tmpRoot())
        await store.insert(modelID: model, tokens: [1, 2], snapshot: makeSnapshot(tokenCount: 2))
        await store.insert(
            modelID: model, tokens: [1, 2, 3, 4], snapshot: makeSnapshot(tokenCount: 4))

        // The [1,2] entry was collapsed into the longer, trimmable one.
        let count = await store.residentCount
        XCTAssertEqual(count, 1)

        // [1,2] is still served — the longer entry trims down to cover it.
        let hit = await store.fetchNearest(modelID: model, tokens: [1, 2])
        XCTAssertNotNil(hit)
        XCTAssertEqual(hit?.reusedTokenCount, 1)
    }

    // MARK: - cold tier

    func testEvictionDemotesToColdAndRestores() async throws {
        try requireMLXRuntimeOrSkip()
        let root = tmpRoot()
        let store = PromptCacheStore(root: root, maxEntries: 1)

        await store.insert(modelID: model, tokens: [1, 2], snapshot: makeSnapshot(tokenCount: 2))
        await store.insert(modelID: model, tokens: [3, 4], snapshot: makeSnapshot(tokenCount: 2))

        // [1,2] evicted from hot → safetensors on disk.
        let coldFile = PromptCacheKey(modelID: model, tokens: [1, 2]).shardedFileURL(under: root)
        XCTAssertTrue(FileManager.default.fileExists(atPath: coldFile.path))

        // And an exact cold lookup restores it.
        let restored = await store.fetchNearest(modelID: model, tokens: [1, 2])
        XCTAssertNotNil(restored)
    }

    /// M1: a cold hit promotes the restored entry back into the hot tier, so a
    /// repeated hit is served from memory rather than re-reading disk. Proven
    /// by deleting the cold file after the first (promoting) hit: the second hit
    /// can then only come from the hot tier.
    func testColdHitPromotesToHotTier() async throws {
        try requireMLXRuntimeOrSkip()
        let root = tmpRoot()
        let store = PromptCacheStore(root: root, maxEntries: 1)

        // Push [1,2] out to cold by inserting a second entry over the budget.
        await store.insert(modelID: model, tokens: [1, 2], snapshot: makeSnapshot(tokenCount: 2))
        await store.insert(modelID: model, tokens: [3, 4], snapshot: makeSnapshot(tokenCount: 2))
        let coldFile = PromptCacheKey(modelID: model, tokens: [1, 2]).shardedFileURL(under: root)
        XCTAssertTrue(FileManager.default.fileExists(atPath: coldFile.path))

        // First fetch: cold hit, which promotes [1,2] back into the hot tier.
        let first = await store.fetchNearest(modelID: model, tokens: [1, 2])
        XCTAssertNotNil(first)

        // Remove the cold file. A second [1,2] hit now proves memory residency —
        // the cold path would find nothing and return nil.
        try FileManager.default.removeItem(at: coldFile)
        let second = await store.fetchNearest(modelID: model, tokens: [1, 2])
        XCTAssertNotNil(second)
        XCTAssertEqual(second?.reusedTokenCount, 1)
    }

    func testClearAllDropsBothTiers() async throws {
        try requireMLXRuntimeOrSkip()
        let root = tmpRoot()
        let store = PromptCacheStore(root: root, maxEntries: 1)
        await store.insert(modelID: model, tokens: [1, 2], snapshot: makeSnapshot(tokenCount: 2))
        await store.insert(modelID: model, tokens: [3, 4], snapshot: makeSnapshot(tokenCount: 2))

        await store.clearAll()

        let count = await store.residentCount
        XCTAssertEqual(count, 0)
        // [3,4] (was hot) is gone, and [1,2]'s cold file was wiped.
        let hot = await store.fetchNearest(modelID: model, tokens: [3, 4])
        XCTAssertNil(hot)
        let cold = await store.fetchNearest(modelID: model, tokens: [1, 2])
        XCTAssertNil(cold)
    }
}
