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

    /// Consistent non-nil weight fingerprints for the cold-tier tests. Wave 2a
    /// makes ``PromptCacheStore`` reject a restore whose stamped fingerprint
    /// doesn't match the current one (and reject a nil current fingerprint
    /// outright), so every demote+restore test threads a stable value; the
    /// mismatch case uses the second.
    private let fpA = "fingerprint-A"
    private let fpB = "fingerprint-B"

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

    /// Total bytes of every regular file under `root` — the cold tier's on-disk
    /// footprint, for asserting the byte-cap invariant.
    private func coldDirBytes(_ root: URL) -> Int {
        guard
            let en = FileManager.default.enumerator(
                at: root, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey])
        else { return 0 }
        var total = 0
        for case let url as URL in en {
            let v = try? url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            if v?.isRegularFile == true, let s = v?.fileSize { total += s }
        }
        return total
    }

    /// Count of regular files under `root` (cold-tier file count).
    private func coldFileCount(_ root: URL) -> Int {
        guard
            let en = FileManager.default.enumerator(
                at: root, includingPropertiesForKeys: [.isRegularFileKey])
        else { return 0 }
        var count = 0
        for case let url as URL in en {
            let v = try? url.resourceValues(forKeys: [.isRegularFileKey])
            if v?.isRegularFile == true { count += 1 }
        }
        return count
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

        await store.insert(
            modelID: model, tokens: [1, 2], snapshot: makeSnapshot(tokenCount: 2),
            modelFingerprint: fpA)
        await store.insert(
            modelID: model, tokens: [3, 4], snapshot: makeSnapshot(tokenCount: 2),
            modelFingerprint: fpA)

        // [1,2] evicted from hot → safetensors on disk.
        let coldFile = PromptCacheKey(modelID: model, tokens: [1, 2]).shardedFileURL(under: root)
        XCTAssertTrue(FileManager.default.fileExists(atPath: coldFile.path))

        // And an exact cold lookup with the matching fingerprint restores it.
        let restored = await store.fetchNearest(
            modelID: model, tokens: [1, 2], modelFingerprint: fpA)
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
        await store.insert(
            modelID: model, tokens: [1, 2], snapshot: makeSnapshot(tokenCount: 2),
            modelFingerprint: fpA)
        await store.insert(
            modelID: model, tokens: [3, 4], snapshot: makeSnapshot(tokenCount: 2),
            modelFingerprint: fpA)
        let coldFile = PromptCacheKey(modelID: model, tokens: [1, 2]).shardedFileURL(under: root)
        XCTAssertTrue(FileManager.default.fileExists(atPath: coldFile.path))

        // First fetch: cold hit (matching fingerprint), which promotes [1,2] back
        // into the hot tier.
        let first = await store.fetchNearest(modelID: model, tokens: [1, 2], modelFingerprint: fpA)
        XCTAssertNotNil(first)

        // Remove the cold file. A second [1,2] hit now proves memory residency —
        // the cold path would find nothing and return nil.
        try FileManager.default.removeItem(at: coldFile)
        let second = await store.fetchNearest(modelID: model, tokens: [1, 2], modelFingerprint: fpA)
        XCTAssertNotNil(second)
        XCTAssertEqual(second?.reusedTokenCount, 1)
    }

    // MARK: - cold-tier byte budget

    /// The core Wave 1 fix: with a byte cap set, the cold directory stays within
    /// it as entries keep spilling over. `maxEntries: 1` demotes the prior entry
    /// on every insert, so six inserts try to write five cold files; pruning
    /// (mtime-LRU) keeps only what fits.
    func testColdCapBudgetPrunesOldestColdFiles() async throws {
        try requireMLXRuntimeOrSkip()
        // Learn one cold file's on-disk footprint by demoting a single entry.
        let probeRoot = tmpRoot()
        let probe = PromptCacheStore(root: probeRoot, maxEntries: 1)
        await probe.insert(
            modelID: model, tokens: [100, 101], snapshot: makeSnapshot(tokenCount: 2),
            modelFingerprint: fpA)
        await probe.insert(
            modelID: model, tokens: [200, 201], snapshot: makeSnapshot(tokenCount: 2),
            modelFingerprint: fpA)
        let oneColdFile = coldDirBytes(probeRoot)
        XCTAssertGreaterThan(oneColdFile, 0)

        // Cap at ~2.5 files and force five demotions.
        let root = tmpRoot()
        let cap = oneColdFile * 2 + oneColdFile / 2
        let store = PromptCacheStore(root: root, maxEntries: 1, coldCapBytes: cap)
        for i in 0..<6 {
            await store.insert(
                modelID: model, tokens: [i * 2, i * 2 + 1],
                snapshot: makeSnapshot(tokenCount: 2),
                modelFingerprint: fpA)
        }

        // Invariant: the cold directory never exceeds its byte cap.
        XCTAssertLessThanOrEqual(coldDirBytes(root), cap)
        // Pruning actually happened (five entries demoted, but the ~2.5-file cap
        // holds only two) — proof this isn't merely a small working set. Asserted
        // by count, not by naming a specific victim, so equal-mtime write ticks
        // can't make it flaky (oldest-first ordering is locked down separately in
        // the pure-function tests).
        XCTAssertLessThanOrEqual(coldFileCount(root), 2)
        XCTAssertGreaterThanOrEqual(coldFileCount(root), 1)
        // The most-recently demoted entry ([8,9], protected as the last write)
        // survives its own prune — protection is by identity, not mtime.
        let newest = PromptCacheKey(modelID: model, tokens: [8, 9]).shardedFileURL(under: root)
        XCTAssertTrue(FileManager.default.fileExists(atPath: newest.path))
    }

    /// Wired the way `MLXSwiftEngine` builds the store from a `PromptCacheConfig`:
    /// a tight hot BYTE budget with a generous entry ceiling. The byte budget —
    /// not the 1024-entry cap — must drive eviction, and the victim demotes to
    /// the cold tier.
    func testWiredHotByteBudgetEvictsToCold() async throws {
        try requireMLXRuntimeOrSkip()
        let probe = PromptCacheStore(root: tmpRoot())
        await probe.insert(modelID: model, tokens: [1, 2, 3], snapshot: makeSnapshot(tokenCount: 3))
        let oneEntryBytes = await probe.residentBytes

        let root = tmpRoot()
        let config = PromptCacheConfig(
            hotBytes: oneEntryBytes, maxEntries: 1024, coldCapBytes: .max, coldEnabled: true)
        let store = PromptCacheStore(
            root: root, maxEntries: config.maxEntries, maxBytes: config.hotBytes,
            coldCapBytes: config.coldCapBytes, coldEnabled: config.coldEnabled)
        await store.insert(
            modelID: model, tokens: [1, 2, 3], snapshot: makeSnapshot(tokenCount: 3),
            modelFingerprint: fpA)
        await store.insert(
            modelID: model, tokens: [4, 5, 6], snapshot: makeSnapshot(tokenCount: 3),
            modelFingerprint: fpA)

        let count = await store.residentCount
        let bytes = await store.residentBytes
        XCTAssertEqual(count, 1)
        XCTAssertLessThanOrEqual(bytes, oneEntryBytes)
        let coldFile = PromptCacheKey(modelID: model, tokens: [1, 2, 3]).shardedFileURL(under: root)
        XCTAssertTrue(FileManager.default.fileExists(atPath: coldFile.path))
    }

    /// With the master toggle off, eviction drops entries without ever touching
    /// disk — a pure hot cache. The evicted prefix becomes a clean miss.
    func testDisabledColdTierNeverDemotesOnEviction() async throws {
        try requireMLXRuntimeOrSkip()
        let root = tmpRoot()
        let store = PromptCacheStore(root: root, maxEntries: 1, coldEnabled: false)
        // Valid fingerprints throughout, so this isolates the `coldEnabled` guard
        // rather than the nil-fingerprint spill skip.
        await store.insert(
            modelID: model, tokens: [1, 2], snapshot: makeSnapshot(tokenCount: 2),
            modelFingerprint: fpA)
        await store.insert(
            modelID: model, tokens: [3, 4], snapshot: makeSnapshot(tokenCount: 2),
            modelFingerprint: fpA)

        let coldFile = PromptCacheKey(modelID: model, tokens: [1, 2]).shardedFileURL(under: root)
        XCTAssertFalse(FileManager.default.fileExists(atPath: coldFile.path))
        // Evicted entry is gone for good (no cold fallback); the resident one serves.
        let miss = await store.fetchNearest(modelID: model, tokens: [1, 2], modelFingerprint: fpA)
        XCTAssertNil(miss)
        let hot = await store.fetchNearest(modelID: model, tokens: [3, 4], modelFingerprint: fpA)
        XCTAssertNotNil(hot)
    }

    /// A zero cold budget means "no cold tier": an eviction must leave NO cold file
    /// behind. Guards the fix for the zero-cap loophole where a spill would write a
    /// file the prune then keeps as the just-written protected entry.
    func testZeroColdCapNeverSpills() async throws {
        try requireMLXRuntimeOrSkip()
        let root = tmpRoot()
        let store = PromptCacheStore(root: root, maxEntries: 1, coldCapBytes: 0)
        // Valid fingerprints throughout, so this isolates the zero-cap guard
        // rather than the nil-fingerprint spill skip.
        await store.insert(
            modelID: model, tokens: [1, 2], snapshot: makeSnapshot(tokenCount: 2),
            modelFingerprint: fpA)
        await store.insert(
            modelID: model, tokens: [3, 4], snapshot: makeSnapshot(tokenCount: 2),
            modelFingerprint: fpA)

        let coldFile = PromptCacheKey(modelID: model, tokens: [1, 2]).shardedFileURL(under: root)
        XCTAssertFalse(FileManager.default.fileExists(atPath: coldFile.path))
        let miss = await store.fetchNearest(modelID: model, tokens: [1, 2], modelFingerprint: fpA)
        XCTAssertNil(miss)
        let hot = await store.fetchNearest(modelID: model, tokens: [3, 4], modelFingerprint: fpA)
        XCTAssertNotNil(hot)
    }

    func testClearAllDropsBothTiers() async throws {
        try requireMLXRuntimeOrSkip()
        let root = tmpRoot()
        let store = PromptCacheStore(root: root, maxEntries: 1)
        // Valid fingerprints so [1,2] is genuinely written to cold — otherwise the
        // "clearAll wiped the cold file" assertion below would be vacuous.
        await store.insert(
            modelID: model, tokens: [1, 2], snapshot: makeSnapshot(tokenCount: 2),
            modelFingerprint: fpA)
        await store.insert(
            modelID: model, tokens: [3, 4], snapshot: makeSnapshot(tokenCount: 2),
            modelFingerprint: fpA)
        // Precondition: [1,2] really did spill to cold before clearAll runs.
        let coldFile = PromptCacheKey(modelID: model, tokens: [1, 2]).shardedFileURL(under: root)
        XCTAssertTrue(FileManager.default.fileExists(atPath: coldFile.path))

        await store.clearAll()

        let count = await store.residentCount
        XCTAssertEqual(count, 0)
        // [3,4] (was hot) is gone, and [1,2]'s cold file was wiped.
        let hot = await store.fetchNearest(modelID: model, tokens: [3, 4], modelFingerprint: fpA)
        XCTAssertNil(hot)
        let cold = await store.fetchNearest(modelID: model, tokens: [1, 2], modelFingerprint: fpA)
        XCTAssertNil(cold)
    }

    // MARK: - cold-tier fingerprint guard (Wave 2a)

    /// Demote a `PromptCacheStore` local helper: push `tokens` out to cold under
    /// `fingerprint` by inserting it then a disjoint second entry over a
    /// `maxEntries: 1` budget. Returns the store's root and the cold-file URL.
    private func demoteToCold(
        tokens: [Int], fingerprint: String
    ) async -> (root: URL, coldFile: URL, store: PromptCacheStore) {
        let root = tmpRoot()
        let store = PromptCacheStore(root: root, maxEntries: 1)
        await store.insert(
            modelID: model, tokens: tokens, snapshot: makeSnapshot(tokenCount: tokens.count),
            modelFingerprint: fingerprint)
        // A disjoint second entry evicts `tokens` to the cold tier.
        await store.insert(
            modelID: model, tokens: [900, 901], snapshot: makeSnapshot(tokenCount: 2),
            modelFingerprint: fingerprint)
        let coldFile = PromptCacheKey(modelID: model, tokens: tokens).shardedFileURL(under: root)
        return (root, coldFile, store)
    }

    /// (a) A cold entry stamped with fingerprint "A" is restored when fetched
    /// with the SAME fingerprint.
    func testColdRestoresOnMatchingFingerprint() async throws {
        try requireMLXRuntimeOrSkip()
        let (_, coldFile, store) = await demoteToCold(tokens: [1, 2], fingerprint: fpA)
        XCTAssertTrue(FileManager.default.fileExists(atPath: coldFile.path))

        let restored = await store.fetchNearest(modelID: model, tokens: [1, 2], modelFingerprint: fpA)
        XCTAssertNotNil(restored)
        XCTAssertEqual(restored?.reusedTokenCount, 1)
        // A matching fingerprint restores AND keeps the cold file (it is not deleted).
        XCTAssertTrue(FileManager.default.fileExists(atPath: coldFile.path))
    }

    /// (b) A cold entry stamped with "A" is REJECTED and its file DELETED when
    /// fetched with a different fingerprint "B" — the weights-changed-under-the-
    /// same-path case that this whole feature exists to catch.
    func testColdRejectsAndDeletesOnFingerprintMismatch() async throws {
        try requireMLXRuntimeOrSkip()
        let (_, coldFile, store) = await demoteToCold(tokens: [1, 2], fingerprint: fpA)
        XCTAssertTrue(FileManager.default.fileExists(atPath: coldFile.path))

        let rejected = await store.fetchNearest(modelID: model, tokens: [1, 2], modelFingerprint: fpB)
        XCTAssertNil(rejected)
        // Reject-AND-delete: the stale file is reclaimed, not left to be re-rejected.
        XCTAssertFalse(FileManager.default.fileExists(atPath: coldFile.path))
    }

    /// (c) A cold entry stamped with "A" is rejected (and deleted) when fetched
    /// with a NIL current fingerprint — a model with no readable `config.json`
    /// can never safely reuse cold, so nil is never a wildcard match.
    func testColdRejectsOnNilFingerprint() async throws {
        try requireMLXRuntimeOrSkip()
        let (_, coldFile, store) = await demoteToCold(tokens: [1, 2], fingerprint: fpA)
        XCTAssertTrue(FileManager.default.fileExists(atPath: coldFile.path))

        let rejected = await store.fetchNearest(
            modelID: model, tokens: [1, 2], modelFingerprint: nil)
        XCTAssertNil(rejected)
        XCTAssertFalse(FileManager.default.fileExists(atPath: coldFile.path))
    }
}
