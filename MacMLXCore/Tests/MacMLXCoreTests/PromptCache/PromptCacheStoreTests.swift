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

    /// Total bytes of the cold tier's `*.safetensors` payload files under `root`
    /// — the footprint the byte cap actually governs. The `index.json` manifest
    /// is deliberately excluded: like ``PromptCacheStore/pruneColdDirectory`` it
    /// is bookkeeping outside the cache-payload budget, so counting it would
    /// misstate the cap invariant.
    private func coldDirBytes(_ root: URL) -> Int {
        guard
            let en = FileManager.default.enumerator(
                at: root, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey])
        else { return 0 }
        var total = 0
        for case let url as URL in en where url.pathExtension == "safetensors" {
            let v = try? url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            if v?.isRegularFile == true, let s = v?.fileSize { total += s }
        }
        return total
    }

    /// Count of the cold tier's `*.safetensors` payload files under `root` (the
    /// manifest is excluded, matching the byte-budget's scope).
    private func coldFileCount(_ root: URL) -> Int {
        guard
            let en = FileManager.default.enumerator(
                at: root, includingPropertiesForKeys: [.isRegularFileKey])
        else { return 0 }
        var count = 0
        for case let url as URL in en where url.pathExtension == "safetensors" {
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
        await store.drainInFlight()  // Stage 3a: await the detached demote write.

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
        await store.drainInFlight()  // Stage 3a: await the detached demote write.
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
        await probe.drainInFlight()  // Stage 3a: land the probe's detached write.
        let oneColdFile = coldDirBytes(probeRoot)
        XCTAssertGreaterThan(oneColdFile, 0)

        // Cap at ~2.5 files and force five demotions. Stage 3a: drain after EACH
        // insert so the detached writes land in demote order — the file mtime-LRU
        // the cap prunes by then matches the demote order this test asserts
        // ("newest survives"). Draining also lets each landed write re-enforce the
        // cap (finishWrite), so the ~2.5-file budget holds across the run rather
        // than only converging on the next demote.
        let root = tmpRoot()
        let cap = oneColdFile * 2 + oneColdFile / 2
        let store = PromptCacheStore(root: root, maxEntries: 1, coldCapBytes: cap)
        for i in 0..<6 {
            await store.insert(
                modelID: model, tokens: [i * 2, i * 2 + 1],
                snapshot: makeSnapshot(tokenCount: 2),
                modelFingerprint: fpA)
            await store.drainInFlight()
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
        await store.drainInFlight()  // Stage 3a: await the detached demote write.

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
        await store.drainInFlight()  // Stage 3a: await the detached demote write.
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
        // Stage 3a: the demote's safetensors write is detached. Drain it so the
        // file (and manifest) are on disk before callers assert `fileExists` or
        // construct a "restarted" store over the same root.
        await store.drainInFlight()
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

    /// (d) A HOT entry stamped with fingerprint "A" is NOT served when the request
    /// carries a different fingerprint "B" — the same weight-safety guard the cold
    /// tier applies, made uniform across both tiers so an `unload()` →
    /// swap-weights-at-same-path → reload that keeps this store alive can't serve KV
    /// built from the old weights (the exact silent-wrong-output this feature prevents).
    func testHotHitRejectedOnFingerprintMismatch() async throws {
        try requireMLXRuntimeOrSkip()
        let root = tmpRoot()
        let store = PromptCacheStore(root: root, maxEntries: 8)
        await store.insert(
            modelID: model, tokens: [1, 2], snapshot: makeSnapshot(tokenCount: 2),
            modelFingerprint: fpA)

        // Matching fingerprint → hot hit.
        let hit = await store.fetchNearest(modelID: model, tokens: [1, 2], modelFingerprint: fpA)
        XCTAssertNotNil(hit)
        // Different fingerprint (weights swapped under the same path) → miss, even
        // though (modelID, tokens) match.
        let miss = await store.fetchNearest(modelID: model, tokens: [1, 2], modelFingerprint: fpB)
        XCTAssertNil(miss)
    }

    /// (e) A cold file written BEFORE fingerprints existed (Wave-1 metadata, no
    /// `modelFingerprint` key) is rejected AND deleted on the next fetch — the
    /// stampless legacy entry auto-migrates away rather than being trusted.
    func testColdRejectsAndDeletesLegacyStamplessEntry() async throws {
        try requireMLXRuntimeOrSkip()
        let root = tmpRoot()
        let store = PromptCacheStore(root: root, maxEntries: 8)

        // Write a cold file the Wave-1 way: `{modelID, tokenCount}` only, NO
        // "modelFingerprint" key, at the exact-key sharded path fetchFromCold probes.
        let coldFile = PromptCacheKey(modelID: model, tokens: [1, 2]).shardedFileURL(under: root)
        try FileManager.default.createDirectory(
            at: coldFile.deletingLastPathComponent(), withIntermediateDirectories: true)
        try savePromptCache(
            url: coldFile,
            cache: makeSnapshot(tokenCount: 2).caches,
            metadata: ["modelID": model, "tokenCount": "2"])
        XCTAssertTrue(FileManager.default.fileExists(atPath: coldFile.path))

        // A fetch with a valid fingerprint finds the stampless file, cannot validate
        // it (no stored fingerprint), rejects, and reclaims the file.
        let rejected = await store.fetchNearest(
            modelID: model, tokens: [1, 2], modelFingerprint: fpA)
        XCTAssertNil(rejected)
        XCTAssertFalse(FileManager.default.fileExists(atPath: coldFile.path))
    }

    // MARK: - cold-tier cross-session restart (Wave 2b)
    //
    // A "restart" is simulated by constructing a SECOND ``PromptCacheStore`` over
    // the SAME root after the first has demoted an entry: the fresh store's
    // `init` rebuilds the cold trie from the `index.json` manifest the first
    // store flushed. `demoteToCold(tokens:fingerprint:)` (above) is the base — it
    // leaves `tokens` on disk (and in the manifest) while a disjoint `[900,901]`
    // stays hot in the now-discarded first store.

    /// The persisted cold entry answers an EXACT cross-session query after a
    /// restart — the manifest-rebuilt cold trie serves it (reuse == tokenCount-1).
    func testColdSurvivesRestartExactHit() async throws {
        try requireMLXRuntimeOrSkip()
        let (root, coldFile, _) = await demoteToCold(tokens: [1, 2, 3, 4], fingerprint: fpA)
        XCTAssertTrue(FileManager.default.fileExists(atPath: coldFile.path))

        let restarted = PromptCacheStore(root: root)
        let hit = await restarted.fetchNearest(
            modelID: model, tokens: [1, 2, 3, 4], modelFingerprint: fpA)
        XCTAssertNotNil(hit)
        XCTAssertEqual(hit?.reusedTokenCount, 3)
    }

    /// FLAGSHIP: the whole point of Wave 2b. A follow-up turn that EXTENDS a
    /// prompt persisted in an earlier session reuses the shared 4-token prefix
    /// from disk (longest-common-prefix across a restart), prefilling only the
    /// 2-token delta — no full cold prefill.
    func testColdSurvivesRestartLCPExtendedPrompt() async throws {
        try requireMLXRuntimeOrSkip()
        let (root, coldFile, _) = await demoteToCold(tokens: [1, 2, 3, 4], fingerprint: fpA)
        XCTAssertTrue(FileManager.default.fileExists(atPath: coldFile.path))

        let restarted = PromptCacheStore(root: root)
        let hit = await restarted.fetchNearest(
            modelID: model, tokens: [1, 2, 3, 4, 5, 6], modelFingerprint: fpA)
        XCTAssertNotNil(hit)
        XCTAssertEqual(hit?.reusedTokenCount, 4)
        // Incremental prefill is only the newly-appended suffix.
        XCTAssertEqual(6 - (hit?.reusedTokenCount ?? 0), 2)
    }

    /// A manifest whose format version has moved on is discarded wholesale: the
    /// cold trie stays empty so cross-session LCP is lost (extended query nil),
    /// but the content-addressed exact-hash ``fetchFromCold`` still serves the
    /// on-disk file, so an exact re-hit survives.
    func testRestartFormatVersionMismatch() async throws {
        try requireMLXRuntimeOrSkip()
        let (root, coldFile, _) = await demoteToCold(tokens: [1, 2, 3, 4], fingerprint: fpA)
        XCTAssertTrue(FileManager.default.fileExists(atPath: coldFile.path))

        // Bump the stamped format version so the rebuild rejects the manifest.
        let indexURL = root.appending(path: "index.json")
        let original = ColdIndex.load(from: indexURL)
        XCTAssertNotNil(original)
        ColdIndex.write(
            ColdIndexManifest(formatVersion: 999, entries: original?.entries ?? []),
            to: indexURL)

        let restarted = PromptCacheStore(root: root)
        // Degraded: no cross-session LCP for an extended prompt.
        let extended = await restarted.fetchNearest(
            modelID: model, tokens: [1, 2, 3, 4, 5, 6], modelFingerprint: fpA)
        XCTAssertNil(extended)
        // But the exact-hash fallback still restores the persisted entry.
        let exact = await restarted.fetchNearest(
            modelID: model, tokens: [1, 2, 3, 4], modelFingerprint: fpA)
        XCTAssertNotNil(exact)
        XCTAssertEqual(exact?.reusedTokenCount, 3)
    }

    /// After a restart, an EXTENDED query carrying a different fingerprint
    /// (weights swapped under the same path) is rejected AND the stale cold file
    /// is deleted — the Wave 2a weight-safety guard, now applied on the cold-trie
    /// LCP path too.
    func testRestartFingerprintMismatchRejectsAndDeletes() async throws {
        try requireMLXRuntimeOrSkip()
        let (root, coldFile, _) = await demoteToCold(tokens: [1, 2, 3, 4], fingerprint: fpA)
        XCTAssertTrue(FileManager.default.fileExists(atPath: coldFile.path))

        let restarted = PromptCacheStore(root: root)
        let hit = await restarted.fetchNearest(
            modelID: model, tokens: [1, 2, 3, 4, 5, 6], modelFingerprint: fpB)
        XCTAssertNil(hit)
        XCTAssertFalse(FileManager.default.fileExists(atPath: coldFile.path))
    }

    /// With the manifest deleted but the safetensors kept, the rebuild finds no
    /// index so the cold trie starts empty (no LCP for an extended prompt), yet
    /// the content-addressed exact hash still restores the persisted entry.
    func testRestartManifestMissingFallsBackToExactHash() async throws {
        try requireMLXRuntimeOrSkip()
        let (root, coldFile, _) = await demoteToCold(tokens: [1, 2, 3, 4], fingerprint: fpA)
        XCTAssertTrue(FileManager.default.fileExists(atPath: coldFile.path))

        try FileManager.default.removeItem(at: root.appending(path: "index.json"))

        let restarted = PromptCacheStore(root: root)
        let extended = await restarted.fetchNearest(
            modelID: model, tokens: [1, 2, 3, 4, 5, 6], modelFingerprint: fpA)
        XCTAssertNil(extended)
        let exact = await restarted.fetchNearest(
            modelID: model, tokens: [1, 2, 3, 4], modelFingerprint: fpA)
        XCTAssertNotNil(exact)
        XCTAssertEqual(exact?.reusedTokenCount, 3)
    }

    /// REGRESSION (Wave 2b): the cross-session LCP trim must derive its held
    /// length from the matched trie key (hash-anchored `candidate.heldCount`),
    /// NEVER the manifest's parallel `tokenCount` scalar. Here `index.json` is
    /// hand-corrupted so an entry keeps valid `tokens` + `hashString` — it still
    /// passes ``ColdIndex/isConsistent`` (hash == SHA(modelID, tokens)) and is
    /// admitted by the rebuild — but carries a LYING `tokenCount` (6, not 4).
    /// The restored 4-position cache must be reused whole, not trimmed against
    /// the lie: under the pre-fix code `heldCount` was that 6, so a `toTrim` of
    /// 6 − 4 = 2 silently trimmed the real cache down to 2 positions while still
    /// reporting reuse 4 — a RoPE-misaligning silent wrong output. Asserting the
    /// snapshot actually holds 4 positions (`offset`) pins that shut; the reuse
    /// count alone would not (the pre-fix code reported 4 there too).
    func testColdTrieLCPIgnoresCorruptManifestTokenCount() async throws {
        try requireMLXRuntimeOrSkip()
        let (root, coldFile, _) = await demoteToCold(tokens: [1, 2, 3, 4], fingerprint: fpA)
        XCTAssertTrue(FileManager.default.fileExists(atPath: coldFile.path))

        // Poison ONLY the count; keep tokens + hashString so the hash gate — the
        // only check the rebuild applies — still passes (the fix must not lean
        // on isConsistent rejecting it).
        let indexURL = root.appending(path: "index.json")
        let original = try XCTUnwrap(ColdIndex.load(from: indexURL))
        let poisoned = original.entries.map { e in
            ColdIndexEntry(
                hashString: e.hashString, modelID: e.modelID, tokens: e.tokens,
                tokenCount: e.tokenCount + 2, modelFingerprint: e.modelFingerprint,
                nbytes: e.nbytes, mtime: e.mtime, isTrimmable: e.isTrimmable)
        }
        XCTAssertTrue(poisoned.allSatisfy(ColdIndex.isConsistent))
        ColdIndex.write(
            ColdIndexManifest(formatVersion: original.formatVersion, entries: poisoned),
            to: indexURL)

        let restarted = PromptCacheStore(root: root)
        let hit = await restarted.fetchNearest(
            modelID: model, tokens: [1, 2, 3, 4, 5, 6], modelFingerprint: fpA)
        XCTAssertNotNil(hit)
        XCTAssertEqual(hit?.reusedTokenCount, 4)
        // The crux: the restored cache truly holds 4 positions, not 2.
        XCTAssertEqual(offset(hit), 4)
    }

    // MARK: - Stage 3a: detached cold-tier writes

    /// Push `tokens` out to the cold tier under `fpA` on a `maxEntries: 1` store,
    /// leaving the demote's detached write PARKED at the barrier — file not yet on
    /// disk, `inFlight` populated. Returns the store and the gate that releases the
    /// write, so the caller drives the exact interleaving.
    private func evictHoldingWrite(
        tokens: [Int], root: URL
    ) async -> (store: PromptCacheStore, gate: WriteGate, coldFile: URL) {
        let store = PromptCacheStore(root: root, maxEntries: 1)
        let gate = WriteGate()
        await store.installWriteBarrier { _ in await gate.wait() }
        await store.insert(
            modelID: model, tokens: tokens, snapshot: makeSnapshot(tokenCount: tokens.count),
            modelFingerprint: fpA)
        // A disjoint second entry evicts `tokens`, whose detached write then parks.
        await store.insert(
            modelID: model, tokens: [900, 901], snapshot: makeSnapshot(tokenCount: 2),
            modelFingerprint: fpA)
        await gate.awaitEntered()  // the write for `tokens` is now held at the barrier
        let coldFile = PromptCacheKey(modelID: model, tokens: tokens).shardedFileURL(under: root)
        return (store, gate, coldFile)
    }

    /// I1: a demote's detached write, still in flight, must serve a concurrent
    /// `fetchNearest` as a HIT — not the phantom miss the old `fileExists` guard
    /// produced. Closes the bug the await-on-`inFlight` gate exists to fix.
    func testDetachedWriteInFlightServesFetchAsHitNotPhantomMiss() async throws {
        try requireMLXRuntimeOrSkip()
        let root = tmpRoot()
        let (store, gate, coldFile) = await evictHoldingWrite(tokens: [1, 2], root: root)

        // The write is held open: the file is NOT on disk yet, though the entry is
        // registered in-flight. The pre-fix code would `dropColdRecord` + miss here.
        XCTAssertFalse(FileManager.default.fileExists(atPath: coldFile.path))

        // A concurrent fetch consults the in-flight registry, awaits the write, and
        // re-validates. The file is renamed into place strictly before `inFlight`
        // clears, so there is NO phantom-miss window: the result is deterministically
        // a hit regardless of how fetch and the write interleave. Hoist the fetch's
        // arguments into Sendable locals so the child task captures no `self`.
        let modelID = model
        let fingerprint = fpA
        let fetch = Task {
            await store.fetchNearest(
                modelID: modelID, tokens: [1, 2], modelFingerprint: fingerprint)
        }
        await Task.yield()  // bias toward exercising the suspend-on-in-flight path
        await gate.open()
        let hit = await fetch.value
        XCTAssertNotNil(hit)
        XCTAssertEqual(hit?.reusedTokenCount, 1)  // exact hit reuses all but the last

        // The record was NOT phantom-dropped: [1,2] is still retrievable (that hit
        // promoted it into the hot tier), so a follow-up fetch hits too.
        await store.installWriteBarrier(nil)
        let again = await store.fetchNearest(modelID: model, tokens: [1, 2], modelFingerprint: fpA)
        XCTAssertNotNil(again)
    }

    /// I2: when a detached write fails, `finishWrite` reconciles the phantom index
    /// record the demote optimistically wrote — a later fetch is a clean miss, not
    /// a hit against an entry whose safetensors file never landed.
    func testDetachedWriteFailureReconcilesRecord() async throws {
        try requireMLXRuntimeOrSkip()
        let root = tmpRoot()
        // Sabotage the write target: a regular FILE where [1,2]'s shard directory
        // must go, so the write's `savePromptCache` (into a temp under that shard)
        // cannot create its file and throws — driving the `finishWrite` failure path.
        let shard = String(PromptCacheKey(modelID: model, tokens: [1, 2]).hashString.prefix(1))
        try Data("blocker".utf8).write(
            to: root.appending(path: shard, directoryHint: .notDirectory))

        let store = PromptCacheStore(root: root, maxEntries: 1)
        await store.insert(
            modelID: model, tokens: [1, 2], snapshot: makeSnapshot(tokenCount: 2),
            modelFingerprint: fpA)
        await store.insert(
            modelID: model, tokens: [900, 901], snapshot: makeSnapshot(tokenCount: 2),
            modelFingerprint: fpA)
        await store.drainInFlight()  // run the failing write + its reconciliation

        // The phantom record is gone from BOTH cold paths: an exact fetch and an
        // LCP-extended fetch for [1,2] are both clean misses.
        let exact = await store.fetchNearest(modelID: model, tokens: [1, 2], modelFingerprint: fpA)
        XCTAssertNil(exact)
        let extended = await store.fetchNearest(
            modelID: model, tokens: [1, 2, 3], modelFingerprint: fpA)
        XCTAssertNil(extended)
    }

    /// I4: `clearAll` cancels an in-flight write, which must abort at its
    /// pre-rename cancellation check so no file resurrects past the wipe.
    func testClearAllCancelsInFlightWriteNoResurrection() async throws {
        try requireMLXRuntimeOrSkip()
        let root = tmpRoot()
        let (store, gate, coldFile) = await evictHoldingWrite(tokens: [1, 2], root: root)
        XCTAssertFalse(FileManager.default.fileExists(atPath: coldFile.path))

        // Capture the write task BEFORE clearAll empties the registry, so its
        // (cancelled) completion can be awaited deterministically afterwards.
        let pending = await store.inFlightTasksSnapshot()
        XCTAssertEqual(pending.count, 1)

        await store.clearAll()  // cancels the in-flight write; wipes both tiers
        await gate.open()       // release it — it must abort before the rename
        for task in pending { await task.value }

        // Nothing resurrected: no file at the canonical path, and [1,2] is a miss.
        XCTAssertFalse(FileManager.default.fileExists(atPath: coldFile.path))
        let miss = await store.fetchNearest(modelID: model, tokens: [1, 2], modelFingerprint: fpA)
        XCTAssertNil(miss)
    }

    /// Happy path: the detached write eventually lands the exact sharded file and
    /// an exact cold restore reuses all but the last token — the on-disk outcome
    /// is identical to the old synchronous write, just produced asynchronously.
    func testDetachedWriteEventuallyLandsFileAndExactHit() async throws {
        try requireMLXRuntimeOrSkip()
        let root = tmpRoot()
        let store = PromptCacheStore(root: root, maxEntries: 1)
        await store.insert(
            modelID: model, tokens: [1, 2, 3], snapshot: makeSnapshot(tokenCount: 3),
            modelFingerprint: fpA)
        await store.insert(
            modelID: model, tokens: [900, 901], snapshot: makeSnapshot(tokenCount: 2),
            modelFingerprint: fpA)
        await store.drainInFlight()

        let coldFile = PromptCacheKey(modelID: model, tokens: [1, 2, 3]).shardedFileURL(under: root)
        XCTAssertTrue(FileManager.default.fileExists(atPath: coldFile.path))
        let hit = await store.fetchNearest(modelID: model, tokens: [1, 2, 3], modelFingerprint: fpA)
        XCTAssertNotNil(hit)
        XCTAssertEqual(hit?.reusedTokenCount, 2)  // tokens.count - 1
    }

    /// FIX 1 regression (concurrency-critic review): a STALE write completion —
    /// `finishWrite` finally running for a write that was in flight during a
    /// `clearAll` — must NOT clobber a FRESH in-flight registration spawned for
    /// the SAME hash after that `clearAll`.
    ///
    /// Sequence: demote K = [1,2] (task1 becomes the barrier's 1ST call, parking
    /// on gate 0; `inFlight[K] = task1`) → `clearAll()` (epoch 0→1, task1
    /// cancelled, `inFlight` emptied) → re-demote the SAME K (task2 becomes the
    /// barrier's 2ND call, parking on gate 1; `inFlight[K] = task2`) → release
    /// gate 0 so task1 resumes, observes its OWN cancellation inside
    /// `ColdTierWriter.write` (right before the rename), fails, and runs its
    /// STALE `finishWrite(epoch: 0, …)` while `self.epoch == 1`. Without FIX 1's
    /// top-level `guard epoch == self.epoch` (checked BEFORE `inFlight[hash] =
    /// nil`), that stale completion would unconditionally null `inFlight[K]` —
    /// clobbering task2's live registration even though task2 is still genuinely
    /// running. Asserting `inFlight` holds EXACTLY task2 immediately after task1
    /// settles is the crux.
    ///
    /// Deterministic throughout, no `Task.sleep`/wall-clock: a ``SequencedGate``
    /// routes the barrier's 1st and 2nd calls to two independently-addressable
    /// gates (keyed by CALL ORDER, which this test controls explicitly via
    /// `awaitEntered`/`open` before each next step), so task1 and task2 can each
    /// be released and observed independently regardless of scheduler timing.
    func testClearAllStaleWriteCompletionDoesNotClobberFreshInFlightRegistration() async throws {
        try requireMLXRuntimeOrSkip()
        let root = tmpRoot()
        let store = PromptCacheStore(root: root, maxEntries: 1)
        let gate = SequencedGate(gateCount: 2)
        await store.installWriteBarrier { _ in await gate.wait() }
        let modelID = model
        let fingerprint = fpA

        // Demote K = [1,2]: its detached write (task1) is the barrier's 1st
        // call, parking on gate 0.
        await store.insert(
            modelID: modelID, tokens: [1, 2], snapshot: makeSnapshot(tokenCount: 2),
            modelFingerprint: fingerprint)
        await store.insert(
            modelID: modelID, tokens: [900, 901], snapshot: makeSnapshot(tokenCount: 2),
            modelFingerprint: fingerprint)
        await gate.awaitEntered(0)
        let afterFirstDemote = await store.inFlightTasksSnapshot()
        XCTAssertEqual(afterFirstDemote.count, 1, "task1 must be registered before clearAll")

        // clearAll: epoch 0→1, cancels task1 (still parked on gate 0 —
        // cancellation is cooperative and does not interrupt a suspended barrier
        // await), empties `inFlight`.
        await store.clearAll()

        // Re-demote the SAME K: `inFlight[K]` is empty post-wipe (dedup doesn't
        // block), so a fresh task2 spawns — the barrier's 2nd call, parking on
        // gate 1.
        await store.insert(
            modelID: modelID, tokens: [1, 2], snapshot: makeSnapshot(tokenCount: 2),
            modelFingerprint: fingerprint)
        await store.insert(
            modelID: modelID, tokens: [900, 901], snapshot: makeSnapshot(tokenCount: 2),
            modelFingerprint: fingerprint)
        await gate.awaitEntered(1)
        let afterSecondDemote = await store.inFlightTasksSnapshot()
        XCTAssertEqual(afterSecondDemote.count, 1, "task2 must be registered for the same key")

        // Release task1 (stale, cancelled): it resumes past the barrier, its
        // write observes cancellation inside `ColdTierWriter.write`, fails, and
        // runs `finishWrite(epoch: 0, …)` — the STALE completion FIX 1 must
        // no-op. Await task1 itself (not `drainInFlight`, which reads the
        // CURRENT `inFlight` and would legitimately return task2's task instead).
        await gate.open(0)
        for task in afterFirstDemote { await task.value }

        // THE CRUX: task2's live registration must have survived task1's stale
        // completion untouched. Pre-FIX-1, task1's unconditional
        // `inFlight[hash] = nil` would have cleared it here even though task2 is
        // still genuinely running.
        let afterStaleCompletion = await store.inFlightTasksSnapshot()
        XCTAssertEqual(
            afterStaleCompletion.count, 1,
            "a stale write's finishWrite must not clobber a fresh in-flight registration for the same key"
        )

        // Clean up: release task2 and let it land normally.
        await gate.open(1)
        for task in afterSecondDemote { await task.value }
    }
}

/// A one-shot gate for deterministic write-barrier ordering in the Stage 3a tests,
/// with no `Task.sleep`/wall-clock. The store's write task awaits ``wait()`` (which
/// parks until ``open()``); a test awaits ``awaitEntered()`` to observe that the
/// write has reached the barrier — i.e. is held open with its file not yet written.
private actor WriteGate {
    private var isOpen = false
    private var hasEntered = false
    private var openWaiters: [CheckedContinuation<Void, Never>] = []
    private var enteredWaiters: [CheckedContinuation<Void, Never>] = []

    /// Called from the write task AT the barrier: signal entry, then park until
    /// ``open()`` (returning immediately if already open).
    func wait() async {
        hasEntered = true
        for waiter in enteredWaiters { waiter.resume() }
        enteredWaiters.removeAll()
        if isOpen { return }
        await withCheckedContinuation { openWaiters.append($0) }
    }

    /// Suspend until the write task has reached ``wait()`` (barrier entered).
    func awaitEntered() async {
        if hasEntered { return }
        await withCheckedContinuation { enteredWaiters.append($0) }
    }

    /// Release the gate; any parked ``wait()`` proceeds, as do all future calls.
    func open() {
        isOpen = true
        for waiter in openWaiters { waiter.resume() }
        openWaiters.removeAll()
    }
}

/// Routes SUCCESSIVE calls to ``wait()`` to their own private ``WriteGate``, in
/// CALL order, so a test can independently control which numbered invocation of
/// a SHARED `writeBarrier` releases — e.g. distinguishing a STALE (pre-
/// `clearAll`) write's barrier call from a FRESH re-write's for the same hash
/// (see `testClearAllStaleWriteCompletionDoesNotClobberFreshInFlightRegistration`).
/// A call past the number of gates provided passes through immediately rather
/// than hang, so a test only needs to reserve as many gates as it cares to
/// address individually.
private actor SequencedGate {
    private var gates: [WriteGate]
    private var nextIndex = 0

    init(gateCount: Int) {
        gates = (0..<gateCount).map { _ in WriteGate() }
    }

    /// Route this call to the Nth-call gate (0-indexed, in the order `wait()` is
    /// invoked) and park until that specific gate opens.
    func wait() async {
        let index = nextIndex
        nextIndex += 1
        guard index < gates.count else { return }
        await gates[index].wait()
    }

    /// Suspend until the `index`-th call has reached its gate (barrier entered).
    func awaitEntered(_ index: Int) async {
        guard index < gates.count else { return }
        await gates[index].awaitEntered()
    }

    /// Release the `index`-th call's gate.
    func open(_ index: Int) async {
        guard index < gates.count else { return }
        await gates[index].open()
    }
}
