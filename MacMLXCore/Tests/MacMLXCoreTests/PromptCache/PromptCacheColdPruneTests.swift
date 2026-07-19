// Copyright © 2026 macMLX. English comments only.

import Foundation
import XCTest

@testable import MacMLXCore

/// Pure-Swift (MLX-free) tests for the cold-tier byte budget: the mtime-LRU
/// victim selection, the real-`FileManager` prune driver, the disabled-tier
/// no-op, and the `Settings` → ``PromptCacheConfig`` unit conversion. None of
/// this touches MLX, so it all runs under bare `swift test`.
final class PromptCacheColdPruneTests: XCTestCase {

    // MARK: - Helpers

    private func tmpRoot() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "kvprune-\(UUID().uuidString)", directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// Write a `size`-byte file at `url` with an explicit modification date, so
    /// mtime ordering is deterministic regardless of filesystem timestamp
    /// granularity.
    @discardableResult
    private func writeFile(_ url: URL, size: Int, mtime: Date) throws -> URL {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(count: size).write(to: url)
        try FileManager.default.setAttributes(
            [.modificationDate: mtime], ofItemAtPath: url.path)
        return url
    }

    private func dirBytes(_ root: URL) -> Int {
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

    private func rec(_ url: URL, _ size: Int, _ mtimeOffset: TimeInterval)
        -> PromptCacheStore.ColdFileRecord
    {
        let base = Date(timeIntervalSince1970: 1_000_000)
        return .init(url: url, size: size, mtime: base.addingTimeInterval(mtimeOffset))
    }

    // MARK: - Pure victim selection

    func testNoVictimsWhenUnderCap() {
        let a = URL(filePath: "/x/a")
        let b = URL(filePath: "/x/b")
        let victims = PromptCacheStore.coldPruneVictims(
            records: [rec(a, 100, 0), rec(b, 100, 1)], capBytes: 1000, protecting: nil)
        XCTAssertTrue(victims.isEmpty)
    }

    func testNoVictimsWhenExactlyAtCap() {
        let a = URL(filePath: "/x/a")
        let b = URL(filePath: "/x/b")
        // total == cap → within budget, nothing to prune (cap is inclusive).
        let victims = PromptCacheStore.coldPruneVictims(
            records: [rec(a, 500, 0), rec(b, 500, 1)], capBytes: 1000, protecting: nil)
        XCTAssertTrue(victims.isEmpty)
    }

    func testEvictsOldestFirstUntilUnderCap() {
        let a = URL(filePath: "/x/a")  // oldest
        let b = URL(filePath: "/x/b")
        let c = URL(filePath: "/x/c")  // newest
        // total 3000, cap 1500 → drop a (→2000), drop b (→1000 ≤ 1500), keep c.
        let victims = PromptCacheStore.coldPruneVictims(
            records: [rec(a, 1000, 0), rec(b, 1000, 10), rec(c, 1000, 20)],
            capBytes: 1500, protecting: nil)
        XCTAssertEqual(victims, [a, b])
    }

    func testUnsortedInputStillEvictsByMtime() {
        let a = URL(filePath: "/x/a")  // oldest (offset 0)
        let b = URL(filePath: "/x/b")  // offset 5
        let c = URL(filePath: "/x/c")  // newest (offset 30)
        // Records deliberately out of mtime order; selection must sort them.
        let victims = PromptCacheStore.coldPruneVictims(
            records: [rec(c, 1000, 30), rec(a, 1000, 0), rec(b, 1000, 5)],
            capBytes: 1000, protecting: nil)
        // Drop a (→2000), drop b (→1000 ≤ 1000), keep c.
        XCTAssertEqual(victims, [a, b])
    }

    func testProtectedFileIsNeverEvictedEvenIfOldest() {
        let a = URL(filePath: "/x/a")  // oldest, but protected
        let b = URL(filePath: "/x/b")
        let c = URL(filePath: "/x/c")
        // total 3000, cap 900. `a` is oldest yet protected, so b then c go.
        let victims = PromptCacheStore.coldPruneVictims(
            records: [rec(a, 1000, 0), rec(b, 1000, 10), rec(c, 1000, 20)],
            capBytes: 900, protecting: a)
        XCTAssertEqual(victims, [b, c])
        XCTAssertFalse(victims.contains(a))
    }

    // MARK: - Real-FileManager prune driver

    func testPruneColdDirectoryEnforcesCapAndRemovesOldest() throws {
        let root = tmpRoot()
        let base = Date(timeIntervalSince1970: 2_000_000)
        // Sharded layout like the real cold tier; four 1000-byte files.
        let a = try writeFile(
            root.appending(path: "a/aaa.safetensors"), size: 1000, mtime: base)
        let b = try writeFile(
            root.appending(path: "b/bbb.safetensors"), size: 1000,
            mtime: base.addingTimeInterval(10))
        let c = try writeFile(
            root.appending(path: "c/ccc.safetensors"), size: 1000,
            mtime: base.addingTimeInterval(20))
        let d = try writeFile(
            root.appending(path: "d/ddd.safetensors"), size: 1000,
            mtime: base.addingTimeInterval(30))
        XCTAssertEqual(dirBytes(root), 4000)

        // Cap 2500 → drop a (→3000), drop b (→2000 ≤ 2500). c, d survive.
        PromptCacheStore.pruneColdDirectory(root: root, capBytes: 2500, protecting: nil)

        let fm = FileManager.default
        XCTAssertFalse(fm.fileExists(atPath: a.path))
        XCTAssertFalse(fm.fileExists(atPath: b.path))
        XCTAssertTrue(fm.fileExists(atPath: c.path))
        XCTAssertTrue(fm.fileExists(atPath: d.path))
        XCTAssertLessThanOrEqual(dirBytes(root), 2500)
    }

    func testPruneColdDirectoryNeverDeletesProtectedFile() throws {
        let root = tmpRoot()
        let base = Date(timeIntervalSince1970: 3_000_000)
        // `a` is the oldest AND the file we just wrote — it must survive even
        // though the cap forces two deletions.
        let a = try writeFile(
            root.appending(path: "a/aaa.safetensors"), size: 1000, mtime: base)
        let b = try writeFile(
            root.appending(path: "b/bbb.safetensors"), size: 1000,
            mtime: base.addingTimeInterval(10))
        let c = try writeFile(
            root.appending(path: "c/ccc.safetensors"), size: 1000,
            mtime: base.addingTimeInterval(20))

        PromptCacheStore.pruneColdDirectory(root: root, capBytes: 1500, protecting: a)

        let fm = FileManager.default
        XCTAssertTrue(fm.fileExists(atPath: a.path), "the just-written entry must survive")
        XCTAssertFalse(fm.fileExists(atPath: b.path))
        XCTAssertFalse(fm.fileExists(atPath: c.path))
    }

    func testPruneColdDirectoryUnboundedCapIsNoOp() throws {
        let root = tmpRoot()
        let a = try writeFile(
            root.appending(path: "a/aaa.safetensors"), size: 1000, mtime: Date())
        PromptCacheStore.pruneColdDirectory(root: root, capBytes: .max, protecting: nil)
        XCTAssertTrue(FileManager.default.fileExists(atPath: a.path))
    }

    // MARK: - Wave 2b: prune returns deletions + skips the manifest

    /// The prune driver now RETURNS exactly the URLs it deleted, so the actor
    /// wrapper can reconcile those files out of the in-memory cold index. Oldest
    /// first, and only the files actually removed.
    func testPruneColdDirectoryReturnsDeletedURLs() throws {
        let root = tmpRoot()
        let base = Date(timeIntervalSince1970: 4_000_000)
        let a = try writeFile(
            root.appending(path: "a/aaa.safetensors"), size: 1000, mtime: base)
        let b = try writeFile(
            root.appending(path: "b/bbb.safetensors"), size: 1000,
            mtime: base.addingTimeInterval(10))
        let c = try writeFile(
            root.appending(path: "c/ccc.safetensors"), size: 1000,
            mtime: base.addingTimeInterval(20))

        // Cap 1500 → drop a (→2000) then b (→1000). c survives.
        let deleted = PromptCacheStore.pruneColdDirectory(
            root: root, capBytes: 1500, protecting: nil)

        // Reported deletions are the two oldest, and match what's gone on disk.
        XCTAssertEqual(deleted.map { $0.lastPathComponent }, ["aaa.safetensors", "bbb.safetensors"])
        let fm = FileManager.default
        XCTAssertFalse(fm.fileExists(atPath: a.path))
        XCTAssertFalse(fm.fileExists(atPath: b.path))
        XCTAssertTrue(fm.fileExists(atPath: c.path))
    }

    /// The `index.json` manifest must never be counted toward the byte cap nor
    /// selected as a victim — deleting it would drop cross-session LCP. Here the
    /// safetensors alone (2000) exceed the cap while the manifest (extra 5000)
    /// is ignored: only a safetensors file is pruned, and the manifest survives
    /// and is never reported deleted.
    func testPruneColdDirectorySkipsIndexManifest() throws {
        let root = tmpRoot()
        let base = Date(timeIntervalSince1970: 5_000_000)
        let old = try writeFile(
            root.appending(path: "a/aaa.safetensors"), size: 1000, mtime: base)
        let new = try writeFile(
            root.appending(path: "b/bbb.safetensors"), size: 1000,
            mtime: base.addingTimeInterval(10))
        // A large, oldest manifest — if it were counted/prunable it would be the
        // first victim and inflate the footprint.
        let manifest = try writeFile(
            root.appending(path: "index.json"), size: 5000,
            mtime: base.addingTimeInterval(-100))

        let deleted = PromptCacheStore.pruneColdDirectory(
            root: root, capBytes: 1500, protecting: nil)

        let fm = FileManager.default
        XCTAssertTrue(fm.fileExists(atPath: manifest.path), "manifest must never be pruned")
        XCTAssertFalse(deleted.contains { $0.lastPathComponent == "index.json" })
        // Exactly the oldest safetensors went; the newest safetensors stayed.
        XCTAssertEqual(deleted.map { $0.lastPathComponent }, ["aaa.safetensors"])
        XCTAssertFalse(fm.fileExists(atPath: old.path))
        XCTAssertTrue(fm.fileExists(atPath: new.path))
    }

    // MARK: - Wave 2b: startup rebuild drops stale manifest entries (MLX-free)

    /// Constructing a store scans the manifest and rebuilds the cold index.
    /// Entries whose hash is inconsistent (corruption / a naming-scheme change)
    /// or whose safetensors file is gone are dropped, and the pruned manifest is
    /// reflushed. Observed here entirely MLX-free via the reloaded manifest —
    /// building a `PromptCacheStore` performs no MLX work.
    func testRebuildDropsInconsistentAndAbsentEntries() throws {
        let root = tmpRoot()
        let indexURL = root.appending(path: "index.json")
        let model = "M"

        // A good entry (consistent hash, file present).
        let goodKey = PromptCacheKey(modelID: model, tokens: [1, 2, 3, 4])
        try writeFile(goodKey.shardedFileURL(under: root), size: 32, mtime: Date())
        let good = ColdIndexEntry(
            hashString: goodKey.hashString, modelID: model, tokens: [1, 2, 3, 4],
            tokenCount: 4, modelFingerprint: "fp", nbytes: 32, mtime: Date(), isTrimmable: true)

        // An inconsistent entry: hash doesn't match its own tokens → dropped
        // regardless of any file.
        let badHash = ColdIndexEntry(
            hashString: "deadbeef", modelID: model, tokens: [5, 6],
            tokenCount: 2, modelFingerprint: "fp", nbytes: 16, mtime: Date(), isTrimmable: true)

        // A consistent entry whose file was never written → dropped on the
        // fileExists check.
        let absentKey = PromptCacheKey(modelID: model, tokens: [7, 8, 9])
        let absent = ColdIndexEntry(
            hashString: absentKey.hashString, modelID: model, tokens: [7, 8, 9],
            tokenCount: 3, modelFingerprint: "fp", nbytes: 24, mtime: Date(), isTrimmable: true)

        ColdIndex.write(
            ColdIndexManifest(
                formatVersion: ColdIndex.coldFormatVersion, entries: [good, badHash, absent]),
            to: indexURL)

        // Construct the store — its init prunes (a no-op under the default cap)
        // then rebuilds, dropping the two stale entries and reflushing.
        _ = PromptCacheStore(root: root)

        let reloaded = ColdIndex.load(from: indexURL)
        XCTAssertEqual(reloaded?.entries.count, 1)
        XCTAssertEqual(reloaded?.entries.first?.hashString, goodKey.hashString)
        XCTAssertEqual(reloaded?.entries.first?.tokens, [1, 2, 3, 4])
    }

    /// A foreign format version makes the rebuild discard the whole manifest
    /// (degraded mode). Because that path returns before any drop, `init` does
    /// NOT reflush, so the on-disk manifest is left untouched — proof the gate
    /// rejected it wholesale rather than partially rebuilding.
    func testRebuildRejectsForeignFormatVersion() throws {
        let root = tmpRoot()
        let indexURL = root.appending(path: "index.json")
        let key = PromptCacheKey(modelID: "M", tokens: [1, 2, 3])
        try writeFile(key.shardedFileURL(under: root), size: 32, mtime: Date())
        let foreign = ColdIndexManifest(
            formatVersion: 999,
            entries: [
                ColdIndexEntry(
                    hashString: key.hashString, modelID: "M", tokens: [1, 2, 3],
                    tokenCount: 3, modelFingerprint: "fp", nbytes: 32, mtime: Date(),
                    isTrimmable: true)
            ])
        ColdIndex.write(foreign, to: indexURL)

        _ = PromptCacheStore(root: root)

        // Untouched: still v999, still one entry (no partial rebuild, no reflush).
        XCTAssertEqual(ColdIndex.load(from: indexURL), foreign)
    }

    // MARK: - Disabled cold tier is a no-op (MLX-free)

    func testDemoteToColdIsNoOpWhenColdDisabled() async {
        let root = tmpRoot()
        let store = PromptCacheStore(root: root, coldEnabled: false)
        // `coldEnabled == false` returns before any filesystem or serialisation
        // work, so an empty (MLX-free) cache array is all that's needed to prove
        // nothing is written. A non-nil fingerprint isolates the `coldEnabled`
        // guard from the separate nil-fingerprint spill skip.
        await store.demoteToCold(
            modelID: "M", tokens: [1, 2, 3], caches: [], fingerprint: "test-fp")

        let file = PromptCacheKey(modelID: "M", tokens: [1, 2, 3]).shardedFileURL(under: root)
        XCTAssertFalse(FileManager.default.fileExists(atPath: file.path))
        // Not even the shard subdirectory should have been created.
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: file.deletingLastPathComponent().path))
    }

    // MARK: - Settings → PromptCacheConfig

    func testConfigDefaultsMatchSettingsDefaults() {
        let c = PromptCacheConfig()
        XCTAssertEqual(c.hotBytes, 512 * 1024 * 1024)
        XCTAssertEqual(c.coldCapBytes, 20 * 1024 * 1024 * 1024)
        XCTAssertEqual(c.maxEntries, PromptCacheConfig.defaultMaxEntries)
        XCTAssertTrue(c.coldEnabled)
    }

    func testConfigFromSettingsConvertsUnits() {
        var s = Settings.default
        s.kvCacheHotMB = 256
        s.kvCacheColdGB = 5
        s.kvCacheColdEnabled = false
        let c = PromptCacheConfig(from: s)
        XCTAssertEqual(c.hotBytes, 256 * 1024 * 1024)
        XCTAssertEqual(c.coldCapBytes, 5 * 1024 * 1024 * 1024)
        XCTAssertFalse(c.coldEnabled)
        // Entry ceiling is a fixed secondary budget, not a persisted setting.
        XCTAssertEqual(c.maxEntries, PromptCacheConfig.defaultMaxEntries)
    }

    func testConfigFromSettingsClampsNegativeToZero() {
        var s = Settings.default
        s.kvCacheHotMB = -1
        s.kvCacheColdGB = -1
        let c = PromptCacheConfig(from: s)
        XCTAssertEqual(c.hotBytes, 0)
        XCTAssertEqual(c.coldCapBytes, 0)
    }
}
