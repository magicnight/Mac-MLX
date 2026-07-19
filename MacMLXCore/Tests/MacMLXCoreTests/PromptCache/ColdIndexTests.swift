// Copyright © 2026 macMLX. English comments only.

import Foundation
import XCTest

@testable import MacMLXCore

/// Pure-Swift (MLX-free) tests for the cold-tier manifest (``ColdIndex`` /
/// ``ColdIndexManifest`` / ``ColdIndexEntry``) and the shared, pure
/// ``PromptCacheStore/resolveReuse(_:tokenCount:)`` arbitration. None of this
/// touches MLX, so it all runs under bare `swift test`.
final class ColdIndexTests: XCTestCase {

    private let model = "M"

    private func tmpRoot() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "coldidx-\(UUID().uuidString)", directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// Build an entry whose `hashString` is the real key hash for its tokens
    /// (i.e. a consistent record), so round-trip and integrity tests start from
    /// a valid baseline.
    private func entry(tokens: [Int], fingerprint: String = "fp", trimmable: Bool = true)
        -> ColdIndexEntry
    {
        let key = PromptCacheKey(modelID: model, tokens: tokens)
        return ColdIndexEntry(
            hashString: key.hashString, modelID: model, tokens: tokens,
            tokenCount: tokens.count, modelFingerprint: fingerprint,
            nbytes: tokens.count * 8, mtime: Date(timeIntervalSince1970: 1_700_000_000),
            isTrimmable: trimmable)
    }

    // MARK: - manifest round-trip

    func testManifestRoundTripPreservesTokensAndVersion() {
        let url = tmpRoot().appending(path: "index.json")
        let manifest = ColdIndexManifest(
            formatVersion: ColdIndex.coldFormatVersion,
            entries: [entry(tokens: [1, 2, 3, 4]), entry(tokens: [9, 8, 7])])
        ColdIndex.write(manifest, to: url)

        let loaded = ColdIndex.load(from: url)
        XCTAssertEqual(loaded, manifest)
        // The load-bearing field survives verbatim: the raw tokens live nowhere
        // else on disk, so a lossy round-trip would silently break LCP rebuild.
        XCTAssertEqual(loaded?.entries.first?.tokens, [1, 2, 3, 4])
        XCTAssertEqual(loaded?.formatVersion, ColdIndex.coldFormatVersion)
    }

    func testAtomicWriteReloadReflectsLatest() {
        let url = tmpRoot().appending(path: "index.json")
        ColdIndex.write(
            ColdIndexManifest(formatVersion: ColdIndex.coldFormatVersion, entries: [entry(tokens: [1])]),
            to: url)
        // A second atomic write fully replaces the first — no stale merge.
        let second = ColdIndexManifest(
            formatVersion: ColdIndex.coldFormatVersion,
            entries: [entry(tokens: [5, 6]), entry(tokens: [7, 8, 9])])
        ColdIndex.write(second, to: url)

        XCTAssertEqual(ColdIndex.load(from: url), second)
        XCTAssertEqual(ColdIndex.load(from: url)?.entries.count, 2)
    }

    // MARK: - load failure modes

    func testLoadMissingFileReturnsNil() {
        let url = tmpRoot().appending(path: "does-not-exist.json")
        XCTAssertNil(ColdIndex.load(from: url))
    }

    func testLoadGarbageReturnsNil() throws {
        let url = tmpRoot().appending(path: "index.json")
        try Data("not json at all {{{".utf8).write(to: url)
        XCTAssertNil(ColdIndex.load(from: url))
    }

    func testLoadTruncatedReturnsNil() throws {
        let url = tmpRoot().appending(path: "index.json")
        // Valid-looking prefix, then cut off mid-object → decode fails → nil.
        try Data(#"{"formatVersion":1,"entries":[{"hashStr"#.utf8).write(to: url)
        XCTAssertNil(ColdIndex.load(from: url))
    }

    /// A manifest stamped with a foreign format version. `ColdIndex.load` is a
    /// pure decode, so it still returns the manifest — the VERSION GATE lives in
    /// the caller (`PromptCacheStore.rebuiltColdIndex`), which compares against
    /// ``ColdIndex/coldFormatVersion`` and discards a mismatch (the end-to-end
    /// degraded behaviour is proven by `testRestartFormatVersionMismatch`). This
    /// test pins the gate's decision input: a v999 manifest is recognisably NOT
    /// the current version.
    func testForeignFormatVersionDecodesButFailsGate() {
        let url = tmpRoot().appending(path: "index.json")
        ColdIndex.write(
            ColdIndexManifest(formatVersion: 999, entries: [entry(tokens: [1, 2])]), to: url)
        let loaded = ColdIndex.load(from: url)
        XCTAssertEqual(loaded?.formatVersion, 999)
        XCTAssertNotEqual(loaded?.formatVersion, ColdIndex.coldFormatVersion)
    }

    // MARK: - integrity

    func testIsConsistentAcceptsRealKeyHash() {
        XCTAssertTrue(ColdIndex.isConsistent(entry(tokens: [3, 1, 4, 1, 5])))
    }

    func testIsConsistentRejectsTamperedHash() {
        let good = entry(tokens: [1, 2, 3])
        let tampered = ColdIndexEntry(
            hashString: good.hashString + "00", modelID: good.modelID, tokens: good.tokens,
            tokenCount: good.tokenCount, modelFingerprint: good.modelFingerprint,
            nbytes: good.nbytes, mtime: good.mtime, isTrimmable: good.isTrimmable)
        XCTAssertFalse(ColdIndex.isConsistent(tampered))
    }

    func testIsConsistentRejectsWrongTokensForHash() {
        // Hash belongs to [1,2,3] but the record claims tokens [1,2,4]: the
        // rebuild would name the wrong file, so the record is inconsistent.
        let key = PromptCacheKey(modelID: model, tokens: [1, 2, 3])
        let mislabelled = ColdIndexEntry(
            hashString: key.hashString, modelID: model, tokens: [1, 2, 4],
            tokenCount: 3, modelFingerprint: "fp", nbytes: 8,
            mtime: Date(), isTrimmable: true)
        XCTAssertFalse(ColdIndex.isConsistent(mislabelled))
    }

    /// The prune-reconciliation mapping: a cold file's basename is
    /// "<hash>.safetensors", so its stem is exactly the hash that keys the cold
    /// index. This is what `pruneCold` relies on to drop the right records given
    /// the URLs `pruneColdDirectory` reports as deleted.
    func testColdFileStemIsTheHash() {
        let key = PromptCacheKey(modelID: model, tokens: [1, 2, 3, 4])
        let url = key.shardedFileURL(under: URL(filePath: "/cold/root"))
        XCTAssertEqual(url.deletingPathExtension().lastPathComponent, key.hashString)
    }

    // MARK: - shared reuse arbitration (parity with PromptTrieTests semantics)

    /// Run a real ``PromptTrie`` search then resolve it, so the arbitration is
    /// checked against the actual four-state search output — the same source
    /// `PromptTrieTests` pins.
    private func resolve(stored: [[Int]], query: [Int]) -> [PromptCacheStore.ReuseCandidate] {
        let trie = PromptTrie<Int>()
        for (i, tokens) in stored.enumerated() { trie.add(model: model, tokens: tokens, value: i) }
        let search = trie.search(model: model, tokens: query)
        return PromptCacheStore.resolveReuse(search, tokenCount: query.count)
    }

    func testResolveReuseExactIsLoneCandidate() {
        let candidates = resolve(stored: [[1, 2, 3]], query: [1, 2, 3])
        XCTAssertEqual(
            candidates,
            [.init(key: [1, 2, 3], heldCount: 3, targetReuse: 2, requiresTrim: false)])
    }

    func testResolveReuseShorterWholesaleNoTrim() {
        // [1,2,3,4] stored; query extends it → reuse the whole 4-token prefix,
        // no trim. This is the cross-session-LCP-extended shape.
        let candidates = resolve(stored: [[1, 2, 3, 4]], query: [1, 2, 3, 4, 5, 6])
        XCTAssertEqual(
            candidates,
            [.init(key: [1, 2, 3, 4], heldCount: 4, targetReuse: 4, requiresTrim: false)])
    }

    func testResolveReuseLongerTrimsToSharedPrefixAndRequiresTrim() {
        // A longer stored path than the query → trim it back to the shared
        // prefix (capped at tokenCount-1) and flag it needs trimmability.
        let candidates = resolve(stored: [[1, 2, 3, 4, 5]], query: [1, 2])
        XCTAssertEqual(
            candidates,
            [.init(key: [1, 2, 3, 4, 5], heldCount: 5, targetReuse: 1, requiresTrim: true)])
    }

    func testResolveReuseLongerThenShorterFallthroughOrder() {
        // [1,2] (shorter, len 2) and [1,2,3,4] (longer) both relate to query
        // [1,2,3]: longer shares 3 > shorter's 2, so longer is tried first, then
        // shorter as the fall-through — the exact order the caller gate needs.
        let candidates = resolve(stored: [[1, 2], [1, 2, 3, 4]], query: [1, 2, 3])
        XCTAssertEqual(candidates.count, 2)
        XCTAssertEqual(
            candidates[0],
            .init(key: [1, 2, 3, 4], heldCount: 4, targetReuse: 2, requiresTrim: true))
        XCTAssertEqual(
            candidates[1],
            .init(key: [1, 2], heldCount: 2, targetReuse: 2, requiresTrim: false))
    }

    func testResolveReuseShorterWinsWhenCommonPrefixNotGreater() {
        // [1,2] stored plus a divergent [1,2,9]: query [1,2,3] has commonPrefix
        // 2, not > shorter length 2, so the longer branch is gated OUT and only
        // the shorter [1,2] candidate remains (matches the hot path's
        // `commonPrefix > shortLength`).
        let candidates = resolve(stored: [[1, 2], [1, 2, 9]], query: [1, 2, 3])
        XCTAssertEqual(
            candidates,
            [.init(key: [1, 2], heldCount: 2, targetReuse: 2, requiresTrim: false)])
    }

    func testResolveReuseEmptyOnMiss() {
        XCTAssertTrue(resolve(stored: [[1, 2, 3]], query: [7, 8]).isEmpty)
    }
}
