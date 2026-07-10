// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLX
import MLXLMCommon

/// Two-tier prompt-cache store.
///
/// **Hot tier** — an in-memory ``PromptTrie`` of token prefix → KV snapshot,
/// evolved from the earlier exact-key dictionary into mlx-lm's trie-backed
/// `LRUPromptCache` design (`mlx_lm/models/cache.py`). It answers
/// longest-common-prefix queries (``fetchNearest(modelID:tokens:)``) so a
/// follow-up turn that merely *extends* a previous prompt — the Claude Code
/// tool-loop / multi-turn-agent shape — reuses the shared prefix's KV state and
/// prefills only the new suffix, instead of paying a full cold prefill on every
/// turn. Residency is bounded by two budgets simultaneously: an entry count
/// (`maxEntries`) and a byte ceiling (`maxBytes`), with a classified LRU
/// (``PromptCacheClassifiedLRU``) choosing victims.
///
/// **Cold tier** — safetensors files on disk under `root/<shard>/<hash>`,
/// round-tripped through mlx-swift-lm's `savePromptCache` / `loadPromptCache`.
/// Entries evicted from the hot tier are demoted here; a hot miss falls back to
/// an *exact-key* cold lookup. The cold tier stays content-addressed by full
/// token hash (``PromptCacheKey``), so it offers persistence and exact re-hits
/// across sessions but not prefix matching — the LCP win lives entirely in the
/// hot trie, which is where recent same-conversation caches sit.
public actor PromptCacheStore {

    /// One resident KV snapshot plus the bookkeeping the budgets need.
    private struct CacheEntry {
        let caches: [any KVCache]
        let nbytes: Int
        let cacheType: PromptCacheType
    }

    private let root: URL
    private let maxEntries: Int
    private let maxBytes: Int

    private var trie = PromptTrie<CacheEntry>()
    private var lru = PromptCacheClassifiedLRU<PromptCacheEntryKey>()
    private var nBytes = 0

    /// - Parameters:
    ///   - root: cold-tier directory (created if missing).
    ///   - maxEntries: hot-tier resident entry ceiling (count-based eviction).
    ///   - maxBytes: hot-tier resident byte ceiling. Defaults to effectively
    ///     unbounded, so count is the only active budget unless a caller opts
    ///     into a byte cap.
    public init(root: URL, maxEntries: Int = 8, maxBytes: Int = .max) {
        self.root = root
        self.maxEntries = maxEntries
        self.maxBytes = maxBytes
        try? FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true
        )
    }

    /// Current resident hot-tier size in bytes (sum of entry KV `state`).
    public var residentBytes: Int { nBytes }

    /// Current resident hot-tier entry count.
    public var residentCount: Int { lru.count }

    // MARK: - Insert

    /// Store `snapshot` under `tokens` for `modelID`. The snapshot's KV offset
    /// is expected to equal `tokens.count` (i.e. it holds exactly these tokens'
    /// state). After insertion, redundant strict-prefix entries are dropped and
    /// both budgets are enforced (evicted entries demote to the cold tier).
    public func insert(
        modelID: String,
        tokens: [Int],
        snapshot: PromptCacheSnapshot,
        cacheType: PromptCacheType = .assistant
    ) {
        store(modelID: modelID, tokens: tokens, caches: snapshot.caches, cacheType: cacheType)
    }

    /// Hot-tier insertion shared by the public ``insert`` entry point and
    /// cold-tier promotion (``fetchFromCold``): add `caches` under `tokens` in
    /// the trie + LRU, collapse now-redundant strict-prefix entries, then
    /// enforce both budgets. The store takes ownership of `caches` — callers
    /// that also hand the same state out to a generator must pass an
    /// independent copy.
    private func store(
        modelID: String, tokens: [Int], caches: [any KVCache], cacheType: PromptCacheType
    ) {
        let entry = CacheEntry(
            caches: caches,
            nbytes: Self.cacheBytes(caches),
            cacheType: cacheType
        )
        let key = PromptCacheEntryKey(modelID: modelID, tokens: tokens)

        nBytes += entry.nbytes
        if let previous = trie.add(model: modelID, tokens: tokens, value: entry) {
            nBytes -= previous.nbytes
            lru.remove(key)
        }
        lru.push(key, type: cacheType)

        // A trimmable cache can serve any of its own prefixes via trim(), so
        // stored strict-prefix entries are pure overhead — drop them.
        if MLXLMCommon.canTrimPromptCache(caches) {
            for (prefixLength, removed) in trie.popPrefixes(model: modelID, tokens: tokens) {
                nBytes -= removed.nbytes
                lru.remove(
                    PromptCacheEntryKey(modelID: modelID, tokens: Array(tokens[0..<prefixLength])))
            }
        }

        // `evictOne` returns false once the hot tier is empty, so a degenerate
        // budget (≤ 0) drains the tier instead of spinning forever.
        while lru.count > maxEntries, evictOne() {}
        while nBytes > maxBytes, evictOne() {}
    }

    // MARK: - Fetch

    /// Find the nearest usable cached prefix for `tokens`, or `nil` on a full
    /// miss. On a hit the returned snapshot is an independent copy already
    /// trimmed to `reusedTokenCount`; feed `tokens[reusedTokenCount...]` to the
    /// generator. `reusedTokenCount ≤ tokens.count - 1` always holds, so the
    /// suffix is never empty.
    public func fetchNearest(modelID: String, tokens: [Int]) -> PromptCacheHit? {
        guard !tokens.isEmpty else { return nil }
        let result = trie.search(model: modelID, tokens: tokens)

        // Exact hit: reuse the whole prefix but leave the last token to feed.
        if let exact = result.exact, let entry = trie.get(model: modelID, tokens: exact) {
            return makeHit(
                caches: entry.caches, heldCount: tokens.count,
                targetReuse: tokens.count - 1, tokenCount: tokens.count)
        }

        let shortLength = result.shorter?.count ?? 0

        // Longer continuation: trim it back to the shared prefix and reuse.
        // Preferred over a shorter prefix when it shares strictly more tokens.
        if let longer = result.longer, result.commonPrefix > shortLength,
            let entry = trie.get(model: modelID, tokens: longer),
            MLXLMCommon.canTrimPromptCache(entry.caches)
        {
            let prefix = min(tokens.count - 1, result.commonPrefix)
            return makeHit(
                caches: entry.caches, heldCount: longer.count,
                targetReuse: prefix, tokenCount: tokens.count)
        }

        // Shorter strict prefix: reuse it wholesale (no trim needed).
        if let shorter = result.shorter, shortLength > 0,
            let entry = trie.get(model: modelID, tokens: shorter)
        {
            return makeHit(
                caches: entry.caches, heldCount: shortLength,
                targetReuse: shortLength, tokenCount: tokens.count)
        }

        // Hot miss → exact-key cold fallback.
        return fetchFromCold(modelID: modelID, tokens: tokens)
    }

    // MARK: - Clear

    /// Blow away both tiers. The hot trie/LRU are reset and the cold-tier
    /// directory is removed wholesale and re-created empty. Invoked from the
    /// Settings → "Clear All KV Caches" button via
    /// `MLXSwiftEngine.clearPromptCache()` and
    /// `EngineCoordinator.clearPromptCache()`.
    public func clearAll() {
        trie = PromptTrie<CacheEntry>()
        lru = PromptCacheClassifiedLRU<PromptCacheEntryKey>()
        nBytes = 0
        let root = self.root
        try? FileManager.default.removeItem(at: root)
        try? FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true
        )
    }

    // MARK: - Private

    /// Copy `caches` — which hold exactly `heldCount` tokens of KV state — trim
    /// the copy back to `targetReuse` positions (clamped to `tokenCount - 1` so
    /// a suffix always remains for the iterator to feed), and wrap it as a hit.
    ///
    /// `heldCount` is the matched trie key's token length (or, for the cold
    /// tier, the exact token count restored from disk) — the copy's true
    /// logical length. It is deliberately NOT read from `copies.first?.offset`:
    /// hybrid/recurrent caches (a `CacheList` wrapping a non-trimmable
    /// `MambaCache`) report a structural `offset` of 0 even while holding N
    /// tokens, so an offset-derived `toTrim` would come out ≤ 0 on an exact hit
    /// and skip the whole trim block — INCLUDING its trimmability guard —
    /// handing back an untrimmed full cache flagged as if `tokenCount - 1`
    /// tokens were reused. The engine would then feed only the final token into
    /// a cache already holding all N: RoPE-misaligned positions plus a
    /// double-advanced recurrent state, i.e. silent wrong output.
    ///
    /// Deriving `toTrim` from the known key length keeps the guard on the real
    /// trim path, so a non-trimmable exact hit correctly returns `nil` and the
    /// caller falls back to a cold/full prefill (matching mlx-lm `cache.py`'s
    /// trim accounting, which measures against the cached prefix length).
    private func makeHit(
        caches: [any KVCache], heldCount: Int, targetReuse: Int, tokenCount: Int
    ) -> PromptCacheHit? {
        let reuse = max(0, min(targetReuse, tokenCount - 1))
        let copies = caches.map { $0.copy() }
        let toTrim = heldCount - reuse
        if toTrim > 0 {
            guard MLXLMCommon.canTrimPromptCache(copies) else { return nil }
            MLXLMCommon.trimPromptCache(copies, numTokens: toTrim)
        }
        return PromptCacheHit(snapshot: PromptCacheSnapshot(copies), reusedTokenCount: reuse)
    }

    /// Pop the LRU victim, remove it from the trie, drop its bytes, and demote
    /// it to the cold tier. Returns `false` when nothing remained to evict.
    @discardableResult
    private func evictOne() -> Bool {
        guard let key = lru.pop(),
            let entry = trie.pop(model: key.modelID, tokens: key.tokens)
        else { return false }
        nBytes -= entry.nbytes
        demoteToCold(modelID: key.modelID, tokens: key.tokens, caches: entry.caches)
        return true
    }

    /// Persist an evicted entry to disk under its exact-token hash.
    ///
    /// v1 accepted tradeoff: `savePromptCache` serialises synchronously on the
    /// actor's executor, so an eviction can stall this actor for the hundreds of
    /// milliseconds a multi-hundred-MB KV snapshot takes to write. Moving the
    /// blocking IO onto a detached task (and reconciling the resulting
    /// ordering/ownership with concurrent `insert`/`fetchNearest` calls) is a
    /// deliberate follow-up rather than part of this pass.
    private func demoteToCold(modelID: String, tokens: [Int], caches: [any KVCache]) {
        let key = PromptCacheKey(modelID: modelID, tokens: tokens)
        let url = key.shardedFileURL(under: root)
        let parent = url.deletingLastPathComponent()
        try? FileManager.default.createDirectory(
            at: parent,
            withIntermediateDirectories: true
        )
        let metadata: [String: String] = [
            "modelID": key.modelID,
            "tokenCount": String(key.tokenCount),
        ]
        try? savePromptCache(url: url, cache: caches, metadata: metadata)
    }

    /// Exact-key cold lookup. Restores the on-disk snapshot for `tokens`,
    /// promotes it back into the hot tier, and — like an exact hot hit — hands
    /// the caller an independent copy trimmed back one token so a suffix
    /// remains.
    ///
    /// The restored entry is re-inserted into the hot tier so a repeated cold
    /// hit is served from memory (and can participate in longest-common-prefix
    /// matching) instead of re-reading disk every time. `makeHit` copies the
    /// restored caches internally, so the pristine `restored` array is what gets
    /// promoted while the returned hit owns a separate, trimmed copy. Promotion
    /// is skipped when the entry can't be served (`makeHit == nil`, e.g. a
    /// non-trimmable hybrid cache), since caching it hot would never help.
    /// Provenance isn't persisted in the cold tier, so the promoted entry
    /// re-enters as `.assistant` (the most-evictable, correct-by-default class
    /// for a speculatively restored prefix).
    ///
    /// v1 accepted tradeoff: `loadPromptCache` reads synchronously on the
    /// actor's executor — the same hundreds-of-ms stall surface documented on
    /// `demoteToCold`; detached IO is the same deferred follow-up.
    private func fetchFromCold(modelID: String, tokens: [Int]) -> PromptCacheHit? {
        let key = PromptCacheKey(modelID: modelID, tokens: tokens)
        let url = key.shardedFileURL(under: root)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let restored: [any KVCache]
        do {
            (restored, _) = try loadPromptCache(url: url)
        } catch {
            return nil
        }
        // Cold entries are content-addressed by the full token hash, so the
        // restored snapshot holds exactly `tokens.count` positions.
        guard
            let hit = makeHit(
                caches: restored, heldCount: tokens.count,
                targetReuse: tokens.count - 1, tokenCount: tokens.count)
        else { return nil }
        store(modelID: modelID, tokens: tokens, caches: restored, cacheType: .assistant)
        return hit
    }

    /// Byte footprint of a KV snapshot: sum of its serialised `state` arrays,
    /// matching how mlx-swift-lm's wired-memory accounting measures a cache.
    private static func cacheBytes(_ caches: [any KVCache]) -> Int {
        caches.flatMap { $0.state }.reduce(0) { $0 + $1.nbytes }
    }
}
