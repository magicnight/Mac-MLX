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
/// Entries evicted from the hot tier are demoted here. The files stay
/// content-addressed by full token hash (``PromptCacheKey``), but the tier ALSO
/// carries a persisted `index.json` manifest (``ColdIndex``) recording each
/// entry's raw tokens. At startup that manifest rebuilds a parallel cold
/// ``PromptTrie``, so a hot miss answers longest-common-prefix against DISK too
/// (``fetchFromColdTrie``): a follow-up turn that extends a prompt persisted in
/// an earlier session reuses the shared prefix instead of a full prefill. The
/// content-addressed exact-hash lookup (``fetchFromCold``) remains the degraded
/// fallback for when the manifest is absent or its format version has moved on.
/// The hot and cold indexes are kept disjoint: promotion moves an entry out of
/// cold and into hot; eviction moves it back.
public actor PromptCacheStore {

    /// One resident KV snapshot plus the bookkeeping the budgets need.
    private struct CacheEntry {
        let caches: [any KVCache]
        let nbytes: Int
        let cacheType: PromptCacheType
        /// Weight-identity fingerprint (``ModelFingerprint``) of the model this
        /// snapshot was built against, carried on the hot entry so that when it
        /// is evicted, ``demoteToCold`` stamps the cold file with the right
        /// value. `nil` for an entry whose model had no readable `config.json`
        /// — such an entry is never spilled to the cold tier.
        let fingerprint: String?
    }

    /// The cold trie's payload: everything a cold-trie fetch needs WITHOUT
    /// reading the safetensors file. `hashString` locates the file (and keys
    /// ``coldEntries``); `fingerprint` is the Wave 2a weight stamp; `isTrimmable`
    /// lets a cold *longer* hit pre-gate a trim before paying for a load.
    ///
    /// It deliberately carries NO token count. The restored cache's true logical
    /// length is the matched trie key's length — `candidate.heldCount`, anchored
    /// to the same hash that locates the file — so ``fetchFromColdTrie`` feeds
    /// that to ``makeHit`` exactly as the hot path does, never a parallel scalar
    /// a corrupt manifest could diverge from the real length.
    private struct ColdPointer {
        let hashString: String
        let fingerprint: String
        let isTrimmable: Bool
    }

    private let root: URL
    private let maxEntries: Int
    private let maxBytes: Int
    private let coldCapBytes: Int
    private let coldEnabled: Bool

    private var trie = PromptTrie<CacheEntry>()
    private var lru = PromptCacheClassifiedLRU<PromptCacheEntryKey>()
    private var nBytes = 0

    /// Parallel cold-tier index, mirroring the hot ``trie`` but for on-disk
    /// entries so cross-session longest-common-prefix survives a restart.
    ///
    /// ``coldEntries`` (hash → ``ColdIndexEntry``) is the single source of
    /// truth; the ``coldTrie`` and the persisted `index.json` manifest are both
    /// derived from it. The hot and cold structures are kept DISJOINT: a cold
    /// entry promoted into the hot tier is removed from here, and a hot entry
    /// evicted to disk is added here — an entry is at most one of hot / cold.
    ///
    /// A deliberately separate trie (rather than an enum in the hot ``trie``)
    /// keeps the hot path byte-for-byte unchanged.
    private var coldTrie = PromptTrie<ColdPointer>()
    private var coldEntries: [String: ColdIndexEntry] = [:]

    /// Cold-tier on-disk layout version — see ``ColdIndex/coldFormatVersion``.
    /// Aliased here so the store's version gate and manifest stamp read from one
    /// canonical constant.
    static let coldFormatVersion = ColdIndex.coldFormatVersion

    /// The cold-tier manifest path (`<root>/index.json`). It lives at the cold
    /// root so ``clearAll``'s root wipe removes it, but it is NOT a cache-payload
    /// file: ``pruneColdDirectory`` scopes the byte budget to `*.safetensors`
    /// and never counts or prunes this manifest.
    ///
    /// `nonisolated` because it reads only the Sendable `let root`, so the
    /// nonisolated `init` can address it alongside the isolated methods.
    private nonisolated var indexURL: URL {
        root.appending(path: "index.json", directoryHint: .notDirectory)
    }

    /// - Parameters:
    ///   - root: cold-tier directory (created if missing).
    ///   - maxEntries: hot-tier resident entry ceiling (count-based eviction).
    ///   - maxBytes: hot-tier resident byte ceiling. Defaults to effectively
    ///     unbounded, so count is the only active budget unless a caller opts
    ///     into a byte cap.
    ///   - coldCapBytes: cold-tier on-disk byte cap, enforced by ``pruneCold``
    ///     (mtime-LRU). Defaults to effectively unbounded — the production
    ///     budget is supplied via ``PromptCacheConfig`` at the engine/app
    ///     construction sites, keeping this low-level default permissive so
    ///     direct callers (and existing tests) see the pre-budget behaviour.
    ///   - coldEnabled: master opt-in for the cold (safetensors) tier. When
    ///     `false` the store is a pure hot cache: ``demoteToCold`` and
    ///     ``fetchFromCold`` short-circuit to no-ops before any file work.
    public init(
        root: URL,
        maxEntries: Int = 8,
        maxBytes: Int = .max,
        coldCapBytes: Int = PromptCacheConfig.defaultColdCapBytes,
        coldEnabled: Bool = true
    ) {
        self.root = root
        self.maxEntries = maxEntries
        self.maxBytes = maxBytes
        self.coldCapBytes = coldCapBytes
        self.coldEnabled = coldEnabled
        try? FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true
        )
        // Enforce the cold budget once at startup: a directory that grew past
        // the cap in an earlier run (or before budgets were wired at all) gets
        // trimmed on load rather than only after the next eviction.
        if coldEnabled {
            Self.pruneColdDirectory(root: root, capBytes: coldCapBytes, protecting: nil)
            // Rebuild the cold trie from the on-disk manifest so cross-session
            // LCP survives this restart. Runs AFTER the prune so a manifest
            // entry whose file the prune just deleted is dropped by the
            // rebuild's `fileExists` check rather than resurrected. The rebuild
            // is a nonisolated `static` producing locals (an actor's own
            // nonisolated `init` can't call its isolated methods), which `init`
            // then installs; a rebuild that dropped a stale entry is reflushed.
            let rebuilt = Self.rebuiltColdIndex(root: root)
            coldEntries = rebuilt.entries
            coldTrie = rebuilt.trie
            if rebuilt.changed {
                ColdIndex.write(
                    ColdIndexManifest(
                        formatVersion: Self.coldFormatVersion,
                        entries: Array(rebuilt.entries.values)),
                    to: indexURL)
            }
        }
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
        cacheType: PromptCacheType = .assistant,
        modelFingerprint: String? = nil
    ) {
        store(
            modelID: modelID, tokens: tokens, caches: snapshot.caches,
            cacheType: cacheType, fingerprint: modelFingerprint)
    }

    /// Hot-tier insertion shared by the public ``insert`` entry point and
    /// cold-tier promotion (``fetchFromCold``): add `caches` under `tokens` in
    /// the trie + LRU, collapse now-redundant strict-prefix entries, then
    /// enforce both budgets. The store takes ownership of `caches` — callers
    /// that also hand the same state out to a generator must pass an
    /// independent copy.
    private func store(
        modelID: String, tokens: [Int], caches: [any KVCache], cacheType: PromptCacheType,
        fingerprint: String?
    ) {
        let entry = CacheEntry(
            caches: caches,
            nbytes: Self.cacheBytes(caches),
            cacheType: cacheType,
            fingerprint: fingerprint
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
    public func fetchNearest(
        modelID: String, tokens: [Int], modelFingerprint: String? = nil
    ) -> PromptCacheHit? {
        guard !tokens.isEmpty else { return nil }
        let result = trie.search(model: modelID, tokens: tokens)

        // A hot entry is reused only when its stamped weight fingerprint matches
        // the model on disk now — the same weight-identity guard the cold tier
        // applies, made uniform across both tiers. Without it, an `unload()` →
        // swap-weights-at-same-path → reload that keeps this store alive would
        // serve KV state built from the OLD weights (silently wrong output — the
        // exact hazard this feature prevents). A stale entry simply misses here and
        // is re-evicted by the LRU as fresh entries arrive. `nil == nil` holds, so a
        // model with no config.json still gets hot LCP within its process lifetime.
        //
        // Arbitration (exact → longer → shorter) is factored into the pure,
        // shared ``resolveReuse`` so the hot and cold paths CANNOT drift. Each
        // resolved candidate is gated here against the live hot entry: identical
        // fingerprint, plus — for the *longer* branch only — real trimmability of
        // the actual cache. A failed gate falls through to the next candidate,
        // exactly as the three explicit `if` branches did before. An exact hit
        // still commits: its `makeHit` may return `nil` for a non-trimmable
        // cache, and that `nil` propagates out (rather than falling to a cold
        // lookup that would only restore the same non-trimmable state), because
        // `resolveReuse` reports `exact` as the lone candidate.
        for candidate in Self.resolveReuse(result, tokenCount: tokens.count) {
            guard let entry = trie.get(model: modelID, tokens: candidate.key),
                entry.fingerprint == modelFingerprint
            else { continue }
            if candidate.requiresTrim, !MLXLMCommon.canTrimPromptCache(entry.caches) {
                continue
            }
            return makeHit(
                caches: entry.caches, heldCount: candidate.heldCount,
                targetReuse: candidate.targetReuse, tokenCount: tokens.count)
        }

        // Hot miss → cold. The cold TRIE — cross-session LCP, rebuilt from the
        // manifest at startup and kept live by ``demoteToCold`` — is tried
        // first; the content-addressed exact-hash ``fetchFromCold`` is the
        // degraded fallback for when the trie can't serve it (e.g. a missing or
        // version-mismatched manifest left the cold trie empty while the
        // safetensors file is still on disk). Both gate the restore on the
        // fingerprint.
        return fetchFromColdTrie(modelID: modelID, tokens: tokens, fingerprint: modelFingerprint)
            ?? fetchFromCold(modelID: modelID, tokens: tokens, fingerprint: modelFingerprint)
    }

    /// A reuse decision derived purely from a ``PromptTrieSearch`` and the query
    /// length: which stored key to reuse, how many tokens it holds, how far to
    /// trim it back, and whether serving it needs a trimmable cache.
    struct ReuseCandidate: Equatable {
        /// The stored trie key to look up.
        let key: [Int]
        /// The stored key's token length — the held/restored cache's true
        /// logical length, fed to ``makeHit`` as `heldCount`. H1 discipline:
        /// this is a KNOWN key length, NEVER a cache `offset` (a hybrid cache
        /// reports offset 0 while holding N tokens).
        let heldCount: Int
        /// How many leading tokens to keep after trimming.
        let targetReuse: Int
        /// `true` only for the *longer* branch, whose reuse trims a longer stored
        /// cache back to the shared prefix and therefore needs a trimmable cache;
        /// the caller falls through to the next candidate when the real cache
        /// isn't trimmable. Exact and shorter don't pre-gate — exact's one-token
        /// trim is guarded inside ``makeHit``, and shorter trims nothing.
        let requiresTrim: Bool
    }

    /// Pure arbitration shared by the hot and cold fetch paths. Given a trie
    /// search and the query length, return the eligible reuse candidates in the
    /// SAME priority order the hot tier has always used: `exact` alone, else
    /// `longer` (only when it shares strictly more tokens than the shorter
    /// prefix) followed by `shorter`. Impure gates — entry lookup, fingerprint
    /// match, trimmability of the actual cache — are applied by each caller
    /// against its own store, since they need the stored value, not just the
    /// search. Factoring this keeps the two paths from ever drifting apart.
    static func resolveReuse(_ search: PromptTrieSearch, tokenCount: Int) -> [ReuseCandidate] {
        // A `PromptTrie` search reports `exact` mutually exclusively with
        // shorter/longer, so an exact match is the lone candidate: reuse the
        // whole prefix but leave the last token to feed.
        if let exact = search.exact {
            return [
                ReuseCandidate(
                    key: exact, heldCount: tokenCount,
                    targetReuse: tokenCount - 1, requiresTrim: false)
            ]
        }
        var candidates: [ReuseCandidate] = []
        let shortLength = search.shorter?.count ?? 0
        // Longer continuation: trim it back to the shared prefix. Preferred over
        // a shorter prefix when it shares strictly more tokens; needs a trimmable
        // cache (caller-gated).
        if let longer = search.longer, search.commonPrefix > shortLength {
            candidates.append(
                ReuseCandidate(
                    key: longer, heldCount: longer.count,
                    targetReuse: min(tokenCount - 1, search.commonPrefix), requiresTrim: true))
        }
        // Shorter strict prefix: reuse it wholesale (no trim needed).
        if let shorter = search.shorter, shortLength > 0 {
            candidates.append(
                ReuseCandidate(
                    key: shorter, heldCount: shortLength,
                    targetReuse: shortLength, requiresTrim: false))
        }
        return candidates
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
        // Reset the cold tier too. Removing the root also removes `index.json`,
        // so the in-memory index and its on-disk manifest are wiped together.
        coldTrie = PromptTrie<ColdPointer>()
        coldEntries = [:]
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
        demoteToCold(
            modelID: key.modelID, tokens: key.tokens, caches: entry.caches,
            fingerprint: entry.fingerprint)
        return true
    }

    /// Persist an evicted entry to disk under its exact-token hash.
    ///
    /// `internal` (not `private`) purely so the disabled-tier no-op is unit
    /// testable without MLX — see `PromptCacheColdPruneTests`.
    ///
    /// v1 accepted tradeoff: `savePromptCache` serialises synchronously on the
    /// actor's executor, so an eviction can stall this actor for the hundreds of
    /// milliseconds a multi-hundred-MB KV snapshot takes to write. Moving the
    /// blocking IO onto a detached task (and reconciling the resulting
    /// ordering/ownership with concurrent `insert`/`fetchNearest` calls) is a
    /// deliberate follow-up rather than part of this pass.
    func demoteToCold(
        modelID: String, tokens: [Int], caches: [any KVCache], fingerprint: String?
    ) {
        // Master opt-in: a disabled cold tier makes this a pure hot cache.
        // Check first, before any filesystem or serialisation work, so the
        // no-op is cheap and observable (no shard directory is even created).
        guard coldEnabled else { return }
        // A zero (or negative) cold budget means "no cold tier": don't spill at
        // all, rather than write a file the prune must then keep as the protected
        // just-written entry — which would leave one file over a 0-byte cap.
        guard coldCapBytes > 0 else { return }
        // No fingerprint ⇒ the model had no readable `config.json`, so a restore
        // could never be safely validated later (``fetchFromCold`` rejects a
        // nil current fingerprint anyway). Don't write an entry that is
        // guaranteed to be rejected-and-deleted on the next fetch — skip the
        // spill entirely so the cold tier only ever holds restorable entries.
        guard let fingerprint else { return }
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
            "modelFingerprint": fingerprint,
        ]
        try? savePromptCache(url: url, cache: caches, metadata: metadata)

        // Record the entry into the cold index (source of truth) + cold trie so
        // an in-process cold-trie fetch can find it now, and a restart can
        // rebuild cross-session LCP from the manifest. The full token array is
        // the load-bearing field — the tokens live nowhere else on disk.
        // Trimmability is stamped from the live caches so a cold *longer* fetch
        // pre-gates a trim without a load. `Date.now` is fine inside the actor.
        let isTrimmable = MLXLMCommon.canTrimPromptCache(caches)
        let entry = ColdIndexEntry(
            hashString: key.hashString, modelID: key.modelID, tokens: tokens,
            tokenCount: key.tokenCount, modelFingerprint: fingerprint,
            nbytes: Self.cacheBytes(caches), mtime: Date.now, isTrimmable: isTrimmable)
        coldEntries[key.hashString] = entry
        coldTrie.add(
            model: modelID, tokens: tokens,
            value: ColdPointer(
                hashString: key.hashString,
                fingerprint: fingerprint, isTrimmable: isTrimmable))

        // Enforce the cold-tier disk budget by mtime-LRU. `protecting: url`
        // guarantees the entry we just wrote is never the one pruned, even if a
        // clock skew made it look older than a peer. The wrapper reconciles any
        // file the prune deleted out of the cold index + trie.
        pruneCold(protecting: url)
        // Persist the reconciled index so a restart rebuilds an accurate cold
        // trie. A small JSON write, dwarfed by the safetensors write above.
        flushColdIndex()
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
    private func fetchFromCold(
        modelID: String, tokens: [Int], fingerprint: String?
    ) -> PromptCacheHit? {
        // Master opt-in: a disabled cold tier has nothing on disk to restore.
        // Short-circuit before any filesystem work so a hot miss stays a pure
        // in-memory miss.
        guard coldEnabled else { return nil }
        let key = PromptCacheKey(modelID: modelID, tokens: tokens)
        let url = key.shardedFileURL(under: root)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let restored: [any KVCache]
        let meta: [String: String]
        do {
            (restored, meta) = try loadPromptCache(url: url)
        } catch {
            return nil
        }
        // Weight-identity guard (Wave 2a): the cold file names a directory + token
        // prefix, NOT the weights those tokens were prefilled against. Only reuse
        // it when the stamped fingerprint matches the model currently on disk.
        // Reject-AND-DELETE on any of: a differing fingerprint (weights changed
        // under the same path), a missing stamp (a Wave-1 file predating this key
        // — auto-migrated away), or a nil current fingerprint (a model with no
        // readable `config.json` can never safely reuse cold). Deleting reclaims
        // the now-unusable file instead of re-rejecting it on every future fetch.
        guard let stored = meta["modelFingerprint"],
            let current = fingerprint,
            stored == current
        else {
            try? FileManager.default.removeItem(at: url)
            return nil
        }
        // Cold entries are content-addressed by the full token hash, so the
        // restored snapshot holds exactly `tokens.count` positions.
        guard
            let hit = makeHit(
                caches: restored, heldCount: tokens.count,
                targetReuse: tokens.count - 1, tokenCount: tokens.count)
        else { return nil }
        // Promote back into the hot tier under the SAME (validated) fingerprint,
        // so a later re-eviction demotes it with the correct stamp.
        store(
            modelID: modelID, tokens: tokens, caches: restored, cacheType: .assistant,
            fingerprint: current)
        // Hot ⊕ cold disjoint: the entry now lives in the hot tier, so drop its
        // cold index record + trie node. The safetensors file stays as the
        // content-addressed backing (a re-eviction rewrites the record).
        dropColdRecord(modelID: modelID, tokens: tokens)
        return hit
    }

    /// Cross-session cold lookup via the ``coldTrie`` — the Wave 2b capability.
    /// Where ``fetchFromCold`` re-hits only the identical token prefix, this
    /// answers longest-common-prefix against the rebuilt cold index, so a
    /// follow-up turn that merely EXTENDS a persisted prompt reuses the shared
    /// prefix from disk instead of a full cold prefill.
    ///
    /// It shares the hot path's ``resolveReuse`` arbitration verbatim, then for
    /// the winning candidate: locates the file by hash; drops the record and
    /// falls through if the file is gone (a lazy miss under cross-process drift);
    /// loads it (a load failure rejects + deletes + drops + falls through);
    /// applies the Wave 2a fingerprint gate identically to ``fetchFromCold``
    /// (reject-AND-delete + drop on mismatch / nil / stampless); builds the hit
    /// with `heldCount = candidate.heldCount` — the matched trie key's length,
    /// anchored to the very hash that located the file and identical to what the
    /// hot path feeds ``makeHit`` (NEVER an offset, and NEVER a parallel manifest
    /// scalar a corrupt `index.json` could diverge from the real restored
    /// length); then promotes into the hot tier and removes the cold record so
    /// the tiers stay disjoint. A cold *longer* hit pre-gates on the stamped
    /// `isTrimmable` so a non-trimmable continuation is skipped before it is
    /// ever loaded.
    private func fetchFromColdTrie(
        modelID: String, tokens: [Int], fingerprint: String?
    ) -> PromptCacheHit? {
        guard coldEnabled else { return nil }
        let search = coldTrie.search(model: modelID, tokens: tokens)
        for candidate in Self.resolveReuse(search, tokenCount: tokens.count) {
            guard let pointer = coldTrie.get(model: modelID, tokens: candidate.key) else {
                continue
            }
            // Pre-gate a *longer* continuation on its stamped trimmability so a
            // cache we could never trim isn't loaded just to be rejected.
            if candidate.requiresTrim, !pointer.isTrimmable { continue }

            let url = PromptCacheKey(modelID: modelID, tokens: candidate.key)
                .shardedFileURL(under: root)
            // The manifest is a hint: a record whose file has vanished (pruned by
            // another process, deleted) degrades to a lazy miss — drop it and try
            // the next candidate.
            guard FileManager.default.fileExists(atPath: url.path) else {
                dropColdRecord(modelID: modelID, tokens: candidate.key)
                continue
            }
            let restored: [any KVCache]
            let meta: [String: String]
            do {
                (restored, meta) = try loadPromptCache(url: url)
            } catch {
                // A corrupt/unreadable file: reject, reclaim, drop, fall through.
                try? FileManager.default.removeItem(at: url)
                dropColdRecord(modelID: modelID, tokens: candidate.key)
                continue
            }
            // Weight-identity guard, identical to ``fetchFromCold``: reuse only
            // when the stamped fingerprint matches the current one. Reject-AND-
            // delete (and drop the record) on a differing stamp, a missing stamp,
            // or a nil current fingerprint.
            guard let stored = meta["modelFingerprint"],
                let current = fingerprint,
                stored == current
            else {
                try? FileManager.default.removeItem(at: url)
                dropColdRecord(modelID: modelID, tokens: candidate.key)
                continue
            }
            // heldCount is the matched trie key's length (`candidate.heldCount`),
            // anchored to the hash that located `restored` — so it always equals
            // the restored cache's true logical length, exactly as the hot path
            // derives it (H1 discipline — never a cache offset, and never the
            // parallel `tokenCount` scalar, which a corrupt `index.json` could
            // diverge from the real length while still passing the hash gate).
            // `makeHit` may still return nil for a non-trimmable exact/longer
            // restore (e.g. `isTrimmable` was stale under cross-process drift);
            // fall through in that case.
            guard
                let hit = makeHit(
                    caches: restored, heldCount: candidate.heldCount,
                    targetReuse: candidate.targetReuse, tokenCount: tokens.count)
            else { continue }
            // Promote into the hot tier under the validated fingerprint, then
            // drop the cold record + node so the tiers stay disjoint.
            store(
                modelID: modelID, tokens: candidate.key, caches: restored,
                cacheType: .assistant, fingerprint: current)
            dropColdRecord(modelID: modelID, tokens: candidate.key)
            return hit
        }
        return nil
    }

    /// Drop a cold entry's ``coldTrie`` node and ``coldEntries`` record together,
    /// keeping the two in lockstep. Popping the trie yields the pointer whose
    /// `hashString` keys the record. The safetensors file is left untouched —
    /// callers that also want the file gone remove it separately.
    private func dropColdRecord(modelID: String, tokens: [Int]) {
        if let pointer = coldTrie.pop(model: modelID, tokens: tokens) {
            coldEntries[pointer.hashString] = nil
        }
    }

    /// Rebuild the cold index + trie from the on-disk manifest at startup so
    /// cross-session LCP survives a restart. A missing, unreadable, garbage, or
    /// version-mismatched manifest yields an EMPTY result — degraded mode: LCP
    /// is lost this session, but the content-addressed ``fetchFromCold`` still
    /// serves exact re-hits and no wrong output is ever possible.
    ///
    /// Each manifest entry is materialised only when it passes an integrity check
    /// (its stored hash matches the current key scheme) AND its safetensors file
    /// is still on disk; an entry that fails either is dropped and `changed` is
    /// set so `init` reflushes the pruned manifest. `nonisolated`/`static` so the
    /// actor's own nonisolated `init` can call it and install the returned locals
    /// (the actor isn't usable from within its initialiser); MLX-free.
    private static func rebuiltColdIndex(
        root: URL
    ) -> (entries: [String: ColdIndexEntry], trie: PromptTrie<ColdPointer>, changed: Bool) {
        let trie = PromptTrie<ColdPointer>()
        var entries: [String: ColdIndexEntry] = [:]
        let indexURL = root.appending(path: "index.json", directoryHint: .notDirectory)
        guard let manifest = ColdIndex.load(from: indexURL),
            manifest.formatVersion == coldFormatVersion
        else { return (entries, trie, false) }

        let fm = FileManager.default
        var changed = false
        for entry in manifest.entries {
            guard ColdIndex.isConsistent(entry) else {
                changed = true
                continue
            }
            let url = PromptCacheKey(modelID: entry.modelID, tokens: entry.tokens)
                .shardedFileURL(under: root)
            guard fm.fileExists(atPath: url.path) else {
                changed = true
                continue
            }
            entries[entry.hashString] = entry
            trie.add(
                model: entry.modelID, tokens: entry.tokens,
                value: ColdPointer(
                    hashString: entry.hashString,
                    fingerprint: entry.modelFingerprint, isTrimmable: entry.isTrimmable))
        }
        return (entries, trie, changed)
    }

    /// Serialise the current ``coldEntries`` (the source of truth) to the
    /// `index.json` manifest, stamped with the current format version. Called
    /// after every demote so a restart rebuilds an accurate cold trie.
    private func flushColdIndex() {
        let manifest = ColdIndexManifest(
            formatVersion: Self.coldFormatVersion,
            entries: Array(coldEntries.values))
        ColdIndex.write(manifest, to: indexURL)
    }

    // MARK: - Cold-tier pruning

    /// Enforce ``coldCapBytes`` over the cold directory by mtime-LRU, then
    /// reconcile the deletions out of the in-memory cold index + trie. The
    /// nonisolated scan/select/delete (``pruneColdDirectory``) is shared with
    /// `init`; this actor-isolated wrapper additionally drops the records for
    /// the files it removed, which `init` does not need (its rebuild runs after
    /// and validates every entry's file with `fileExists`).
    private func pruneCold(protecting: URL?) {
        let deleted = Self.pruneColdDirectory(
            root: root, capBytes: coldCapBytes, protecting: protecting)
        // Reconcile the deletions out of the in-memory index + trie. A cold
        // file's basename is "<hash>.safetensors", so its stem is the hash that
        // keys ``coldEntries``; drop the record and its trie node together.
        for url in deleted {
            let hash = url.deletingPathExtension().lastPathComponent
            guard let entry = coldEntries[hash] else { continue }
            coldEntries[hash] = nil
            coldTrie.pop(model: entry.modelID, tokens: entry.tokens)
        }
    }

    /// One cold-tier file's pruning-relevant facts. `internal` so the pure
    /// victim selection is unit-testable without touching the filesystem.
    struct ColdFileRecord {
        let url: URL
        let size: Int
        let mtime: Date
    }

    /// Pure selection: given the current cold files, the byte cap, and the file
    /// to protect, return the URLs to delete — oldest (mtime) first — until the
    /// remaining total is within `capBytes`. The `protecting` URL (the entry a
    /// caller just wrote) is never selected. MLX-free, so it is exhaustively
    /// unit-testable on its own.
    ///
    /// The protection check compares symlink-resolved paths, not raw `URL`
    /// values: on a symlinked cache root (macOS maps `/var` → `/private/var`,
    /// and temp dirs live under it) the directory enumerator hands back the
    /// resolved form while `demoteToCold` protects the unresolved one, so a raw
    /// `==` would fail to recognise the just-written entry and prune it.
    static func coldPruneVictims(
        records: [ColdFileRecord],
        capBytes: Int,
        protecting: URL?
    ) -> [URL] {
        var total = records.reduce(0) { $0 + $1.size }
        guard total > capBytes else { return [] }
        // Oldest first; the protected entry is off the table entirely. Only pay
        // the per-record path canonicalisation when something is protected.
        let candidates: [ColdFileRecord]
        if let protectedPath = protecting?.resolvingSymlinksInPath().standardizedFileURL.path {
            candidates =
                records
                .filter { $0.url.resolvingSymlinksInPath().standardizedFileURL.path != protectedPath }
                .sorted { $0.mtime < $1.mtime }
        } else {
            candidates = records.sorted { $0.mtime < $1.mtime }
        }
        var victims: [URL] = []
        for record in candidates {
            if total <= capBytes { break }
            victims.append(record.url)
            total -= record.size
        }
        return victims
    }

    /// Scan `root`, then delete the mtime-LRU victims chosen by
    /// ``coldPruneVictims`` until the directory is within `capBytes`. Mirrors
    /// the store's existing leniency: an unreadable file is skipped (never
    /// counted, never fatal) and a failed delete is skipped (best-effort — the
    /// next prune re-scans and retries). `nonisolated`/`static` so `init` can
    /// call it before the actor is fully initialised.
    /// Serialises cold-directory pruning across ALL `PromptCacheStore` instances.
    /// Every store shares the one `~/.mac-mlx/kv-cache` root, so without this two
    /// pool engines pruning concurrently could scan-and-delete the same files; the
    /// lock makes each prune's scan+delete atomic with respect to the others.
    private static let pruneLock = NSLock()

    @discardableResult
    static func pruneColdDirectory(root: URL, capBytes: Int, protecting: URL?) -> [URL] {
        // An unbounded cap can never be exceeded — skip the scan entirely.
        guard capBytes != .max else { return [] }
        // One prune at a time across every store sharing this cold root.
        pruneLock.lock()
        defer { pruneLock.unlock() }
        let keys: Set<URLResourceKey> = [
            .isRegularFileKey, .fileSizeKey, .contentModificationDateKey,
        ]
        let fm = FileManager.default
        guard
            let enumerator = fm.enumerator(
                at: root,
                includingPropertiesForKeys: Array(keys),
                options: [.skipsHiddenFiles],
                errorHandler: { _, _ in true }  // skip unreadable subtrees, keep going
            )
        else { return [] }

        var records: [ColdFileRecord] = []
        for case let url as URL in enumerator {
            // The byte budget governs cache-payload files only, so count solely
            // "*.safetensors" entries. This deliberately excludes the
            // `index.json` manifest — which must never be a prune victim
            // (deleting it drops cross-session LCP) nor inflate the measured
            // footprint — and any transient temp file an atomic manifest write
            // leaves in the directory mid-write.
            guard url.pathExtension == "safetensors" else { continue }
            guard
                let values = try? url.resourceValues(forKeys: keys),
                values.isRegularFile == true,
                let size = values.fileSize,
                let mtime = values.contentModificationDate
            else { continue }  // unreadable / not a regular file → leniently skip
            records.append(ColdFileRecord(url: url, size: size, mtime: mtime))
        }

        var deleted: [URL] = []
        for victim in coldPruneVictims(records: records, capBytes: capBytes, protecting: protecting) {
            // A failed delete is skipped, never fatal, and NOT reported deleted:
            // the file survives, so its index record stays valid and the next
            // prune re-scans and retries.
            do {
                try fm.removeItem(at: victim)
                deleted.append(victim)
            } catch {
                continue
            }
        }
        return deleted
    }

    /// Byte footprint of a KV snapshot: sum of its serialised `state` arrays,
    /// matching how mlx-swift-lm's wired-memory accounting measures a cache.
    private static func cacheBytes(_ caches: [any KVCache]) -> Int {
        caches.flatMap { $0.state }.reduce(0) { $0 + $1.nbytes }
    }
}
