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

    // MARK: Detached cold-tier writes (Wave 3 Stage 3a)

    /// Off-executor serialiser for the blocking safetensors write. A demote hands
    /// it the evicted snapshot and awaits the write on ITS executor, leaving this
    /// store's executor free for concurrent `fetchNearest`/`insert`.
    private let coldWriter = ColdTierWriter()

    /// Detached writes currently in flight, keyed by ``PromptCacheKey/hashString``.
    /// A cold fetch consults this so a still-being-written entry reads as a HIT,
    /// not a phantom miss at the `fileExists` guard (invariant I1); `clearAll`
    /// cancels every entry (I4); `demoteToCold` dedupes and backpressures on it.
    private var inFlight: [String: Task<Void, Never>] = [:]

    /// Bumped by ``clearAll()`` to invalidate outstanding writes: a write task
    /// captures the epoch at spawn and `finishWrite` skips reconciliation when it
    /// no longer matches, so a `clearAll` that already wiped both tiers is never
    /// undone by a late-completing write it had cancelled (I4).
    private var epoch: Int = 0

    /// Bounded backpressure: once this many detached writes are in flight, a
    /// further demote degrades to a synchronous inline write rather than spawning
    /// an unbounded number of background writers each pinning a KV snapshot in
    /// memory. Bounded memory beats bounded latency for a spill path.
    private let maxInFlightWrites = 2

    /// Test-only ordering hook (nil in production). When set, each detached write
    /// awaits it — keyed by the entry's hash — from inside the write task, AFTER
    /// the store has recorded the index entry and registered the in-flight task
    /// but BEFORE the file is written. Lets a test hold a write "open" (file not
    /// yet on disk, `inFlight` populated) to exercise the cold-fetch await path
    /// deterministically without any `Task.sleep`/wall-clock. Set via
    /// ``installWriteBarrier(_:)``; `internal` so `@testable` tests reach it.
    var writeBarrier: (@Sendable (String) async -> Void)?

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
            // Reclaim any write temp orphaned by a hard kill between a detached
            // write's temp `savePromptCache` and its atomic rename (Wave 3 Stage
            // 3a) — see ``ColdTierWriter/sweepStaleColdTemporaries(root:)`` for why
            // this can't be left to the byte-cap prune below, which deliberately
            // SKIPS hidden files and would otherwise never reclaim it. Must run
            // BEFORE that prune/the manifest rebuild: at startup nothing is
            // mid-write, so every such temp is unambiguously a crash orphan.
            ColdTierWriter.sweepStaleColdTemporaries(root: root)
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

    // MARK: - Test hooks (internal, reached via `@testable`)

    /// Install (or clear) the deterministic ``writeBarrier`` ordering hook.
    /// Actor-isolated state can't be assigned from outside, so tests set it here.
    func installWriteBarrier(_ barrier: (@Sendable (String) async -> Void)?) {
        writeBarrier = barrier
    }

    /// Await every currently in-flight detached write to completion (including its
    /// `finishWrite` reconciliation, which runs before the task returns). Lets a
    /// test observe the on-disk outcome of the now-asynchronous demote path
    /// deterministically. Snapshots the set first so awaiting doesn't fight
    /// concurrent mutation of `inFlight`.
    func drainInFlight() async {
        for task in Array(inFlight.values) { await task.value }
    }

    /// Snapshot the currently in-flight write tasks. `Task<Void, Never>` is
    /// Sendable, so a test can capture them here BEFORE ``clearAll()`` empties the
    /// registry and then await their (cancelled) completion — the only way to
    /// deterministically observe that a cancelled write left nothing behind (I4).
    func inFlightTasksSnapshot() -> [Task<Void, Never>] {
        Array(inFlight.values)
    }

    /// The set of hashes currently registered as in flight. Test-only: lets a
    /// test confirm a specific key's write is (or isn't) registered without
    /// needing `Task` identity comparison (`Task` has no public `Equatable`).
    func inFlightHashes() -> Set<String> {
        Set(inFlight.keys)
    }

    /// Snapshot of the current cold-tier index records, keyed by hash.
    /// Test-only: lets a test verify a deduped re-demote (Bugbot #4) left the
    /// ORIGINAL record byte-identical (``ColdIndexEntry`` is `Equatable`,
    /// including `mtime`) rather than re-stamping it.
    func coldEntriesSnapshot() -> [String: ColdIndexEntry] {
        coldEntries
    }

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
    ///
    /// `async` as of Wave 3 Stage 3a: the HOT path below stays fully synchronous
    /// (no suspension), but the cold fallback may `await` an in-flight detached
    /// write so a mid-write entry serves as a hit rather than a phantom miss.
    /// Every call site already `await`s this actor method, so its public shape is
    /// source-unchanged.
    public func fetchNearest(
        modelID: String, tokens: [Int], modelFingerprint: String? = nil
    ) async -> PromptCacheHit? {
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
        if let hit = await fetchFromColdTrie(
            modelID: modelID, tokens: tokens, fingerprint: modelFingerprint)
        {
            return hit
        }
        return await fetchFromCold(
            modelID: modelID, tokens: tokens, fingerprint: modelFingerprint)
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
    ///
    /// Known accepted residue: a detached write can land its atomic `rename`
    /// (see ``ColdTierWriter/write(snapshot:to:metadata:)``) in the same narrow
    /// window this method's root-removal races against it, leaving ONE file on
    /// disk with no ``coldEntries`` record after the wipe. This is bounded (at
    /// most one file per write that was mid-rename at wipe time), fingerprint-
    /// safe (a content-addressed, parse-validated file is never served as wrong
    /// output — an unindexed file is simply never looked up), and reclaimed by
    /// the next byte-cap prune like any ordinary `*.safetensors` file. Closing it
    /// would mean awaiting every in-flight write here, which would turn this
    /// synchronous API `async` and add exactly the suspension point Stage 3a's
    /// design avoids — not worth it for a residue this narrow and harmless.
    public func clearAll() {
        // Invalidate outstanding detached writes (I4). Bumping `epoch` makes any
        // late-completing write's `finishWrite` skip reconciliation (it would
        // otherwise re-touch a tier we're about to wipe), and cancelling each task
        // makes its pre-rename `Task.checkCancellation` abort before a file lands.
        // Combined with the root wipe below, no entry can resurrect. Deliberately
        // NOT awaited — cancel-and-let-them-abort avoids adding a suspension point
        // to what is otherwise a synchronous wipe.
        epoch += 1
        for task in inFlight.values { task.cancel() }
        inFlight.removeAll()

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

    /// Persist an evicted entry to the cold tier, with the blocking safetensors
    /// write moved OFF this actor's executor (Wave 3 Stage 3a).
    ///
    /// `internal` (not `private`) purely so the disabled-tier no-op is unit
    /// testable without MLX — see `PromptCacheColdPruneTests`.
    ///
    /// Order of operations: guards, THEN the same-key dedup check (Bugbot #4 —
    /// deliberately BEFORE any recording, so a dedup return never re-stamps the
    /// index with content that won't be what actually lands), THEN the index +
    /// trie record, prune, flush — so a concurrent cold-trie fetch and a restart
    /// both see the entry immediately. Only the FILE write differs by path: the
    /// common case hands it to ``coldWriter`` on a detached task registered in
    /// ``inFlight``; a bounded backpressure fallback (too many detached writes
    /// already in flight) writes synchronously inline instead, mirroring
    /// ``finishWrite``'s success/failure reconciliation itself, since it
    /// bypasses that method entirely (Bugbot #1, #2). This method stays
    /// synchronous (the barrier is awaited inside the detached task, not here),
    /// so the `insert` → `store` → `evictOne` chain and `insert`'s public
    /// signature are untouched.
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
        let hash = key.hashString
        let metadata: [String: String] = [
            "modelID": key.modelID,
            "tokenCount": String(key.tokenCount),
            "modelFingerprint": fingerprint,
        ]

        // Dedup (Bugbot #4): a write for this exact content-addressed key is
        // already in flight, and its index record (recorded when THAT demote
        // ran, below) already stands. Returning HERE — before this call would
        // re-record coldEntries/coldTrie/the manifest — keeps the index
        // consistent with the file that will actually land on disk. Recording
        // again here (the old order) would re-stamp the record with THIS call's
        // metadata (e.g. a fresh `mtime`) while the in-flight write lands the
        // ORIGINAL call's content — index and file would disagree about what's
        // on disk. A same-fingerprint re-demote is identical content anyway
        // (nothing lost by skipping); a fingerprint-divergent one leaves
        // index == file == the in-flight (original) content, which a later
        // fetch validates against the FILE's own embedded fingerprint and
        // rejects if the weights have since changed — never a silent
        // divergence, never wrong output. This also avoids registering a
        // second concurrent same-hash write, which would risk an intra-epoch
        // `inFlight` clobber.
        if inFlight[hash] != nil { return }

        // Record the entry into the cold index (source of truth) + cold trie so
        // an in-process cold-trie fetch can find it now, and a restart can
        // rebuild cross-session LCP from the manifest. The full token array is
        // the load-bearing field — the tokens live nowhere else on disk.
        // Trimmability is stamped from the live caches so a cold *longer* fetch
        // pre-gates a trim without a load. `Date.now` is fine inside the actor.
        let isTrimmable = MLXLMCommon.canTrimPromptCache(caches)
        let entry = ColdIndexEntry(
            hashString: hash, modelID: key.modelID, tokens: tokens,
            tokenCount: key.tokenCount, modelFingerprint: fingerprint,
            nbytes: Self.cacheBytes(caches), mtime: Date.now, isTrimmable: isTrimmable)
        coldEntries[hash] = entry
        coldTrie.add(
            model: modelID, tokens: tokens,
            value: ColdPointer(
                hashString: hash, fingerprint: fingerprint, isTrimmable: isTrimmable))

        // Enforce the cold-tier disk budget by mtime-LRU, then persist the index.
        // ORDERING NUANCE (Stage 3a): the write below hasn't landed yet, so this
        // prune scans a directory that does NOT yet hold the in-flight file — it
        // is therefore neither counted nor a prune victim this round. The cap is
        // re-enforced when the write lands (``finishWrite`` for the detached
        // path, or inline below for the backpressure path) and on the next
        // demote; this transient under-count is acceptable. `flushColdIndex`
        // persists the entry as expected-to-land; a later write failure is
        // reconciled + reflushed (by ``finishWrite``, or inline below).
        pruneCold(protecting: url)
        flushColdIndex()

        // ---- File write. ----
        // Carry the evicted caches to the writer via the module's Sendable KV
        // carrier. ``evictOne`` just `pop`-ed these out of the hot trie, so the
        // store holds the SOLE reference: exclusive ownership transfers to the
        // writer with no copy and no aliasing — exactly ``PromptCacheSnapshot``'s
        // documented `@unchecked Sendable` invariant. (A raw `sending [any
        // KVCache]` transfer is not expressible: a value read back out of the
        // actor-isolated trie is permanently region-merged with this actor, so the
        // region checker rejects every attempt to extract it as `sending`.)
        let snapshot = PromptCacheSnapshot(caches)

        // Backpressure: with `maxInFlightWrites` background writers already each
        // pinning a KV snapshot in memory, degrade to a synchronous inline write
        // (blocks this executor for one write) rather than spawn an unbounded
        // number more. Bounded memory beats bounded latency for a spill path.
        // This bypasses ``finishWrite`` entirely, so it mirrors BOTH of that
        // method's outcomes inline (Bugbot #1, #2):
        if inFlight.count >= maxInFlightWrites {
            do {
                try ColdTierWriter.writeSynchronously(
                    snapshot: snapshot, to: url, metadata: metadata)
                // #2: the file has landed now, so re-enforce the cap the
                // demote-time prune above could not yet count (mirrors
                // ``finishWrite``'s `.success` branch).
                if pruneCold(protecting: url) { flushColdIndex() }
            } catch {
                // #1: reconcile the phantom index record for a write that never
                // landed (mirrors ``finishWrite``'s `.failure` branch). No epoch
                // guard needed here — unlike a detached write's delayed
                // completion, this runs synchronously inside `demoteToCold` on
                // THIS actor's executor with no intervening suspension, so no
                // `clearAll` can have run between the record above and this catch.
                dropColdRecord(modelID: modelID, tokens: tokens)
                try? FileManager.default.removeItem(at: url)
                flushColdIndex()
            }
            return
        }

        // Spawn the detached write and register it. The task inherits this actor's
        // isolation, but its `await`s (the test barrier, then the writer actor)
        // free THIS executor the instant it suspends — the blocking
        // `savePromptCache` runs on the writer's executor. `finishWrite` then runs
        // back on this actor to clear the registry and reconcile a failed write.
        let capturedEpoch = epoch
        let barrier = writeBarrier
        let task = Task { [coldWriter] in
            // Barrier (nil in production) gates the write AFTER `inFlight` is
            // populated, so a concurrent cold fetch observes the in-flight entry
            // and awaits it rather than phantom-missing (I1).
            await barrier?(hash)
            let result: Result<Void, Error>
            do {
                try await coldWriter.write(snapshot: snapshot, to: url, metadata: metadata)
                result = .success(())
            } catch {
                result = .failure(error)
            }
            self.finishWrite(hash: hash, epoch: capturedEpoch, result: result)
        }
        inFlight[hash] = task
    }

    /// Retire a detached write. Runs back on this actor after the write task
    /// completes (success, failure, or cancellation).
    ///
    /// - FIRST, and above all else: a STALE completion — one whose captured
    ///   `epoch` no longer matches `self.epoch` because a ``clearAll()`` ran
    ///   while this write was in flight — does NOTHING and returns immediately.
    ///   This is not an optimisation; it is required for correctness. Consider:
    ///   task1 (epoch 0) is in flight for hash `h` → `clearAll()` bumps the epoch
    ///   to 1, cancels task1, and empties `inFlight` → a fresh eviction of the
    ///   SAME key `h` spawns task2, registering `inFlight[h] = task2` → only THEN
    ///   does the cancelled task1 finally resume and call `finishWrite(epoch: 0,
    ///   …)`. If that call proceeded, its unconditional `inFlight[h] = nil` would
    ///   CLOBBER task2's live registration: a concurrent fetch would no longer
    ///   find task2 to await (reintroducing the I1 phantom-miss transiently), and
    ///   a later `clearAll` could no longer cancel task2 (weakening I4). The
    ///   epoch guard makes task1's stale completion a no-op — `clearAll` already
    ///   removed *its* `inFlight` entry and already wiped the tiers it would have
    ///   touched — leaving task2 completely undisturbed. A normal (non-stale)
    ///   completion is unaffected: `epoch == self.epoch` holds, so it falls
    ///   through exactly as before.
    /// - Only past that guard: clear the ``inFlight`` registration — this task is
    ///   done and, epoch having matched, it is still the live entry for `hash`.
    /// - On `.success`: re-enforces the cold cap now that the file has actually
    ///   landed (the demote-time prune under-counted it — see the ordering nuance
    ///   there), flushing only if that changed the index. Leaves the index record
    ///   otherwise as-is — it may have been legitimately dropped by a promotion
    ///   since the demote, so it is never re-added here.
    /// - On `.failure`: reconciles the phantom record the demote optimistically
    ///   wrote, when the record still exists: drop it, best-effort delete any
    ///   leftover file, and reflush (I2).
    private func finishWrite(hash: String, epoch: Int, result: Result<Void, Error>) {
        guard epoch == self.epoch else { return }
        inFlight[hash] = nil
        switch result {
        case .success:
            // The file exists now, so the cap can finally count it. Protect the
            // just-landed file (newest mtime anyway) so it survives its own prune.
            let landed = shardedURL(forHash: hash)
            if pruneCold(protecting: landed) { flushColdIndex() }
        case .failure:
            guard let entry = coldEntries[hash] else { return }
            dropColdRecord(modelID: entry.modelID, tokens: entry.tokens)
            let url = PromptCacheKey(modelID: entry.modelID, tokens: entry.tokens)
                .shardedFileURL(under: root)
            try? FileManager.default.removeItem(at: url)
            flushColdIndex()
        }
    }

    /// The sharded cold-file URL for a bare `hashString`, mirroring
    /// ``PromptCacheKey/shardedFileURL(under:)`` — usable where only the hash is in
    /// hand (e.g. ``finishWrite`` reconciling a landed write).
    private func shardedURL(forHash hash: String) -> URL {
        root
            .appending(path: String(hash.prefix(1)), directoryHint: .isDirectory)
            .appending(path: "\(hash).safetensors", directoryHint: .notDirectory)
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
    /// `loadPromptCache` still reads synchronously on this executor (~1.3 ms —
    /// cheap; moving the READ off-actor is Stage 3c, deferred). The Stage 3a
    /// change: a missing file is no longer an immediate miss — if a detached
    /// WRITE for this exact key is in flight, await it and re-check, so a
    /// mid-write entry is a hit not a phantom miss (I1). Hardened defensively
    /// (Bugbot #3): the await is UNCONDITIONAL on `inFlight`, not gated on
    /// `fileExists` first — even when a file already appears to exist, a
    /// pending write for the same key might be mid-rename (content-addressed,
    /// so it can only be replacing the file with identical bytes — never wrong
    /// output either way, but this avoids ever reading a file a write is
    /// actively touching).
    private func fetchFromCold(
        modelID: String, tokens: [Int], fingerprint: String?
    ) async -> PromptCacheHit? {
        // Master opt-in: a disabled cold tier has nothing on disk to restore.
        // Short-circuit before any filesystem work so a hot miss stays a pure
        // in-memory miss.
        guard coldEnabled else { return nil }
        let key = PromptCacheKey(modelID: modelID, tokens: tokens)
        let url = key.shardedFileURL(under: root)
        // Await any in-flight write for this key BEFORE reading, regardless of
        // whether a file currently appears to exist. Reentrancy discipline: the
        // await is a suspension point, so `fileExists` below is a FRESH read,
        // never trusted from before the await.
        if let task = inFlight[key.hashString] { await task.value }
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
    ///
    /// `async` as of Stage 3a: a candidate whose file is absent may just be a
    /// detached WRITE still in flight, so the vanished-file branch awaits any such
    /// write and re-validates before treating absence as a lazy miss (I1).
    /// Hardened defensively (Bugbot #3): the await is UNCONDITIONAL on
    /// `inFlight` per candidate, not gated on `fileExists` first — see
    /// ``fetchFromCold`` for why a file appearing to exist doesn't make the
    /// await skippable.
    private func fetchFromColdTrie(
        modelID: String, tokens: [Int], fingerprint: String?
    ) async -> PromptCacheHit? {
        guard coldEnabled else { return nil }
        let search = coldTrie.search(model: modelID, tokens: tokens)
        for candidate in Self.resolveReuse(search, tokenCount: tokens.count) {
            guard let pointer = coldTrie.get(model: modelID, tokens: candidate.key) else {
                continue
            }
            // Pre-gate a *longer* continuation on its stamped trimmability so a
            // cache we could never trim isn't loaded just to be rejected.
            if candidate.requiresTrim, !pointer.isTrimmable { continue }

            let candidateKey = PromptCacheKey(modelID: modelID, tokens: candidate.key)
            let url = candidateKey.shardedFileURL(under: root)
            let h = candidateKey.hashString
            // The manifest is a hint: a record whose file has vanished (pruned by
            // another process, deleted) degrades to a lazy miss. A missing file
            // may instead be a detached WRITE still in flight (I1). Defensive
            // hardening (Bugbot #3): await any in-flight write for this key
            // UNCONDITIONALLY — not only when the file is currently missing —
            // then RE-VALIDATE from scratch. Reentrancy discipline: the `await`
            // is a suspension point, so re-read `coldEntries`/`fileExists` fresh
            // and never trust the pre-await `search`/`pointer`. Only a file still
            // absent after the write settles — a genuine vanish, or a write that
            // `finishWrite` already reconciled away — is dropped as a lazy miss.
            if let task = inFlight[h] { await task.value }
            guard coldEntries[h] != nil,
                FileManager.default.fileExists(atPath: url.path)
            else {
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
    ///
    /// Returns `true` when it dropped at least one in-memory record, so callers
    /// that need the manifest to match (``finishWrite``) can flush conditionally.
    @discardableResult
    private func pruneCold(protecting: URL?) -> Bool {
        let deleted = Self.pruneColdDirectory(
            root: root, capBytes: coldCapBytes, protecting: protecting)
        // Reconcile the deletions out of the in-memory index + trie. A cold
        // file's basename is "<hash>.safetensors", so its stem is the hash that
        // keys ``coldEntries``; drop the record and its trie node together.
        var changed = false
        for url in deleted {
            let hash = url.deletingPathExtension().lastPathComponent
            guard let entry = coldEntries[hash] else { continue }
            coldEntries[hash] = nil
            coldTrie.pop(model: entry.modelID, tokens: entry.tokens)
            changed = true
        }
        return changed
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
