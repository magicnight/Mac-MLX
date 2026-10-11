import Foundation
import os

/// Actor managing multiple resident `InferenceEngine` instances with
/// LRU + explicit pinning + byte-budget auto-evict, plus the auxiliary
/// residents (the server's reranker and embedder, #130) whose bytes it
/// budgets without holding their engines. Use cases:
///
/// - Swap between chat models without re-reading weights from disk
/// - External API cold-swap without unloading the GUI's current model
/// - Keep a small always-ready model pinned alongside a big one that
///   auto-evicts on memory pressure
///
/// Load path is serialised under `loadTasks` to avoid two concurrent
/// requests double-loading the same weights — the second caller awaits
/// the first's completion.
public actor ModelPool {

    public typealias EngineFactory = @Sendable (LocalModel) -> any InferenceEngine

    // MARK: - State

    /// An unload the pool runs for an entry it reclaims: an engine's
    /// `unload()`, or the release an auxiliary resident's owner registered.
    public typealias UnloadAction = @Sendable () async -> Void

    /// Currently resident engines, keyed by model ID.
    private var engines: [String: any InferenceEngine] = [:]
    /// Auxiliary residents (#130), keyed by `auxiliaryKey(_:_:)`: models the
    /// pool did not load and holds no engine for (the server's reranker and
    /// embedder), only the action that makes their owner let go. Their
    /// `entries` row takes part in the byte budget, LRU eviction, the idle
    /// sweep and in-flight protection like any engine's.
    private var auxiliaryUnloads: [String: UnloadAction] = [:]

    /// The unload actions a load still has to run, handed to its task. Each
    /// action is dropped as soon as it has run, so nothing it captured lives
    /// on into the new allocation (POOL-5 is about memory, and a captured
    /// engine would keep its weights resident).
    private final class PendingUnloads: Sendable {
        private let remaining: OSAllocatedUnfairLock<[UnloadAction]>

        init(_ actions: [UnloadAction]) {
            remaining = OSAllocatedUnfairLock(initialState: actions)
        }

        func next() -> UnloadAction? {
            remaining.withLock { $0.popLast() }
        }
    }

    /// Bookkeeping keyed by model ID.
    private var entries: [String: PooledEngineEntry] = [:]
    /// In-flight loads so concurrent callers deduplicate.
    private var loadTasks: [String: Task<any InferenceEngine, Error>] = [:]
    /// Byte reservations for in-flight loads, keyed by model ID (POOL-4).
    /// A load reserves its cost here BEFORE suspending on the load task —
    /// the entry only lands in `entries` after the load completes, so
    /// without this reservation `currentResidentBytes()` would ignore
    /// in-flight loads entirely. Two concurrent loads of DIFFERENT models
    /// would then each `evict(toFit:)` blind to the other and let combined
    /// residency exceed `maxBytes` (OOM risk). Converted into a real entry
    /// on success; rolled back on failure.
    private var pendingBytes: [String: Int64] = [:]
    /// Tombstones for models whose `unload(_:)` raced an in-flight `load(_:)`
    /// (P3-5). While a load is in flight its engine isn't in `engines`/`entries`
    /// yet — it's inserted only after the load task completes — so an `unload`
    /// arriving mid-load would otherwise find nothing, no-op, and let the
    /// completing load resurrect the very entry the caller asked to remove.
    /// `unload` drops a tombstone here; the completing load checks it (before
    /// inserting) and instead unloads its fresh engine + releases the byte
    /// reservation, so the model ends up NOT resident. Cleared by the load that
    /// observes it (or on load failure).
    private var unloadTombstones: Set<String> = []

    private let engineFactory: EngineFactory

    // MARK: - Budget

    /// Maximum total estimated bytes that may be resident. Exceeding
    /// this triggers LRU eviction (pinned entries are spared).
    public var maxBytes: Int64

    public init(
        maxBytes: Int64,
        engineFactory: @escaping EngineFactory
    ) {
        self.maxBytes = maxBytes
        self.engineFactory = engineFactory
    }

    public func setMaxBytes(_ bytes: Int64) {
        self.maxBytes = bytes
    }

    // MARK: - Public

    /// Every resident model: engines under their model id, auxiliary residents
    /// under `auxiliaryKey(_:_:)`.
    public func residentModelIDs() -> [String] {
        Array(entries.keys).sorted()
    }

    /// The key an auxiliary resident lives under: `"reranker:<model id>"`,
    /// so a reranker and a chat model with the same id never share an entry.
    public static func auxiliaryKey(_ kind: AuxiliaryKind, _ modelID: String) -> String {
        "\(kind.rawValue):\(modelID)"
    }

    public func engine(for modelID: String) -> (any InferenceEngine)? {
        guard let e = engines[modelID] else { return nil }
        // Touch LRU timestamp.
        if var entry = entries[modelID] {
            entry.lastAccess = Date()
            entries[modelID] = entry
        }
        return e
    }

    public func setPinned(_ modelID: String, _ pinned: Bool) {
        guard var entry = entries[modelID] else { return }
        entry.isPinned = pinned
        entries[modelID] = entry
    }

    public func isPinned(_ modelID: String) -> Bool {
        entries[modelID]?.isPinned ?? false
    }

    /// Admit an auxiliary resident (#130): a model the pool does not load
    /// itself but whose bytes it must account for. Same order as `load(_:)`:
    /// reclaim idle entries, evict to fit `model`'s cost, reserve that cost,
    /// unload the victims, and only then run `load` (the owner's, serialised
    /// by the owner). On success the reservation becomes an entry whose
    /// unload is `release`; the pool calls `release` when it reclaims the
    /// entry — under budget pressure, in the idle sweep, or on `unload(_:)`
    /// — and never otherwise. With `inUse` the entry is in use from the
    /// moment it exists (the request admitting it is about to compute on it,
    /// and an eviction between the admission and a separate mark would take
    /// it), until the owner's `endUsingAuxiliary`. On failure
    /// the reservation is rolled back and the error rethrown. An entry
    /// already under this key is dropped without its release first: the
    /// owner is replacing what it had. An `unload(_:)` of this key while the
    /// admission is in flight finds no entry yet and does nothing; no caller
    /// does that.
    public func admitAuxiliary(
        _ kind: AuxiliaryKind,
        _ model: LocalModel,
        ttlSeconds: Int? = nil,
        inUse: Bool = false,
        load: @Sendable () async throws -> Void,
        release: @escaping UnloadAction
    ) async throws {
        let key = Self.auxiliaryKey(kind, model.id)
        forgetAuxiliary(kind, model.id)
        sweepIdle()
        let cost = model.sizeBytes > 0 ? model.sizeBytes : estimateModelSize(at: model.directory)
        var victims = evict(toFit: cost)
        pendingBytes[key] = cost
        // Each victim's unload runs, and is dropped, before the owner's load
        // allocates (POOL-5), so nothing an action captured lives on.
        while let victim = victims.popLast() {
            await victim()
        }
        do {
            try await load()
        } catch {
            pendingBytes.removeValue(forKey: key)
            throw error
        }
        pendingBytes.removeValue(forKey: key)
        auxiliaryUnloads[key] = release
        entries[key] = PooledEngineEntry(
            modelID: key,
            estimatedBytes: cost,
            ttlSeconds: ttlSeconds
        )
        if inUse { beginGenerating(key) }
    }

    /// Drop an auxiliary resident's entry WITHOUT running its release: the
    /// owner has already let go of the model (a swap, its unload door, a
    /// stop). No-op when nothing is resident under that key.
    public func forgetAuxiliary(_ kind: AuxiliaryKind, _ modelID: String) {
        let key = Self.auxiliaryKey(kind, modelID)
        guard auxiliaryUnloads.removeValue(forKey: key) != nil else { return }
        entries.removeValue(forKey: key)
    }

    /// POOL-3 for an auxiliary resident: a request is about to compute on it,
    /// so eviction and the idle sweep spare it until `endUsingAuxiliary`, and
    /// its LRU clock is refreshed in the same step, so a hit needs no separate
    /// touch. Returns false, marking nothing, when the pool no longer holds
    /// the entry: it was evicted or swept and its release is on its way to
    /// the owner, or has reached it, who must not compute on the model as
    /// resident. An admission with `inUse` starts in use the same way.
    public func beginUsingAuxiliary(_ kind: AuxiliaryKind, _ modelID: String) -> Bool {
        let key = Self.auxiliaryKey(kind, modelID)
        // Keyed on the release, like `forgetAuxiliary`: an entry under this
        // key with no release is a chat model whose directory happens to be
        // named like one, not an auxiliary resident.
        guard auxiliaryUnloads[key] != nil else { return false }
        beginGenerating(key)
        return true
    }

    /// The request has finished with the model: one end per admission with
    /// `inUse` or `beginUsingAuxiliary` that returned true (an end for an
    /// entry the pool has since let go is a no-op).
    public func endUsingAuxiliary(_ kind: AuxiliaryKind, _ modelID: String) {
        endGenerating(Self.auxiliaryKey(kind, modelID))
    }

    /// Register the start of a generation against `id` (A3). Increments the
    /// entry's in-flight refcount, so while ANY generation is streaming the
    /// entry is exempt from both `sweepIdle` and `evict(toFit:)` — a
    /// concurrent `load(_:)` (GUI cold-swap, or another server request)
    /// can't reclaim a model mid-stream. Also refreshes `lastAccess`, so a
    /// just-started generation counts as fresh for LRU/TTL. No-op when `id`
    /// isn't resident. Balance every call with exactly one `endGenerating`.
    public func beginGenerating(_ id: String) {
        guard var entry = entries[id] else { return }
        entry.generatingCount += 1
        entry.lastAccess = Date()
        entries[id] = entry
    }

    /// Register the end of a generation against `id` (A3). Decrements the
    /// in-flight refcount, clamped at 0 so an unbalanced end (or an end for
    /// an entry whose count is already 0) can never underflow into a
    /// negative value that would wrongly keep the entry protected forever.
    /// The entry becomes eligible for sweep/evict again only once the count
    /// returns to 0. No-op when `id` isn't resident.
    public func endGenerating(_ id: String) {
        guard var entry = entries[id] else { return }
        if entry.generatingCount > 0 {
            entry.generatingCount -= 1
        }
        entries[id] = entry
    }

    public func unload(_ modelID: String) async {
        // P3-5: if a load for this id is in flight, its engine isn't resident
        // yet (inserted only when the load task completes). Drop a tombstone so
        // the completing load discards its fresh engine and releases its byte
        // reservation instead of resurrecting the entry we're unloading. Setting
        // it here — synchronously, while the load is still suspended on
        // `await task.value` — is what makes the fix deterministic: the load's
        // post-await insertion block runs without an intervening `await`, so it
        // always observes a tombstone set before it resumes.
        if loadTasks[modelID] != nil {
            unloadTombstones.insert(modelID)
        }
        // The entry leaves the resident set before the unload runs, so an
        // admission under the same key that lands during the await is kept.
        if let unload = detach(modelID) {
            await unload()
        }
    }

    /// Return an engine with `model.id` loaded. Reuses an existing
    /// entry when possible. Evicts LRU entries as needed to stay
    /// within `maxBytes`. Concurrent loads of the same ID share.
    @discardableResult
    public func load(
        _ model: LocalModel,
        ttlSeconds: Int? = nil
    ) async throws -> any InferenceEngine {
        // Already loaded? Touch and return.
        if let e = engines[model.id] {
            if var entry = entries[model.id] {
                entry.lastAccess = Date()
                entries[model.id] = entry
            }
            return e
        }
        // In-flight load by another caller? Join it.
        if let pending = loadTasks[model.id] {
            return try await pending.value
        }

        // Reclaim idle models past their TTL before we take the byte
        // budget into account (v0.5.1). Safe here: `model.id` is not
        // resident (guarded above), so this can never sweep the model
        // we're about to load. Sweep-on-load is the MVP — a background
        // timer is a deliberate follow-up.
        sweepIdle()

        // Evict to fit before starting the load, using the model's
        // sizeBytes (or our estimate) as the cost. `evict` detaches victims
        // synchronously (removing them from `entries`/`engines`) and returns
        // their engines for us to unload — it no longer fires the unloads
        // itself (POOL-5).
        let cost = model.sizeBytes > 0 ? model.sizeBytes : estimateModelSize(at: model.directory)
        // Straight into the task's queue: this frame keeps no reference to the
        // victims' actions, which would keep what they captured alive.
        let pending = PendingUnloads(evict(toFit: cost))

        // POOL-4: reserve this load's cost NOW — synchronously, before the
        // first `await` below — so a concurrent `load(_:)` of a DIFFERENT
        // model sees these bytes when it `evict(toFit:)`s. The reservation
        // is placed AFTER our own `evict` (so we don't evict to fit our own
        // cost) and is converted into a real entry on success / rolled back
        // on failure.
        pendingBytes[model.id] = cost

        // Everything from the `loadTasks[model.id]` join-check above to the
        // assignment below runs without an `await`, so the dedup stays
        // atomic: a concurrent caller for the SAME id can't slip past the
        // join-check while this load is only half-registered. The victim
        // unloads (POOL-5) therefore run INSIDE the task — before the new
        // engine allocates — rather than here, where an `await` would open
        // that gap.
        let factory = engineFactory
        let task = Task { () throws -> any InferenceEngine in
            // POOL-5: free the evicted engines' memory BEFORE the new engine
            // allocates, so we never transiently hold both resident (double
            // residency compounds POOL-4's budget accounting). Each action is
            // dropped once run, so the task keeps no victim alive.
            while let victim = pending.next() {
                await victim()
            }
            let engine = factory(model)
            try await engine.load(model)
            return engine
        }
        loadTasks[model.id] = task
        do {
            let engine = try await task.value
            loadTasks.removeValue(forKey: model.id)
            pendingBytes.removeValue(forKey: model.id)  // reservation → real entry
            // P3-5: an `unload(_:)` that raced this in-flight load dropped a
            // tombstone. Honor it — discard the freshly-loaded engine instead of
            // resurrecting the entry the caller asked to unload. The byte
            // reservation was already released just above, so there is no leak.
            // These statements run with NO `await` between the resume above and
            // the insertion below, so `unload` can only have set the tombstone
            // during the load's suspension — never mid-insert.
            if unloadTombstones.remove(model.id) != nil {
                try? await engine.unload()
                return engine
            }
            engines[model.id] = engine
            entries[model.id] = PooledEngineEntry(
                modelID: model.id,
                estimatedBytes: cost,
                ttlSeconds: ttlSeconds
            )
            return engine
        } catch {
            loadTasks.removeValue(forKey: model.id)
            pendingBytes.removeValue(forKey: model.id)  // roll back on failure
            unloadTombstones.remove(model.id)  // no stale tombstone survives a failed load
            throw error
        }
    }

    // MARK: - Eviction

    /// Total bytes the budget must account for: resident entries PLUS
    /// in-flight load reservations (POOL-4). Including `pendingBytes` is
    /// what lets two concurrent loads of different models see each other's
    /// cost and evict-to-fit against the true combined footprint instead of
    /// each ignoring the other and blowing past `maxBytes`.
    private func currentResidentBytes() -> Int64 {
        let resident = entries.values.map(\.estimatedBytes).reduce(0, +)
        let pending = pendingBytes.values.reduce(0, +)
        return resident + pending
    }

    /// Detach `modelID` from the resident set WITHOUT unloading it, returning
    /// the action that unloads it (if resident): the engine's `unload()`, or
    /// an auxiliary resident's release. The entry leaves `entries` and its
    /// table synchronously; the caller decides when to run the action — the
    /// load path (POOL-5) awaits it before allocating the new model, while
    /// `removeAndUnload` keeps the fire-and-forget shape for the sweep.
    @discardableResult
    private func detach(_ modelID: String) -> UnloadAction? {
        entries.removeValue(forKey: modelID)
        if let engine = engines.removeValue(forKey: modelID) {
            return { try? await engine.unload() }
        }
        return auxiliaryUnloads.removeValue(forKey: modelID)
    }

    /// Detach `modelID` and fire-and-forget its async unload. Used only by
    /// `sweepIdle(now:)`: a TTL reclaim isn't racing an imminent allocation
    /// (unlike the load path, which awaits its victims for POOL-5), so
    /// awaiting each victim here would only serialise sweeps for no
    /// memory-safety benefit. The entry leaves the resident set
    /// synchronously; only the physical free is deferred.
    private func removeAndUnload(_ modelID: String) {
        if let unload = detach(modelID) {
            Task { await unload() }
        }
    }

    /// Evict LRU non-pinned, non-generating entries until (currentBytes +
    /// incoming) fits, and RETURN the detached victims' unload actions so
    /// the caller (`load`, `admitAuxiliary`) can `await` them before the new
    /// model allocates (POOL-5). Skipping `isGenerating` mirrors the guard
    /// `sweepIdle` already applies (POOL-3) — without it, a concurrent
    /// `load(_:)` (GUI cold-swap, or another server request) could evict a
    /// model out from under an in-flight generation: bytes not actually
    /// freed (ARC keeps weights alive via the generation's captured
    /// container) so the budget is silently violated, and the entry
    /// disappearing feeds POOL-1 stale-ready / SRV-1 bricking.
    private func evict(toFit incoming: Int64) -> [UnloadAction] {
        var target = maxBytes - incoming
        if target < 0 { target = 0 }

        // Candidates: non-pinned, non-generating, oldest first.
        let candidates = entries.values
            .filter { !$0.isPinned && !$0.isGenerating }
            .sorted { $0.lastAccess < $1.lastAccess }

        var current = currentResidentBytes()
        var iterator = candidates.makeIterator()
        var victims: [UnloadAction] = []
        while current > target, let victim = iterator.next() {
            if let unload = detach(victim.modelID) {
                victims.append(unload)
            }
            current -= victim.estimatedBytes
        }
        return victims
    }

    /// Unload every non-pinned, non-generating resident model whose
    /// `ttlSeconds` is set and whose idle time (`now - lastAccess`)
    /// exceeds it — even while inside the byte budget (v0.5.1). Pinned,
    /// in-flight (`isGenerating`), and nil-TTL entries are never swept.
    /// Called at the top of `load(_:)`; `now` is injectable for tests.
    ///
    /// Mid-use hazard (A4) — now handled: `lastAccess` is refreshed on
    /// `engine(for:)` / `load()` but NOT during a long-running generation,
    /// so a concurrent `load(_:)`'s sweep could otherwise unload a model
    /// that is actively streaming. Callers bracket the stream with
    /// `beginGenerating(_:)` / `endGenerating(_:)`, and the filter below
    /// skips any entry whose in-flight refcount is non-zero
    /// (`isGenerating == true`).
    public func sweepIdle(now: Date = Date()) {
        let expired = entries.values.filter { entry in
            guard !entry.isPinned, !entry.isGenerating, let ttl = entry.ttlSeconds else { return false }
            return now.timeIntervalSince(entry.lastAccess) > Double(ttl)
        }
        for victim in expired {
            removeAndUnload(victim.modelID)
        }
    }
}
