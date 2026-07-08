import Foundation

/// Actor managing multiple resident `InferenceEngine` instances with
/// LRU + explicit pinning + byte-budget auto-evict. Use cases:
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

    /// Currently resident engines, keyed by model ID.
    private var engines: [String: any InferenceEngine] = [:]
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

    public func residentModelIDs() -> [String] {
        Array(engines.keys).sorted()
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
        if let e = engines.removeValue(forKey: modelID) {
            try? await e.unload()
        }
        entries.removeValue(forKey: modelID)
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
        let victims = evict(toFit: cost)

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
            // residency compounds POOL-4's budget accounting).
            for victim in victims {
                try? await victim.unload()
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
    /// its engine (if resident). The entry leaves `entries`/`engines`
    /// synchronously; the caller decides when to perform the async unload —
    /// the load path (POOL-5) awaits it before allocating the new model,
    /// while `removeAndUnload` keeps the fire-and-forget shape for the sweep.
    @discardableResult
    private func detach(_ modelID: String) -> (any InferenceEngine)? {
        let engine = engines.removeValue(forKey: modelID)
        entries.removeValue(forKey: modelID)
        return engine
    }

    /// Detach `modelID` and fire-and-forget its async unload. Used only by
    /// `sweepIdle(now:)`: a TTL reclaim isn't racing an imminent allocation
    /// (unlike the load path, which awaits its victims for POOL-5), so
    /// awaiting each victim here would only serialise sweeps for no
    /// memory-safety benefit. The entry leaves the resident set
    /// synchronously; only the physical free is deferred.
    private func removeAndUnload(_ modelID: String) {
        if let e = detach(modelID) {
            Task { try? await e.unload() }
        }
    }

    /// Evict LRU non-pinned, non-generating entries until (currentBytes +
    /// incoming) fits, and RETURN the detached victims' engines so the
    /// caller (`load`) can `await` their unloads before the new model
    /// allocates (POOL-5). Skipping `isGenerating` mirrors the guard
    /// `sweepIdle` already applies (POOL-3) — without it, a concurrent
    /// `load(_:)` (GUI cold-swap, or another server request) could evict a
    /// model out from under an in-flight generation: bytes not actually
    /// freed (ARC keeps weights alive via the generation's captured
    /// container) so the budget is silently violated, and the entry
    /// disappearing feeds POOL-1 stale-ready / SRV-1 bricking.
    private func evict(toFit incoming: Int64) -> [any InferenceEngine] {
        var target = maxBytes - incoming
        if target < 0 { target = 0 }

        // Candidates: non-pinned, non-generating, oldest first.
        let candidates = entries.values
            .filter { !$0.isPinned && !$0.isGenerating }
            .sorted { $0.lastAccess < $1.lastAccess }

        var current = currentResidentBytes()
        var iterator = candidates.makeIterator()
        var victims: [any InferenceEngine] = []
        while current > target, let victim = iterator.next() {
            if let engine = detach(victim.modelID) {
                victims.append(engine)
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
