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
    public func load(_ model: LocalModel) async throws -> any InferenceEngine {
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

        // Evict to fit before starting the load, using the model's
        // sizeBytes (or our estimate) as the cost.
        let cost = model.sizeBytes > 0 ? model.sizeBytes : estimateModelSize(at: model.directory)
        evict(toFit: cost)

        let factory = engineFactory
        let task = Task { () throws -> any InferenceEngine in
            let engine = factory(model)
            try await engine.load(model)
            return engine
        }
        loadTasks[model.id] = task
        do {
            let engine = try await task.value
            loadTasks.removeValue(forKey: model.id)
            engines[model.id] = engine
            entries[model.id] = PooledEngineEntry(
                modelID: model.id,
                estimatedBytes: cost
            )
            return engine
        } catch {
            loadTasks.removeValue(forKey: model.id)
            throw error
        }
    }

    // MARK: - Eviction

    private func currentResidentBytes() -> Int64 {
        entries.values.map(\.estimatedBytes).reduce(0, +)
    }

    /// Evict LRU non-pinned entries until (currentBytes + incoming) fits.
    private func evict(toFit incoming: Int64) {
        var target = maxBytes - incoming
        if target < 0 { target = 0 }

        // Candidates: non-pinned, oldest first.
        let candidates = entries.values
            .filter { !$0.isPinned }
            .sorted { $0.lastAccess < $1.lastAccess }

        var current = currentResidentBytes()
        var iterator = candidates.makeIterator()
        while current > target, let victim = iterator.next() {
            if let e = engines.removeValue(forKey: victim.modelID) {
                Task { try? await e.unload() }
            }
            entries.removeValue(forKey: victim.modelID)
            current -= victim.estimatedBytes
        }
    }
}
