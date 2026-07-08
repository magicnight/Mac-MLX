import Foundation

/// Bookkeeping struct held by `ModelPool` per resident model.
/// The engine itself is not stored here (it's an actor in the pool's
/// dict); this is the value-type metadata.
public struct PooledEngineEntry: Sendable, Equatable {
    /// Model identifier (matches `LocalModel.id`).
    public let modelID: String
    /// Estimated memory cost — sum of safetensors file sizes in
    /// the model directory. Rough but stable for budget math;
    /// actual MLX allocator usage can exceed this by 10–30%.
    public let estimatedBytes: Int64
    /// Wall-clock time of last `engine(for:)` or `load(_:)` access.
    public var lastAccess: Date
    /// Pinned entries are never evicted by the LRU sweeper.
    public var isPinned: Bool
    /// Optional idle time-to-live in seconds (v0.5.1). When set,
    /// `ModelPool.sweepIdle(now:)` unloads this entry once it has been
    /// idle longer than `ttlSeconds` — even within the byte budget.
    /// `nil` means "never idle-unload"; pinned entries are exempt.
    public var ttlSeconds: Int?
    /// In-flight marker (v0.5.1 A4). `true` while a generation is actively
    /// streaming against this entry's engine. `ModelPool.sweepIdle(now:)`
    /// never unloads an entry with `isGenerating == true`, so a concurrent
    /// `load(_:)`'s idle sweep can't evict a model mid-stream. Toggled by
    /// `ModelPool.setGenerating(_:_:)` around the generation.
    public var isGenerating: Bool = false

    public init(
        modelID: String,
        estimatedBytes: Int64,
        lastAccess: Date = Date(),
        isPinned: Bool = false,
        ttlSeconds: Int? = nil
    ) {
        self.modelID = modelID
        self.estimatedBytes = estimatedBytes
        self.lastAccess = lastAccess
        self.isPinned = isPinned
        self.ttlSeconds = ttlSeconds
    }
}

/// Sum of `.safetensors` files under `directory`. Rough proxy for
/// how much memory the model needs when loaded. Returns 0 on any
/// filesystem error.
public func estimateModelSize(at directory: URL) -> Int64 {
    guard let files = try? FileManager.default.contentsOfDirectory(
        at: directory,
        includingPropertiesForKeys: [.fileSizeKey]
    ) else {
        return 0
    }
    return files
        .filter { $0.pathExtension.lowercased() == "safetensors" }
        .compactMap { url -> Int64? in
            guard let values = try? url.resourceValues(forKeys: [.fileSizeKey]),
                  let size = values.fileSize else { return nil }
            return Int64(size)
        }
        .reduce(0, +)
}
