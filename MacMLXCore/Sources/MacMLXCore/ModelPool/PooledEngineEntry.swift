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

    public init(
        modelID: String,
        estimatedBytes: Int64,
        lastAccess: Date = Date(),
        isPinned: Bool = false
    ) {
        self.modelID = modelID
        self.estimatedBytes = estimatedBytes
        self.lastAccess = lastAccess
        self.isPinned = isPinned
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
