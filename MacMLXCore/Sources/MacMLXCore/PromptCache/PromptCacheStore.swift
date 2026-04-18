import Foundation
import MLX
import MLXLMCommon

/// Sendable wrapper that lets a `[any KVCache]` cross actor-isolation
/// boundaries. `KVCache` is a reference-type protocol without a
/// `Sendable` conformance in mlx-swift-lm — in practice we hand the
/// snapshot off to the generation pipeline which owns it exclusively
/// until generation ends, so an unchecked conformance is safe.
public struct PromptCacheSnapshot: @unchecked Sendable {
    public let caches: [any KVCache]
    public init(_ caches: [any KVCache]) {
        self.caches = caches
    }
}

/// Two-tier prompt-cache store. Hot = in-memory LRU dict of
/// `PromptCacheKey → [any KVCache]`. Cold = safetensors files
/// on disk under `root/<shard>/<hash>.safetensors`, round-tripped
/// through mlx-swift-lm's `savePromptCache` / `loadPromptCache`.
///
/// MVP LRU is strict — full eviction, no partial. v0.4.1+ may add
/// size-based (byte-count) eviction instead of count-based.
public actor PromptCacheStore {

    private let root: URL
    private let hotCapacity: Int

    /// Ordered pair list simulates an LRU. Head = oldest.
    /// Dictionary gives O(1) lookup; `order` gives O(n) touch but
    /// `hotCapacity` is small (default 8), so linear scans are fine.
    private var hot: [PromptCacheKey: [any KVCache]] = [:]
    private var order: [PromptCacheKey] = []

    public init(root: URL, hotCapacity: Int = 8) {
        self.root = root
        self.hotCapacity = hotCapacity
        try? FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true
        )
    }

    /// Insert or refresh. Evicts to disk if hot is full.
    public func put(key: PromptCacheKey, snapshot: PromptCacheSnapshot) {
        let cache = snapshot.caches
        if hot[key] != nil {
            touch(key)
            hot[key] = cache
            return
        }
        while hot.count >= hotCapacity, let oldest = order.first {
            demote(oldest)
        }
        hot[key] = cache
        order.append(key)
    }

    /// Blow away both tiers. Hot dict is cleared, the cold-tier
    /// directory is removed wholesale and re-created empty. Invoked
    /// from the Settings → "Clear All KV Caches" button via
    /// `MLXSwiftEngine.clearPromptCache()` and
    /// `EngineCoordinator.clearPromptCache()`.
    public func clearAll() {
        hot.removeAll()
        order.removeAll()
        let root = self.root
        try? FileManager.default.removeItem(at: root)
        try? FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true
        )
    }

    /// Return a cache snapshot, preferring the hot tier. On cold-hit,
    /// promote into hot (possibly evicting another entry).
    public func get(_ key: PromptCacheKey) -> PromptCacheSnapshot? {
        if let cache = hot[key] {
            touch(key)
            return PromptCacheSnapshot(cache)
        }
        let url = key.shardedFileURL(under: root)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return nil
        }
        do {
            let (caches, _) = try loadPromptCache(url: url)
            // Promote.
            while hot.count >= hotCapacity, let oldest = order.first {
                demote(oldest)
            }
            hot[key] = caches
            order.append(key)
            return PromptCacheSnapshot(caches)
        } catch {
            return nil
        }
    }

    // MARK: - Private

    private func touch(_ key: PromptCacheKey) {
        order.removeAll { $0 == key }
        order.append(key)
    }

    /// Persist an entry to disk + remove from hot.
    private func demote(_ key: PromptCacheKey) {
        guard let cache = hot.removeValue(forKey: key) else {
            order.removeAll { $0 == key }
            return
        }
        order.removeAll { $0 == key }
        let url = key.shardedFileURL(under: root)
        let parent = url.deletingLastPathComponent()
        try? FileManager.default.createDirectory(
            at: parent,
            withIntermediateDirectories: true
        )
        let metadata: [String: String] = [
            "modelID": key.modelID,
            "tokenCount": String(key.tokenCount)
        ]
        try? savePromptCache(url: url, cache: cache, metadata: metadata)
    }
}
