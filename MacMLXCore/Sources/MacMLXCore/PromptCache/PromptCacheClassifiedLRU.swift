// Copyright © 2026 macMLX. English comments only.

/// Per-type LRU ordering, a pure-Swift port of the `CacheOrder` nested class in
/// mlx-lm's `LRUPromptCache`. MLX-free.
///
/// Maintains one FIFO queue per ``PromptCacheType``; the front of a queue is
/// its least-recently-used member. ``pop()`` implements upstream's balancing
/// eviction: it walks the type ordering and evicts from the first queue that is
/// non-empty and at least as populated as the next one, so no single class
/// starves the others while still honouring the ``PromptCacheType`` priority.
struct PromptCacheClassifiedLRU<Key: Hashable> {

    private let ordering: [PromptCacheType]
    private var queues: [PromptCacheType: [Key]]

    init(ordering: [PromptCacheType] = PromptCacheType.evictionOrdering) {
        self.ordering = ordering
        var queues: [PromptCacheType: [Key]] = [:]
        for type in ordering {
            queues[type] = []
        }
        self.queues = queues
    }

    /// Total number of tracked keys across all queues.
    var count: Int {
        queues.values.reduce(0) { $0 + $1.count }
    }

    /// Number of keys currently tracked for `type`.
    func count(of type: PromptCacheType) -> Int {
        queues[type]?.count ?? 0
    }

    /// Record `key` as most-recently-used within its class.
    mutating func push(_ key: Key, type: PromptCacheType) {
        queues[type, default: []].append(key)
    }

    /// Remove `key` from whichever queue holds it (no-op if absent).
    mutating func remove(_ key: Key) {
        for type in ordering {
            if let index = queues[type]?.firstIndex(of: key) {
                queues[type]?.remove(at: index)
                return
            }
        }
    }

    /// Evict and return the next key to drop, or `nil` if empty.
    mutating func pop() -> Key? {
        var index = 0
        while index + 1 < ordering.count {
            let here = queues[ordering[index]] ?? []
            let next = queues[ordering[index + 1]] ?? []
            if !here.isEmpty && here.count >= next.count {
                return popFront(ordering[index])
            }
            index += 1
        }
        guard let last = ordering.last else { return nil }
        return popFront(last)
    }

    private mutating func popFront(_ type: PromptCacheType) -> Key? {
        guard var queue = queues[type], !queue.isEmpty else { return nil }
        let front = queue.removeFirst()
        queues[type] = queue
        return front
    }
}
