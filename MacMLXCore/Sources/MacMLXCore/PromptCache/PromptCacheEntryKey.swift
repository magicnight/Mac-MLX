// Copyright © 2026 macMLX. English comments only.

/// Identity of a cached prefix inside ``PromptCacheStore``: a `(model, tokens)`
/// pair, mirroring the tuples mlx-lm's `LRUPromptCache` tracks in its LRU
/// queues. Distinct from ``PromptCacheKey`` (which is the hashed, sharded
/// filename used for the on-disk cold tier) — this key carries the raw tokens
/// the in-memory trie needs.
public struct PromptCacheEntryKey: Hashable, Sendable {
    public let modelID: String
    public let tokens: [Int]

    public init(modelID: String, tokens: [Int]) {
        self.modelID = modelID
        self.tokens = tokens
    }
}
