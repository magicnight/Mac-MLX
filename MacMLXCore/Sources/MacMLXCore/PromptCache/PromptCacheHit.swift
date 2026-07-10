// Copyright © 2026 macMLX. English comments only.

/// Result of a nearest-prefix cache lookup
/// (``PromptCacheStore/fetchNearest(modelID:tokens:)``).
///
/// The `snapshot` is an independent copy whose KV state has already been
/// trimmed to `reusedTokenCount` positions, so the caller must prefill only the
/// query's remaining suffix — `tokens[reusedTokenCount...]`. The store
/// guarantees `reusedTokenCount ≤ tokens.count - 1`, i.e. at least one token is
/// always left to feed the token iterator.
public struct PromptCacheHit: @unchecked Sendable {
    public let snapshot: PromptCacheSnapshot
    public let reusedTokenCount: Int

    public init(snapshot: PromptCacheSnapshot, reusedTokenCount: Int) {
        self.snapshot = snapshot
        self.reusedTokenCount = reusedTokenCount
    }
}
