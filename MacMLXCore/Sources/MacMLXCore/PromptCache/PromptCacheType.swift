// Copyright © 2026 macMLX. English comments only.

/// Provenance class of a cached KV prefix, used to prioritise which entries a
/// budget-constrained ``PromptCacheStore`` evicts first. Mirrors the string
/// tags mlx-lm's `LRUPromptCache` keys its per-type LRU queues by.
///
/// The declaration order of ``evictionOrdering`` — assistant, then user, then
/// system — is the eviction preference: the balancing pop favours dropping the
/// larger queue biased toward `assistant`, keeping the more broadly reusable
/// `system`/`user` prefixes resident longer.
public enum PromptCacheType: String, Sendable, Hashable, CaseIterable {
    case assistant
    case user
    case system

    /// Eviction priority order (least-precious queue considered first).
    public static let evictionOrdering: [PromptCacheType] = [.assistant, .user, .system]
}
