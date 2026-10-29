// Copyright © 2026 macMLX. English comments only.

/// What the history-boundary prefill (see ``HistoryBoundary``) hands back out
/// of `container.perform`: the copy to store, the key length it belongs
/// under, and what the split prefill cost, which the request's usage and
/// phase summary add back.
struct HistoryBoundarySnapshot: Sendable {
    /// The number of leading prompt tokens the snapshot holds; its key is
    /// `prompt[0..<boundary]`.
    let boundary: Int
    /// Tokens prefilled before the split, `boundary` minus what the prompt
    /// cache had already served.
    let prefilledTokens: Int
    /// Wall time of that prefill.
    let seconds: Double
    /// An evaluated copy of the cache at the boundary, independent of the
    /// working cache generation continues on.
    let snapshot: PromptCacheSnapshot
}
