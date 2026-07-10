// Copyright © 2026 macMLX. English comments only.

/// Outcome of a ``PromptTrie`` prefix lookup, mirroring mlx-lm's
/// `PromptTrieResult` (`mlx_lm/models/cache.py`).
///
/// Exactly describes how the queried token sequence relates to the paths
/// stored in the trie. The four fields are not mutually exclusive in general,
/// but ``PromptCacheStore/fetchNearest(modelID:tokens:)`` consumes them in a
/// fixed priority order (exact → longer → shorter).
public struct PromptTrieSearch: Equatable, Sendable {
    /// The queried tokens, present verbatim, when a stored entry matches the
    /// whole sequence. `nil` otherwise.
    public let exact: [Int]?

    /// The longest stored entry that is a strict prefix of the query and has
    /// length ≥ 2. `nil` when no such prefix exists. (Length-1 prefixes are
    /// deliberately not reported, matching the upstream `last_index > 0` gate.)
    public let shorter: [Int]?

    /// The shortest stored entry that extends *beyond* the query (i.e. the
    /// query is a strict prefix of it), sharing ``commonPrefix`` leading
    /// tokens. `nil` when no stored path continues past the query.
    public let longer: [Int]?

    /// Length of the longest path in the trie that matches the query token by
    /// token — how far a walk from the root could descend before diverging.
    public let commonPrefix: Int

    public init(exact: [Int]?, shorter: [Int]?, longer: [Int]?, commonPrefix: Int) {
        self.exact = exact
        self.shorter = shorter
        self.longer = longer
        self.commonPrefix = commonPrefix
    }
}
