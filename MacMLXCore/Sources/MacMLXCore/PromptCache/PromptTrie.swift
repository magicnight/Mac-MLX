// Copyright © 2026 macMLX. English comments only.

/// Token-prefix trie, a pure-Swift port of mlx-lm's `PromptTrie`
/// (`mlx_lm/models/cache.py`). MLX-free: it stores opaque `Value`s keyed by
/// integer token sequences and answers longest-common-prefix queries so a
/// caller can reuse a KV-cache prefix instead of re-prefilling from scratch.
///
/// Entries are namespaced by an opaque model identifier (a different model
/// gives the same token ids different meaning, so their subtrees are kept
/// disjoint), matching upstream's per-`model` sub-dictionaries.
///
/// Not `Sendable` and not internally synchronised — intended to live behind an
/// actor (see ``PromptCacheStore``).
final class PromptTrie<Value> {

    /// One node per consumed token. `value` is the payload stored at a path
    /// that terminates here (a node can be both an interior branch *and* hold
    /// a value when a shorter sequence is a prefix of a longer stored one).
    private final class Node {
        var children: [Int: Node] = [:]
        var value: Value?
    }

    private var roots: [String: Node] = [:]

    init() {}

    /// Store `value` at `tokens` for `model`, returning the previous value at
    /// that exact path if one existed (so a caller can reconcile byte counts).
    @discardableResult
    func add(model: String, tokens: [Int], value: Value) -> Value? {
        let root: Node
        if let existing = roots[model] {
            root = existing
        } else {
            root = Node()
            roots[model] = root
        }
        var current = root
        for token in tokens {
            if let next = current.children[token] {
                current = next
            } else {
                let next = Node()
                current.children[token] = next
                current = next
            }
        }
        let previous = current.value
        current.value = value
        return previous
    }

    /// Fetch the value stored at exactly `tokens`, or `nil`.
    func get(model: String, tokens: [Int]) -> Value? {
        guard var current = roots[model] else { return nil }
        for token in tokens {
            guard let next = current.children[token] else { return nil }
            current = next
        }
        return current.value
    }

    /// Remove and return the value stored at exactly `tokens`, pruning any
    /// now-empty nodes on the path bottom-up. Returns `nil` if absent.
    @discardableResult
    func pop(model: String, tokens: [Int]) -> Value? {
        guard let root = roots[model] else { return nil }
        var path: [Node] = [root]
        for token in tokens {
            guard let next = path[path.count - 1].children[token] else { return nil }
            path.append(next)
        }
        let leaf = path[path.count - 1]
        let value = leaf.value
        leaf.value = nil

        // Prune empty nodes from the leaf upward: a node is dead once it holds
        // neither a value nor any children.
        var depth = tokens.count
        while depth > 0 {
            let node = path[depth]
            if node.children.isEmpty && node.value == nil {
                path[depth - 1].children[tokens[depth - 1]] = nil
            } else {
                break
            }
            depth -= 1
        }
        if root.children.isEmpty && root.value == nil {
            roots[model] = nil
        }
        return value
    }

    /// Remove and return every value stored at a *strict prefix* of `tokens`
    /// (as `(prefixLength, value)` pairs, shortest first). The value at
    /// `tokens` itself is left untouched. Used to drop entries made redundant
    /// by a longer, trimmable insertion.
    @discardableResult
    func popPrefixes(model: String, tokens: [Int]) -> [(Int, Value)] {
        guard let root = roots[model] else { return [] }
        var removed: [(Int, Value)] = []
        var current = root
        for (index, token) in tokens.enumerated() {
            if let value = current.value {
                removed.append((index, value))
                current.value = nil
            }
            guard let next = current.children[token] else { break }
            current = next
        }
        return removed
    }

    /// Classify `tokens` against the stored paths — see ``PromptTrieSearch``.
    func search(model: String, tokens: [Int]) -> PromptTrieSearch {
        guard let root = roots[model] else {
            return PromptTrieSearch(exact: nil, shorter: nil, longer: nil, commonPrefix: 0)
        }

        var current = root

        if tokens.isEmpty {
            if current.value != nil {
                return PromptTrieSearch(exact: [], shorter: nil, longer: nil, commonPrefix: 0)
            }
            return PromptTrieSearch(exact: nil, shorter: nil, longer: nil, commonPrefix: 0)
        }

        // Walk as far down the query as the trie permits, remembering the
        // deepest index at which a stored value sits on the path.
        var lastValueIndex = -1
        var index = 0
        while index < tokens.count, let next = current.children[tokens[index]] {
            current = next
            if current.value != nil {
                lastValueIndex = index
            }
            index += 1
        }

        // Exact hit: a value sits on the node reached after consuming every
        // query token.
        if lastValueIndex == tokens.count - 1 && lastValueIndex >= 0 {
            return PromptTrieSearch(exact: tokens, shorter: nil, longer: nil, commonPrefix: 0)
        }

        // Longest strict prefix (length ≥ 2, per upstream's `> 0` gate).
        var shorter: [Int]?
        if lastValueIndex > 0 {
            shorter = Array(tokens[0...lastValueIndex])
        }

        // Shortest stored continuation past the query, if the walk reached a
        // live node with descendants.
        var longer: [Int]?
        let commonPrefix = index
        if index > 0 {
            var best: [Int]?
            var stack: [(Node, [Int])] = [(current, [])]
            while let (node, extra) = stack.popLast() {
                if node.value != nil {
                    if let current = best {
                        if extra.count < current.count { best = extra }
                    } else {
                        best = extra
                    }
                } else {
                    let shorterThanBest: Bool
                    if let current = best {
                        shorterThanBest = extra.count < current.count
                    } else {
                        shorterThanBest = true
                    }
                    if shorterThanBest {
                        for (token, child) in node.children {
                            stack.append((child, extra + [token]))
                        }
                    }
                }
            }
            if let best {
                longer = Array(tokens[0..<index]) + best
            }
        }

        return PromptTrieSearch(
            exact: nil, shorter: shorter, longer: longer, commonPrefix: commonPrefix)
    }
}
