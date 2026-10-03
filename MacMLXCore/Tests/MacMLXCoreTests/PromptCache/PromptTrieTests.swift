// Copyright © 2026 macMLX. English comments only.

import XCTest

@testable import MacMLXCore

/// Pure-Swift (MLX-free) parity tests for ``PromptTrie``, checked token-by-token
/// against the four-state semantics of mlx-lm's `PromptTrie.search`
/// (`mlx_lm/models/cache.py`). No Metal required — runs under bare `swift test`.
final class PromptTrieTests: XCTestCase {

    private let model = "M"

    // MARK: - search: the four states

    func testEmptyTrieReturnsAllNil() {
        let trie = PromptTrie<Int>()
        let r = trie.search(model: model, tokens: [1, 2, 3])
        XCTAssertNil(r.exact)
        XCTAssertNil(r.shorter)
        XCTAssertNil(r.longer)
        XCTAssertEqual(r.commonPrefix, 0)
    }

    func testExactMatch() {
        let trie = PromptTrie<Int>()
        trie.add(model: model, tokens: [1, 2, 3], value: 42)
        let r = trie.search(model: model, tokens: [1, 2, 3])
        XCTAssertEqual(r.exact, [1, 2, 3])
        XCTAssertNil(r.shorter)
        XCTAssertNil(r.longer)
        XCTAssertEqual(r.commonPrefix, 0)
    }

    func testSingleTokenExactMatch() {
        let trie = PromptTrie<Int>()
        trie.add(model: model, tokens: [5], value: 1)
        let r = trie.search(model: model, tokens: [5])
        XCTAssertEqual(r.exact, [5])
    }

    /// Query is a strict prefix of a longer stored entry → `longer`, no `exact`.
    func testLongerContinuation() {
        let trie = PromptTrie<Int>()
        trie.add(model: model, tokens: [1, 2, 3, 4, 5], value: 1)
        let r = trie.search(model: model, tokens: [1, 2])
        XCTAssertNil(r.exact)
        XCTAssertNil(r.shorter)
        XCTAssertEqual(r.longer, [1, 2, 3, 4, 5])
        XCTAssertEqual(r.commonPrefix, 2)
    }

    /// Query extends past a stored entry that itself has a divergent branch →
    /// `shorter` reports the stored prefix (length ≥ 2).
    func testShorterPrefix() {
        let trie = PromptTrie<Int>()
        trie.add(model: model, tokens: [1, 2], value: 1)
        trie.add(model: model, tokens: [1, 2, 9], value: 2)
        let r = trie.search(model: model, tokens: [1, 2, 3])
        XCTAssertNil(r.exact)
        XCTAssertEqual(r.shorter, [1, 2])
        XCTAssertEqual(r.commonPrefix, 2)
        // Upstream quirk: stopping on a valued node also reports it as `longer`
        // with an empty extension. Harmless — `fetchNearest` gates the longer
        // branch on `commonPrefix > shorter.count` (2 > 2 is false), so the
        // shorter branch wins.
        XCTAssertEqual(r.longer, [1, 2])
    }

    /// A length-1 stored prefix is deliberately NOT reported as `shorter`
    /// (upstream's `last_index > 0` gate).
    func testLengthOnePrefixNotReportedAsShorter() {
        let trie = PromptTrie<Int>()
        trie.add(model: model, tokens: [5], value: 1)
        let r = trie.search(model: model, tokens: [5, 6])
        XCTAssertNil(r.exact)
        XCTAssertNil(r.shorter)
        XCTAssertEqual(r.longer, [5])
        XCTAssertEqual(r.commonPrefix, 1)
    }

    /// Query diverges after sharing one token with a longer stored path.
    func testBranchingMiss() {
        let trie = PromptTrie<Int>()
        trie.add(model: model, tokens: [1, 2, 3], value: 1)
        let r = trie.search(model: model, tokens: [1, 9])
        XCTAssertNil(r.exact)
        XCTAssertNil(r.shorter)
        XCTAssertEqual(r.longer, [1, 2, 3])
        XCTAssertEqual(r.commonPrefix, 1)
    }

    func testCompletelyDisjointQuery() {
        let trie = PromptTrie<Int>()
        trie.add(model: model, tokens: [1, 2, 3], value: 1)
        let r = trie.search(model: model, tokens: [7, 8])
        XCTAssertNil(r.exact)
        XCTAssertNil(r.shorter)
        XCTAssertNil(r.longer)
        XCTAssertEqual(r.commonPrefix, 0)
    }

    /// `longer` picks the SHORTEST continuation among branches.
    func testLongerPrefersShortestExtension() {
        let trie = PromptTrie<Int>()
        trie.add(model: model, tokens: [1, 2, 3, 4, 5, 6], value: 1)
        trie.add(model: model, tokens: [1, 2, 7], value: 2)
        let r = trie.search(model: model, tokens: [1, 2])
        // Both [1,2,7] (extra [7], len 1) and [1,2,3,4,5,6] (extra len 4) extend
        // the query; the shorter extension wins.
        XCTAssertEqual(r.longer, [1, 2, 7])
        XCTAssertEqual(r.commonPrefix, 2)
    }

    // MARK: - empty-token boundary

    func testEmptyTokensWithStoredEmptyEntryIsExact() {
        let trie = PromptTrie<Int>()
        trie.add(model: model, tokens: [], value: 99)
        let r = trie.search(model: model, tokens: [])
        XCTAssertEqual(r.exact, [])
        XCTAssertNil(r.shorter)
        XCTAssertNil(r.longer)
    }

    func testEmptyTokensWithoutStoredEmptyEntry() {
        let trie = PromptTrie<Int>()
        trie.add(model: model, tokens: [1], value: 1)
        let r = trie.search(model: model, tokens: [])
        XCTAssertNil(r.exact)
        XCTAssertNil(r.shorter)
        XCTAssertNil(r.longer)
        XCTAssertEqual(r.commonPrefix, 0)
    }

    // MARK: - add / get / pop / popPrefixes

    func testAddReturnsPreviousValue() {
        let trie = PromptTrie<Int>()
        XCTAssertNil(trie.add(model: model, tokens: [1], value: 10))
        XCTAssertEqual(trie.add(model: model, tokens: [1], value: 20), 10)
        XCTAssertEqual(trie.get(model: model, tokens: [1]), 20)
    }

    func testGetMissReturnsNil() {
        let trie = PromptTrie<Int>()
        trie.add(model: model, tokens: [1, 2], value: 1)
        XCTAssertNil(trie.get(model: model, tokens: [1, 2, 3]))
        XCTAssertNil(trie.get(model: model, tokens: [9]))
    }

    func testPopRemovesValueAndPrunes() {
        let trie = PromptTrie<Int>()
        trie.add(model: model, tokens: [1, 2, 3], value: 7)
        XCTAssertEqual(trie.pop(model: model, tokens: [1, 2, 3]), 7)
        XCTAssertNil(trie.get(model: model, tokens: [1, 2, 3]))
        // Fully pruned → the model subtree is gone, so search is a clean miss.
        let r = trie.search(model: model, tokens: [1, 2, 3])
        XCTAssertNil(r.exact)
        XCTAssertNil(r.longer)
        XCTAssertEqual(r.commonPrefix, 0)
    }

    func testPopKeepsSharedPrefixEntry() {
        let trie = PromptTrie<Int>()
        trie.add(model: model, tokens: [1, 2], value: 12)
        trie.add(model: model, tokens: [1, 2, 3], value: 123)
        XCTAssertEqual(trie.pop(model: model, tokens: [1, 2, 3]), 123)
        // Popping the longer entry must not disturb the shorter one.
        XCTAssertEqual(trie.get(model: model, tokens: [1, 2]), 12)
        XCTAssertEqual(trie.search(model: model, tokens: [1, 2]).exact, [1, 2])
    }

    func testPopMissReturnsNil() {
        let trie = PromptTrie<Int>()
        trie.add(model: model, tokens: [1, 2], value: 1)
        XCTAssertNil(trie.pop(model: model, tokens: [1, 2, 3]))
        XCTAssertNil(trie.pop(model: "other", tokens: [1]))
    }

    func testPopPrefixesRemovesStrictPrefixesOnly() {
        let trie = PromptTrie<Int>()
        trie.add(model: model, tokens: [1], value: 1)
        trie.add(model: model, tokens: [1, 2], value: 12)
        trie.add(model: model, tokens: [1, 2, 3], value: 123)
        let removed = trie.popPrefixes(model: model, tokens: [1, 2, 3])
        XCTAssertEqual(removed.map { $0.0 }, [1, 2])
        XCTAssertEqual(removed.map { $0.1 }, [1, 12])
        // The full entry survives; the strict prefixes are gone.
        XCTAssertEqual(trie.get(model: model, tokens: [1, 2, 3]), 123)
        XCTAssertNil(trie.get(model: model, tokens: [1]))
        XCTAssertNil(trie.get(model: model, tokens: [1, 2]))
    }

    // MARK: - multi-model isolation

    func testModelsAreIsolated() {
        let trie = PromptTrie<Int>()
        trie.add(model: "A", tokens: [1, 2], value: 1)
        XCTAssertNil(trie.get(model: "B", tokens: [1, 2]))
        let r = trie.search(model: "B", tokens: [1, 2])
        XCTAssertNil(r.exact)
        XCTAssertEqual(r.commonPrefix, 0)
        // Model A still resolves.
        XCTAssertEqual(trie.search(model: "A", tokens: [1, 2]).exact, [1, 2])
    }

    // MARK: - Teardown

    /// A stored sequence is a chain of one node per token, so releasing a trie
    /// used to nest one `deinit` per token: a few hundred levels overflowed the
    /// 512 KiB stack of the cooperative-pool thread the owning actor runs on,
    /// which is how `macmlx serve` died on SIGTERM and the Track G smokes died
    /// at teardown (SIGBUS in `PromptTrie.Node.deinit`). The teardown is now
    /// an explicit worklist. This test releases a 200,000-token chain on a
    /// cooperative thread; the only way it fails is by killing the runner.
    func testDeepChainTearsDownWithoutRecursion() async {
        let depth = 200_000
        let built = await Task.detached(priority: .userInitiated) { () -> Int in
            let trie = PromptTrie<Int>()
            trie.add(model: "M", tokens: Array(0..<depth), value: 1)
            // A side branch off the deep chain, so the worklist has to fan out
            // as well as descend.
            trie.add(model: "M", tokens: Array(0..<(depth / 2)) + [-1, -2, -3], value: 2)
            let found = trie.search(model: "M", tokens: Array(0..<depth)).exact?.count ?? 0
            return found
            // `trie` is released here, on the cooperative thread.
        }.value
        XCTAssertEqual(built, depth)

        // Dropping a whole subtree through `pop` pruning and through replacing
        // the trie (the `clearAll` shape) must be just as safe.
        let replaced = await Task.detached { () -> Bool in
            var trie = PromptTrie<Int>()
            trie.add(model: "M", tokens: Array(0..<depth), value: 1)
            trie = PromptTrie<Int>()
            return trie.search(model: "M", tokens: [0]).commonPrefix == 0
        }.value
        XCTAssertTrue(replaced)
    }
}
