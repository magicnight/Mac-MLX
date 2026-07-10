// Copyright © 2026 macMLX. English comments only.

import XCTest

@testable import MacMLXCore

/// Pure-Swift (MLX-free) tests for ``PromptCacheClassifiedLRU``, the per-type
/// balancing LRU ported from mlx-lm's `LRUPromptCache.CacheOrder`. No Metal
/// required.
final class PromptCacheClassifiedLRUTests: XCTestCase {

    func testEmptyPopReturnsNil() {
        var lru = PromptCacheClassifiedLRU<String>()
        XCTAssertNil(lru.pop())
        XCTAssertEqual(lru.count, 0)
    }

    func testSingleTypeIsPlainFIFO() {
        var lru = PromptCacheClassifiedLRU<String>()
        lru.push("a1", type: .assistant)
        lru.push("a2", type: .assistant)
        lru.push("a3", type: .assistant)
        XCTAssertEqual(lru.count, 3)
        // Oldest first (least-recently-used).
        XCTAssertEqual(lru.pop(), "a1")
        XCTAssertEqual(lru.pop(), "a2")
        XCTAssertEqual(lru.pop(), "a3")
        XCTAssertNil(lru.pop())
    }

    func testRemove() {
        var lru = PromptCacheClassifiedLRU<String>()
        lru.push("a1", type: .assistant)
        lru.push("a2", type: .assistant)
        lru.push("u1", type: .user)
        lru.remove("a1")
        XCTAssertEqual(lru.count, 2)
        XCTAssertEqual(lru.pop(), "a2")
        XCTAssertEqual(lru.pop(), "u1")
    }

    func testRemoveMissingIsNoOp() {
        var lru = PromptCacheClassifiedLRU<String>()
        lru.push("a1", type: .assistant)
        lru.remove("nope")
        XCTAssertEqual(lru.count, 1)
        XCTAssertEqual(lru.pop(), "a1")
    }

    /// With balanced queues, `assistant` is evicted before `system`.
    func testAssistantEvictedBeforeSystemWhenBalanced() {
        var lru = PromptCacheClassifiedLRU<String>()
        lru.push("s1", type: .system)
        lru.push("a1", type: .assistant)
        XCTAssertEqual(lru.pop(), "a1")
        XCTAssertEqual(lru.pop(), "s1")
    }

    /// Full balancing sequence: assistant=[a1], user=[u1,u2], system=[s1].
    /// The algorithm drains the larger, lower-priority queue first while
    /// keeping the classes balanced — locked here token-for-token against the
    /// upstream `CacheOrder.pop` walk.
    func testBalancingEvictionOrder() {
        var lru = PromptCacheClassifiedLRU<String>()
        lru.push("a1", type: .assistant)
        lru.push("u1", type: .user)
        lru.push("u2", type: .user)
        lru.push("s1", type: .system)
        XCTAssertEqual(lru.count, 4)

        // user(2) >= system(1) → u1; then assistant(1) >= user(1) → a1;
        // then user(1) >= system(1) → u2; finally system → s1.
        XCTAssertEqual(lru.pop(), "u1")
        XCTAssertEqual(lru.pop(), "a1")
        XCTAssertEqual(lru.pop(), "u2")
        XCTAssertEqual(lru.pop(), "s1")
        XCTAssertNil(lru.pop())
    }

    func testCountByType() {
        var lru = PromptCacheClassifiedLRU<String>()
        lru.push("a1", type: .assistant)
        lru.push("a2", type: .assistant)
        lru.push("u1", type: .user)
        XCTAssertEqual(lru.count(of: .assistant), 2)
        XCTAssertEqual(lru.count(of: .user), 1)
        XCTAssertEqual(lru.count(of: .system), 0)
    }
}
