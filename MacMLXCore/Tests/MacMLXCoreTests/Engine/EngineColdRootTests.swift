// Copyright © 2026 macMLX. English comments only.

import Foundation
import Testing

@testable import MacMLXCore

/// An engine built in a test, the way the tests build one, never resolves the
/// user's real `~/.mac-mlx/kv-cache` as its cold-tier root: building a store
/// sweeps and prunes its root, and `clearPromptCache()` deletes it. Checks
/// the resolved path only; nothing is cleared. The XCTest runner has its own
/// copy, `EngineColdRootXCTests`.
@Suite("Engine cold root under swift-testing")
struct EngineColdRootTests {

    @Test("A test engine's cold root is a temporary directory of its own")
    func aTestEngineNeverUsesTheRealColdRoot() async {
        #expect(TestHost.isActive)
        let real = DataRoot.macMLX("kv-cache").standardizedFileURL.path
        let first = await MLXSwiftEngine(promptCache: PromptCacheConfig(coldEnabled: false)).promptCacheRoot
        let second = await MLXSwiftEngine(promptCache: PromptCacheConfig(coldEnabled: false)).promptCacheRoot
        for root in [first, second] {
            #expect(!root.standardizedFileURL.path.hasPrefix(real))
            #expect(!root.standardizedFileURL.path.hasPrefix(DataRoot.macMLX.standardizedFileURL.path))
        }
        #expect(first != second, "one directory per engine")
    }

    @Test("A test engine whose cold tier stayed empty removes its directory", .timeLimit(.minutes(10)))
    func anEmptyTestRootIsRemovedWithItsEngine() async throws {
        var engine: MLXSwiftEngine? = MLXSwiftEngine(promptCache: PromptCacheConfig(coldEnabled: false))
        let root = try #require(await engine?.promptCacheRoot)
        #expect(FileManager.default.fileExists(atPath: root.path), "the store creates its root")
        engine = nil
        // Polled, since the last reference may be released a moment after this
        // one and nothing reports the removal. No deadline: a wait on the clock
        // is a wait on the rest of the test process. A directory that is never
        // removed ends at the time limit, which cancels the sleep.
        while FileManager.default.fileExists(atPath: root.path) {
            try await Task.sleep(for: .milliseconds(10))
        }
    }
}
