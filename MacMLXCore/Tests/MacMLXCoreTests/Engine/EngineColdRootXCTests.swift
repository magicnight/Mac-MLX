// Copyright © 2026 macMLX. English comments only.

import Foundation
import XCTest

@testable import MacMLXCore

/// `EngineColdRootTests` under the XCTest runner, where the gated real-model
/// tests run: an engine built as they build one never resolves the user's
/// real `~/.mac-mlx/kv-cache` as its cold-tier root. Checks the resolved
/// path only; nothing is cleared.
final class EngineColdRootXCTests: XCTestCase {

    func testATestEngineNeverUsesTheRealColdRoot() async {
        XCTAssertTrue(TestHost.isActive)
        let root = await MLXSwiftEngine(promptCache: PromptCacheConfig(coldEnabled: false)).promptCacheRoot
        XCTAssertFalse(
            root.standardizedFileURL.path.hasPrefix(DataRoot.macMLX.standardizedFileURL.path),
            "\(root.path) is under the real ~/.mac-mlx")
    }
}
