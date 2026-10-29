// Copyright © 2026 macMLX. English comments only.

import Foundation

/// Whether this process is a test runner: an XCTest bundle (under xcodebuild,
/// `swift test`'s `xctest`, or hosted inside the app) or swift-testing's
/// `swift test` helper. Code that would touch the user's real data under
/// `~/.mac-mlx` by default asks this to pick a throwaway location instead,
/// so a test cannot reach that data unless it passes the real location
/// explicitly. Always false in the app and the CLI.
enum TestHost {
    static let isActive: Bool = {
        let process = ProcessInfo.processInfo
        return NSClassFromString("XCTestCase") != nil
            || process.environment["XCTestConfigurationFilePath"] != nil
            || process.environment["XCTestSessionIdentifier"] != nil
            || ["xctest", "swiftpm-testing-helper"].contains(process.processName)
    }()
}
