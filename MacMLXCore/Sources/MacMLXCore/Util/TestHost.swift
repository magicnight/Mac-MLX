// Copyright © 2026 macMLX. English comments only.

import Foundation

/// Whether this process is a test runner: an XCTest bundle (under xcodebuild,
/// `swift test`'s `xctest`, or hosted inside the app) or swift-testing's
/// `swift test` helper. Code that would touch the user's real data under
/// `~/.mac-mlx` by default asks this to pick a throwaway location instead,
/// so a test cannot reach that data unless it passes the real location
/// explicitly.
///
/// Detected in Debug builds only, the configuration both `swift test` and
/// `xcodebuild test` build. A Release build always answers false, so a
/// shipped app or CLI can never mistake itself for a test runner and stop
/// persisting the user's data. The flip side: tests run in Release
/// (`swift test -c release`, `xcodebuild -configuration Release test`) get
/// no isolation and would use the real `~/.mac-mlx` again, so don't run the
/// suite that way.
enum TestHost {
    #if DEBUG
    static let isActive: Bool = {
        let process = ProcessInfo.processInfo
        return NSClassFromString("XCTestCase") != nil
            || process.environment["XCTestConfigurationFilePath"] != nil
            || process.environment["XCTestSessionIdentifier"] != nil
            || ["xctest", "swiftpm-testing-helper"].contains(process.processName)
    }()
    #else
    static let isActive = false
    #endif
}
