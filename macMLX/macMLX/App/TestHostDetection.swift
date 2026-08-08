// TestHostDetection.swift
// macMLX
//
// Detects that this process was launched as the XCTest host for the
// `macMLXTests` bundle rather than by a user.
//
// The app is the unit tests' TEST_HOST, so `xcodebuild test` launches the real
// app before injecting the test bundle. Two of its launch behaviours are
// hostile to that and are skipped while hosting tests:
//
//   * the single-instance check in `AppDelegate` terminates this process when
//     another macMLX is already running — which kills the test run outright on
//     any machine where the developer has the app open;
//   * `AppState.bootstrap()` runs against the user's REAL settings: it can
//     bind the HTTP server port and re-load their last model (gigabytes of
//     weights) as a side effect of running unit tests.
//
// Compiled out of Release entirely, so the shipping app is byte-for-byte what
// it was before this seam existed.

#if DEBUG
import Foundation

enum TestHostDetection {

    /// `true` when an XCTest bundle is being hosted by this process.
    ///
    /// Both signals are checked because which one is present depends on how
    /// the run was started: the injection dylib links XCTest before `main`, so
    /// `XCTestCase` resolves; `xcodebuild` additionally exports the test
    /// configuration path. Either alone is conclusive.
    static var isHostingTests: Bool {
        if NSClassFromString("XCTestCase") != nil { return true }
        let environment = ProcessInfo.processInfo.environment
        return environment["XCTestConfigurationFilePath"] != nil
            || environment["XCTestBundlePath"] != nil
    }
}
#endif
