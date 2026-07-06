import XCTest

extension XCTestCase {
    /// Skip an MLX-backed test when it can't reach the Metal backend.
    ///
    /// MLX runs fine under **xcodebuild** (which builds/JITs the Metal
    /// shaders — requires the Metal Toolchain component installed), but
    /// under bare **`swift test`** the SPM test binary has no metallib
    /// and MLX aborts the process with a `fatalError` on the first op.
    /// A `fatalError` can't be caught, so we detect the SPM case *before*
    /// touching MLX and skip.
    ///
    /// Discriminator: the SPM test bundle lives under `.build/`
    /// (`…/.build/arm64-apple-macosx/debug/…PackageTests.xctest`),
    /// whereas the xcodebuild bundle lives under `DerivedData/`.
    ///
    /// CI: the "Xcode App Build" job could grow an
    /// `xcodebuild test -scheme MacMLXCore` step to actually run these;
    /// today they run locally via xcodebuild and skip in the SPM job.
    func requireMLXRuntimeOrSkip(
        _ message: String = "MLX Metal backend unavailable under `swift test` — run via xcodebuild",
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let underSPM = Bundle(for: Self.self).bundlePath.contains("/.build/")
        if underSPM {
            throw XCTSkip(message, file: file, line: line)
        }
    }
}
