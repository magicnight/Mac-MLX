import Foundation
import XCTest

/// Anchor for locating the host (`.xctest`) bundle via `Bundle(for:)`.
private final class MLXTestSupportBundleToken {}

/// True when mlx-swift's Metal library (`default.metallib`) is reachable, so
/// real `MLXArray` creation/evaluation can run without aborting the process.
///
/// mlx-swift ships its Metal kernels in a nested `mlx-swift_Cmlx.bundle`
/// resource bundle. `xcodebuild` embeds it under the `.xctest` bundle's
/// `Resources`; bare `swift test` binaries frequently omit it — and MLX
/// `fatalError`s on first use, which cannot be caught. Swift-testing suites
/// gate with `.enabled(if: mlxMetallibIsAvailable, …)`; XCTest suites use
/// `requireMLXRuntimeOrSkip()` below.
let mlxMetallibIsAvailable: Bool = {
    func hasMetallib(_ bundle: Bundle?) -> Bool {
        bundle?.url(forResource: "default", withExtension: "metallib") != nil
    }
    let host = Bundle(for: MLXTestSupportBundleToken.self)
    if let nested = host.url(forResource: "mlx-swift_Cmlx", withExtension: "bundle"),
        hasMetallib(Bundle(url: nested))
    {
        return true
    }
    if hasMetallib(Bundle(identifier: "mlx-swift_Cmlx.resources")) { return true }
    return Bundle.allBundles.contains { $0.bundlePath.contains("Cmlx") && hasMetallib($0) }
}()

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

    /// Gate for **strict numeric-parity** tests (`allClose` at 1e-4 against
    /// Python-captured fixtures): additionally skip when Metal is present but
    /// NOT trustworthy for tight tolerances.
    ///
    /// GitHub-hosted macOS runners are VMs with *paravirtualized* Metal —
    /// matmul/softmax accumulation there diverges from real Apple Silicon
    /// beyond 1e-4 (observed: attention prefill, decoder layer, and
    /// full-model parity fail on `macos-26` runners while passing on real
    /// hardware). The CI Metal job sets `MACMLX_UNTRUSTED_METAL=1`
    /// (via the `TEST_RUNNER_` prefix); local xcodebuild runs — and any
    /// future self-hosted Apple Silicon runner — leave it unset, so parity
    /// stays enforced wherever the numbers are meaningful. Behavioral MLX
    /// tests (exact ops: top-k selection, sanitize shape transforms, cache
    /// round-trips) keep the plain gate and still run on CI Metal.
    func requireTrustworthyMetalOrSkip(
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        try requireMLXRuntimeOrSkip(file: file, line: line)
        if ProcessInfo.processInfo.environment["MACMLX_UNTRUSTED_METAL"] == "1" {
            throw XCTSkip(
                "Strict numeric parity skipped on untrusted (paravirtualized) Metal",
                file: file, line: line)
        }
    }
}
