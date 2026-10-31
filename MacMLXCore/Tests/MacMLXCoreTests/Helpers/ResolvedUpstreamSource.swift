import Foundation

/// Source files of the mlx-swift-lm checkout SwiftPM resolved for this build,
/// for tests that pin a copy of something upstream keeps private (a registry's
/// keys, a state key's id) against the code that actually runs.
///
/// The checkout sits under `checkouts/` in the scratch directory, found by
/// walking up from the test target's resource bundle, which sits in that
/// directory under any `--scratch-path`. Not from this source file: a tree's
/// own `.build` may hold an older checkout than the one this build resolved.
/// Under xcodebuild there is none (it resolves into DerivedData), so the tests
/// that read it skip there. The spm job, in CI and ci-local, sets
/// `MACMLX_REQUIRE_RESOLVED_SOURCE=1`, which those tests turn into a failure,
/// so a gate cannot pass by skipping.
enum ResolvedUpstreamSource {

    static var isRequired: Bool {
        ProcessInfo.processInfo.environment["MACMLX_REQUIRE_RESOLVED_SOURCE"] == "1"
    }

    /// The text of `path`, relative to the mlx-swift-lm checkout root (e.g.
    /// `Libraries/MLXVLM/VLMModelFactory.swift`), or nil with no checkout.
    static func mlxSwiftLM(_ path: String) -> String? {
        let relative = "checkouts/mlx-swift-lm/\(path)"
        var directory = Bundle.module.bundleURL.deletingLastPathComponent()
        for _ in 0..<8 {
            if let text = try? String(contentsOf: directory.appending(path: relative), encoding: .utf8) {
                return text
            }
            directory.deleteLastPathComponent()
        }
        return nil
    }
}
