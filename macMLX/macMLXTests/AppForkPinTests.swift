import Testing
import Foundation

/// The app has its own version of the identity collision `CLIForkPinTests`
/// describes, and it bit harder.
///
/// `magicnight/mlx-swift` (our controlled fork) and the `ml-explore/mlx-swift`
/// that `mlx-swift-lm` depends on share the SwiftPM package identity
/// `mlx-swift`, and the ROOT package's declaration picks the winner. For the
/// app the root is `macMLX.xcodeproj`, which declared only Sparkle — so the
/// fork, declared one level down in `MacMLXCore/Package.swift`, lost every
/// time. Verified physically: every `macMLX-*` DerivedData on this machine
/// from 2026-07-06 onward had checked out upstream 0.31.6, meaning the shipped
/// app never ran a single one of the fork's cherry-picks.
///
/// The project now declares the fork itself, by revision. These tests fail if
/// that declaration is dropped or drifts from Core's, which is the only cheap
/// way to catch it: resolving upstream instead produces no build error and no
/// other test failure, just an app quietly running unpatched kernels.
///
/// A declaration is necessary but not sufficient — what actually ships is what
/// SwiftPM resolved. CI asserts the resolved graph separately, after the app's
/// package resolution, because only that catches a resolution that ignores the
/// declaration.
@Suite("App fork pin matches Core")
struct AppForkPinTests {

    private static let forkURL = "https://github.com/magicnight/mlx-swift.git"

    /// Repo root, located relative to this source file so the test works from
    /// any working directory.
    private static var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // macMLXTests
            .deletingLastPathComponent()   // macMLX
            .deletingLastPathComponent()   // repo root
    }

    private static func contents(_ relativePath: String) throws -> String {
        try String(contentsOf: repoRoot.appending(path: relativePath), encoding: .utf8)
    }

    /// The revision pinned for `forkURL` in `MacMLXCore/Package.swift`.
    /// Deliberately a plain scan rather than a manifest parse: the point is to
    /// read what a human reading the file would read.
    private static func corePinnedRevision(in manifest: String) -> String? {
        guard let forkRange = manifest.range(of: forkURL) else { return nil }
        let after = manifest[forkRange.upperBound...]
        guard let revisionKeyword = after.range(of: "revision:") else { return nil }
        let afterKeyword = after[revisionKeyword.upperBound...]
        guard let openQuote = afterKeyword.firstIndex(of: "\""),
              let closeQuote = afterKeyword[afterKeyword.index(after: openQuote)...]
                .firstIndex(of: "\"")
        else { return nil }
        return String(afterKeyword[afterKeyword.index(after: openQuote)..<closeQuote])
    }

    /// The revision pinned for `forkURL` in `project.pbxproj`, which spells the
    /// same thing as an unquoted `revision = <sha>;` inside the package
    /// reference's `requirement` block.
    private static func projectPinnedRevision(in project: String) -> String? {
        guard let forkRange = project.range(of: forkURL) else { return nil }
        let after = project[forkRange.upperBound...]
        guard let revisionKeyword = after.range(of: "revision = ") else { return nil }
        let value = after[revisionKeyword.upperBound...]
            .prefix { $0.isHexDigit }
        return value.isEmpty ? nil : String(value)
    }

    @Test("the Xcode project declares the controlled mlx-swift fork itself")
    func projectDeclaresTheFork() throws {
        let project = try Self.contents("macMLX/macMLX.xcodeproj/project.pbxproj")
        #expect(
            Self.projectPinnedRevision(in: project) != nil,
            """
            macMLX.xcodeproj no longer pins \(Self.forkURL) by revision. \
            Without a project-level declaration the app resolves upstream \
            mlx-swift and ships without the fork's cherry-picks.
            """)
    }

    @Test("the app and Core pin the same fork revision")
    func appPinMatchesCorePin() throws {
        let project = try Self.contents("macMLX/macMLX.xcodeproj/project.pbxproj")
        let core = try Self.contents("MacMLXCore/Package.swift")

        let appRevision = Self.projectPinnedRevision(in: project)
        let coreRevision = Self.corePinnedRevision(in: core)

        #expect(coreRevision != nil, "MacMLXCore/Package.swift no longer pins \(Self.forkURL)")
        #expect(
            appRevision == coreRevision,
            """
            mlx-swift fork revision drifted — app: \(appRevision ?? "none"), \
            Core: \(coreRevision ?? "none"). The app would run a different \
            engine than MacMLXCore's own tests exercise.
            """)
    }
}
