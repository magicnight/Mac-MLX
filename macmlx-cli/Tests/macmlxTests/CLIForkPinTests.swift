import Testing
import Foundation

/// `magicnight/mlx-swift` (our controlled fork) and the `ml-explore/mlx-swift`
/// that `mlx-swift-lm` depends on share the SwiftPM package identity
/// `mlx-swift`. When identities collide, the ROOT package's declaration picks
/// the winner — and `macmlx-cli` is its own root whenever the CLI is built
/// (`scripts/package-cli.sh`, and CI's
/// `swift package --package-path macmlx-cli resolve`).
///
/// Declaring the fork only in `MacMLXCore/Package.swift` was therefore not
/// enough: the CLI resolved upstream instead and shipped without the fork's
/// cherry-picks, so the Homebrew tarball ran a different engine than
/// macMLX.app — silently, since both build and run fine.
///
/// Both manifests now pin the same revision. These tests fail if that stops
/// being true, which is the only cheap way to catch the drift: a mismatch
/// produces no build error and no test failure anywhere else, just a CLI
/// quietly running unpatched kernels again.
@Suite("CLI fork pin matches Core")
struct CLIForkPinTests {

    private static let forkURL = "https://github.com/magicnight/mlx-swift.git"

    /// Both `Package.swift` manifests, located relative to this source file so
    /// the test works from any working directory.
    private static func manifest(_ relativePath: String) throws -> String {
        // …/macmlx-cli/Tests/macmlxTests/CLIForkPinTests.swift → repo root
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // macmlxTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // macmlx-cli
            .deletingLastPathComponent()   // repo root
        return try String(contentsOf: repoRoot.appending(path: relativePath), encoding: .utf8)
    }

    /// The revision pinned for `forkURL` in a manifest, or nil when the fork
    /// is not declared at all. Deliberately a plain scan rather than a manifest
    /// parse: the point is to read what a human reading the file would read.
    private static func pinnedRevision(in manifest: String) -> String? {
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

    @Test("the CLI declares the controlled mlx-swift fork itself")
    func cliDeclaresTheFork() throws {
        let cli = try Self.manifest("macmlx-cli/Package.swift")
        #expect(
            Self.pinnedRevision(in: cli) != nil,
            """
            macmlx-cli/Package.swift no longer pins \(Self.forkURL) by revision. \
            Without a root-level declaration the CLI resolves upstream \
            mlx-swift and ships without the fork's cherry-picks.
            """)
    }

    @Test("the CLI and Core pin the same fork revision")
    func cliPinMatchesCorePin() throws {
        let cli = try Self.manifest("macmlx-cli/Package.swift")
        let core = try Self.manifest("MacMLXCore/Package.swift")

        let cliRevision = Self.pinnedRevision(in: cli)
        let coreRevision = Self.pinnedRevision(in: core)

        #expect(coreRevision != nil, "MacMLXCore/Package.swift no longer pins \(Self.forkURL)")
        #expect(
            cliRevision == coreRevision,
            """
            mlx-swift fork revision drifted between manifests — \
            CLI: \(cliRevision ?? "none"), Core: \(coreRevision ?? "none"). \
            The CLI and the app would run different engines.
            """)
    }
}
