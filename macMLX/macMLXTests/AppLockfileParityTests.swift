import Foundation
import Testing

/// The app ships the graph its workspace lockfile pins, and until #149 that
/// graph was whatever the build machine resolved on the day: the lockfile was
/// gitignored, so a release archive on a fresh runner took the newest release
/// each manifest allowed. The v0.9.0 DMG shipped swift-jinja 2.4.2 while every
/// chat-template test had run on the 2.3.6 that MacMLXCore's lockfile pinned.
///
/// The lockfile is committed now, and this suite holds it to MacMLXCore's:
/// every package Core pins, the app pins at the same location, version and
/// revision. `scripts/check-lockfile-parity.py` makes the same comparison in
/// CI and the local gate after the app's build, next to the step that proves
/// the build left the file unchanged; this suite catches a drift earlier,
/// inside Xcode, with the files as a human would read them.
@Suite("App lockfile matches Core")
struct AppLockfileParityTests {

    private static let appLockfile = "macMLX/macMLX.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved"
    private static let coreLockfile = "MacMLXCore/Package.resolved"
    private static let forkURL = "https://github.com/magicnight/mlx-swift.git"

    /// Packages only MacMLXCore's tests depend on never reach the app's graph
    /// (SwiftPM ignores a dependency's test-only requirements when another
    /// root resolves), so they are exempt from "present in the app lockfile".
    /// Empty today: every such package also arrives through a library
    /// dependency. Keep it in step with `CORE_TEST_ONLY` in
    /// `scripts/check-lockfile-parity.py`.
    private static let coreTestOnly: Set<String> = []

    private struct Lockfile: Decodable {
        let pins: [Pin]
    }

    private struct Pin: Decodable {
        struct State: Decodable {
            let version: String?
            let revision: String?
        }

        let identity: String
        let location: String
        let state: State
    }

    /// Repo root, located relative to this source file so the test works from
    /// any working directory.
    private static var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // macMLXTests
            .deletingLastPathComponent()   // macMLX
            .deletingLastPathComponent()   // repo root
    }

    private static func pins(_ relativePath: String) throws -> [String: Pin] {
        let data = try Data(contentsOf: repoRoot.appending(path: relativePath))
        let lockfile = try JSONDecoder().decode(Lockfile.self, from: data)
        return Dictionary(lockfile.pins.map { ($0.identity, $0) }, uniquingKeysWith: { first, _ in first })
    }

    /// One spelling for a package URL: lockfiles differ in case and `.git`.
    private static func normalized(_ location: String) -> String {
        var trimmed = location.trimmingCharacters(in: .whitespaces).lowercased()
        while trimmed.hasSuffix("/") { trimmed.removeLast() }
        if trimmed.hasSuffix(".git") { trimmed.removeLast(4) }
        return trimmed
    }

    @Test("every package MacMLXCore pins, the app pins the same way")
    func appPinsMatchCore() throws {
        let core = try Self.pins(Self.coreLockfile)
        let app = try Self.pins(Self.appLockfile)
        #expect(core.count >= 30, "MacMLXCore's lockfile names the whole engine graph")
        for (identity, expected) in core.sorted(by: { $0.key < $1.key }) where !Self.coreTestOnly.contains(identity) {
            let actual = try #require(app[identity], "the app lockfile does not name \(identity), which MacMLXCore pins")
            #expect(
                Self.normalized(actual.location) == Self.normalized(expected.location),
                "\(identity): the app resolved \(actual.location), MacMLXCore \(expected.location)"
            )
            #expect(
                actual.state.version == expected.state.version && actual.state.revision == expected.state.revision,
                "\(identity): the app pins \(actual.state.version ?? "") \(actual.state.revision ?? ""), MacMLXCore pins \(expected.state.version ?? "") \(expected.state.revision ?? "")"
            )
        }
    }

    @Test("the app lockfile names the controlled mlx-swift fork at the revision Core pins")
    func appLockfileNamesTheFork() throws {
        let core = try Self.pins(Self.coreLockfile)
        let app = try Self.pins(Self.appLockfile)
        let corePin = try #require(core["mlx-swift"], "MacMLXCore's lockfile names no mlx-swift")
        let appPin = try #require(app["mlx-swift"], "the app lockfile names no mlx-swift")
        #expect(Self.normalized(corePin.location) == Self.normalized(Self.forkURL), "MacMLXCore resolved \(corePin.location), not the fork")
        #expect(Self.normalized(appPin.location) == Self.normalized(Self.forkURL), "the app resolved \(appPin.location), not the fork")
        #expect(appPin.state.revision == corePin.state.revision, "the app pins the fork at \(appPin.state.revision ?? ""), MacMLXCore at \(corePin.state.revision ?? "")")
    }
}
