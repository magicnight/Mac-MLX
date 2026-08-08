// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "macmlx",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "macmlx", targets: ["macmlx"]),
    ],
    dependencies: [
        .package(path: "../MacMLXCore"),
        // Repeat MacMLXCore's controlled mlx-swift fork pin — see the comment
        // on the matching declaration in ../MacMLXCore/Package.swift for what
        // the fork carries and when to drop it.
        //
        // This has to be declared HERE, not only in MacMLXCore, because
        // `magicnight/mlx-swift` and the `ml-explore/mlx-swift` that
        // mlx-swift-lm depends on share the SwiftPM identity `mlx-swift`.
        // When that identity collides, the ROOT package's declaration decides
        // the winner — and `macmlx-cli` is its own root whenever the CLI is
        // built (`scripts/package-cli.sh`, and CI's
        // `swift package --package-path macmlx-cli resolve`). Without this the
        // CLI silently resolved upstream and shipped WITHOUT the fork's fixes,
        // so the tarball ran a different engine than macMLX.app. Keep the
        // revision identical to MacMLXCore's; `CLIForkPinTests` fails if the
        // two ever drift.
        .package(
            url: "https://github.com/magicnight/mlx-swift.git",
            revision: "4a9db6cee379727898c538a376c16ff3b147d7d2"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.8.2"),
        // MCP server MVP (v0.4.0). Pin per-minor — SDK is still pre-1.0.
        .package(url: "https://github.com/modelcontextprotocol/swift-sdk.git", from: "0.12.1"),
        // SwiftTUI removed in v0.3.5 — upstream (rensbreur/SwiftTUI) has
        // been unmaintained for over a year and its nonisolated `View`
        // protocol is incompatible with Swift 6 strict concurrency. The
        // three TUI dashboards now render with our own ANSI helper
        // (`CLITerm.swift`). If SwiftTUI catches up to Swift 6 and
        // matches SwiftUI's pace, consider reintroducing it for richer
        // dashboards (tracked as historical note in
        // .claude/features/cli-tui.md).
    ],
    targets: [
        .executableTarget(
            name: "macmlx",
            dependencies: [
                "MacMLXCore",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "MCP", package: "swift-sdk"),
            ]
        ),
        .testTarget(
            name: "macmlxTests",
            dependencies: ["macmlx"]
        ),
    ]
)
