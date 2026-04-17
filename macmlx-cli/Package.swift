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
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.7.1"),
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
            ]
        ),
        .testTarget(
            name: "macmlxTests",
            dependencies: ["macmlx"]
        ),
    ]
)
