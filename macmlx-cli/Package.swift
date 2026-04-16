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
        .package(url: "https://github.com/rensbreur/SwiftTUI.git", revision: "537133031bc2b2731048d00748c69700e1b48185"),
    ],
    targets: [
        .executableTarget(
            name: "macmlx",
            dependencies: [
                "MacMLXCore",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "SwiftTUI", package: "SwiftTUI"),
            ]
        ),
        .testTarget(
            name: "macmlxTests",
            dependencies: ["macmlx"]
        ),
    ]
)
