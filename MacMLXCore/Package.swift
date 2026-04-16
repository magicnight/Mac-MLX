// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MacMLXCore",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "MacMLXCore", targets: ["MacMLXCore"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "3.31.3"),
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.22.0"),
    ],
    targets: [
        .target(
            name: "MacMLXCore",
            dependencies: [
                // Imports added in Stage 2 — declared above so SPM resolves them now.
                // .product(name: "MLXLLM", package: "mlx-swift-lm"),
                // .product(name: "Hummingbird", package: "hummingbird"),
            ]
        ),
        .testTarget(
            name: "MacMLXCoreTests",
            dependencies: ["MacMLXCore"]
        ),
    ]
)
