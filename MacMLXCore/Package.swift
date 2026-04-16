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
        .package(url: "https://github.com/kean/Pulse.git", from: "5.0.0"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "0.1.0"),
    ],
    targets: [
        .target(
            name: "MacMLXCore",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Transformers", package: "swift-transformers"),
                // .product(name: "Hummingbird", package: "hummingbird"),  // TODO: v0.2
            ]
        ),
        .testTarget(
            name: "MacMLXCoreTests",
            dependencies: ["MacMLXCore"]
        ),
    ]
)
