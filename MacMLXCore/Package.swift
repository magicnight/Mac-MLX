// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MacMLXCore",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "MacMLXCore", targets: ["MacMLXCore"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "3.31.4"),
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.25.0"),
        .package(url: "https://github.com/kean/Pulse.git", from: "5.2.3"),
        // Use 1.3.x series: avoids 0.1.24's pin on swift-argument-parser 1.4.x
        // which conflicts with our CLI's argparse 1.8.x requirement.
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.3.3"),
        // MCP client pool (v0.5+). Pinned per-minor — SDK is still
        // pre-1.0. macmlx-cli already pulls the same package for the
        // v0.4.0 server-side MCP feature, but Core needs its own
        // declaration so GUI / HummingbirdServer can speak MCP too.
        .package(url: "https://github.com/modelcontextprotocol/swift-sdk.git", from: "0.12.0"),
    ],
    targets: [
        .target(
            name: "MacMLXCore",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "Pulse", package: "Pulse"),
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "MCP", package: "swift-sdk"),
            ]
        ),
        .testTarget(
            name: "MacMLXCoreTests",
            dependencies: ["MacMLXCore"],
            resources: [
                // Numerical-parity fixtures captured from the Python
                // mlx-lm reference (weights + inputs + expected output).
                .copy("Fixtures"),
            ]
        ),
    ]
)
