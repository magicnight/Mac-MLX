// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MacMLXCore",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "MacMLXCore", targets: ["MacMLXCore"]),
    ],
    dependencies: [
        // CONTROLLED MINIMAL FORK (master plan §1.1 as revised 2026-07-10):
        // upstream mlx-swift 0.31.6 plus ONE cherry-pick — ml-explore/mlx#3498
        // (batched single-token RoPE fix, in mlx-core 0.32.0 but not yet
        // vendored by any mlx-swift release; see ml-explore/mlx-swift#441).
        // The fork carries no API changes. Drop this override and return to
        // the upstream package as soon as mlx-swift vendors core >= 0.32
        // (the inverted tripwire in BatchPositionedCacheWrapperTests guards
        // the switch-back). Pinned by revision so it can never drift.
        .package(
            url: "https://github.com/magicnight/mlx-swift.git",
            revision: "283e9917e209075390b449594a3520cb5ec1907f"),
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
        // swift-jinja is resolved transitively (swift-transformers pins
        // `from: "2.0.0"`). Declared directly with a RAISED floor of 2.4.0 to
        // lock in the parser/filter fixes we upstreamed and that these tests
        // depend on — huggingface/swift-jinja #62 (integer-keyed object
        // literals, Seed-OSS), #63 (literal `}}`, Command R7B), and #64
        // (`strip(arg)` argument handling, Hunyuan `<answer>`) — so every
        // checkpoint chat template renders natively with no built-in override.
        // Also linked into the render-parity TEST target so it can render the
        // checkpoint templates through the exact engine production uses. Not
        // linked into the MacMLXCore library itself.
        .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.4.0"),
    ],
    targets: [
        .target(
            name: "MacMLXCore",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "Pulse", package: "Pulse"),
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "MCP", package: "swift-sdk"),
            ]
        ),
        .testTarget(
            name: "MacMLXCoreTests",
            dependencies: [
                "MacMLXCore",
                // Render checkpoint chat templates through swift-jinja (the same
                // engine swift-transformers uses) to prove, ungated, that they
                // match the Python reference render. See the
                // *ChatTemplateParityTests.
                .product(name: "Jinja", package: "swift-jinja"),
            ],
            resources: [
                // Numerical-parity fixtures captured from the Python
                // mlx-lm reference (weights + inputs + expected output).
                .copy("Fixtures"),
            ]
        ),
    ]
)
