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
        // upstream mlx-swift 0.31.6 plus NINE cherry-picks, all already in
        // mlx-core but not yet vendored by any mlx-swift release (see
        // ml-explore/mlx-swift#441). All but #4043 and #3632 produce silently
        // wrong numbers rather than an error, which is why the ones we could
        // reproduce here carry a parity test rather than being left to surface
        // on their own:
        //
        //   - ml-explore/mlx#3498 — batched single-token RoPE fix.
        //   - ml-explore/mlx#4043 — the Metal kernel cache looked itself up
        //     with operator[] while holding only a shared lock, so a lookup
        //     miss inserted into the map while other threads read it. We serve
        //     concurrent requests, so two of them compiling different kernels
        //     at once is ordinary traffic, not a corner case.
        //   - ml-explore/mlx#3810 — NAX split-K GEMM instantiated the JIT
        //     kernel with the output dtype instead of the input dtype, so a
        //     bf16 matmul read its inputs as float: out-of-bounds loads and
        //     silent garbage, no crash and no NaN guard. Reachable on NAX
        //     hardware for a non-float32 matmul with batch 1, M*N >= 2048^2,
        //     K >= 10240 and K >= 3*max(M, N). Pinned by
        //     NAXSplitKGemmParityTests.
        //   - ml-explore/mlx#3497 — qvm mishandles a stride-0 batch dimension,
        //     the shape GQA produces when a quantized KV cache is broadcast
        //     across repeat groups. Only the non-transposed product (scores @
        //     values) reaches it, and only for M >= 2, which is why ordinary
        //     single-token decode never showed it and speculative decoding
        //     does. Not hardware-gated.
        //   - ml-explore/mlx#3631 — NAX qmm edge-tile bounds computed in
        //     short(), wrapping past 32767, so an unaligned N above 2^15
        //     leaves a column band corrupted. Reached through an lm_head whose
        //     vocab is unaligned — MiniCPM3-4B's 73448 is one, and one we
        //     list as supported.
        //   - ml-explore/mlx#3560 — steel GEMM safe load read past the edge of
        //     an unaligned tile. Carried preventively.
        //   - ml-explore/mlx#3361 — the NAX attention kernel held its tile
        //     positions in short, so a KV sequence past 32767 wrapped negative
        //     and the mask was read from the wrong offsets. Reproduced here on
        //     an M5 Max: cosine 0.0 and non-finite output at KV=40960, correct
        //     at KV=8192. This is the flagship long-context path — the tiered
        //     SSD KV cache exists precisely to reach these lengths. Pinned by
        //     NAXAttentionAndAddMMParityTests.
        //   - ml-explore/mlx#3960 — sorted gather_mm derived the activation row
        //     stride from a dimension that can be a singleton, which is the
        //     shape MoE expert dispatch produces during single-token decode.
        //   - ml-explore/mlx#3632 — gather_qmm built a NAX kernel name the
        //     library does not export, so quantized MoE on NAX hardware fails
        //     to load its kernel. Loud rather than silent, unlike the rest.
        //
        // #3497 and #3631 are pinned by QuantizedMatmulParityTests, #3560 by
        // SteelGemmSafeLoadParityTests.
        //
        // Deliberately NOT carried: ml-explore/mlx#3422 ("AddMM was completely
        // broken on NAX"). That defect arrived with upstream's NAXFrag refactor
        // after v0.31.1 — the epilogue bound its destination fragment by value
        // and dropped the bias. Our vendored tree predates the refactor and
        // binds a reference, so it never had the bug; the rest of that commit
        // is split-K tuning against a code shape we do not have. The addmm case
        // in NAXAttentionAndAddMMParityTests holds that property.
        //
        // The fork carries no API changes. Drop this override and return to
        // the upstream package as soon as mlx-swift vendors core >= 0.32
        // (the inverted tripwire in BatchPositionedCacheWrapperTests guards
        // the switch-back). Pinned by revision so it can never drift.
        .package(
            url: "https://github.com/magicnight/mlx-swift.git",
            revision: "049bb52f24fed0a0b26b183edbd0de78fcf9da3e"),
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
        // swift-jinja is already resolved transitively (swift-transformers pins
        // `from: "2.0.0"`, currently 2.3.6). Declared directly with the SAME
        // requirement — so no new version is introduced — purely so the
        // chat-template render-parity TEST target can render the Seed-OSS
        // override through the exact engine production uses. Not linked into the
        // MacMLXCore library itself.
        .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.0.0"),
        // Audio (v0.9 W1a): MIT, Swift-native STT (Whisper/Parakeet family) and
        // TTS (Kokoro family) on top of MLX. Adds NO new transitive package —
        // it only depends on mlx-swift / mlx-swift-lm / swift-transformers /
        // swift-huggingface, all already in this graph. Its own mlx-swift
        // requirement (`.upToNextMajor(from: "0.30.6")`) is LOOSER than ours, so
        // the controlled fork revision pin above still wins.
        //
        // NOTE: the package declares swift-tools-version 6.2 while this manifest
        // is 6.0. That is legal — a dependency may use a newer tools version than
        // its consumer; only the toolchain in use has to be new enough to parse it.
        .package(url: "https://github.com/Blaizzy/mlx-audio-swift.git", from: "0.1.3"),
        // Already resolved transitively (mlx-audio-swift pins
        // `.upToNextMajor(from: "0.8.1")`, currently 0.9.0). Declared directly
        // with the SAME requirement — so no new version is introduced — purely
        // because `AudioEngine` has to name `HubCache` to redirect audio model
        // downloads into `~/.mac-mlx/`, and Swift does not re-export it through
        // MLXAudioSTT/TTS. Same pattern as the swift-jinja declaration above.
        .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.8.1"),
    ],
    targets: [
        // Runtime bridge to Apple's private IOReport framework (silicon metrics,
        // v0.7 W1). Pure C, no link-time dependency on the private framework — it
        // dlopen/dlsym-resolves the symbols so consumers need no linker flags and the
        // app degrades gracefully if IOReport ever disappears. See the target header.
        .target(
            name: "CIOReport"
        ),
        .target(
            name: "MacMLXCore",
            dependencies: [
                "CIOReport",
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "Pulse", package: "Pulse"),
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "MCP", package: "swift-sdk"),
                // Audio (v0.9 W1a) — see the package note above.
                .product(name: "MLXAudioCore", package: "mlx-audio-swift"),
                .product(name: "MLXAudioSTT", package: "mlx-audio-swift"),
                .product(name: "MLXAudioTTS", package: "mlx-audio-swift"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ]
        ),
        .testTarget(
            name: "MacMLXCoreTests",
            dependencies: [
                "MacMLXCore",
                // Render the Seed-OSS chat-template override through swift-jinja
                // (the same engine swift-transformers uses) to prove, ungated,
                // that it matches the Python reference render of the ORIGINAL
                // template. See SeedOssChatTemplateParityTests.
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
