// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MacMLXCore",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "MacMLXCore", targets: ["MacMLXCore"]),
    ],
    dependencies: [
        // CONTROLLED MINIMAL FORK (master plan §1.1, rebased 2026-10-02):
        // upstream mlx-swift 0.32.3, which vendors core v0.32.2, plus TWO
        // cherry-picks that landed in core v0.32.3 and are not yet in any
        // mlx-swift release. Both produce silently wrong numbers on NAX
        // hardware and both are pinned by QuantizedMatmulParityTests:
        //
        //   - ml-explore/mlx#3922 — the sorted-RHS affine NAX kernel narrowed
        //     its remaining-row count to short before clamping it to the tile,
        //     so past 32768 rows early tiles left output rows unwritten.
        //     Reproduced here: 32 of 32769 rows came back all-zero, exactly
        //     one tile height. One row per (token, expert) pair means an
        //     8K-token prompt through top-4 routing already sits on the seam.
        //   - ml-explore/mlx#4009 — the same kernel got its K tail wrong when
        //     K is not a multiple of the 64-wide tile: the activation tile was
        //     bounded by the full tile rather than the remainder, and the
        //     weight loader zeroed the wrong axis. Upstream measured 92–97% of
        //     output elements wrong at K=160. Only group size 32 admits such a
        //     K, so the window is a `--q-group-size 32` checkpoint whose hidden
        //     or MoE-intermediate size is 32 mod 64; the result reads like a
        //     bad model rather than a bug.
        //
        // The twelve fixes the previous base (0.31.6, core v0.31.1) carried —
        // #3498 #3810 #3497 #3631 #3560 #3632 #3960 #4043 #3361 #4251 #3167
        // #3873 — are all in core v0.32.2, each checked by ancestry and, for
        // the kernel-header ones, line by line in upstream's shipped generated
        // copies. Their parity tests stay as regression guards:
        // NAXSplitKGemmParityTests (#3810), NAXAttentionAndAddMMParityTests
        // (#3361, and the addmm property that #3422 — now in the base — must
        // keep), SteelGemmSafeLoadParityTests (#3560), and
        // QuantizedMatmulParityTests (#3497, #3631, #4251).
        //
        // Two things this base fixed that the old one had and we had not
        // noticed: mlx#4372's second half removed a trailing underscore from
        // the "gather_qmm_t_nax_" kernel-name strings in quantized.cpp, a
        // JIT-only lookup failure on the quantized MoE gather path that had
        // been live since the fork was created (the PR was excluded by its
        // title; the half we had read did not apply, the half we had not did).
        // And mlx#3963 makes the kernels compile under the Metal 4.1 language
        // version that core v0.32.2 requests on macOS 27 — the runtime JIT on
        // the old base capped itself at 4.0 and never asked, which is why
        // nothing broke, and also why a developer machine on Xcode 27 is the
        // first place that path runs; CI and release pin Xcode 26.4.1.
        //
        // The stream model changed with this base. 0.32.3 creates every stream
        // through mlx_stream_new_thread_unsafe and resolves the default from a
        // task-local with a global fallback, so evaluating from a second OS
        // thread no longer aborts (ml-explore/mlx-swift#457, closed upstream).
        // CrossThreadEvalTripwireTests holds that line. BatchPositionedCacheWrapper,
        // the shim over the batched single-token RoPE defect fixed by #3498,
        // was deleted with this move as its own tests had scheduled.
        //
        // The fork carries no API changes; its .gitmodules points the mlx
        // submodule at the magicnight mirror so the carried commits resolve.
        // Drop this override and return to the upstream package as soon as an
        // mlx-swift release vendors core >= v0.32.3. Pinned by revision so it
        // can never drift; CLIForkPinTests, AppForkPinTests and the CI step
        // that reads the app's resolved graph keep the three roots in step.
        .package(
            url: "https://github.com/magicnight/mlx-swift.git",
            revision: "1026d239d831ac7ddd154c970bb6c1bff08c0e0c"),
        // Minor-pinned: a minor bump of mlx-swift-lm raises its mlx-swift floor,
        // which the fork's revision pin can satisfy at resolution but not at the
        // source level. Move both together, never one.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", .upToNextMinor(from: "3.32.3")),
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.25.0"),
        .package(url: "https://github.com/kean/Pulse.git", from: "5.2.3"),
        // Use 1.3.x series: avoids 0.1.24's pin on swift-argument-parser 1.4.x
        // which conflicts with our CLI's argparse 1.8.x requirement.
        // 1.3.4 is the floor on purpose: its own manifest requires swift-jinja
        // >= 2.4.2, and Transformers is linked into the MacMLXCore LIBRARY, so
        // that requirement reaches every root that depends on this package —
        // the CLI and the app included. The swift-jinja declaration further
        // down does NOT do that job (see its comment).
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.3.4"),
        // MCP client pool (v0.5+). Pinned per-minor — SDK is still
        // pre-1.0. macmlx-cli already pulls the same package for the
        // v0.4.0 server-side MCP feature, but Core needs its own
        // declaration so GUI / HummingbirdServer can speak MCP too.
        .package(url: "https://github.com/modelcontextprotocol/swift-sdk.git", from: "0.12.0"),
        // swift-jinja is what swift-transformers renders chat templates with.
        // The parser/filter fixes we reported upstream — huggingface/swift-jinja
        // #62 (integer-keyed object literals, Seed-OSS), #63 (literal `}}`,
        // Command R7B) and #64 (`strip(arg)` argument handling, Hunyuan
        // `<answer>`) — are in 2.4.0; 2.4.0 also changed `Value.object`'s keys
        // and broke swift-transformers, and 2.4.1 restored String-keyed source
        // compatibility, and 2.4.2 is the floor swift-transformers 1.3.4
        // declares. Every checkpoint chat template now renders natively, with
        // no built-in override.
        //
        // This declaration links Jinja into the render-parity TEST target only,
        // so the tests render through the exact engine production uses. It is
        // NOT what guarantees the version downstream: SwiftPM ignores a
        // dependency package's test-only requirements when another root (the
        // CLI, the app) resolves its graph. The library-level guarantee is the
        // swift-transformers 1.3.4 floor above, whose manifest requires
        // swift-jinja >= 2.4.2. Keep the two in step.
        .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.4.2"),
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
        // mlx-audio-swift v0.1.3 plus two compatibility commits, carried on a
        // fork because the package does not compile against mlx-swift-lm 3.32.3
        // as released: that version added a complete `compile()` overload
        // matrix to MLXLMCommon with `@Sendable` bodies (#589), which hijacks
        // Parakeet's two bare `compile` calls, and made `newCache(parameters:)`
        // throwing, which Marvis's non-throwing cache resets did not expect.
        // Both patches are one-line qualifications; nothing else differs from
        // the upstream tag. Drop the fork once upstream tags a release that
        // builds against 3.32.x. Pinned by revision so it can never drift.
        .package(
            url: "https://github.com/magicnight/mlx-audio-swift.git",
            revision: "d13853250a7e4eda66fe686d23f0d5cfd3cc86da"),
        // Already resolved transitively (mlx-audio-swift pins
        // `.upToNextMajor(from: "0.8.1")`, currently 0.9.0). Declared directly
        // with the SAME requirement — so no new version is introduced — purely
        // because `AudioEngine` has to name `HubCache` to redirect audio model
        // downloads into `~/.mac-mlx/`, and Swift does not re-export it through
        // MLXAudioSTT/TTS.
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
