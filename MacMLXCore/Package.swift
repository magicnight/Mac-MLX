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
        // upstream mlx-swift 0.32.3, which vendors core v0.32.2, plus FIVE
        // upstream cherry-picks that no mlx-swift release carries. Each
        // produces silently wrong numbers and each was reproduced here
        // against the base before it was carried. The first two landed in
        // core v0.32.3, are NAX-only, and are pinned by
        // QuantizedMatmulParityTests:
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
        // Three more were added on 2026-10-29, none of them hardware-gated:
        //
        //   - ml-explore/mlx#4660 (core v0.32.4) — the pad-for-conv paths in
        //     conv.cpp freed the zero scalar they pad channels with before the
        //     GPU read it, so a scalar allocated in that window became the
        //     padding. Reachable through the Conv3d(3 → hidden) patch embeds of
        //     Qwen2/2.5/3-VL and GLM-OCR. Reproduced here: a 3-channel conv3d
        //     of ones under asyncEval returned 128 instead of 24 in 297 of 300
        //     runs. ConvChannelPaddingRaceTests.
        //   - ml-explore/mlx#4431 (core v0.32.3) — sdpa_vector_2pass_1_gqa read
        //     batch 0's keys and values for every batch element (one query, no
        //     mask, GQA factor 8, head dim 64 or 128, >= 8192 keys, B > 1).
        //     Reproduced here: row 1 off by 0.61. No production path reaches
        //     it today, because batched decode always passes an array mask.
        //     SDPAVectorGQABatchParityTests.
        //   - ml-explore/mlx#4775 (mlx main, in no release yet) — GPU affine
        //     quantize started its maximum at 0, so an all-negative group was
        //     quantized over [min, 0]. Reachable through the quantized KV cache
        //     (kv_bits). Reproduced here: every group size and bit width
        //     restored a unit-spaced negative grid with max |diff| 1.0, the
        //     CPU stream with 0. AffineQuantizeSignedGroupTests.
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
        // mlx-swift release vendors a core that contains all five; #4775 is
        // after core v0.32.4, so v0.32.4 is not enough. Pinned by revision so it
        // can never drift; CLIForkPinTests, AppForkPinTests and the CI step
        // that reads the app's resolved graph keep the three roots in step.
        .package(
            url: "https://github.com/magicnight/mlx-swift.git",
            revision: "420dd0552f7abf383a60977c9be040c6a0216163"),
        // Minor-pinned: a minor bump of mlx-swift-lm raises its mlx-swift floor,
        // which the fork's revision pin can satisfy at resolution but not at the
        // source level. Move both together, never one.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", .upToNextMinor(from: "3.32.3")),
        // 2.26.0 fixed three denial-of-service advisories (GHSA-wqpw-qq6j-p4x2,
        // GHSA-62cf-93wq-244m, GHSA-97r6-mq85-c3rv); 2.28.0 also raises its
        // swift-nio-ssl and swift-nio-http2 floors past their July advisories.
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.28.0"),
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
        // mlx-audio-swift v0.1.5 plus the commits on the fork's branch
        // `v0.1.5-macmlx`, all of them about where audio models land, where
        // their requests go and what the Hub's listing certifies (#135, #149):
        // upstream's pending PR Blaizzy/mlx-audio-swift#257, so a cached
        // snapshot counts as complete only when it holds the files a loader's
        // patterns name, reworked so the Hub's listing is what certifies a
        // pattern and a load fetches, one by one, only the listed files not
        // yet on disk, refusing a listed path it cannot write under the model
        // directory; and the caller's `HubCache` passed through the six STT
        // loaders and the TTS loaders that fetched without it: Chatterbox and
        // OmniVoice (their whole model, Chatterbox's S3 tokenizer with it),
        // the Llama and Qwen3 voices and Echo (their codecs); the token
        // clients follow `HF_ENDPOINT` like the plain one (the cache fixes are
        // reported upstream in Blaizzy/mlx-audio-swift#279 and on #257); and
        // the listing is read to its last page through `listAllTree`
        // (swift-huggingface 0.11.0 and later), where `listFiles` returned the
        // first; the snapshot download's own listing is still the library's
        // single request, so a fresh fetch of a repository past one page
        // brings the first page's files, the next resolve completes what a
        // loader's own patterns name, and files only the default patterns
        // name, weights included, do not come from a later page. Nothing
        // else differs from the upstream tag; the two small compatibility
        // commits the v0.1.3 base needed for mlx-swift-lm 3.32.3 are
        // upstream's own since v0.1.5. The previous pin is tagged
        // `v0.1.3-macmlx-1` on the fork.
        // Drop the fork once a release passes the cache through, completes a
        // cached snapshot by the Hub's listing rather than by what a load
        // asked for, reads that listing past its first page, and sends its
        // token clients to `HF_ENDPOINT` too. Pinned by revision so it can
        // never drift.
        .package(
            url: "https://github.com/magicnight/mlx-audio-swift.git",
            revision: "504b338c8f8e7f2bebd1e7eb973a5cc64630dcae"),
        // Already resolved transitively (the mlx-audio-swift fork requires
        // `.upToNextMajor(from: "0.11.0")`). Declared directly with the SAME
        // requirement — so no new version is introduced — purely because
        // `AudioEngine` has to name `HubCache` to redirect audio model
        // downloads into `~/.mac-mlx/`, and Swift does not re-export it through
        // MLXAudioSTT/TTS. The floor is the fork's: its completeness listing
        // reads every page of a repository tree through `listAllTree`, which
        // 0.11.0 added. Every root's lockfile holds the package at 0.13.0
        // (#149); 1.0.0 is a breaking release (`HTTPClientError`, `Provider`,
        // `Capability` and `Bucket.Region` become structs, swift-xet 1.0 is
        // required) and needs its own pass.
        .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.11.0"),
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
                .product(name: "MLXRerankers", package: "mlx-swift-lm"),
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
