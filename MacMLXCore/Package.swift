// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MacMLXCore",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "MacMLXCore", targets: ["MacMLXCore"]),
    ],
    dependencies: [
        // CONTROLLED MINIMAL FORK (master plan §1.1, rebased 2026-11-21):
        // upstream mlx-swift 0.33.0, which vendors core v0.32.4, plus ONE
        // upstream cherry-pick that no mlx-swift release carries:
        //
        //   - ml-explore/mlx#4775 (mlx main, after core v0.32.4) — GPU affine
        //     quantize started its maximum at 0, so an all-negative group was
        //     quantized over [min, 0]. Reachable through the quantized KV cache
        //     (kv_bits). Reproduced here against upstream 0.33.0: every group
        //     size and bit width restored a unit-spaced negative grid with max
        //     |diff| 1.0, the CPU stream with 0. AffineQuantizeSignedGroupTests.
        //     The fix is in a kernel header, so the fork changes the generated
        //     copies under Source/Cmlx/mlx-generated as well as the submodule
        //     pointer (.claude/distribution.md says why both).
        //
        // The other four fixes the previous base (0.32.3, core v0.32.2) carried
        // are in core v0.32.4, each checked by ancestry: #3922 and #4009 (the
        // sorted-RHS affine NAX kernel: rows past 32768 left unwritten, and the
        // K tail when K is not a multiple of 64), #4660 (the pad-for-conv paths
        // freed their zero scalar before the GPU read it) and #4431
        // (sdpa_vector_2pass_1_gqa read batch 0's keys for every batch
        // element). Their tests stay as regression guards:
        // QuantizedMatmulParityTests, ConvChannelPaddingRaceTests and
        // SDPAVectorGQABatchParityTests. So do the guards for the twelve fixes
        // that reached the base at core v0.32.2: NAXSplitKGemmParityTests
        // (#3810), NAXAttentionAndAddMMParityTests (#3361, and the addmm
        // property that #3422 must keep), SteelGemmSafeLoadParityTests (#3560),
        // and QuantizedMatmulParityTests (#3497, #3631, #4251).
        //
        // Since 0.32.3 every stream is created through
        // mlx_stream_new_thread_unsafe and the default is resolved from a
        // task-local with a global fallback, so evaluating from a second OS
        // thread does not abort (ml-explore/mlx-swift#457, closed upstream).
        // CrossThreadEvalTripwireTests holds that line.
        //
        // The fork carries no API changes; its .gitmodules points the mlx
        // submodule at the magicnight mirror so the carried commit resolves.
        // Drop this override and return to the upstream package as soon as an
        // mlx-swift release vendors a core that contains #4775. Pinned by
        // revision so it can never drift; CLIForkPinTests, AppForkPinTests and
        // the CI step that reads the app's resolved graph keep the three roots
        // in step.
        .package(
            url: "https://github.com/magicnight/mlx-swift.git",
            revision: "8a9422e6385078199ba3d96a23560dbbd322c7bf"),
        // Minor-pinned: a minor bump of mlx-swift-lm raises its mlx-swift floor,
        // which the fork's revision pin can satisfy at resolution but not at the
        // source level. Move both together, never one.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", .upToNextMinor(from: "3.33.0")),
        // 2.26.0 fixed three denial-of-service advisories (GHSA-wqpw-qq6j-p4x2,
        // GHSA-62cf-93wq-244m, GHSA-97r6-mq85-c3rv); 2.28.0 also raises its
        // swift-nio-ssl and swift-nio-http2 floors past their July advisories.
        // 2.28.1 stops `consumeWithInboundCloseHandler` spinning on a stream
        // that is already closed. 2.29.0 makes `LocalFileSystem` stop following
        // symbolic links by default; the server serves no files from disk, so
        // that does not reach it.
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.29.0"),
        .package(url: "https://github.com/kean/Pulse.git", from: "5.2.3"),
        // Use the 1.x series: avoids 0.1.24's pin on swift-argument-parser 1.4.x
        // which conflicts with our CLI's argparse 1.8.x requirement.
        // 1.4.0 follows Transformers 5.7.0 on decoding: cleanup of tokenization
        // spaces is off unless the tokenizer config asks for it, and BPE
        // tokenizers skip it unless the config forces it, so `x != y` no longer
        // decodes as `x!= y`. The release also changes encoding in places (code
        // point matching in Split, Replace and BPE merges, `ignore_merges`,
        // `fuse_unk`), so do not assume ids are stable across it: GLM-OCR's
        // moved on CRLF text, the other local checkpoints' did not. Its manifest requires swift-jinja
        // 2.4.2 ..< 4.0.0, and Transformers is linked into the MacMLXCore
        // LIBRARY, so that requirement reaches every root that depends on this
        // package — the CLI and the app included. The swift-jinja declaration
        // further down does NOT do that job (see its comment), so that range,
        // not our 3.2.1 floor, is all the manifests ask of the CLI and the app;
        // the committed lockfiles and the parity check hold them at Core's pin.
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.4.0"),
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
        // compatibility, and 2.4.2 is the floor swift-transformers 1.4.0
        // declares. Every checkpoint chat template now renders natively, with
        // no built-in override.
        //
        // 3.x renders much closer to Python's Jinja2, which is what the
        // checkpoints' templates were written for. Against transformers 5.23.0
        // on 19 local templates and 20 renders each, 2.5.1 differed on 140 and
        // 3.2.1 on 66; the 66 are all tool schemas in a different key order (a
        // tool arrives as an unordered Swift dictionary and swift-jinja's
        // `Value(any:)` sorts the keys; Python keeps the client's order). What
        // 3.x fixed: stray
        // spaces in Gemma 4's tool declarations, `true` for `True` in MiniCPM3,
        // Seed-OSS and Qwen 3.5 tool text, and a trailing newline the reranker
        // templates should not end with. What 3.x costs: a macro may call
        // itself at most 16 deep (`maximumCallDepth`, which swift-transformers
        // does not expose), so Gemma 4's and MiniCPM3's recursive schema macros
        // throw on a tool schema nested about 15 levels.
        //
        // This declaration links Jinja into the render-parity TEST target only,
        // so the tests render through the exact engine production uses. It is
        // NOT what guarantees the version downstream: SwiftPM ignores a
        // dependency package's test-only requirements when another root (the
        // CLI, the app) resolves its graph. The library-level guarantee is the
        // swift-transformers 1.4.0 floor above, whose manifest requires
        // swift-jinja 2.4.2 ..< 4.0.0. That range is wider than this floor, so
        // moving this one also means moving the pin in all three lockfiles; the
        // app's does not move on its own (see "Moving a dependency").
        .package(url: "https://github.com/huggingface/swift-jinja.git", from: "3.2.1"),
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
        // else differs from the upstream tag but one three-line commit that
        // lets Parakeet build against mlx-swift 0.33, whose compile() returns
        // plain closures (upstream made the same change in v0.2.0). Upstream
        // v0.2.1 is the next base; it needs swift-huggingface 1.x, so that
        // move is its own change. Earlier heads are tagged `v0.1.3-macmlx-1`
        // and `v0.1.5-macmlx-1` on the fork.
        // Drop the fork once a release passes the cache through, completes a
        // cached snapshot by the Hub's listing rather than by what a load
        // asked for, reads that listing past its first page, and sends its
        // token clients to `HF_ENDPOINT` too. Pinned by revision so it can
        // never drift.
        .package(
            url: "https://github.com/magicnight/mlx-audio-swift.git",
            revision: "e9ecc7b07257ef2e403fe88965a9c33cbfd25671"),
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
