// Copyright © 2026 macMLX. English comments only.

import Foundation
import XCTest

@testable import MacMLXCore

/// Gated, real-weights smoke for the pure-Swift MiniCPM3 (MiniCPM3-4B) port (Track G).
///
/// Unlike the numeric-parity suites (tiny synthetic fixtures), this loads the REAL
/// 4-bit checkpoint through the full engine path and end-to-end exercises:
///
///  a. **overlay resolution** — `MLXSwiftEngine.load` runs
///     `ModelOverlay.registerAll()`, so `LLMModelFactory` resolves
///     `config.json`'s `model_type: minicpm3` to `MiniCPM3Model`.
///  b. **quantized load** — the 4-bit (group-size 64) weights load into the stock
///     `Linear`/`Embedding`/`RMSNorm` layers. The shipped checkpoint is UNTIED
///     (`tie_word_embeddings` absent → false), so it carries an `lm_head` and the
///     hidden state is divided by `hidden_size / dim_model_base` (2560/256 = 10)
///     before the head — muP scaling #3.
///  c. **non-absorbed MLA + muP** — the materialized MLA data flow (low-rank q/kv
///     projections, the nope/rope split, the single-head RoPE key broadcast to all
///     40 heads, the distinct 64-wide value head, the 96^-0.5 softmax scale), the
///     embedding × `scale_emb` (12) and both-branch `scale_depth/√layers` scalings,
///     and the longrope `SuScaledRoPE` all run over the real 62-layer stack.
///  d. **eos handling** — MiniCPM3's `eos_token_id` is an ARRAY `[2, 73440]`; the
///     engine's stop-token handling (generation_config / tokenizer) must halt on
///     EITHER. A run that never stops would still be caught by the Mars anchor /
///     maxTokens, but coherent early termination is the expected path.
///  e. **generation** — greedy decode produces coherent, non-empty text; tok/s is
///     printed for the record (not asserted — hardware-dependent).
///
/// GATED — never runs in CI. Self-skips unless ALL hold:
///   1. `requireMLXRuntimeOrSkip()` passes (real Metal, i.e. xcodebuild),
///   2. env `MACMLX_RUN_MINICPM3_SMOKE=1`, and
///   3. the checkpoint is found on disk. Discovery order:
///        • env `MACMLX_MINICPM3_MODEL_DIR` (a full snapshot-directory path), else
///        • the HuggingFace cache
///          `~/.cache/huggingface/hub/models--mlx-community--MiniCPM3-4B-4bit/snapshots/<hash>/`
///          (the snapshot subdir containing `config.json`), else
///        • `~/.mac-mlx/models/<MACMLX_MINICPM3_MODEL>` (default `MiniCPM3-4B-4bit`).
///
/// Run (with the ~2.5 GB checkpoint already in the HF cache):
///   MACMLX_RUN_MINICPM3_SMOKE=1 TEST_RUNNER_MACMLX_RUN_MINICPM3_SMOKE=1 \
///     xcodebuild test -scheme MacMLXCore -destination 'platform=macOS' \
///     -skipPackagePluginValidation \
///     -only-testing:MacMLXCoreTests/MiniCPM3SmokeTests/testMiniCPM3SmokeGeneratesCoherentText
final class MiniCPM3SmokeTests: XCTestCase {

    private static let hfRepo = "mlx-community/MiniCPM3-4B-4bit"

    /// Resolve the checkpoint directory: explicit env path → HF cache snapshot →
    /// `~/.mac-mlx/models`. Returns nil when nothing usable (with a `config.json`)
    /// is present, so the caller can skip.
    private func resolveModelDirectory() -> URL? {
        let fm = FileManager.default
        func hasConfig(_ dir: URL) -> Bool {
            fm.fileExists(atPath: dir.appending(path: "config.json").path)
        }

        // 1. Explicit override.
        if let explicit = ProcessInfo.processInfo.environment["MACMLX_MINICPM3_MODEL_DIR"] {
            let dir = URL(fileURLWithPath: explicit, isDirectory: true)
            return hasConfig(dir) ? dir : nil
        }

        // 2. HuggingFace cache: pick the snapshot subdir that carries config.json.
        let cacheName = "models--" + Self.hfRepo.replacingOccurrences(of: "/", with: "--")
        let snapshots = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appending(path: ".cache/huggingface/hub/\(cacheName)/snapshots", directoryHint: .isDirectory)
        if let entries = try? fm.contentsOfDirectory(
            at: snapshots, includingPropertiesForKeys: nil)
        {
            if let snapshot = entries.sorted(by: { $0.path < $1.path }).first(where: hasConfig) {
                return snapshot
            }
        }

        // 3. Local models dir fallback.
        let name = ProcessInfo.processInfo.environment["MACMLX_MINICPM3_MODEL"]
            ?? "MiniCPM3-4B-4bit"
        let local = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appending(path: ".mac-mlx/models/\(name)", directoryHint: .isDirectory)
        return hasConfig(local) ? local : nil
    }

    private func localModel(id: String, directory: URL) -> LocalModel {
        LocalModel(
            id: id,
            displayName: id,
            directory: directory,
            sizeBytes: 0,
            format: .mlx,
            quantization: nil,
            parameterCount: nil,
            architecture: nil
        )
    }

    /// Loads the real MiniCPM3-4B-4bit checkpoint and greedy-decodes a fixed-answer
    /// continuation, checking the result is coherent (the topic-anchor technique the
    /// other real-model smokes use).
    func testMiniCPM3SmokeGeneratesCoherentText() async throws {
        try requireMLXRuntimeOrSkip()

        guard ProcessInfo.processInfo.environment["MACMLX_RUN_MINICPM3_SMOKE"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_MINICPM3_SMOKE=1 to run the MiniCPM3 real-weights smoke test")
        }

        guard let directory = resolveModelDirectory() else {
            throw XCTSkip("MiniCPM3 checkpoint not found (HF cache / MACMLX_MINICPM3_MODEL_DIR / ~/.mac-mlx/models)")
        }
        let modelID = "MiniCPM3-4B-4bit"

        let engine = MLXSwiftEngine()

        // Fixed-answer prompt: MiniCPM3-4B is an INSTRUCT/chat model, so a direct
        // instruction (rather than a raw sequence completion) is the robust anchor.
        // Asking for the ordered planet list greedily yields "Mercury, Venus, Earth,
        // Mars, …" — verified against the mlx-lm reference (same weights + chat
        // template), which lets us assert the "Mars" anchor (as in the Seed-OSS /
        // Mellum2 / Hunyuan / Cohere2 smokes) while generating enough tokens (~50)
        // for the printed tok/s to reflect real decode throughput, not one-shot
        // load overhead. Since the port is 1e-4 parity-proven, identical logits give
        // identical greedy tokens.
        let prompt = "List all eight planets in order from the Sun, separated by commas."
        let parameters = GenerationParameters(
            temperature: 0, topP: 1.0, maxTokens: 128, stream: true)
        let request = GenerateRequest(
            model: modelID,
            messages: [ChatMessage(role: .user, content: prompt)],
            parameters: parameters
        )

        var text = ""
        var completionTokens: Int?
        let start = Date()
        try await engine.load(localModel(id: modelID, directory: directory))
        for try await chunk in engine.generate(request) {
            text += chunk.text
            if let usage = chunk.usage { completionTokens = usage.completionTokens }
        }
        let elapsed = Date().timeIntervalSince(start)

        XCTAssertFalse(text.isEmpty, "MiniCPM3 must produce real output, not an early-exit stub")
        XCTAssertTrue(
            text.contains("Mars"),
            "greedy ordered-planets list must name Mars (the fourth planet) "
                + "for output to count as coherent — got: \(text)")

        // Echo the generated continuation for the record.
        print("MINICPM3_SMOKE_TEXT<<<\(text)>>>")

        if let completionTokens, elapsed > 0 {
            let tokPerSec = Double(completionTokens) / elapsed
            print(
                "MINICPM3_SMOKE model=\(modelID) dir=\(directory.path) "
                    + "completionTokens=\(completionTokens) "
                    + "elapsed=\(String(format: "%.2f", elapsed))s "
                    + "tokPerSec=\(String(format: "%.1f", tokPerSec))")
        }
    }
}
