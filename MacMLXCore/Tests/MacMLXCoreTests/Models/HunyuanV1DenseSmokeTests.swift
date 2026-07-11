// Copyright © 2026 macMLX. English comments only.

import Foundation
import XCTest

@testable import MacMLXCore

/// Gated, real-weights smoke for the pure-Swift Hunyuan V1 Dense port (Track G).
///
/// Unlike the numeric-parity suites (tiny synthetic fixtures), this loads the
/// REAL 4-bit checkpoint through the full engine path and end-to-end exercises:
///
///  a. **overlay resolution** — `MLXSwiftEngine.load` runs
///     `ModelOverlay.registerAll()`, so `LLMModelFactory` resolves
///     `config.json`'s `model_type: hunyuan_v1_dense` to `HunyuanV1DenseModel`.
///  b. **quantized load + sanitize** — the 4-bit (group-size 64) weights load
///     into the stock `Linear`/`Embedding` layers (no attention/MLP bias in this
///     checkpoint; per-head q/k norm present); the TIED embeddings mean `sanitize`
///     drops `lm_head.weight` and logits come from `embed_tokens.asLinear`.
///  c. **chat template** — the checkpoint's own `chat_template.jinja` compiles and
///     renders under swift-jinja with NO macMLX override (proven byte-for-byte on
///     the standard path by `HunyuanV1DenseChatTemplateParityTests`); a template
///     failure here would surface as a load/generation error and fail loudly.
///  d. **generation** — greedy decode produces coherent, non-empty text; tok/s is
///     printed for the record (not asserted — hardware-dependent).
///
/// GATED — never runs in CI. Self-skips unless ALL hold:
///   1. `requireMLXRuntimeOrSkip()` passes (real Metal, i.e. xcodebuild),
///   2. env `MACMLX_RUN_HUNYUAN_V1_SMOKE=1`, and
///   3. the checkpoint is found on disk. Discovery order:
///        • env `MACMLX_HUNYUAN_V1_MODEL_DIR` (a full snapshot-directory path), else
///        • the HuggingFace cache
///          `~/.cache/huggingface/hub/models--mlx-community--Hunyuan-1.8B-Instruct-4bit/snapshots/<hash>/`
///          (the snapshot subdir containing `config.json`), else
///        • `~/.mac-mlx/models/<MACMLX_HUNYUAN_V1_MODEL>` (default
///          `Hunyuan-1.8B-Instruct-4bit`).
///
/// Run (with the ~1 GB checkpoint already in the HF cache):
///   MACMLX_RUN_HUNYUAN_V1_SMOKE=1 TEST_RUNNER_MACMLX_RUN_HUNYUAN_V1_SMOKE=1 \
///     xcodebuild test -scheme MacMLXCore -destination 'platform=macOS' \
///     -skipPackagePluginValidation \
///     -only-testing:MacMLXCoreTests/HunyuanV1DenseSmokeTests/testHunyuanV1DenseSmokeGeneratesCoherentText
final class HunyuanV1DenseSmokeTests: XCTestCase {

    private static let hfRepo = "mlx-community/Hunyuan-1.8B-Instruct-4bit"

    /// Resolve the checkpoint directory: explicit env path → HF cache snapshot →
    /// `~/.mac-mlx/models`. Returns nil when nothing usable (with a `config.json`)
    /// is present, so the caller can skip.
    private func resolveModelDirectory() -> URL? {
        let fm = FileManager.default
        func hasConfig(_ dir: URL) -> Bool {
            fm.fileExists(atPath: dir.appending(path: "config.json").path)
        }

        // 1. Explicit override.
        if let explicit = ProcessInfo.processInfo.environment["MACMLX_HUNYUAN_V1_MODEL_DIR"] {
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
        let name = ProcessInfo.processInfo.environment["MACMLX_HUNYUAN_V1_MODEL"]
            ?? "Hunyuan-1.8B-Instruct-4bit"
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

    /// Loads the real Hunyuan-1.8B-Instruct-4bit checkpoint and greedy-decodes a
    /// fixed-answer continuation, checking the result is coherent (the topic-
    /// anchor technique the other real-model smokes use).
    ///
    /// Hunyuan supports a thinking flow, so the budget is generous (512 tokens)
    /// and the "Mars" anchor is searched over the FULL raw stream (reasoning +
    /// answer), matching the project-memory discipline for thinking models
    /// (Seed-OSS / Mellum2 / Qwen3.6 smokes).
    func testHunyuanV1DenseSmokeGeneratesCoherentText() async throws {
        try requireMLXRuntimeOrSkip()

        guard ProcessInfo.processInfo.environment["MACMLX_RUN_HUNYUAN_V1_SMOKE"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_HUNYUAN_V1_SMOKE=1 to run the Hunyuan real-weights smoke test")
        }

        guard let directory = resolveModelDirectory() else {
            throw XCTSkip("Hunyuan checkpoint not found (HF cache / MACMLX_HUNYUAN_V1_MODEL_DIR / ~/.mac-mlx/models)")
        }
        let modelID = "Hunyuan-1.8B-Instruct-4bit"

        let engine = MLXSwiftEngine()

        // Fixed-answer prompt: greedy decoding of a competent model must continue
        // the planet sequence with Mars. Same anchor technique the Seed-OSS /
        // Mellum2 / Qwen3.6 smokes use.
        let prompt = "Planets in order from the Sun: Mercury, Venus, Earth,"
        let parameters = GenerationParameters(
            temperature: 0, topP: 1.0, maxTokens: 512, stream: true)
        let request = GenerateRequest(
            model: modelID,
            messages: [ChatMessage(role: .user, content: prompt)],
            parameters: parameters
        )

        var text = ""
        var completionTokens: Int?
        let start = Date()
        try await engine.load(localModel(id: modelID, directory: directory))
        // The chat template is compiled lazily during generate's input-prep. The
        // checkpoint's own template renders under swift-jinja with no override, so
        // this compiles cleanly and generates — any load OR generation error fails
        // the test loudly.
        for try await chunk in engine.generate(request) {
            text += chunk.text
            if let usage = chunk.usage { completionTokens = usage.completionTokens }
        }
        let elapsed = Date().timeIntervalSince(start)

        XCTAssertFalse(text.isEmpty, "Hunyuan must produce real output, not an early-exit stub")
        XCTAssertTrue(
            text.contains("Mars"),
            "greedy continuation of 'Mercury, Venus, Earth,' must name the next planet "
                + "for output to count as coherent — got: \(text)")

        // Echo the generated continuation for the record.
        print("HUNYUAN_V1_SMOKE_TEXT<<<\(text)>>>")

        if let completionTokens, elapsed > 0 {
            let tokPerSec = Double(completionTokens) / elapsed
            print(
                "HUNYUAN_V1_SMOKE model=\(modelID) dir=\(directory.path) "
                    + "completionTokens=\(completionTokens) "
                    + "elapsed=\(String(format: "%.2f", elapsed))s "
                    + "tokPerSec=\(String(format: "%.1f", tokPerSec))")
        }
    }
}
