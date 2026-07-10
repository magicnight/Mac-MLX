// Copyright © 2026 macMLX. English comments only.

import Foundation
import XCTest

@testable import MacMLXCore

/// Gated, real-weights smoke for the pure-Swift Mellum 2 port (Track G).
///
/// Unlike the numeric-parity suites (tiny synthetic fixtures), this loads the
/// REAL 4-bit checkpoint through the full engine path and end-to-end exercises:
///
///  a. **overlay resolution** — `MLXSwiftEngine.load` runs
///     `ModelOverlay.registerAll()`, so `LLMModelFactory` resolves
///     `config.json`'s `model_type: mellum` to `Mellum2Model`.
///  b. **quantized load + sanitize** — the mixed-precision quantized weights
///     (gate/attention 8-bit, `switch_mlp` 4-bit, per the checkpoint's
///     `quantization` block) load into the stock `Linear`/`SwitchGLU`/
///     `Embedding` layers; the pre-stacked `switch_mlp` short-circuits
///     `sanitize`.
///  c. **mixed cache + generation** — `newCache` builds the per-layer mix
///     (`RotatingKVCache` for sliding layers, `KVCacheSimple` for full), and
///     greedy decode produces coherent, non-empty text. tok/s is printed for
///     the record (not asserted — hardware-dependent).
///
/// GATED — never runs in CI (7.36 GB download). Self-skips unless ALL hold:
///   1. `requireMLXRuntimeOrSkip()` passes (real Metal, i.e. xcodebuild),
///   2. env `MACMLX_RUN_MELLUM_SMOKE=1`, and
///   3. the model directory exists on disk (env `MACMLX_MELLUM_MODEL` overrides
///      the directory name under `~/.mac-mlx/models`; default
///      `Mellum2-12B-A2.5B-Thinking-4bit`).
///
/// Run (once `jedisct1/Mellum2-12B-A2.5B-Thinking-mlx-4bit` is downloaded to
/// `~/.mac-mlx/models/Mellum2-12B-A2.5B-Thinking-4bit`):
///   MACMLX_RUN_MELLUM_SMOKE=1 TEST_RUNNER_MACMLX_RUN_MELLUM_SMOKE=1 \
///     xcodebuild test -scheme MacMLXCore -destination 'platform=macOS' \
///     -skipPackagePluginValidation \
///     -only-testing:MacMLXCoreTests/Mellum2SmokeTests/testMellum2SmokeGeneratesCoherentText
final class Mellum2SmokeTests: XCTestCase {

    private func modelDirectory(_ name: String) -> URL {
        URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appending(path: ".mac-mlx/models/\(name)", directoryHint: .isDirectory)
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

    /// Loads the real Mellum2-12B-A2.5B-Thinking-4bit checkpoint and greedy-
    /// decodes a fixed-answer continuation, checking the result is coherent
    /// (the topic-anchor technique the other real-model smokes use).
    ///
    /// Mellum 2 is a THINKING model: the chat template opens a reasoning flow
    /// before the answer, so the budget is generous (256 tokens) and the anchor
    /// is searched over the FULL raw stream (reasoning + answer) rather than a
    /// parsed content field — the project-memory discipline for thinking models.
    func testMellum2SmokeGeneratesCoherentText() async throws {
        try requireMLXRuntimeOrSkip()

        guard ProcessInfo.processInfo.environment["MACMLX_RUN_MELLUM_SMOKE"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_MELLUM_SMOKE=1 to run the Mellum 2 real-weights smoke test")
        }

        let modelID =
            ProcessInfo.processInfo.environment["MACMLX_MELLUM_MODEL"]
            ?? "Mellum2-12B-A2.5B-Thinking-4bit"
        let directory = modelDirectory(modelID)
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw XCTSkip("Mellum 2 model dir not found: \(directory.path)")
        }

        let engine = MLXSwiftEngine()
        try await engine.load(localModel(id: modelID, directory: directory))

        // Fixed-answer prompt: greedy decoding of a competent model must
        // continue the planet sequence with Mars. Same anchor technique the
        // Qwen3.6 / speculative-decoding smokes use.
        let prompt = "Planets in order from the Sun: Mercury, Venus, Earth,"
        let parameters = GenerationParameters(
            temperature: 0, topP: 1.0, maxTokens: 256, stream: true)
        let request = GenerateRequest(
            model: modelID,
            messages: [ChatMessage(role: .user, content: prompt)],
            parameters: parameters
        )

        var text = ""
        var completionTokens: Int?
        let start = Date()
        for try await chunk in engine.generate(request) {
            text += chunk.text
            if let usage = chunk.usage { completionTokens = usage.completionTokens }
        }
        let elapsed = Date().timeIntervalSince(start)

        XCTAssertFalse(text.isEmpty, "Mellum 2 must produce real output, not an early-exit stub")
        XCTAssertTrue(
            text.contains("Mars"),
            "greedy continuation of 'Mercury, Venus, Earth,' must name the next planet "
                + "for output to count as coherent — got: \(text)")

        if let completionTokens, elapsed > 0 {
            let tokPerSec = Double(completionTokens) / elapsed
            print(
                "MELLUM2_SMOKE model=\(modelID) completionTokens=\(completionTokens) "
                    + "elapsed=\(String(format: "%.2f", elapsed))s "
                    + "tokPerSec=\(String(format: "%.1f", tokPerSec))")
        }
    }
}
