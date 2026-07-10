// Copyright © 2026 macMLX. English comments only.

import Foundation
import XCTest

@testable import MacMLXCore

/// Gated, real-model compatibility validation for Qwen3.6 (Track G).
///
/// Qwen3.6 is NOT a new architecture port. Its `config.json` reports
/// `model_type: "qwen3_5"` (dense) / `"qwen3_5_moe"` (MoE) — the exact same
/// `model_type` strings Qwen3.5 already uses. mlx-swift-lm 3.31.4's
/// `Qwen35Model` / `Qwen35MoEModel` (registered in `LLMModelFactory` under
/// those two keys) and macMLX's own format classifier
/// (`ModelLibraryManager.knownVLMTypes`, which does NOT list bare `qwen3_5`)
/// therefore already resolve Qwen3.6 checkpoints with zero code changes —
/// this test exists to PROVE that against real downloaded weights, not to
/// add support.
///
/// GATED — never runs in CI (15 GB download). Self-skips unless ALL hold:
///   1. `requireMLXRuntimeOrSkip()` passes (real Metal, i.e. xcodebuild),
///   2. env `MACMLX_RUN_QWEN36_SMOKE=1`, and
///   3. the model directory exists on disk (env `MACMLX_QWEN36_MODEL`
///      overrides the directory name under `~/.mac-mlx/models`; default
///      `Qwen3.6-27B-4bit` — the smallest Qwen3.6 checkpoint available.
///      Qwen shipped only two Qwen3.6 base sizes, 27B dense and 35B-A3B
///      MoE — no smaller dense variant exists on the Hub).
///
/// Run (once `mlx-community/Qwen3.6-27B-4bit` is downloaded locally):
///   MACMLX_RUN_QWEN36_SMOKE=1 TEST_RUNNER_MACMLX_RUN_QWEN36_SMOKE=1 \
///     xcodebuild test -scheme MacMLXCore -destination 'platform=macOS' \
///     -skipPackagePluginValidation \
///     -only-testing:MacMLXCoreTests/Qwen36ModelTests/testQwen36SmokeGeneratesCoherentText
final class Qwen36ModelTests: XCTestCase {

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

    /// Loads the real Qwen3.6-27B-4bit checkpoint, greedy-decodes a short
    /// continuation of a fixed-answer prompt, and checks the result is
    /// coherent — the topic-anchor technique `BatchServingE2ETests` uses for
    /// the same "prove it's not garbage without a golden reference" problem.
    /// End-to-end this exercises:
    ///
    ///  a. **config parse** — `MLXSwiftEngine.load` decodes `config.json`
    ///     (`Qwen35Configuration` → `Qwen35TextConfiguration`) without error,
    ///     including 27B-specific values (`hidden_size: 5120`,
    ///     `num_hidden_layers: 64`, `linear_num_value_heads: 48`, …) absent
    ///     from the smaller Qwen3.5 checkpoints already in the test fleet.
    ///  b. **load + sanitize** — weight loading and `Qwen35Model.sanitize`
    ///     (vision-tower key drop, `conv1d` axis fix, MTP-key filter) succeed
    ///     against the REAL quantized weights, not synthetic fixtures.
    ///  c. **generation smoke** — greedy decode produces coherent, non-empty
    ///     text; tok/s is printed for the record (not asserted — hardware-
    ///     dependent).
    ///
    /// (No MoE routing check: `mlx-community/Qwen3.6-27B-4bit` is the DENSE
    /// checkpoint. `Qwen3.6-35B-A3B-4bit`, ~20 GB, `qwen3_5_moe`, would be
    /// the MoE counterpart; not downloaded per the size-gated selection.)
    func testQwen36SmokeGeneratesCoherentText() async throws {
        try requireMLXRuntimeOrSkip()

        guard ProcessInfo.processInfo.environment["MACMLX_RUN_QWEN36_SMOKE"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_QWEN36_SMOKE=1 to run the Qwen3.6 compatibility smoke test")
        }

        let modelID = ProcessInfo.processInfo.environment["MACMLX_QWEN36_MODEL"] ?? "Qwen3.6-27B-4bit"
        let directory = modelDirectory(modelID)
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw XCTSkip("Qwen3.6 model dir not found: \(directory.path)")
        }

        let engine = MLXSwiftEngine()
        try await engine.load(localModel(id: modelID, directory: directory))

        // Same fixed-answer prompt `SpeculativeDecodingModelTests` uses — greedy
        // decoding a competent model must continue the planet sequence with Mars.
        // Qwen3.6 is a THINKING model: the chat template opens a reasoning flow
        // before the answer, so the budget must be large enough to reach "Mars"
        // (it appears in the reasoning and/or the final answer; 32 tokens only
        // covered the first thinking sentence). Engine-level chunk.text carries
        // the full raw stream including the think block.
        let prompt = "Planets in order: Mercury, Venus, Earth,"
        let parameters = GenerationParameters(temperature: 0, topP: 1.0, maxTokens: 256, stream: true)
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

        XCTAssertFalse(text.isEmpty, "Qwen3.6 must produce real output, not an early-exit stub")
        XCTAssertTrue(
            text.contains("Mars"),
            "greedy continuation of 'Mercury, Venus, Earth,' must name the next planet "
                + "for output to count as coherent — got: \(text)")

        if let completionTokens, elapsed > 0 {
            let tokPerSec = Double(completionTokens) / elapsed
            print(
                "QWEN36_SMOKE model=\(modelID) completionTokens=\(completionTokens) "
                    + "elapsed=\(String(format: "%.2f", elapsed))s "
                    + "tokPerSec=\(String(format: "%.1f", tokPerSec))")
        }
    }
}
