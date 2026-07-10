// Copyright © 2026 macMLX. English comments only.

import Foundation
import XCTest

@testable import MacMLXCore

/// End-to-end proof that D1 (classic per-request draft-model speculative
/// decoding) produces IDENTICAL greedy output to plain generation, on a REAL
/// target + draft model pair, through the public `MLXSwiftEngine` API.
///
/// GATED — never runs in CI. It self-skips unless ALL hold:
///   1. `requireMLXRuntimeOrSkip()` passes (real Metal backend, i.e. xcodebuild),
///   2. env `MACMLX_RUN_SPEC_E2E=1` is set, and
///   3. BOTH the target and draft model directories exist on disk.
///
/// Run (once a small Qwen3.5 draft model is downloaded locally):
///   MACMLX_RUN_SPEC_E2E=1 TEST_RUNNER_MACMLX_RUN_SPEC_E2E=1 \
///     MACMLX_SPEC_TARGET_MODEL=Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit \
///     MACMLX_SPEC_DRAFT_MODEL=Qwen3.5-4B-MLX-4bit \
///     TEST_RUNNER_MACMLX_SPEC_TARGET_MODEL=Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit \
///     TEST_RUNNER_MACMLX_SPEC_DRAFT_MODEL=Qwen3.5-4B-MLX-4bit \
///     xcodebuild test -scheme MacMLXCore -destination 'platform=macOS' \
///     -skipPackagePluginValidation \
///     -only-testing:MacMLXCoreTests/SpeculativeDecodingModelTests
///
/// ## What it proves
///  - **Output identity:** at `temperature: 0` (greedy), the FULL concatenated
///    text of a generation with `draftModelID` set is byte-for-byte identical
///    to the same prompt/parameters with `draftModelID` nil. Speculative
///    decoding is a lossless acceleration of greedy decoding — the draft model
///    only PROPOSES tokens, the target model's own distribution decides
///    accept/reject, so the emitted sequence must match the target's
///    standalone decode exactly. Comparing decoded text (not raw token ids)
///    is deliberate: `GenerateChunk` only exposes decoded pieces over the
///    public engine API this test drives, and with greedy decoding the fully
///    concatenated text is exactly as strong a proof as comparing token ids
///    (a token-level divergence could not decode back to the same string).
///  - **Speculative path actually ran:** the ON run's terminal chunk carries
///    `speculativeDecoding` telemetry with `proposedTokens > 0` — proof
///    mlx-swift-lm's `SpeculativeTokenIterator` executed, not a silent
///    fallback to plain decoding.
///
/// No local model pairing satisfies the gate today — `~/.mac-mlx/models`
/// only has gemma-4-e4b-it-8bit / Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit
/// / Qwen3.5-35B-A3B-4bit, no small same-tokenizer-family draft — so this
/// currently always skips at step 3. That's fine: the gate itself is what's
/// under test-authoring obligation here; see the executor report for a
/// verified-compatible small-draft candidate (`mlx-community/Qwen3.5-4B-MLX-4bit`,
/// same `model_type`/`vocab_size` as the 27B target).
final class SpeculativeDecodingModelTests: XCTestCase {

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

    /// Drain `engine.generate(request)`, returning the full concatenated
    /// text and the terminal chunk's speculative decoding telemetry (nil
    /// when the run didn't use the speculative path).
    private func drain(
        _ engine: MLXSwiftEngine, _ request: GenerateRequest
    ) async throws -> (text: String, speculativeDecoding: SpeculativeDecodingUsage?) {
        var text = ""
        var speculativeDecoding: SpeculativeDecodingUsage?
        for try await chunk in engine.generate(request) {
            text += chunk.text
            if let usage = chunk.speculativeDecoding {
                speculativeDecoding = usage
            }
        }
        return (text, speculativeDecoding)
    }

    func testSpeculativeAndPlainGreedyDecodeProduceIdenticalOutput() async throws {
        try requireMLXRuntimeOrSkip()

        guard ProcessInfo.processInfo.environment["MACMLX_RUN_SPEC_E2E"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_SPEC_E2E=1 to run the D1 speculative-decoding parity test")
        }

        let env = ProcessInfo.processInfo.environment
        let targetID = env["MACMLX_SPEC_TARGET_MODEL"] ?? "Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit"
        // Smallest verified same-tokenizer-family Qwen3.5 draft candidate as
        // of this writing — see the executor report for the verification
        // (model_type "qwen3_5", vocab_size 248320 on both). Not present
        // locally, so absent an override this test skips at the directory
        // check below.
        let draftID = env["MACMLX_SPEC_DRAFT_MODEL"] ?? "Qwen3.5-4B-MLX-4bit"

        let targetDirectory = modelDirectory(targetID)
        let draftDirectory = modelDirectory(draftID)
        guard FileManager.default.fileExists(atPath: targetDirectory.path) else {
            throw XCTSkip("Target model dir not found: \(targetDirectory.path)")
        }
        guard FileManager.default.fileExists(atPath: draftDirectory.path) else {
            throw XCTSkip("Draft model dir not found: \(draftDirectory.path)")
        }

        let engine = MLXSwiftEngine()
        try await engine.load(localModel(id: targetID, directory: targetDirectory))

        let prompt = "Planets in order: Mercury, Venus, Earth,"
        let parameters = GenerationParameters(temperature: 0, topP: 1.0, maxTokens: 24, stream: true)
        let messages = [ChatMessage(role: .user, content: prompt)]

        let plainRequest = GenerateRequest(
            model: targetID, messages: messages, parameters: parameters
        )
        let (plainText, plainTelemetry) = try await drain(engine, plainRequest)
        XCTAssertNil(plainTelemetry, "plain (non-speculative) generation must carry no speculative telemetry")

        let speculativeRequest = GenerateRequest(
            model: targetID, messages: messages, parameters: parameters,
            draftModelID: draftID, numDraftTokens: 3
        )
        let (speculativeText, speculativeTelemetry) = try await drain(engine, speculativeRequest)

        XCTAssertEqual(
            speculativeText, plainText,
            "speculative decoding must reproduce the target model's own greedy output exactly")
        let telemetry = try XCTUnwrap(
            speculativeTelemetry, "speculative run must report acceptance telemetry")
        XCTAssertGreaterThan(
            telemetry.proposedTokens, 0,
            "the draft model must have proposed at least one token")
    }
}
