// Copyright © 2026 macMLX. English comments only.

import Foundation
import XCTest

@testable import MacMLXCore

/// Gated, real-model coverage for D1 (classic per-request draft-model
/// speculative decoding), through the public `MLXSwiftEngine` API. Two
/// scenarios, two tests:
///
///  - `testSpeculativeAndPlainGreedyDecodeProduceIdenticalOutput`: a DENSE
///    target/draft pair (ordinary, trimmable KV caches) — proves speculative
///    decoding actually ran AND reproduced the target's own greedy output
///    exactly.
///  - `testHybridArchitectureFallsBackToPlainDecoding`: a HYBRID/
///    linear-attention target (Qwen3.5's GatedDeltaNet layers allocate a
///    non-trimmable `MambaCache`) paired with a draft model — proves the
///    engine gracefully falls back to plain decoding (see
///    `MLXSwiftEngine.canUseSpeculativeDecoding`) instead of letting
///    mlx-swift-lm's `SpeculativeTokenIterator.init` "Speculative decoding
///    requires trimmable KV caches." throw bubble out as a request failure.
///    This mirrors a real bug caught via E2E testing: before the fallback
///    precheck existed, a draft-model request against a hybrid target failed
///    the whole request.
///
/// GATED — neither test runs in CI. Each self-skips unless ALL hold:
///   1. `requireMLXRuntimeOrSkip()` passes (real Metal backend, i.e. xcodebuild),
///   2. env `MACMLX_RUN_SPEC_E2E=1` is set, and
///   3. BOTH the target and draft model directories for THAT test exist on disk.
///
/// Run both (once the model pairs referenced by each test's own doc comment
/// are downloaded locally):
///   MACMLX_RUN_SPEC_E2E=1 TEST_RUNNER_MACMLX_RUN_SPEC_E2E=1 \
///     xcodebuild test -scheme MacMLXCore -destination 'platform=macOS' \
///     -skipPackagePluginValidation \
///     -only-testing:MacMLXCoreTests/SpeculativeDecodingModelTests
///
/// See each test's own doc comment for its model pair, env var overrides,
/// and what it proves.
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

    /// Proof that D1 speculative decoding, on a DENSE target/draft pair
    /// (ordinary trimmable KV caches — no hybrid/linear-attention layers),
    /// produces IDENTICAL greedy output to plain generation.
    ///
    /// Run (once the pair below is downloaded locally):
    ///   MACMLX_RUN_SPEC_E2E=1 TEST_RUNNER_MACMLX_RUN_SPEC_E2E=1 \
    ///     MACMLX_SPEC_TARGET_MODEL=Qwen3-4B-4bit \
    ///     MACMLX_SPEC_DRAFT_MODEL=Qwen3-0.6B-4bit \
    ///     TEST_RUNNER_MACMLX_SPEC_TARGET_MODEL=Qwen3-4B-4bit \
    ///     TEST_RUNNER_MACMLX_SPEC_DRAFT_MODEL=Qwen3-0.6B-4bit \
    ///     xcodebuild test -scheme MacMLXCore -destination 'platform=macOS' \
    ///     -skipPackagePluginValidation \
    ///     -only-testing:MacMLXCoreTests/SpeculativeDecodingModelTests/testSpeculativeAndPlainGreedyDecodeProduceIdenticalOutput
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
    ///    fallback to plain decoding. (Contrast with
    ///    `testHybridArchitectureFallsBackToPlainDecoding` below, which
    ///    asserts the opposite on a hybrid-architecture target.)
    func testSpeculativeAndPlainGreedyDecodeProduceIdenticalOutput() async throws {
        try requireMLXRuntimeOrSkip()

        guard ProcessInfo.processInfo.environment["MACMLX_RUN_SPEC_E2E"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_SPEC_E2E=1 to run the D1 speculative-decoding parity test")
        }

        let env = ProcessInfo.processInfo.environment
        // Dense (non-hybrid) same-tokenizer-family pair — both models' KV
        // caches are trimmable, so this exercises the real speculative path
        // end-to-end rather than the fallback. See
        // `testHybridArchitectureFallsBackToPlainDecoding` for the
        // hybrid-architecture fallback counterpart.
        let targetID = env["MACMLX_SPEC_TARGET_MODEL"] ?? "Qwen3-4B-4bit"
        let draftID = env["MACMLX_SPEC_DRAFT_MODEL"] ?? "Qwen3-0.6B-4bit"

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

    /// Regression test for the real bug this fix addresses: requesting
    /// speculative decoding against a HYBRID/linear-attention target model
    /// (Qwen3.5's GatedDeltaNet layers allocate a non-trimmable
    /// `MambaCache`) used to bubble mlx-swift-lm's
    /// `SpeculativeTokenIterator.init` "Speculative decoding requires
    /// trimmable KV caches." throw straight out as a request failure. The
    /// fix (`MLXSwiftEngine.canUseSpeculativeDecoding`, prechecked in
    /// `runLLMGeneration` before ever entering the speculative branch) makes
    /// this gracefully fall back to plain decoding instead.
    ///
    /// Defaults to the same target/draft pairing that reproduced the bug —
    /// both directories are present in a typical local `~/.mac-mlx/models`
    /// today, so this runs locally without any env override beyond the
    /// top-level gate:
    ///
    ///   MACMLX_RUN_SPEC_E2E=1 TEST_RUNNER_MACMLX_RUN_SPEC_E2E=1 \
    ///     xcodebuild test -scheme MacMLXCore -destination 'platform=macOS' \
    ///     -skipPackagePluginValidation \
    ///     -only-testing:MacMLXCoreTests/SpeculativeDecodingModelTests/testHybridArchitectureFallsBackToPlainDecoding
    ///
    /// ## What it proves
    ///  - **No request failure:** `engine.generate(request)` completes
    ///    without throwing, even though `draftModelID` names a draft model
    ///    paired with a hybrid target whose cache can't be trimmed.
    ///  - **Real output:** the concatenated text is non-empty — this is a
    ///    genuine (fallen-back) generation, not an early-exit stub.
    ///  - **Fallback evidence:** the terminal chunk's `speculativeDecoding`
    ///    telemetry is `nil` — proof the engine did NOT run
    ///    `SpeculativeTokenIterator` (which would report telemetry), i.e. it
    ///    genuinely fell back rather than silently degrading some other way.
    func testHybridArchitectureFallsBackToPlainDecoding() async throws {
        try requireMLXRuntimeOrSkip()

        guard ProcessInfo.processInfo.environment["MACMLX_RUN_SPEC_E2E"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_SPEC_E2E=1 to run the D1 speculative-decoding fallback test")
        }

        let env = ProcessInfo.processInfo.environment
        // Hybrid (GatedDeltaNet linear-attention) target paired with a
        // same-tokenizer-family draft — this is the exact pairing that
        // reproduced the "Speculative decoding requires trimmable KV
        // caches." request failure before this fix. Distinct env var names
        // from the dense-pair test above so overriding one test's pair never
        // silently repurposes the other's.
        let targetID = env["MACMLX_SPEC_FALLBACK_TARGET_MODEL"]
            ?? "Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit"
        let draftID = env["MACMLX_SPEC_FALLBACK_DRAFT_MODEL"] ?? "Qwen3.5-4B-MLX-4bit"

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

        let request = GenerateRequest(
            model: targetID, messages: messages, parameters: parameters,
            draftModelID: draftID, numDraftTokens: 3
        )
        let (text, speculativeDecoding) = try await drain(engine, request)

        XCTAssertFalse(text.isEmpty, "fallback generation must still produce real output")
        XCTAssertNil(
            speculativeDecoding,
            "a hybrid-architecture target must fall back to plain decoding — no speculative telemetry")
    }
}
