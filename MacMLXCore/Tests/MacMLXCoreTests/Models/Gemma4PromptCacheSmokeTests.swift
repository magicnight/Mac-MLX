// Copyright © 2026 macMLX. English comments only.

import Foundation
import XCTest

@testable import MacMLXCore

/// Gated, real-weights check of the prompt cache across a turn that ended on
/// a stop token (#142), on a model whose chat template closes the assistant
/// turn with that same token: Gemma 4 stops on `<turn|>` and renders a
/// previous assistant turn as its text followed by `<turn|>`, so the next
/// turn's prompt extends the recorded key through the stop token and the hit
/// reuses every position turn 1 recorded. Turn 1 asks for logprobs, which
/// routes it through the engine's hand-built pipeline (the loop's other entry
/// point), and checks that the stop token is in neither the text, the
/// logprobs nor the completion count. Turn 2's own record is checked too:
/// ending on the stop token, it holds exactly its key's positions, so the
/// shift no longer grows by a position per turn. Both engines run without
/// the cold tier.
///
/// GATED — never runs in CI. Self-skips unless ALL hold:
///   1. `requireMLXRuntimeOrSkip()` passes (real Metal, i.e. xcodebuild),
///   2. env `MACMLX_RUN_GEMMA4_SMOKE=1`, and
///   3. the model directory exists (env `MACMLX_GEMMA4_MODEL` overrides the
///      directory name under `~/.mac-mlx/models`; default `gemma-4-e4b-it-8bit`).
///
/// Run:
///   MACMLX_RUN_GEMMA4_SMOKE=1 TEST_RUNNER_MACMLX_RUN_GEMMA4_SMOKE=1 \
///     xcodebuild test -scheme MacMLXCore -destination 'platform=macOS' \
///     -skipPackagePluginValidation \
///     -only-testing:MacMLXCoreTests/Gemma4PromptCacheSmokeTests
final class Gemma4PromptCacheSmokeTests: XCTestCase {

    private struct Collected {
        var text = ""
        var finish: FinishReason?
        var promptTokens: Int?
        var completionTokens: Int?
        var cachedPromptTokens: Int?
        var logprobCount = 0
    }

    private func collect(_ engine: MLXSwiftEngine, _ request: GenerateRequest) async throws -> Collected {
        var out = Collected()
        for try await chunk in engine.generate(request) {
            out.text += chunk.text
            if let reason = chunk.finishReason { out.finish = reason }
            if let usage = chunk.usage {
                out.promptTokens = usage.promptTokens
                out.completionTokens = usage.completionTokens
                out.cachedPromptTokens = usage.cachedPromptTokens
            }
            out.logprobCount += chunk.logprobs?.count ?? 0
        }
        return out
    }

    private func localModel(id: String, directory: URL) -> LocalModel {
        LocalModel(
            id: id, displayName: id, directory: directory, sizeBytes: 0,
            format: GatedLoad.format, quantization: nil, parameterCount: nil, architecture: nil)
    }

    func testATurnThatEndedOnTheStopTokenIsExtendedByTheNext() async throws {
        try requireMLXRuntimeOrSkip()

        guard ProcessInfo.processInfo.environment["MACMLX_RUN_GEMMA4_SMOKE"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_GEMMA4_SMOKE=1 to run the Gemma 4 real-weights prompt cache smoke")
        }
        let modelID = ProcessInfo.processInfo.environment["MACMLX_GEMMA4_MODEL"] ?? "gemma-4-e4b-it-8bit"
        let directory = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appending(path: ".mac-mlx/models/\(modelID)", directoryHint: .isDirectory)
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw XCTSkip("Gemma 4 model dir not found: \(directory.path)")
        }

        let noColdTier = PromptCacheConfig(coldEnabled: false)
        let engine = MLXSwiftEngine(promptCache: noColdTier)
        await GatedLoad.prepare(engine)
        try await engine.load(localModel(id: modelID, directory: directory))
        try await GatedLoad.requireFullPathIfForced(engine)
        addTeardownBlock { try await engine.unload() }

        let question = ChatMessage(
            role: .user, content: "Answer with one word: what colour is the sky on a clear day?")
        var withLogprobs = GenerationParameters(temperature: 0, topP: 1.0, maxTokens: 64, stream: true)
        withLogprobs.logprobs = true
        let turn1 = GenerateRequest(model: modelID, messages: [question], parameters: withLogprobs)
        let first = try await collect(engine, turn1)
        XCTAssertEqual(
            first.finish, .stop,
            "turn 1 must end on the model's stop token: \(String(describing: first.finish)) after \(first.text.count) characters")
        XCTAssertFalse(first.text.contains("<turn|>"), "the stop token is not part of the text: \(first.text.debugDescription)")
        let recordOrNil = await engine.lastPromptCacheRecord
        let record = try XCTUnwrap(recordOrNil, "turn 1 went through the prompt cache")
        print(
            "[eos-smoke] gemma 4 turn 1: key=\(record.tokenCount) held=\(record.heldPositions) stop=\(record.endedOnStopToken) "
                + "stored=\(record.recorded) prompt=\(first.promptTokens ?? -1) completion=\(first.completionTokens ?? -1) "
                + "logprobs=\(first.logprobCount) text=\(first.text.debugDescription)")
        XCTAssertTrue(record.endedOnStopToken, "the loop delivered the stop token")
        XCTAssertTrue(record.recorded, "the entry is recorded")
        XCTAssertEqual(
            record.heldPositions, record.tokenCount,
            "the cache holds exactly the key's positions, the stop token's included")
        let promptTokens = try XCTUnwrap(first.promptTokens)
        let completionTokens = try XCTUnwrap(first.completionTokens)
        XCTAssertEqual(
            record.tokenCount, promptTokens + completionTokens + 1,
            "the key is the prompt, the tokens that became text, and the stop token")
        XCTAssertEqual(
            first.logprobCount, completionTokens,
            "one logprob per token that became text, none for the stop token")

        let turn2 = GenerateRequest(
            model: modelID,
            messages: [
                question, ChatMessage(role: .assistant, content: first.text),
                ChatMessage(role: .user, content: "And at sunset? One word."),
            ],
            parameters: GenerationParameters(temperature: 0, topP: 1.0, maxTokens: 24, stream: true))
        let warm = try await collect(engine, turn2)
        let secondOrNil = await engine.lastPromptCacheRecord
        let second = try XCTUnwrap(secondOrNil, "turn 2 went through the prompt cache")
        let oracle = MLXSwiftEngine(promptCache: noColdTier)
        await GatedLoad.prepare(oracle)
        try await oracle.load(localModel(id: modelID, directory: directory))
        addTeardownBlock { try await oracle.unload() }
        let cold = try await collect(oracle, turn2)
        print(
            "[eos-smoke] gemma 4 turn 2: warm reused \(warm.cachedPromptTokens ?? 0) of \(record.tokenCount), "
                + "prompt warm=\(warm.promptTokens ?? -1) cold=\(cold.promptTokens ?? -1), key=\(second.tokenCount) "
                + "held=\(second.heldPositions) stop=\(second.endedOnStopToken); "
                + "warm=\(warm.text.debugDescription) cold=\(cold.text.debugDescription)")
        XCTAssertEqual(
            warm.cachedPromptTokens, record.tokenCount,
            "turn 2 extends the recorded key through the stop token and reuses every position turn 1 recorded")
        XCTAssertEqual(warm.promptTokens, cold.promptTokens, "the warm turn reports the full prompt length")
        XCTAssertEqual(warm.text, cold.text, "a turn served through the stop token's position matches the cold oracle")
        // The turn served from the extension hit ends on the stop token too,
        // and records exactly its key's positions: the shift no longer grows
        // by one per turn.
        XCTAssertEqual(warm.finish, .stop, "the one-word follow-up ends on the stop token too")
        XCTAssertTrue(second.recorded)
        XCTAssertTrue(second.endedOnStopToken)
        XCTAssertEqual(second.heldPositions, second.tokenCount, "a turn served through the stop token's position records exactly its key's positions")
    }
}
