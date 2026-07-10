// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLXLMCommon
import XCTest

@testable import MacMLXCore

/// End-to-end proof that Track B's nearest-prefix reuse + incremental prefill
/// is CORRECT on a real model: a 3-turn growing conversation served by a
/// cache-warm engine must produce byte-identical greedy output to the same
/// turns served cold. The oracle engine has its prompt cache cleared before
/// every turn, so it always pays a full cold prefill — an independent reference
/// that shares no cached state with the warm engine. Greedy (temperature 0) is
/// deterministic, so any divergence in the trimmed-cache / incremental-prefill
/// path shows up as a text mismatch.
///
/// The per-turn *incremental* prefill length (the headline acceptance metric —
/// turn N prefills only the newly appended tokens) is asserted deterministically
/// and model-free in `PromptCacheStoreTests.testIncrementalReuseAcrossGrowingTurns`
/// and is visible at runtime in the engine's `Prompt cache HIT — … incremental
/// prefill of N` debug log.
///
/// GATED — never runs in CI. Self-skips unless ALL hold:
///   1. `requireMLXRuntimeOrSkip()` (real Metal, i.e. xcodebuild),
///   2. env `MACMLX_RUN_PROMPT_CACHE_E2E=1`,
///   3. a model is on disk (env `MACMLX_PROMPT_CACHE_MODEL` names a dir under
///      `~/.mac-mlx/models`, else built-in candidates such as `Qwen3-4B-4bit`).
final class PromptCacheIncrementalPrefillE2ETests: XCTestCase {

    private func modelDirectory() -> URL? {
        let root = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appending(path: ".mac-mlx/models", directoryHint: .isDirectory)
        var candidates: [String] = []
        if let override = ProcessInfo.processInfo.environment["MACMLX_PROMPT_CACHE_MODEL"] {
            candidates.append(override)
        }
        candidates.append(contentsOf: ["Qwen3-4B-4bit", "Qwen3-4B-8bit", "Qwen3-1.7B-4bit"])
        for name in candidates {
            let dir = root.appending(path: name, directoryHint: .isDirectory)
            if FileManager.default.fileExists(atPath: dir.path) { return dir }
        }
        return nil
    }

    private func request(_ model: String, _ messages: [ChatMessage]) -> GenerateRequest {
        GenerateRequest(
            model: model,
            messages: messages,
            parameters: .init(temperature: 0, topP: 1, maxTokens: 24, stream: true))
    }

    private func collect(
        _ engine: MLXSwiftEngine, _ request: GenerateRequest
    ) async throws -> (text: String, promptTokens: Int?) {
        var text = ""
        var promptTokens: Int?
        for try await chunk in engine.generate(request) {
            text += chunk.text
            if let usage = chunk.usage { promptTokens = usage.promptTokens }
        }
        return (text, promptTokens)
    }

    func testGrowingConversationReusesPrefixWithIdenticalOutput() async throws {
        try requireMLXRuntimeOrSkip()
        guard ProcessInfo.processInfo.environment["MACMLX_RUN_PROMPT_CACHE_E2E"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_PROMPT_CACHE_E2E=1 to run the Track B prefix-cache E2E")
        }
        guard let modelDir = modelDirectory() else {
            throw XCTSkip("No model found (set MACMLX_PROMPT_CACHE_MODEL to a dir name)")
        }

        let modelID = modelDir.lastPathComponent
        let model = LocalModel(
            id: modelID, displayName: modelID, directory: modelDir,
            sizeBytes: 0, format: .mlx, quantization: nil,
            parameterCount: nil, architecture: nil)

        // Warm engine accumulates cache; oracle engine is cleared each turn.
        let warm = MLXSwiftEngine()
        let oracle = MLXSwiftEngine()
        try await warm.load(model)
        try await oracle.load(model)

        // Build a conversation that GROWS by appending the model's own reply
        // plus a new user turn — the tool-loop / multi-turn-agent shape.
        var messages: [ChatMessage] = [
            ChatMessage(role: .user, content: "Reply with exactly one short sentence about apples.")
        ]
        let followUps = ["Now one about oranges.", "Now one about pears."]

        var lastPromptTokens = 0
        for turn in 0..<3 {
            let req = request(modelID, messages)

            let warmResult = try await collect(warm, req)

            // Force the oracle cold for this exact prompt.
            await oracle.clearPromptCache()
            let oracleResult = try await collect(oracle, req)

            XCTAssertFalse(
                warmResult.text.isEmpty, "turn \(turn): warm output should be non-empty")
            XCTAssertEqual(
                warmResult.text, oracleResult.text,
                "turn \(turn): cache-warm output must match cold oracle output")

            // Full-length `prompt_tokens` semantics: the cache-warm engine must
            // report the SAME prompt token count as the cold oracle, which
            // always pays a full prefill and so is naturally full-length. Their
            // equality is direct proof that the reused prefix is added back into
            // usage rather than the incremental-only prefill count leaking
            // through on the HIT path.
            XCTAssertEqual(
                warmResult.promptTokens, oracleResult.promptTokens,
                "turn \(turn): warm prompt_tokens must equal the cold oracle's full-length count")

            // And the full length still grows turn over turn (the reused prefix
            // expands as the conversation accretes).
            if let pt = warmResult.promptTokens {
                XCTAssertGreaterThan(pt, lastPromptTokens, "turn \(turn): prompt should grow")
                lastPromptTokens = pt
            }

            // Extend the conversation for the next turn.
            if turn < followUps.count {
                messages.append(ChatMessage(role: .assistant, content: warmResult.text))
                messages.append(ChatMessage(role: .user, content: followUps[turn]))
            }
        }
    }
}
