// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLXLMCommon
import XCTest

@testable import MacMLXCore
// MLXLMCommon has a `JSONValue` of its own; the scoped import makes ours win
// (see the same note in PromptCacheStoreTests).
import enum MacMLXCore.JSONValue

/// Gated, real-weights check of the history-boundary prompt-cache snapshot
/// (``HistoryBoundary``) on an agent loop's second turn.
///
/// Turn 1 is a Claude Code-shaped request: a system prompt long enough to
/// clear ``HistoryBoundary/minimumBoundarySnapshotTokens``, a `Bash` tool and a
/// user message. Turn 2 replays it with the assistant's tool call, re-rendered
/// the way the server re-renders it, and the tool's result. On Qwen3.6 neither
/// turn 1's full prompt nor its record (prompt + generated tokens) is a prefix
/// of turn 2, and its hybrid cache cannot be trimmed back to the shared part,
/// so before the snapshot turn 2 reused nothing. The test asserts that turn 2
/// reuses exactly turn 1's history boundary — the chat rendered without the
/// generation prompt, computed here from the model's own tokenizer — and that
/// turn 2's greedy output matches a cold run of the same prompt, prefilled in
/// one piece with the snapshot turned off.
///
/// "Matches" is per token, with the project's near-tie rule. The warm run
/// prefills the same tokens as the cold one, split at a different place, and
/// the model computes in bf16: its logits sit around 16-32, where one bf16
/// step is 0.125, and the measured per-token logprob difference between the
/// two runs is one such step. That is enough to flip a greedy choice between
/// two tokens the model rates equally: on Qwen3.6 the 23rd token of turn 2
/// is " let" or " confirm", logprob -1.5 for both in the warm run, 0.375 apart
/// in the cold one, and the texts part there. So up to the first token the
/// runs disagree on, every logprob must agree within `logprobTolerance`; at
/// that token each run's choice must be within `nearTieTolerance` of the
/// other run's choice in that run's own distribution; after it the
/// continuations differ and are not compared. A reused cache that was not
/// the cache a cold prefill builds misses these bounds by far: a state off
/// by one position moves logprobs by whole units.
///
/// It runs twice: on the stock decode path, then on the hand-built pipeline
/// (`logprobs`, the run the comparison above is made on). Both decode
/// greedily from the same caches, so the stock path's warm and cold texts
/// must equal the hand-built pipeline's.
///
/// Mellum2 (sliding-window, also drops reasoning from history) passes the same
/// checks. Gemma 4 would not pass the equality on `cachedPromptTokens`: its
/// template keeps turn 1's record a prefix of turn 2, so the longer record is
/// what turn 2 reuses.
///
/// GATED — never runs in CI. Self-skips unless ALL hold:
///   1. `requireMLXRuntimeOrSkip()` passes (real Metal, i.e. xcodebuild),
///   2. env `MACMLX_RUN_HISTORY_BOUNDARY_SMOKE=1`, and
///   3. the model directory exists (env `MACMLX_HISTORY_BOUNDARY_MODEL`
///      overrides the directory name under `~/.mac-mlx/models`; default
///      `Qwen3.6-27B-4bit`).
///
/// Run:
///   MACMLX_RUN_HISTORY_BOUNDARY_SMOKE=1 TEST_RUNNER_MACMLX_RUN_HISTORY_BOUNDARY_SMOKE=1 \
///     xcodebuild test -scheme MacMLXCore -destination 'platform=macOS' \
///     -skipPackagePluginValidation \
///     -only-testing:MacMLXCoreTests/HistoryBoundaryPromptCacheSmokeTests
/// and, for Mellum2, add
///   TEST_RUNNER_MACMLX_HISTORY_BOUNDARY_MODEL=Mellum2-12B-A2.5B-Thinking-4bit
final class HistoryBoundaryPromptCacheSmokeTests: XCTestCase {

    /// Bound on the per-token logprob difference between the warm and the
    /// cold turn 2 before they first disagree: two bf16 steps at the logits'
    /// magnitude (measured: one).
    private let logprobTolerance: Float = 0.25

    /// Bound on the gap, in one run's distribution, between its own greedy
    /// token and the other run's at the first token they disagree on: four
    /// bf16 steps (measured: three on Qwen3.6).
    private let nearTieTolerance: Float = 0.5

    private struct Collected {
        var text = ""
        var promptTokens: Int?
        var cachedPromptTokens: Int?
        var logprobs: [TokenLogprob] = []
        var seconds: Double = 0
    }

    private func collect(_ engine: MLXSwiftEngine, _ request: GenerateRequest) async throws -> Collected {
        var out = Collected()
        let start = Date()
        for try await chunk in engine.generate(request) {
            out.text += chunk.text
            out.logprobs += chunk.logprobs ?? []
            if let usage = chunk.usage {
                out.promptTokens = usage.promptTokens
                out.cachedPromptTokens = usage.cachedPromptTokens
            }
        }
        out.seconds = Date().timeIntervalSince(start)
        return out
    }

    /// A Claude Code-shaped system prompt: well past the snapshot minimum on
    /// any tokenizer, and plain prose, so the model reads it as instructions.
    /// Long enough, too, to carry the prompt past Mellum2's 1,024-token
    /// sliding window: short of it a rotating cache can still be trimmed, and
    /// turn 2 would reuse turn 1's record trimmed back instead.
    private static let systemPrompt: String = {
        let rules = [
            "You are an interactive command line agent that helps users with software engineering tasks.",
            "Use the tools available to you to read files, run commands and make changes.",
            "Before running a command, explain briefly what it does and why it is needed.",
            "Prefer small, reversible changes and verify each change before moving on.",
            "Never invent file contents; read a file before you describe or edit it.",
            "When a command fails, read its output and fix the cause instead of retrying blindly.",
            "Keep answers short; the user reads them in a terminal.",
            "Quote paths and identifiers exactly as they appear in the repository.",
        ]
        return (1...10).map { section in
            "Section \(section).\n" + rules.map { "- \($0)" }.joined(separator: "\n")
        }.joined(separator: "\n\n")
    }()

    private static let bashTool: JSONValue = .object([
        "type": .string("function"),
        "function": .object([
            "name": .string("Bash"),
            "description": .string("Run a shell command and return its output."),
            "parameters": .object([
                "type": .string("object"),
                "properties": .object([
                    "command": .object([
                        "type": .string("string"),
                        "description": .string("The command to run."),
                    ])
                ]),
                "required": .array([.string("command")]),
            ]),
        ]),
    ])

    private func turns(
        modelID: String, logprobs: Bool
    ) -> (turn1: GenerateRequest, turn2: GenerateRequest) {
        var parameters = GenerationParameters(temperature: 0, topP: 1.0, maxTokens: 16, stream: true)
        parameters.logprobs = logprobs
        // The other run's token has to be among the alternatives at a
        // near-tie.
        parameters.topLogprobs = logprobs ? 5 : nil
        let user = ChatMessage(role: .user, content: "Run the shell command: echo hi")
        let call = ToolCallRequest(id: "call_1", name: "Bash", arguments: ["command": .string("echo hi")])
        let turn1 = GenerateRequest(
            model: modelID, messages: [user], systemPrompt: Self.systemPrompt,
            parameters: parameters, tools: [Self.bashTool])
        var turn2Parameters = parameters
        turn2Parameters.maxTokens = 24
        let turn2 = GenerateRequest(
            model: modelID,
            messages: [
                user,
                ChatMessage(role: .assistant, content: "", toolCalls: [call]),
                ChatMessage(role: .tool, content: "hi", toolCallID: "call_1"),
            ],
            systemPrompt: Self.systemPrompt, parameters: turn2Parameters, tools: [Self.bashTool])
        return (turn1, turn2)
    }

    /// `request`'s chat rendered through the resident model's tokenizer with
    /// and without the generation prompt: the prompt generation prefills and
    /// its history boundary.
    private func renders(
        _ request: GenerateRequest, tokenizer: TokenizerBridge
    ) throws -> (prompt: Int, history: Int) {
        let input = UserInput(
            chat: request.allMessages.map { MLXSwiftEngine.upstreamChatMessage(from: $0, images: []) },
            tools: request.tools.map { MLXSwiftEngine.toolSpecs(from: $0) },
            additionalContext: nil)
        let messages = DefaultMessageGenerator().generate(from: input)
        let prompt = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: nil, addGenerationPrompt: true)
        let history = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: nil, addGenerationPrompt: false)
        return (prompt.count, history.count)
    }

    /// At the first token two greedy runs disagree on, each run's choice is
    /// among the other's alternatives and within `nearTieTolerance` of the
    /// other's choice there.
    private func assertNearTie(_ warm: TokenLogprob, _ cold: TokenLogprob, at position: Int) {
        for (run, own, other) in [("warm", warm, cold), ("cold", cold, warm)] {
            guard let alternative = own.topLogprobs.first(where: { $0.token == other.token }) else {
                XCTFail(
                    "token \(position): the \(run) run does not rank \(other.token.debugDescription) "
                        + "among its alternatives \(own.topLogprobs.map(\.token))")
                continue
            }
            XCTAssertLessThanOrEqual(
                own.logprob - alternative.logprob, nearTieTolerance,
                "token \(position): the \(run) run picked \(own.token.debugDescription) at \(own.logprob), "
                    + "\(other.token.debugDescription) at \(alternative.logprob)")
        }
    }

    func testTheNextAgentTurnReusesTheHistoryBoundaryAndMatchesAColdRun() async throws {
        try requireMLXRuntimeOrSkip()
        guard ProcessInfo.processInfo.environment["MACMLX_RUN_HISTORY_BOUNDARY_SMOKE"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_HISTORY_BOUNDARY_SMOKE=1 to run the history-boundary prompt cache smoke")
        }
        let modelID = ProcessInfo.processInfo.environment["MACMLX_HISTORY_BOUNDARY_MODEL"] ?? "Qwen3.6-27B-4bit"
        let directory = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appending(path: ".mac-mlx/models/\(modelID)", directoryHint: .isDirectory)
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw XCTSkip("Model dir not found: \(directory.path)")
        }

        let engine = MLXSwiftEngine(promptCache: PromptCacheConfig(coldEnabled: false))
        try await engine.load(
            LocalModel(
                id: modelID, displayName: modelID, directory: directory, sizeBytes: 0,
                format: .mlx, quantization: nil, parameterCount: nil, architecture: nil))
        addTeardownBlock { try await engine.unload() }
        let loaded = try await HuggingFaceTokenizerLoader().load(from: directory)
        let tokenizer = try XCTUnwrap(loaded as? TokenizerBridge)

        var stockTexts: (warm: String, cold: String)?
        for logprobs in [false, true] {
            let path = logprobs ? "custom pipeline" : "stock path"
            await engine.clearPromptCache()
            let (turn1, turn2) = turns(modelID: modelID, logprobs: logprobs)
            let turn1Render = try renders(turn1, tokenizer: tokenizer)
            XCTAssertGreaterThanOrEqual(
                turn1Render.history, HistoryBoundary.minimumBoundarySnapshotTokens,
                "\(path): the system prompt clears the snapshot minimum")

            let first = try await collect(engine, turn1)
            let turn1Bytes = await engine.promptCacheResidentBytes
            XCTAssertEqual(first.promptTokens, turn1Render.prompt, "\(path): the test renders generation's prompt")
            XCTAssertEqual(first.cachedPromptTokens, 0, "\(path): turn 1 starts cold")
            let recordOrNil = await engine.lastPromptCacheRecord
            let record = try XCTUnwrap(recordOrNil, "\(path): turn 1 went through the prompt cache")
            XCTAssertTrue(record.recorded, "\(path): turn 1's own record is still stored")

            let warm = try await collect(engine, turn2)
            // The reference: the same turn, cold and prefilled in one piece.
            await engine.clearPromptCache()
            await engine.setHistoryBoundarySnapshotsEnabled(false)
            let cold = try await collect(engine, turn2)
            await engine.setHistoryBoundarySnapshotsEnabled(true)
            let pairs = Array(zip(warm.logprobs, cold.logprobs))
            let divergence = pairs.firstIndex { $0.0.token != $0.1.token }
            let delta = pairs.prefix(divergence ?? pairs.count)
                .map { abs($0.0.logprob - $0.1.logprob) }.max() ?? 0
            let disagreement = divergence.map { String($0) } ?? "none"
            let timing = String(format: "warm=%.2f s cold=%.2f s", warm.seconds, cold.seconds)
            print(
                "[history-boundary] \(modelID) \(path): turn 1 prompt=\(first.promptTokens ?? -1) "
                    + "boundary=\(turn1Render.history) record=\(record.tokenCount); turn 2 prompt=\(warm.promptTokens ?? -1) "
                    + "reused=\(warm.cachedPromptTokens ?? -1) \(timing) max |dlogprob|=\(delta) "
                    + "first disagreement=\(disagreement); "
                    + "warm=\(warm.text.debugDescription) cold=\(cold.text.debugDescription)")

            XCTAssertEqual(
                warm.cachedPromptTokens, turn1Render.history,
                "\(path): turn 2 reuses exactly turn 1's history boundary")
            XCTAssertEqual(cold.cachedPromptTokens, 0, "\(path): the oracle run is cold")
            XCTAssertEqual(warm.promptTokens, cold.promptTokens, "\(path): the warm turn reports the full prompt length")
            XCTAssertFalse(warm.text.isEmpty, "\(path): turn 2 generated text")
            if !logprobs {
                stockTexts = (warm.text, cold.text)
                // What the snapshot costs: turn 1 again, cold and without it,
                // leaves only its record; the difference is the boundary entry.
                await engine.clearPromptCache()
                await engine.setHistoryBoundarySnapshotsEnabled(false)
                _ = try await collect(engine, turn1)
                await engine.setHistoryBoundarySnapshotsEnabled(true)
                let recordBytes = await engine.promptCacheResidentBytes
                print(
                    "[history-boundary] \(modelID): turn 1 hot tier \(turn1Bytes) bytes with the snapshot, "
                        + "\(recordBytes) without; snapshot \(turn1Bytes - recordBytes) bytes "
                        + "for \(turn1Render.history) tokens")
                XCTAssertGreaterThan(turn1Bytes, recordBytes, "\(path): the snapshot is a second hot-tier entry")
            }
            if logprobs {
                XCTAssertEqual(warm.text, stockTexts?.warm, "the warm run decodes alike on both paths")
                XCTAssertEqual(cold.text, stockTexts?.cold, "the cold run decodes alike on both paths")
                XCTAssertFalse(warm.logprobs.isEmpty, "\(path): logprobs were captured")
                XCTAssertLessThanOrEqual(
                    delta, logprobTolerance, "\(path): the same distributions up to the first disagreement")
                if let divergence {
                    assertNearTie(warm.logprobs[divergence], cold.logprobs[divergence], at: divergence)
                } else {
                    XCTAssertEqual(warm.text, cold.text, "\(path): the reused history does not change the output")
                }
            }
        }
    }
}
