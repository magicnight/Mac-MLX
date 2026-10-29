// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLXLMCommon
import XCTest

@testable import MacMLXCore
// MLXLMCommon has a `JSONValue` of its own; the scoped import makes ours win
// (see the same note in PromptCacheStoreTests).
import enum MacMLXCore.JSONValue

/// Gated, real-weights checks of the history-boundary prompt-cache snapshot
/// (``HistoryBoundary``) on an agent loop's second turn.
///
/// Turn 1 is a Claude Code-shaped request: a system prompt long enough to
/// clear ``HistoryBoundary/minimumBoundarySnapshotTokens``, a `Bash` tool and a
/// user message. Turn 2 replays it with the assistant's tool call, re-rendered
/// the way the server re-renders it, and the tool's result. On Qwen3.6 neither
/// turn 1's full prompt nor its own entry (prompt + generated tokens) is a
/// prefix of turn 2, and its hybrid cache cannot be trimmed back to the shared
/// part, so before the snapshot turn 2 reused nothing. Both models' caches
/// cannot be trimmed, so turn 1 ends with the snapshot as its only hot-tier
/// entry.
///
/// What the main test asserts:
/// - turn 2 reuses exactly turn 1's history boundary, the chat rendered
///   without the generation prompt, computed here from the model's own
///   tokenizer;
/// - a bitwise oracle: turn 2 run cold with its prefill forced to split where
///   the warm run's was split (at turn 1's boundary, then the rest in one
///   piece, as the warm turn ran it: its own boundary gains too little to
///   split at) reproduces the warm run bit for bit, logprobs and top-5
///   alternatives included;
/// - against a cold run prefilled in one piece, turn 2 and turn 1 (whose
///   prefill the snapshot split) match under the project's near-tie rule.
///
/// The near-tie rule exists because a prefill split at a different place does
/// not round alike: the model computes in bf16, its logits sit around 16-32,
/// where one bf16 step is 0.125, and the measured per-token logprob difference
/// is one such step. That is enough to flip a greedy choice between two tokens
/// the model rates equally: on Qwen3.6 the 23rd token of turn 2 is " let" or
/// " confirm", logprob -1.5 for both in the warm run, 0.375 apart in the cold
/// one, and the texts part there. So up to the first token the runs disagree
/// on, every chosen token's logprob, and every top-5 alternative both runs
/// list, must agree within `logprobTolerance`; at that token each run's choice
/// must be within `nearTieTolerance` of the other run's choice in its own
/// distribution; after it the continuations differ and are not compared. A
/// split that feeds a position twice or skips one fails these bounds.
///
/// It runs twice: on the stock decode path, then on the hand-built pipeline
/// (`logprobs`, the run the comparisons are made on). Both decode greedily
/// from the same caches, so the stock path's texts must equal the hand-built
/// pipeline's.
///
/// Mellum2 (sliding-window, also drops reasoning from history) passes the same
/// checks. Gemma 4 E4B would not pass the equality on `cachedPromptTokens`
/// while its prompt is under its 512-token sliding window: its cache can then
/// be trimmed, so no snapshot is taken and turn 2 reuses turn 1's own entry.
///
/// Every engine here keeps its cold tier under a temporary directory, never
/// `~/.mac-mlx/kv-cache`.
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

    /// Bound on the logprob difference between two runs before they first
    /// disagree: two bf16 steps at the logits' magnitude (measured: one).
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

    /// `request` on `engine` with an empty prompt cache, the snapshot on or
    /// off, and the prefill split forced at `forcedBoundary` when given. The
    /// engine is left with the snapshot on and nothing forced.
    private func coldRun(
        _ engine: MLXSwiftEngine, _ request: GenerateRequest, snapshots: Bool, forcedBoundary: Int? = nil
    ) async throws -> Collected {
        await engine.clearPromptCache()
        await engine.setHistoryBoundarySnapshotsEnabled(snapshots)
        await engine.setForcedHistoryBoundary(forcedBoundary)
        let out = try await collect(engine, request)
        await engine.setHistoryBoundarySnapshotsEnabled(true)
        await engine.setForcedHistoryBoundary(nil)
        return out
    }

    private static let rules = [
        "You are an interactive command line agent that helps users with software engineering tasks.",
        "Use the tools available to you to read files, run commands and make changes.",
        "Before running a command, explain briefly what it does and why it is needed.",
        "Prefer small, reversible changes and verify each change before moving on.",
        "Never invent file contents; read a file before you describe or edit it.",
        "When a command fails, read its output and fix the cause instead of retrying blindly.",
        "Keep answers short; the user reads them in a terminal.",
        "Quote paths and identifiers exactly as they appear in the repository.",
    ]

    /// A Claude Code-shaped system prompt of `sections` sections, about 160
    /// tokens each: plain prose, so the model reads it as instructions. Ten
    /// clear the snapshot minimum on any tokenizer and carry the prompt past
    /// Mellum2's 1,024-token sliding window, short of which a rotating cache
    /// can still be trimmed.
    private static func systemPrompt(sections: Int) -> String {
        (1...sections).map { section in
            "Section \(section).\n" + rules.map { "- \($0)" }.joined(separator: "\n")
        }.joined(separator: "\n\n")
    }

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
        modelID: String, logprobs: Bool, sections: Int = 10
    ) -> (turn1: GenerateRequest, turn2: GenerateRequest) {
        var parameters = GenerationParameters(temperature: 0, topP: 1.0, maxTokens: 16, stream: true)
        parameters.logprobs = logprobs
        // The other run's token has to be among the alternatives at a
        // near-tie.
        parameters.topLogprobs = logprobs ? 5 : nil
        let system = Self.systemPrompt(sections: sections)
        let user = ChatMessage(role: .user, content: "Run the shell command: echo hi")
        let call = ToolCallRequest(id: "call_1", name: "Bash", arguments: ["command": .string("echo hi")])
        let turn1 = GenerateRequest(
            model: modelID, messages: [user], systemPrompt: system,
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
            systemPrompt: system, parameters: turn2Parameters, tools: [Self.bashTool])
        return (turn1, turn2)
    }

    private static func userInput(_ request: GenerateRequest) -> UserInput {
        UserInput(
            chat: request.allMessages.map { MLXSwiftEngine.upstreamChatMessage(from: $0, images: []) },
            tools: request.tools.map { MLXSwiftEngine.toolSpecs(from: $0) },
            additionalContext: nil)
    }

    /// `request`'s chat rendered through the model's tokenizer the way the
    /// engine renders it, with and without the generation prompt.
    private func renders(
        _ request: GenerateRequest, tokenizer: TokenizerBridge
    ) throws -> (prompt: Int, history: Int) {
        let input = Self.userInput(request)
        let messages = DefaultMessageGenerator().generate(from: input)
        let prompt = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: nil, addGenerationPrompt: true)
        let history = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: nil, addGenerationPrompt: false)
        return (prompt.count, history.count)
    }

    private func modelDirectory() throws -> (id: String, url: URL) {
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
        return (modelID, directory)
    }

    private func loadedEngine(
        _ model: (id: String, url: URL), coldEnabled: Bool, coldRoot: URL
    ) async throws -> MLXSwiftEngine {
        let engine = MLXSwiftEngine(
            promptCache: PromptCacheConfig(coldEnabled: coldEnabled), promptCacheRoot: coldRoot)
        try await engine.load(
            LocalModel(
                id: model.id, displayName: model.id, directory: model.url, sizeBytes: 0,
                format: .mlx, quantization: nil, parameterCount: nil, architecture: nil))
        addTeardownBlock {
            try await engine.unload()
            try? FileManager.default.removeItem(at: coldRoot)
        }
        return engine
    }

    private func temporaryRoot() -> URL {
        FileManager.default.temporaryDirectory
            .appending(path: "history-boundary-\(UUID().uuidString)", directoryHint: .isDirectory)
    }

    // MARK: - Comparisons

    /// Two greedy runs of the same prompt, compared under the near-tie rule
    /// (see the type's documentation).
    private func assertMatchesUnderNearTie(_ split: Collected, _ unsplit: Collected, _ label: String) {
        XCTAssertFalse(split.logprobs.isEmpty, "\(label): logprobs were captured")
        let pairs = Array(zip(split.logprobs, unsplit.logprobs))
        let divergence = pairs.firstIndex { $0.0.token != $0.1.token }
        for (position, (a, b)) in pairs.prefix(divergence ?? pairs.count).enumerated() {
            XCTAssertLessThanOrEqual(
                abs(a.logprob - b.logprob), logprobTolerance,
                "\(label) token \(position) \(a.token.debugDescription): \(a.logprob) against \(b.logprob)")
            for alternative in a.topLogprobs {
                guard let other = b.topLogprobs.first(where: { $0.token == alternative.token }) else { continue }
                XCTAssertLessThanOrEqual(
                    abs(alternative.logprob - other.logprob), logprobTolerance,
                    "\(label) token \(position), alternative \(alternative.token.debugDescription): "
                        + "\(alternative.logprob) against \(other.logprob)")
            }
        }
        guard let divergence else {
            XCTAssertEqual(split.text, unsplit.text, "\(label): the same output")
            return
        }
        let ends = [
            ("split", split.logprobs[divergence], unsplit.logprobs[divergence]),
            ("unsplit", unsplit.logprobs[divergence], split.logprobs[divergence]),
        ]
        for (run, own, other) in ends {
            guard let alternative = own.topLogprobs.first(where: { $0.token == other.token }) else {
                XCTFail(
                    "\(label) token \(divergence): the \(run) run does not rank \(other.token.debugDescription) "
                        + "among its alternatives \(own.topLogprobs.map(\.token))")
                continue
            }
            XCTAssertLessThanOrEqual(
                own.logprob - alternative.logprob, nearTieTolerance,
                "\(label) token \(divergence): the \(run) run picked \(own.token.debugDescription) at "
                    + "\(own.logprob), \(other.token.debugDescription) at \(alternative.logprob)")
        }
    }

    private static func describe(_ run: Collected) -> String {
        let timing = String(format: "%.2f s", run.seconds)
        return "prompt=\(run.promptTokens ?? -1) reused=\(run.cachedPromptTokens ?? -1) \(timing) "
            + run.text.debugDescription
    }

    // MARK: - Tests

    func testTheNextAgentTurnReusesTheHistoryBoundaryAndMatchesAColdRun() async throws {
        let model = try modelDirectory()
        let engine = try await loadedEngine(model, coldEnabled: false, coldRoot: temporaryRoot())
        let loaded = try await HuggingFaceTokenizerLoader().load(from: model.url)
        let tokenizer = try XCTUnwrap(loaded as? TokenizerBridge)

        var stockTexts: [String] = []
        for logprobs in [false, true] {
            let path = logprobs ? "custom pipeline" : "stock path"
            let (turn1, turn2) = turns(modelID: model.id, logprobs: logprobs)
            let turn1Render = try renders(turn1, tokenizer: tokenizer)
            let boundary = turn1Render.history
            XCTAssertGreaterThanOrEqual(
                boundary, HistoryBoundary.minimumBoundarySnapshotTokens,
                "\(path): the system prompt clears the snapshot minimum")

            // Turn 1 cold and in one piece: its own entry alone.
            let turn1Unsplit = try await coldRun(engine, turn1, snapshots: false)
            let entryBytes = await engine.promptCacheResidentBytes

            // Turn 1 with the snapshot, then turn 2 on the cache it left.
            let first = try await coldRun(engine, turn1, snapshots: true)
            let snapshotBytes = await engine.promptCacheResidentBytes
            let turn1Entries = await engine.promptCacheResidentCount
            let recordOrNil = await engine.lastPromptCacheRecord
            let record = try XCTUnwrap(recordOrNil, "\(path): turn 1 went through the prompt cache")
            let warm = try await collect(engine, turn2)

            let cold = try await coldRun(engine, turn2, snapshots: false)
            let oracle = try await coldRun(engine, turn2, snapshots: true, forcedBoundary: boundary)

            let label = "[history-boundary] \(model.id) \(path):"
            print(
                "\(label) boundary=\(boundary) entry=\(record.tokenCount) tokens; hot tier after turn 1: "
                    + "\(snapshotBytes) bytes (\(turn1Entries) entry) with the snapshot, \(entryBytes) without")
            print("\(label) turn 2 warm    \(Self.describe(warm))")
            print("\(label) turn 2 cold    \(Self.describe(cold))")
            print("\(label) turn 2 oracle  \(Self.describe(oracle))")
            print("\(label) turn 1 split   \(Self.describe(first))")
            print("\(label) turn 1 unsplit \(Self.describe(turn1Unsplit))")

            XCTAssertEqual(first.promptTokens, turn1Render.prompt, "\(path): the test renders generation's prompt")
            XCTAssertEqual(first.cachedPromptTokens, 0, "\(path): turn 1 starts cold")
            XCTAssertFalse(record.recorded, "\(path): turn 1's own entry is not stored beside the snapshot")
            XCTAssertEqual(turn1Entries, 1, "\(path): one hot-tier entry per turn")
            XCTAssertEqual(
                warm.cachedPromptTokens, boundary, "\(path): turn 2 reuses exactly turn 1's history boundary")
            XCTAssertEqual(cold.cachedPromptTokens, 0, "\(path): the reference run is cold")
            XCTAssertEqual(oracle.cachedPromptTokens, 0, "\(path): the oracle run is cold")
            XCTAssertEqual(warm.promptTokens, cold.promptTokens, "\(path): the warm turn reports the full prompt length")
            XCTAssertFalse(warm.text.isEmpty, "\(path): turn 2 generated text")
            XCTAssertEqual(warm.text, oracle.text, "\(path): the warm turn is the cold turn split where it was split")

            let texts = [warm.text, cold.text, first.text, turn1Unsplit.text]
            if logprobs {
                XCTAssertEqual(texts, stockTexts, "the runs decode alike on both paths")
                XCTAssertEqual(
                    warm.logprobs, oracle.logprobs,
                    "turn 2 warm against the forced-split oracle: the same logprobs and alternatives, bit for bit")
                assertMatchesUnderNearTie(warm, cold, "turn 2 warm against the unsplit cold run")
                assertMatchesUnderNearTie(first, turn1Unsplit, "turn 1 split against unsplit")
            } else {
                stockTexts = texts
            }
        }
    }

    /// The SSD tier at an agent session's size: a history whose snapshot is
    /// larger than the in-memory tier, with the cold tier on. The snapshot goes
    /// straight to disk, turn 2 is served from disk without promoting it, and
    /// no turn writes more than one file. Prints each turn's writes.
    func testAHistoryLargerThanTheInMemoryTierIsServedFromTheSSDTier() async throws {
        let model = try modelDirectory()
        let root = temporaryRoot()
        let engine = try await loadedEngine(model, coldEnabled: true, coldRoot: root)
        let loaded = try await HuggingFaceTokenizerLoader().load(from: model.url)
        let tokenizer = try XCTUnwrap(loaded as? TokenizerBridge)
        let (turn1, turn2) = turns(modelID: model.id, logprobs: false, sections: 50)
        let boundary = try renders(turn1, tokenizer: tokenizer).history

        var files = Self.coldFiles(under: root)
        let first = try await collect(engine, turn1)
        await engine.drainPromptCacheWrites()
        let turn1Writes = Self.writes(since: files, under: root)
        files = Self.coldFiles(under: root)
        let turn1Resident = await engine.promptCacheResidentCount
        let warm = try await collect(engine, turn2)
        await engine.drainPromptCacheWrites()
        let turn2Writes = Self.writes(since: files, under: root)
        let turn2Resident = await engine.promptCacheResidentCount

        let label = "[history-boundary] \(model.id) SSD tier:"
        print("\(label) turn 1 \(Self.describe(first)); boundary \(boundary); writes \(turn1Writes) bytes; hot entries \(turn1Resident)")
        print("\(label) turn 2 \(Self.describe(warm)); writes \(turn2Writes) bytes; hot entries \(turn2Resident)")
        XCTAssertEqual(turn1Writes.count, 1, "turn 1 writes the snapshot once")
        XCTAssertGreaterThan(
            turn1Writes.first ?? 0, PromptCacheConfig.defaultHotBytes, "the snapshot is larger than the in-memory tier")
        XCTAssertEqual(turn1Resident, 0, "the snapshot never enters the in-memory tier")
        XCTAssertEqual(warm.cachedPromptTokens, boundary, "turn 2 reuses the snapshot from the SSD tier")
        XCTAssertLessThanOrEqual(turn2Writes.count, 1, "turn 2 writes at most its own entry")
    }

    /// What the one render the snapshot adds costs at an agent session's
    /// size: the chat of a ~17k-token prompt rendered without the generation
    /// prompt, the way the engine renders it. A measurement; it asserts only
    /// that the prompt reached the size.
    func testTheHistoryRenderAtAgentSize() async throws {
        let model = try modelDirectory()
        let loaded = try await HuggingFaceTokenizerLoader().load(from: model.url)
        let tokenizer = try XCTUnwrap(loaded as? TokenizerBridge)
        let request = turns(modelID: model.id, logprobs: false, sections: 108).turn2
        var seconds: [Double] = []
        var count = 0
        for _ in 0..<5 {
            let start = Date()
            let input = Self.userInput(request)
            let messages = DefaultMessageGenerator().generate(from: input)
            count = try tokenizer.applyChatTemplate(
                messages: messages, tools: input.tools, additionalContext: nil, addGenerationPrompt: false
            ).count
            seconds.append(Date().timeIntervalSince(start))
        }
        let sorted = seconds.sorted()
        let timing = String(format: "median %.1f ms, min %.1f ms", sorted[sorted.count / 2] * 1000, sorted[0] * 1000)
        print("[history-boundary] \(model.id) render without the generation prompt: \(count) tokens, \(timing)")
        XCTAssertGreaterThan(count, 16_000)
    }

    // MARK: - Cold-tier files

    private struct ColdFile {
        let size: Int
        let modified: Date
    }

    /// The cold tier's safetensors files under `root`, by path.
    private static func coldFiles(under root: URL) -> [String: ColdFile] {
        var files: [String: ColdFile] = [:]
        let keys: [URLResourceKey] = [.fileSizeKey, .contentModificationDateKey]
        guard let walker = FileManager.default.enumerator(at: root, includingPropertiesForKeys: keys) else {
            return files
        }
        for case let url as URL in walker where url.pathExtension == "safetensors" {
            let values = try? url.resourceValues(forKeys: Set(keys))
            files[url.path] = ColdFile(
                size: values?.fileSize ?? 0, modified: values?.contentModificationDate ?? .distantPast)
        }
        return files
    }

    /// Sizes of the files written under `root` since `before`: new ones, and
    /// ones whose modification time moved; largest first.
    private static func writes(since before: [String: ColdFile], under root: URL) -> [Int] {
        coldFiles(under: root).compactMap { path, now in
            guard let old = before[path] else { return now.size }
            return old.modified == now.modified ? nil : now.size
        }.sorted(by: >)
    }
}
