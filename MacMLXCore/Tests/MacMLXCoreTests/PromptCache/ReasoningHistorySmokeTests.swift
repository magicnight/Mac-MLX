// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLXLMCommon
import XCTest

@testable import MacMLXCore
// MLXLMCommon has a `JSONValue` of its own; the scoped import makes ours win
// (see the same note in PromptCacheStoreTests).
import enum MacMLXCore.JSONValue

/// Gated, real-weights checks that a replayed assistant turn's reasoning
/// reaches the chat template, and what that does to prompt-cache reuse.
///
/// The conversation is an agent tool loop: a long system prompt, a `Bash`
/// tool, a user query, two tool calls whose long results each clear
/// ``HistoryBoundary/minimumBoundarySnapshotTokens``, a final answer, then a
/// new user query. Every assistant turn carries reasoning, as a client
/// replays it.
///
/// What it asserts:
/// - the engine's prompt for the loop's second turn (`countPromptTokens`, and
///   the `promptTokens` generation reports) is the template's render of the
///   conversation with `reasoning_content` set, which contains the reasoning
///   (decoded from the tokens), and is longer than the same turn without
///   reasoning by about the reasoning's own length;
/// - without reasoning the prompt is the `.chat` render the engine always
///   built, token for token;
/// - inside the loop the prompt stays append-only: turns 2 and 3 reuse at
///   least the previous turn's history boundary.
///
/// What it prints: each turn's reuse, and at the new query (turn 4) the reuse
/// with and without `preserve_thinking` in the template kwargs. Qwen3.6 and
/// Mellum2 render reasoning only for the turns after the last user query, so
/// at a new query the loop's turns re-render without it and reuse falls back
/// to the end of the previous query; Qwen3.6's `preserve_thinking: true`
/// renders every turn's reasoning, keeping the render append-only across
/// queries. Mellum2's template has no such switch.
///
/// Every engine here keeps its cold tier off, under a temporary directory,
/// never `~/.mac-mlx/kv-cache`.
///
/// GATED — never runs in CI. Self-skips unless ALL hold:
///   1. `requireMLXRuntimeOrSkip()` passes (real Metal, i.e. xcodebuild),
///   2. env `MACMLX_RUN_REASONING_HISTORY_SMOKE=1`, and
///   3. the model directory exists (env `MACMLX_REASONING_HISTORY_MODEL`
///      overrides the directory name under `~/.mac-mlx/models`; default
///      `Qwen3.6-27B-4bit`).
///
/// Run:
///   TEST_RUNNER_MACMLX_RUN_REASONING_HISTORY_SMOKE=1 \
///     xcodebuild test -scheme MacMLXCore -destination 'platform=macOS' \
///     -skipPackagePluginValidation \
///     -only-testing:MacMLXCoreTests/ReasoningHistorySmokeTests
/// and, for Mellum2, add
///   TEST_RUNNER_MACMLX_REASONING_HISTORY_MODEL=Mellum2-12B-A2.5B-Thinking-4bit
final class ReasoningHistorySmokeTests: XCTestCase {

    private struct Collected {
        var promptTokens: Int?
        var cachedPromptTokens: Int?
    }

    private func collect(_ engine: MLXSwiftEngine, _ request: GenerateRequest) async throws -> Collected {
        var out = Collected()
        for try await chunk in engine.generate(request) {
            if let usage = chunk.usage {
                out.promptTokens = usage.promptTokens
                out.cachedPromptTokens = usage.cachedPromptTokens
            }
        }
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

    /// About 1,600 tokens: clears the snapshot minimum and Mellum2's
    /// 1,024-token sliding window.
    private static let systemPrompt = (1...10).map { section in
        "Section \(section).\n" + rules.map { "- \($0)" }.joined(separator: "\n")
    }.joined(separator: "\n\n")

    private static let bashTool: JSONValue = .object([
        "type": .string("function"),
        "function": .object([
            "name": .string("Bash"),
            "description": .string("Run a shell command and return its output."),
            "parameters": .object([
                "type": .string("object"),
                "properties": .object([
                    "command": .object(["type": .string("string"), "description": .string("The command to run.")])
                ]),
                "required": .array([.string("command")]),
            ]),
        ]),
    ])

    /// A directory listing of `count` lines, about 25 tokens each.
    private static func listing(_ directory: String, count: Int) -> String {
        (1...count).map { n in
            "-rw-r--r--  1 kevin  staff  \(1_000 + n * 37) Oct 30 12:\(String(format: "%02d", n % 60)) "
                + "\(directory)/Source\(n).swift"
        }.joined(separator: "\n")
    }

    private static let reasoning1 =
        "The user wants to know which Swift files are in Sources. I should list the directory with ls -la "
        + "before saying anything about it, then count the files in the output."
    private static let reasoning2 =
        "The listing shows twenty files. The user also mentioned tests, so I should list Tests the same "
        + "way to compare the two directories."
    private static let reasoning3 = "Both listings are in. Sources has twenty files and Tests has twenty; I can answer."

    private static func call(_ id: String, _ command: String) -> ToolCallRequest {
        ToolCallRequest(id: id, name: "Bash", arguments: ["command": .string(command)])
    }

    /// The loop's messages, turn by turn: turn `n` sends the first `n`
    /// entries' messages. `withReasoning` false replays no reasoning.
    private static func conversation(withReasoning: Bool) -> [[ChatMessage]] {
        func reasoning(_ text: String) -> String? { withReasoning ? text : nil }
        return [
            [ChatMessage(role: .user, content: "Which Swift files are in Sources and Tests?")],
            [
                ChatMessage(role: .assistant, content: "", toolCalls: [call("call_1", "ls -la Sources")],
                            reasoning: reasoning(reasoning1)),
                ChatMessage(role: .tool, content: listing("Sources", count: 20), toolCallID: "call_1"),
            ],
            [
                ChatMessage(role: .assistant, content: "", toolCalls: [call("call_2", "ls -la Tests")],
                            reasoning: reasoning(reasoning2)),
                ChatMessage(role: .tool, content: listing("Tests", count: 20), toolCallID: "call_2"),
            ],
            [
                ChatMessage(role: .assistant, content: "Sources and Tests have twenty Swift files each.",
                            reasoning: reasoning(reasoning3)),
                ChatMessage(role: .user, content: "Thanks. Now how many lines are in Source1.swift?"),
            ],
        ]
    }

    private func request(
        _ modelID: String, turn: Int, withReasoning: Bool = true, preserveThinking: Bool = false
    ) -> GenerateRequest {
        let messages = Self.conversation(withReasoning: withReasoning).prefix(turn).flatMap { $0 }
        return GenerateRequest(
            model: modelID, messages: Array(messages), systemPrompt: Self.systemPrompt,
            parameters: GenerationParameters(temperature: 0, topP: 1.0, maxTokens: 4, stream: true),
            templateKwargs: preserveThinking ? ["preserve_thinking": .bool(true)] : nil,
            tools: [Self.bashTool])
    }

    private func modelDirectory() throws -> (id: String, url: URL) {
        try requireMLXRuntimeOrSkip()
        guard ProcessInfo.processInfo.environment["MACMLX_RUN_REASONING_HISTORY_SMOKE"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_REASONING_HISTORY_SMOKE=1 to run the reasoning-history smoke")
        }
        let modelID = ProcessInfo.processInfo.environment["MACMLX_REASONING_HISTORY_MODEL"] ?? "Qwen3.6-27B-4bit"
        let directory = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appending(path: ".mac-mlx/models/\(modelID)", directoryHint: .isDirectory)
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw XCTSkip("Model dir not found: \(directory.path)")
        }
        return (modelID, directory)
    }

    private func loadedEngine(_ model: (id: String, url: URL), coldEnabled: Bool = false) async throws -> MLXSwiftEngine {
        let coldRoot = FileManager.default.temporaryDirectory
            .appending(path: "reasoning-history-\(UUID().uuidString)", directoryHint: .isDirectory)
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

    /// `request` rendered by the model's tokenizer, with and without the
    /// generation prompt, from `input`.
    private func render(
        _ input: UserInput, tokenizer: TokenizerBridge, generationPrompt: Bool = true
    ) throws -> [Int] {
        let messages = DefaultMessageGenerator().generate(from: input)
        return try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext,
            addGenerationPrompt: generationPrompt)
    }

    /// The engine's builder, as generation calls it on an LLM.
    private func engineInput(_ request: GenerateRequest) -> UserInput {
        MLXSwiftEngine.userInput(
            for: request, isVLM: false, messageGenerator: DefaultMessageGenerator(), logsDroppedInputs: false)
    }

    /// The `.chat` input the engine built before reasoning was rendered.
    private func chatInput(_ request: GenerateRequest) -> UserInput {
        UserInput(
            chat: request.allMessages.map { MLXSwiftEngine.upstreamChatMessage(from: $0, images: []) },
            tools: request.tools.map { MLXSwiftEngine.toolSpecs(from: $0) },
            additionalContext: request.templateKwargs?.mapValues { $0.toSendable() })
    }

    /// The loop's four turns on `engine`, from an empty cache: each turn's
    /// reuse and history boundary.
    private func runLoop(
        _ engine: MLXSwiftEngine, _ modelID: String, tokenizer: TokenizerBridge, preserveThinking: Bool,
        label: String
    ) async throws -> (reused: [Int], boundaries: [Int]) {
        await engine.clearPromptCache()
        var boundaries: [Int] = []
        var reused: [Int] = []
        for turn in 1...4 {
            let turnRequest = request(modelID, turn: turn, preserveThinking: preserveThinking)
            let history = try render(engineInput(turnRequest), tokenizer: tokenizer, generationPrompt: false)
            let run = try await collect(engine, turnRequest)
            await engine.drainPromptCacheWrites()
            if turn == 2 {
                let expected = try await engine.countPromptTokens(turnRequest)
                XCTAssertEqual(run.promptTokens, expected, "generation prefills the counted prompt")
            }
            boundaries.append(history.count)
            reused.append(run.cachedPromptTokens ?? -1)
            print("\(label) turn \(turn): prompt=\(run.promptTokens ?? -1) "
                + "reused=\(run.cachedPromptTokens ?? -1) history boundary=\(history.count)")
        }
        XCTAssertEqual(reused[0], 0, "\(label) turn 1 starts cold")
        XCTAssertGreaterThanOrEqual(reused[1], boundaries[0], "\(label) turn 2 reuses turn 1's history boundary")
        XCTAssertGreaterThanOrEqual(reused[2], boundaries[1], "\(label) turn 3 reuses turn 2's history boundary")
        print("\(label) at the new query turn 4 reused \(reused[3]) (turn 3's boundary \(boundaries[2]), "
            + "turn 1's \(boundaries[0]))")
        return (reused, boundaries)
    }

    /// Whether the model's template has Qwen3.6's `preserve_thinking` switch.
    private func templateHasPreserveThinking(_ model: (id: String, url: URL)) -> Bool {
        let template = try? String(contentsOf: model.url.appending(path: "chat_template.jinja"), encoding: .utf8)
        return template?.contains("preserve_thinking") == true
    }

    // MARK: - Tests

    func testReplayedReasoningIsRenderedAndTheLoopStaysAppendOnly() async throws {
        let model = try modelDirectory()
        let engine = try await loadedEngine(model)
        let loaded = try await HuggingFaceTokenizerLoader().load(from: model.url)
        let tokenizer = try XCTUnwrap(loaded as? TokenizerBridge)
        let label = "[reasoning-history] \(model.id):"

        // The render of the loop's second turn.
        let turn2 = request(model.id, turn: 2)
        let turn2Plain = request(model.id, turn: 2, withReasoning: false)
        let withTokens = try render(engineInput(turn2), tokenizer: tokenizer)
        let withoutTokens = try render(engineInput(turn2Plain), tokenizer: tokenizer)
        let renderedText = tokenizer.decode(tokenIds: withTokens, skipSpecialTokens: false)
        let reasoningTokens = tokenizer.encode(text: Self.reasoning1, addSpecialTokens: false).count
        let counted = try await engine.countPromptTokens(turn2)
        let countedPlain = try await engine.countPromptTokens(turn2Plain)
        print("\(label) turn 2 prompt \(withTokens.count) tokens with reasoning, \(withoutTokens.count) without; "
            + "the reasoning alone is \(reasoningTokens)")

        XCTAssertTrue(renderedText.contains(Self.reasoning1), "the replayed reasoning reaches the template")
        XCTAssertFalse(
            tokenizer.decode(tokenIds: withoutTokens, skipSpecialTokens: false).contains(Self.reasoning1),
            "and is absent without it")
        XCTAssertEqual(counted, withTokens.count, "the engine counts the render with reasoning")
        XCTAssertEqual(countedPlain, withoutTokens.count)
        // Plus the think tags around it on a template (Mellum2's) that
        // renders none for a turn without reasoning; Qwen3.6 renders empty
        // ones either way. A token either side for the merge at its edges.
        let growth = counted - countedPlain
        XCTAssertGreaterThanOrEqual(growth, reasoningTokens - 1, "the prompt grows by about the reasoning's length")
        XCTAssertLessThanOrEqual(growth, reasoningTokens + 8, "the prompt grows by about the reasoning's length")
        let chatTokens = try render(chatInput(turn2Plain), tokenizer: tokenizer)
        XCTAssertEqual(withoutTokens, chatTokens, "without reasoning the prompt is the .chat render, token for token")

        // Prompt-cache reuse across the loop, then at a new query, with
        // `preserve_thinking` off and on. The cold tier is off: only the
        // in-memory tier serves.
        _ = try await runLoop(
            engine, model.id, tokenizer: tokenizer, preserveThinking: false,
            label: "\(label) in-memory tier, preserve_thinking=false")
        let preserved = try await runLoop(
            engine, model.id, tokenizer: tokenizer, preserveThinking: true,
            label: "\(label) in-memory tier, preserve_thinking=true")
        if templateHasPreserveThinking(model) {
            XCTAssertGreaterThanOrEqual(
                preserved.reused[3], preserved.boundaries[2],
                "with preserve_thinking the new query reuses the loop's last history boundary")
        }
    }

    /// The new query with the SSD tier on (the default), under a temporary
    /// root: the earlier boundary snapshots the in-memory tier evicted are
    /// still on disk.
    func testANewQueryWithTheSSDTier() async throws {
        let model = try modelDirectory()
        let engine = try await loadedEngine(model, coldEnabled: true)
        let loaded = try await HuggingFaceTokenizerLoader().load(from: model.url)
        let tokenizer = try XCTUnwrap(loaded as? TokenizerBridge)
        let label = "[reasoning-history] \(model.id):"
        _ = try await runLoop(
            engine, model.id, tokenizer: tokenizer, preserveThinking: false,
            label: "\(label) SSD tier, preserve_thinking=false")
    }
}
