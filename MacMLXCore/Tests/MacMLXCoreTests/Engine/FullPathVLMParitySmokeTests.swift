// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MacMLXCore
// MLXLMCommon has a `JSONValue` of its own; the scoped import makes ours win
// (see the same note in PromptCacheStoreTests).
import enum MacMLXCore.JSONValue

/// Gated, real-weights parity gate a family passes before it joins
/// ``FullPathVLM/servedFamilies``: the same checkpoint loaded as text
/// (`.mlx`, `LLMModelFactory`) and on the full path (`.mlxVLM`, the VLM
/// factory served on the text generation path) must serve text requests
/// alike. Gates 1, 2, 3, 6 and 7 of
/// `docs/superpowers/specs/2026-10-31-gemma4-qwen35-vision-design.md`; the
/// image gates are in `FullPathVLMImageGateSmokeTests`.
///
/// - Gate 1, same tokens: each agent-shaped prompt (a system turn, tools,
///   tool calls, tool results, replayed reasoning, over 256 tokens) renders
///   to the same token ids on both loads.
/// - Gate 2, same distribution: on those prompts, over the text load's top
///   20 next tokens, in float32 log-softmax, the largest |Δ log p| is within
///   δ = 4 bf16 ULPs of the text load's top-1 logit, the KL over those
///   tokens is below δ²/2, and a greedy 64-token continuation is
///   token-identical up to the first position where the text load's top-2
///   margin is below δ.
/// - Gate 3, cache: the second turn of an agent conversation reuses the
///   same number of tokens on both loads.
/// - Gate 6, memory: the full-path load is resident with the vision tower's
///   bytes more than the text load, within ``memoryTolerance``.
/// - Gate 7, speculation: with the checkpoint itself as a draft (loaded
///   through `LLMModelFactory`), a text request on the full-path load still
///   speculates, and its greedy output is the one the load gives without a
///   draft.
///
/// The full-path load goes through the forcing seam, so the suite runs for a
/// family before it flips. Every engine keeps its cold tier off. Observed
/// values are printed with their bounds.
///
/// GATED — never runs in CI. Self-skips unless ALL hold:
///   1. `requireMLXRuntimeOrSkip()` passes (real Metal, i.e. xcodebuild),
///   2. env `MACMLX_RUN_FULL_PATH_PARITY=1`, and
///   3. the model directory exists (env `MACMLX_FULL_PATH_PARITY_MODEL`
///      overrides the directory name under `~/.mac-mlx/models`; default
///      `gemma-4-e4b-it-8bit`).
///
/// Run:
///   MACMLX_RUN_FULL_PATH_PARITY=1 TEST_RUNNER_MACMLX_RUN_FULL_PATH_PARITY=1 \
///     xcodebuild test -scheme MacMLXCore -destination 'platform=macOS' \
///     -skipPackagePluginValidation \
///     -only-testing:MacMLXCoreTests/FullPathVLMParitySmokeTests
final class FullPathVLMParitySmokeTests: XCTestCase {

    /// The other chunking the noise-floor read prefills with.
    static let noiseFloorPrefill = PrefillParameters(stepSize: 128, chunking: .remainder)

    /// Gate 6's allowance for what is resident besides the weights.
    static let memoryTolerance = 32 << 20

    // MARK: - Gates 1 and 2

    func testAgentPromptsRenderAlikeAndPredictAlike() async throws {
        let (modelID, directory) = try gateAndResolveModel()
        let text = try await loadedEngine(modelID, directory, fullPath: false)
        let full = try await loadedEngine(modelID, directory, fullPath: true)

        for (name, request) in Self.agentPrompts(modelID) {
            let textRun = try await text.coldNextTokenLogits(request)
            let fullRun = try await full.coldNextTokenLogits(request)
            XCTAssertGreaterThan(textRun.tokens.count, 256, "\(name): the prompt is long enough to snapshot")
            XCTAssertEqual(textRun.tokens, fullRun.tokens, "\(name): gate 1, the loads render different tokens")

            let comparison = try XCTUnwrap(
                LogitComparison(reference: textRun.logits, candidate: fullRun.logits, top: 20))
            print("[full-path-parity] \(modelID) \(name): prompt=\(textRun.tokens.count) \(comparison)")
            // For the record, not asserted: the text load against itself
            // prefilled in other chunks, the size of the difference chunking
            // alone makes on this checkpoint.
            let rechunked = try await text.coldNextTokenLogits(request, prefill: Self.noiseFloorPrefill)
            if let noise = LogitComparison(reference: textRun.logits, candidate: rechunked.logits, top: 20) {
                print("[full-path-parity] \(modelID) \(name): text load rechunked \(noise)")
            }
            XCTAssertLessThanOrEqual(comparison.maxLogProbDelta, comparison.delta, "\(name): gate 2, |Δ log p|")
            XCTAssertLessThan(comparison.kl, comparison.klBound, "\(name): gate 2, KL")

            var greedy = request
            greedy.parameters.maxTokens = 64
            greedy.parameters.logprobs = true
            greedy.parameters.topLogprobs = 2
            await text.clearPromptCache()
            await full.clearPromptCache()
            let textTokens = try await logprobs(text, greedy)
            let fullTokens = try await logprobs(full, greedy)
            let agreed = Self.agreedPrefix(reference: textTokens, candidate: fullTokens, delta: comparison.delta)
            print("[full-path-parity] \(modelID) \(name): greedy \(agreed)")
            XCTAssertNil(agreed.mismatch, "\(name): gate 2, greedy diverges before a near-tie: \(agreed)")
        }
    }

    // MARK: - Classification

    /// The model library makes the gated checkpoint what a user gets: a VLM
    /// once its family is served (read-only scan of its models directory).
    func testTheModelLibraryClassifiesTheCheckpoint() async throws {
        let (modelID, directory) = try gateAndResolveModel()
        let models = try await ModelLibraryManager().scan(directory.deletingLastPathComponent())
        let model = try XCTUnwrap(models.first { $0.id == modelID })
        let config = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appending(path: "config.json")))
                as? [String: Any])
        let family = try XCTUnwrap(FullPathVLM.Family(rawValue: model.architecture?.lowercased() ?? ""))
        print("[full-path-parity] \(modelID) library format: \(model.format)")
        if ProcessInfo.processInfo.environment["MACMLX_FULL_PATH_PARITY_MODEL"] == nil {
            // The checkpoint the gate ran on, asserted outright so a filter
            // that wrongly excluded it would fail here.
            XCTAssertEqual(model.format, .mlxVLM)
        } else {
            let served = FullPathVLM.servedFamilies.contains(family) && family.parityRunCovers(config)
            XCTAssertEqual(model.format, served ? .mlxVLM : .mlx)
        }
    }

    // MARK: - Gate 3

    func testTheSecondAgentTurnReusesTheSameOnBothLoads() async throws {
        let (modelID, directory) = try gateAndResolveModel()
        var reused: [Bool: Int] = [:]
        for fullPath in [false, true] {
            let engine = try await loadedEngine(modelID, directory, fullPath: fullPath)
            let (turn1, turn2) = Self.twoTurns(modelID)
            _ = try await usage(engine, turn1)
            let second = try await usage(engine, turn2)
            reused[fullPath] = second.cachedPromptTokens ?? 0
            print("[full-path-parity] \(modelID) \(fullPath ? "full path" : "text load"): turn 2 "
                + "prompt=\(second.promptTokens) reused=\(second.cachedPromptTokens ?? 0)")
            try await engine.unload()
        }
        XCTAssertGreaterThan(reused[false] ?? 0, 0, "the text load reuses turn 1")
        XCTAssertEqual(reused[true], reused[false], "gate 3: the loads reuse different amounts")
    }

    // MARK: - Gate 6

    func testTheFullPathHoldsOnlyTheVisionTowerMore() async throws {
        let (modelID, directory) = try gateAndResolveModel()
        let vision = try Self.visionTowerBytes(directory)
        var resident: [Bool: Int] = [:]
        for fullPath in [false, true] {
            Memory.clearCache()
            let before = Memory.activeMemory
            let engine = try await loadedEngine(modelID, directory, fullPath: fullPath)
            resident[fullPath] = Memory.activeMemory - before
            try await engine.unload()
            Memory.clearCache()
        }
        let textBytes = try XCTUnwrap(resident[false])
        let fullBytes = try XCTUnwrap(resident[true])
        let growth = fullBytes - textBytes
        print("[full-path-parity] \(modelID) memory: text=\(textBytes) B full=\(fullBytes) B "
            + "growth=\(growth) B vision tower=\(vision) B tolerance=\(Self.memoryTolerance) B")
        XCTAssertGreaterThan(vision, 0, "the checkpoint holds a vision tower")
        XCTAssertLessThanOrEqual(abs(growth - vision), Self.memoryTolerance, "gate 6: growth is not the vision tower")
    }

    // MARK: - Gate 7

    /// Run on the text load too, as the reference for what speculation does
    /// there; the gate is on the full path. A family whose text load cannot
    /// speculate (Qwen3.5's `MambaCache` cannot be trimmed) must not start
    /// to on the full path; one that can must keep doing so.
    func testTheFullPathStillSpeculatesWithADraft() async throws {
        let (modelID, directory) = try gateAndResolveModel()
        var plain = GenerateRequest(
            model: modelID,
            messages: [ChatMessage(role: .user, content: "Write four short lines about a lighthouse at night.")],
            parameters: GenerationParameters(temperature: 0, topP: 1.0, maxTokens: 64, stream: true))
        plain.templateKwargs = nil
        var drafted = plain
        drafted.draftModelID = modelID

        var textLoadSpeculates = false
        for fullPath in [false, true] {
            let engine = try await loadedEngine(modelID, directory, fullPath: fullPath)
            let without = try await textAndUsage(engine, plain)
            await engine.clearPromptCache()
            let with = try await textAndUsage(engine, drafted)
            let speculation = with.speculation
            let load = fullPath ? "full path" : "text load"
            print("[full-path-parity] \(modelID) \(load) speculation: proposed=\(speculation?.proposedTokens ?? 0) "
                + "accepted=\(speculation?.acceptedTokens ?? 0); same text: \(with.text == without.text); "
                + "with draft \(with.text.prefix(24).debugDescription), without \(without.text.prefix(24).debugDescription)")
            try await engine.unload()
            guard fullPath else {
                textLoadSpeculates = (speculation?.proposedTokens ?? 0) > 0
                continue
            }
            XCTAssertEqual(
                (speculation?.proposedTokens ?? 0) > 0, textLoadSpeculates,
                "gate 7: the full path speculates where the text load does not, or the reverse")
            // Gemma 4's caches trim, so its text load must speculate: parity
            // alone would pass if both loads had stopped.
            if Self.modelType(directory) == FullPathVLM.Family.gemma4.rawValue {
                XCTAssertTrue(textLoadSpeculates, "gate 7: the Gemma 4 text load did not speculate")
            }
            XCTAssertEqual(with.text, without.text, "gate 7: greedy output changes with a draft")
        }
    }

    // MARK: - Prompts

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

    /// About 1,600 tokens: clears the snapshot minimum and Gemma 4's
    /// 512-token sliding window.
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

    private static func call(_ id: String, _ command: String) -> ToolCallRequest {
        ToolCallRequest(id: id, name: "Bash", arguments: ["command": .string(command)])
    }

    private static func listing(_ directory: String, count: Int) -> String {
        (1...count).map { n in
            "-rw-r--r--  1 kevin  staff  \(1_000 + n * 37) Oct 30 12:\(String(format: "%02d", n % 60)) "
                + "\(directory)/Source\(n).swift"
        }.joined(separator: "\n")
    }

    private static func agentRequest(_ modelID: String, _ messages: [ChatMessage]) -> GenerateRequest {
        GenerateRequest(
            model: modelID, messages: messages, systemPrompt: systemPrompt,
            parameters: GenerationParameters(temperature: 0, topP: 1.0, maxTokens: 1, stream: true),
            tools: [bashTool])
    }

    /// Three points of one agent loop: the opening question, the turn after a
    /// tool result, and a follow-up after two rounds of tool calls with
    /// replayed reasoning.
    static func agentPrompts(_ modelID: String) -> [(String, GenerateRequest)] {
        let question = ChatMessage(role: .user, content: "Which Swift files are in Sources and Tests?")
        let round1: [ChatMessage] = [
            ChatMessage(
                role: .assistant, content: "", toolCalls: [call("call_1", "ls -la Sources")],
                reasoning: "The user wants the Swift files in Sources. I should list the directory first."),
            ChatMessage(role: .tool, content: listing("Sources", count: 20), toolCallID: "call_1"),
        ]
        let round2: [ChatMessage] = [
            ChatMessage(
                role: .assistant, content: "", toolCalls: [call("call_2", "ls -la Tests")],
                reasoning: "Twenty files in Sources. The user also asked about Tests, so I list that too."),
            ChatMessage(role: .tool, content: listing("Tests", count: 20), toolCallID: "call_2"),
            ChatMessage(
                role: .assistant, content: "Sources and Tests have twenty Swift files each.",
                reasoning: "Both listings are in, twenty files each; I can answer."),
            ChatMessage(role: .user, content: "Thanks. Now how many lines are in Source1.swift?"),
        ]
        return [
            ("opening", agentRequest(modelID, [question])),
            ("after a tool result", agentRequest(modelID, [question] + round1)),
            ("follow-up with reasoning", agentRequest(modelID, [question] + round1 + round2)),
        ]
    }

    /// An agent turn that calls a tool, and the turn after its result.
    static func twoTurns(_ modelID: String) -> (GenerateRequest, GenerateRequest) {
        let question = ChatMessage(role: .user, content: "Run the shell command: echo hi")
        var turn1 = agentRequest(modelID, [question])
        turn1.parameters.maxTokens = 4
        var turn2 = agentRequest(modelID, [
            question,
            ChatMessage(role: .assistant, content: "", toolCalls: [call("call_1", "echo hi")]),
            ChatMessage(role: .tool, content: "hi", toolCallID: "call_1"),
        ])
        turn2.parameters.maxTokens = 4
        return (turn1, turn2)
    }

    // MARK: - Comparison

    /// Gate 2's comparison of two next-token distributions over the
    /// reference's top tokens.
    struct LogitComparison: CustomStringConvertible {
        /// 4 bf16 ULPs of the reference's top-1 logit.
        let delta: Double
        let maxLogProbDelta: Double
        let kl: Double
        var klBound: Double { delta * delta / 2 }

        init?(reference: [Float], candidate: [Float], top: Int) {
            guard reference.count == candidate.count, reference.count >= top,
                let top1 = reference.max(), top1 != 0
            else { return nil }
            delta = 4 * Self.bf16ULP(top1)
            let indices = reference.indices.sorted { reference[$0] > reference[$1] }.prefix(top)
            let p = Self.logSoftmax(reference)
            let q = Self.logSoftmax(candidate)
            maxLogProbDelta = indices.map { abs(p[$0] - q[$0]) }.max() ?? 0
            // Both distributions renormalized over the reference's top tokens.
            let pTop = indices.map { p[$0] }
            let qTop = indices.map { q[$0] }
            let pNorm = Self.logSumExp(pTop)
            let qNorm = Self.logSumExp(qTop)
            kl = zip(pTop, qTop).reduce(0) { sum, pair in
                let lp = pair.0 - pNorm
                return sum + exp(lp) * (lp - (pair.1 - qNorm))
            }
        }

        var description: String {
            String(format: "δ=%.4f max|Δlogp|=%.5f KL=%.2e (bound %.2e)", delta, maxLogProbDelta, kl, klBound)
        }

        /// The spacing of bfloat16 values around `x` (8 significand bits).
        static func bf16ULP(_ x: Float) -> Double {
            pow(2, floor(log2(Double(abs(x)))) - 7)
        }

        static func logSoftmax(_ logits: [Float]) -> [Double] {
            let values = logits.map(Double.init)
            let norm = logSumExp(values)
            return values.map { $0 - norm }
        }

        static func logSumExp(_ values: [Double]) -> Double {
            guard let top = values.max() else { return -.infinity }
            return top + log(values.reduce(0) { $0 + exp($1 - top) })
        }
    }

    /// How far two greedy runs agree: positions compared until the reference's
    /// top-2 margin falls below `delta`, and the first position they differ
    /// at before that, if any.
    struct AgreedPrefix: CustomStringConvertible {
        var compared = 0
        var stoppedAtNearTie: Int?
        var mismatch: Int?
        var lengths: (Int, Int) = (0, 0)
        var description: String {
            "compared=\(compared) near-tie stop=\(stoppedAtNearTie.map(String.init) ?? "none") "
                + "mismatch=\(mismatch.map(String.init) ?? "none") lengths=\(lengths.0)/\(lengths.1)"
        }
    }

    static func agreedPrefix(reference: [TokenLogprob], candidate: [TokenLogprob], delta: Double) -> AgreedPrefix {
        var result = AgreedPrefix(lengths: (reference.count, candidate.count))
        for (position, token) in reference.enumerated() {
            let alternatives = token.topLogprobs
            if alternatives.count >= 2, Double(alternatives[0].logprob - alternatives[1].logprob) < delta {
                result.stoppedAtNearTie = position
                return result
            }
            guard position < candidate.count, candidate[position].token == token.token else {
                result.mismatch = position
                return result
            }
            result.compared = position + 1
        }
        if candidate.count != reference.count { result.mismatch = reference.count }
        return result
    }

    // MARK: - Helpers

    private func gateAndResolveModel() throws -> (id: String, directory: URL) {
        try requireMLXRuntimeOrSkip()
        guard ProcessInfo.processInfo.environment["MACMLX_RUN_FULL_PATH_PARITY"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_FULL_PATH_PARITY=1 to run the full-path parity gate")
        }
        let modelID = ProcessInfo.processInfo.environment["MACMLX_FULL_PATH_PARITY_MODEL"] ?? "gemma-4-e4b-it-8bit"
        let directory = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appending(path: ".mac-mlx/models/\(modelID)", directoryHint: .isDirectory)
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw XCTSkip("Model dir not found: \(directory.path)")
        }
        return (modelID, directory)
    }

    /// An engine with `modelID` resident: as text, or on the full path
    /// through the forcing seam, checked to have taken it.
    private func loadedEngine(_ modelID: String, _ directory: URL, fullPath: Bool) async throws -> MLXSwiftEngine {
        let engine = MLXSwiftEngine(promptCache: PromptCacheConfig(coldEnabled: false))
        await engine.setForcesFullPathVLM(fullPath)
        try await engine.load(LocalModel(
            id: modelID, displayName: modelID, directory: directory, sizeBytes: 0,
            format: fullPath ? .mlxVLM : .mlx, quantization: nil, parameterCount: nil, architecture: nil))
        let served = await engine.servesFullPathVLM
        XCTAssertEqual(served, fullPath, "the load took the wrong path")
        return engine
    }

    private func logprobs(_ engine: MLXSwiftEngine, _ request: GenerateRequest) async throws -> [TokenLogprob] {
        var out: [TokenLogprob] = []
        for try await chunk in engine.generate(request) { out += chunk.logprobs ?? [] }
        return out
    }

    private func usage(_ engine: MLXSwiftEngine, _ request: GenerateRequest) async throws -> TokenUsage {
        let usage = try await textAndUsage(engine, request).usage
        return try XCTUnwrap(usage)
    }

    private func textAndUsage(
        _ engine: MLXSwiftEngine, _ request: GenerateRequest
    ) async throws -> (text: String, usage: TokenUsage?, speculation: SpeculativeDecodingUsage?) {
        var text = ""
        var usage: TokenUsage?
        var speculation: SpeculativeDecodingUsage?
        for try await chunk in engine.generate(request) {
            text += chunk.text
            if let chunkUsage = chunk.usage { usage = chunkUsage }
            if let chunkSpeculation = chunk.speculativeDecoding { speculation = chunkSpeculation }
        }
        return (text, usage, speculation)
    }

    /// The checkpoint's `model_type`, lowercased.
    static func modelType(_ directory: URL) -> String? {
        guard let data = try? Data(contentsOf: directory.appending(path: "config.json")),
            let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return (root["model_type"] as? String)?.lowercased()
    }

    /// The bytes of the vision tower's tensors (`vision_tower.*` and
    /// `embed_vision.*`, after a leading `model.`), read from the
    /// `.safetensors` headers' data offsets.
    static func visionTowerBytes(_ directory: URL) throws -> Int {
        let shards = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
        var total = 0
        for shard in shards {
            let handle = try FileHandle(forReadingFrom: shard.resolvingSymlinksInPath())
            defer { try? handle.close() }
            let prefix = try XCTUnwrap(try handle.read(upToCount: 8))
            let length = prefix.reversed().reduce(0) { $0 << 8 | Int($1) }
            let body = try XCTUnwrap(try handle.read(upToCount: length))
            let header = try XCTUnwrap(try JSONSerialization.jsonObject(with: body) as? [String: Any])
            for (name, value) in header {
                let bare = name.hasPrefix("model.") ? String(name.dropFirst("model.".count)) : name
                guard bare.hasPrefix("vision_tower.") || bare.hasPrefix("embed_vision."),
                    let entry = value as? [String: Any], let offsets = entry["data_offsets"] as? [Int],
                    offsets.count == 2
                else { continue }
                total += offsets[1] - offsets[0]
            }
        }
        return total
    }
}
