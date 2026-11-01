// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import Testing

@testable import MacMLXCore

/// The engine seam that serves a VLM on the text generation path
/// (``FullPathVLM``): which checkpoints it applies to, what it renders with,
/// the state a warm continuation starts from, the input shape the text
/// path slices, and where an image request's reuse stops.
@Suite("FullPathVLM")
struct FullPathVLMTests {

    private func config(_ json: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appending(path: "full-path-vlm-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appending(path: "config.json", directoryHint: .notDirectory)
        try Data(json.utf8).write(to: url)
        return url
    }

    @Test(arguments: FullPathVLM.Family.allCases)
    func aKnownFamilyIsServedWhenForcedOrOnceItIsInTheSet(_ family: FullPathVLM.Family) throws {
        let url = try config(#"{"model_type": "\#(family.rawValue)", "vision_config": {}}"#)
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        #expect(FullPathVLM.resolve(configURL: url, forced: true) == FullPathVLM(family: family))
        #expect(
            (FullPathVLM.resolve(configURL: url, forced: false) != nil)
                == FullPathVLM.servedFamilies.contains(family),
            "unforced, only a served family takes the full path")
    }

    /// Gemma 4 passed its parity gate on `gemma-4-e4b-it-8bit`, Qwen3.5/3.6
    /// on `Qwen3.6-27B-4bit` and `Qwen3.5-4B-MLX-8bit`. A family joins with
    /// its own flip.
    @Test
    func bothFamiliesAreServed() {
        #expect(FullPathVLM.servedFamilies == [.gemma4, .qwen35])
    }

    /// Each family's `processorClass` is the key the VLM factory's processor
    /// registry builds that family's processor under, in the source this
    /// build resolved, and no `model_type` resolver in the loading registry
    /// overrides what a Gemma 4 or Qwen3.5 checkpoint declares, so the class
    /// a checkpoint declares is the processor it gets. (Upstream's Qwen3.5
    /// resolver only supplies metadata to a checkpoint that ships none, and
    /// such a checkpoint stays text.)
    @Test(.enabled(if: ResolvedUpstreamSource.mlxSwiftLM(Self.factorySource) != nil
                   || ResolvedUpstreamSource.isRequired,
                   "no SwiftPM checkout of mlx-swift-lm next to the test bundle"))
    func everyFamilyNamesTheProcessorUpstreamBuildsForIt() throws {
        let factory = try #require(ResolvedUpstreamSource.mlxSwiftLM(Self.factorySource),
                                   "MACMLX_REQUIRE_RESOLVED_SOURCE is set but no checkout was found")
        let start = try #require(factory.range(of: "public enum VLMProcessorTypeRegistry"))
        let entry = /"(\w+)"\s*:\s*create\(\s*\w+\.self\s*,\s*(\w+)\.init\s*\)/
        let processors = Dictionary(
            factory[start.upperBound...].matches(of: entry).map { (String($0.1), String($0.2)) },
            uniquingKeysWith: { first, _ in first })
        #expect(processors[FullPathVLM.Family.gemma4.processorClass] == "Gemma4Processor")
        #expect(processors[FullPathVLM.Family.qwen35.processorClass] == "Qwen3VLProcessor")

        let loading = try #require(ResolvedUpstreamSource.mlxSwiftLM(Self.loadingSource),
                                   "no VLMProcessorLoadingRegistry.swift in the checkout")
        #expect(!loading.contains(#""gemma4":"#), "a resolver now overrides Gemma 4's processor")
        #expect(!loading.contains(#""qwen3_5":"#), "a resolver now overrides Qwen3.5's processor")
        let qwen35Resolver = try #require(loading.range(of: "struct Qwen35ProcessorLoadingResolver"))
        let qwen35Body = String(loading[qwen35Resolver.upperBound...])
            .components(separatedBy: "\n}\n").first ?? ""
        #expect(!qwen35Body.contains("func processorType"), "Qwen3.5's resolver now picks the processor type")
    }

    private static let factorySource = "Libraries/MLXVLM/VLMModelFactory.swift"
    private static let loadingSource = "Libraries/MLXVLM/VLMProcessorLoadingRegistry.swift"

    /// The forcing seam reaches only the families this path knows how to
    /// drive; the siblings that share a generator but not a class, a text
    /// type and an unrelated VLM stay where they are.
    @Test(arguments: ["gemma4_unified", "qwen3_5_moe", "gemma4_text", "qwen3_5_text", "qwen3_vl", "llama"])
    func otherTypesNeverTakeTheFullPath(_ modelType: String) throws {
        let url = try config(#"{"model_type": "\#(modelType)"}"#)
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        #expect(FullPathVLM.resolve(configURL: url, forced: true) == nil)
    }

    @Test
    func theModelTypeIsReadCaseInsensitively() throws {
        let url = try config(#"{"model_type": "Gemma4"}"#)
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        #expect(FullPathVLM.resolve(configURL: url, forced: true)?.family == .gemma4)
    }

    @Test
    func anUnreadableConfigStaysOnTheStockPath() {
        let missing = URL(fileURLWithPath: "/nonexistent/\(UUID().uuidString)/config.json")
        #expect(FullPathVLM.resolve(configURL: missing, forced: true) == nil)
    }

    /// Every family is a key the VLM factory registers and macMLX knows
    /// about, so a family can never name a type nothing loads.
    @Test(arguments: FullPathVLM.Family.allCases)
    func everyFamilyIsAVLMRegistryKey(_ family: FullPathVLM.Family) {
        #expect(
            ModelLibraryManager.vlmTypesServedAsText.contains(family.rawValue)
                || ModelLibraryManager.fullPathVLMTypes.contains(family.rawValue))
    }

    /// The generator each family's processor builds for itself
    /// (`Gemma4Processor.prepare`, `Qwen3VLProcessor.prepare`), so the
    /// engine's own renders (replayed reasoning, the history boundary) match
    /// the prompt.
    @Test
    func eachFamilyRendersWithItsProcessorsGenerator() {
        #expect(FullPathVLM(family: .gemma4).processorMessageGenerator is Gemma4MessageGenerator)
        #expect(FullPathVLM(family: .qwen35).processorMessageGenerator is Qwen3VLMessageGenerator)
    }

    /// A full-path load always gets its generator's dictionaries, with or
    /// without replayed reasoning; an LLM without reasoning keeps `.chat`.
    @Test
    func aFullPathInputIsAlwaysTheGeneratorsMessages() throws {
        let generator = FullPathVLM(family: .gemma4).messageGenerator
        let plain = GenerateRequest(
            model: "m", messages: [ChatMessage(role: .user, content: "Hi.")],
            parameters: GenerationParameters())
        var reasoned = plain
        reasoned.messages = [
            ChatMessage(role: .user, content: "Hi."),
            ChatMessage(role: .assistant, content: "Hello.", reasoning: "Greet back."),
            ChatMessage(role: .user, content: "Again."),
        ]
        func messages(_ input: UserInput) -> [[String: any Sendable]]? {
            if case .messages(let messages) = input.prompt { return messages }
            return nil
        }
        let full = try #require(messages(MLXSwiftEngine.userInput(
            for: plain, isVLM: true, messageGenerator: generator, logsDroppedInputs: false)))
        #expect(full.first?["content"] as? String == "Hi.")
        let withReasoning = try #require(messages(MLXSwiftEngine.userInput(
            for: reasoned, isVLM: true, messageGenerator: generator, logsDroppedInputs: false)))
        #expect(withReasoning[1]["reasoning_content"] as? String == "Greet back.")
        let probe = MLXSwiftEngine.userInput(
            for: plain, isVLM: false, messageGenerator: generator, rendersThroughGenerator: true,
            logsDroppedInputs: false)
        #expect(messages(probe) != nil, "a caller that drops the images still renders through it")
        let llm = MLXSwiftEngine.userInput(
            for: plain, isVLM: false, messageGenerator: DefaultMessageGenerator(), logsDroppedInputs: false)
        #expect(messages(llm) == nil, "an LLM without reasoning keeps `.chat`")
    }

    /// A message without media renders as the text load renders it, string
    /// content and tool metadata, so a tool call with no text keeps empty
    /// content, which Gemma 4's template needs to keep the model turn open
    /// after the tool result. A message with an image gets the family's
    /// parts, the image placeholder among them.
    @Test(arguments: FullPathVLM.Family.allCases)
    func onlyAMessageWithMediaRendersAsParts(_ family: FullPathVLM.Family) throws {
        let generator = FullPathVLM(family: family).messageGenerator
        let call = ToolCall(
            function: .init(name: "Bash", arguments: ["command": "ls"] as [String: any Sendable]), id: "call_1")
        let messages: [Chat.Message] = [
            .system("Be brief."),
            .user("List the files."),
            .assistant("", toolCalls: [call]),
            .tool("a.swift", id: "call_1"),
            .user("What does this say?", images: [.url(URL(fileURLWithPath: "/nonexistent/a.png"))]),
        ]
        let rendered = generator.generate(messages: messages)
        let reference = DefaultMessageGenerator().generate(messages: messages)
        try #require(rendered.count == messages.count)
        for index in 0..<4 {
            #expect(rendered[index]["content"] as? String == reference[index]["content"] as? String)
            #expect(rendered[index].keys.sorted() == reference[index].keys.sorted())
        }
        #expect(rendered[2]["content"] as? String == "")
        #expect(rendered[2]["tool_calls"] != nil)
        let parts = try #require(rendered[4]["content"] as? [[String: any Sendable]])
        #expect(parts.contains { $0["type"] as? String == "image" })
    }

    /// The media ids are read with upstream's keys, so the cap on an image
    /// request's reuse lands where the model finds the block. Qwen3.6 sets a
    /// `vision_start_token_id` other than upstream's default.
    @Test
    func theMediaTokenIDsAreReadFromTheConfig() throws {
        let qwen = try config(#"""
            {"model_type": "qwen3_5", "vision_start_token_id": 248053,
             "image_token_id": 248056, "video_token_id": 248057, "image_token_index": 248060}
            """#)
        let gemma = try config(#"""
            {"model_type": "gemma4", "boi_token_id": 255999, "image_token_id": 258880,
             "video_token_id": 258884, "audio_token_id": 258881}
            """#)
        defer {
            for url in [qwen, gemma] { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        }
        #expect(FullPathVLM.resolve(configURL: qwen, forced: true)?.mediaTokenIDs == [248_053, 248_056, 248_057, 248_060])
        #expect(
            FullPathVLM.resolve(configURL: gemma, forced: true)?.mediaTokenIDs
                == [255_999, 258_880, 258_884, 258_881])
    }

    @Test
    func missingMediaKeysFallBackToUpstreamsDefaults() {
        #expect(FullPathVLM(family: .qwen35).mediaTokenIDs == [248_045, 248_056, 248_057])
        #expect(FullPathVLM(family: .gemma4).mediaTokenIDs == [255_999, 258_880])
    }

    /// Only Gemma 4's bidirectional vision attention breaks on a warm cache
    /// (upstream's overlay falls back to causal attention inside the image
    /// block), and only the value that turns it on.
    @Test(arguments: [
        ("gemma4", #""vision""#, false), ("gemma4", "null", true), ("gemma4", #""all""#, true),
        ("qwen3_5", #""vision""#, true),
    ])
    func onlyBidirectionalVisionAttentionKeepsImageRequestsCold(
        _ modelType: String, _ flag: String, _ reuses: Bool
    ) throws {
        let url = try config(#"""
            {"model_type": "\#(modelType)", "text_config": {"use_bidirectional_attention": \#(flag)}}
            """#)
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        #expect(FullPathVLM.resolve(configURL: url, forced: true)?.reusesPrefixBeforeMedia == reuses)
    }

    /// The key is the prefix before the first media token plus that token,
    /// whichever media id comes first.
    @Test
    func anImageRequestsLookupKeyEndsAtItsFirstMediaToken() {
        let vlm = FullPathVLM(family: .gemma4, mediaTokenIDs: [8, 9])
        #expect(vlm.imageRequestLookupKey([1, 2, 3, 8, 9, 4]) == [1, 2, 3, 8])
        #expect(vlm.imageRequestLookupKey([1, 2, 9, 8, 4]) == [1, 2, 9])
        #expect(vlm.imageRequestLookupKey([1, 2, 3]) == nil, "no block to cap at")
        #expect(vlm.imageRequestLookupKey([8, 1, 2]) == nil, "nothing before the block")
        let cold = FullPathVLM(family: .gemma4, mediaTokenIDs: [8], reusesPrefixBeforeMedia: false)
        #expect(cold.imageRequestLookupKey([1, 2, 8]) == nil)
    }

    /// What the store makes of the key, with a history-boundary entry stored
    /// under exactly the prefix and the text turn's longer entry beside it:
    /// the boundary is the only candidate and is reused whole, with no trim,
    /// which a `MambaCache` (Qwen3.5) or a sliding window past its size
    /// (Gemma 4) needs. A key without the opening token would make the
    /// boundary an exact match, trimmed by one.
    @Test
    func anEntryStoredUnderExactlyThePrefixIsReusedWhole() throws {
        let trie = PromptTrie<Int>()
        trie.add(model: "m", tokens: [1, 2, 3], value: 0)
        trie.add(model: "m", tokens: [1, 2, 3, 4, 5], value: 1)
        let vlm = FullPathVLM(family: .qwen35, mediaTokenIDs: [8])
        let key = try #require(vlm.imageRequestLookupKey([1, 2, 3, 8, 6, 7]))
        #expect(
            PromptCacheStore.resolveReuse(trie.search(model: "m", tokens: key), tokenCount: key.count)
                == [.init(key: [1, 2, 3], heldCount: 3, targetReuse: 3, requiresTrim: false)])
    }

    /// The key id is rebuilt from a string because upstream keeps the key
    /// private; this pins the string to the source this build resolved, as
    /// `ModelLibraryManagerVLMTests` does for the registry. A wrong id would
    /// leave every warm Qwen3.5 continuation without its anchor.
    @Test(.enabled(if: ResolvedUpstreamSource.mlxSwiftLM(Self.qwen35Source) != nil
                   || ResolvedUpstreamSource.isRequired,
                   "no SwiftPM checkout of mlx-swift-lm next to the test bundle"))
    func theQwen35RopeDeltasKeyIsUpstreamsKey() throws {
        let source = try #require(ResolvedUpstreamSource.mlxSwiftLM(Self.qwen35Source),
                                  "MACMLX_REQUIRE_RESOLVED_SOURCE is set but no checkout was found")
        let declaration = /let ropeDeltasKey = LMOutput\.Key<MLXArray>\(\s*"([^"]+)"\s*\)/
        let match = try #require(source.firstMatch(of: declaration), "no ropeDeltasKey in Qwen35.swift")
        #expect(String(match.1) == FullPathVLM.qwen35RopeDeltasKeyID)
    }

    private static let qwen35Source = "Libraries/MLXVLM/Models/Qwen35.swift"
}

/// The MLX-backed half: the state a warm continuation starts from and the
/// shape the text path gets.
@Suite("FullPathVLM inputs", .enabled(if: mlxMetallibIsAvailable, "needs the MLX Metal library"))
struct FullPathVLMInputTests {

    @Test
    func qwen35StartsFromAZeroRopeDeltaAndGemma4FromNothing() throws {
        let state = try #require(FullPathVLM(family: .qwen35).iteratorState())
        let delta = try #require(state[LMOutput.Key<MLXArray>(FullPathVLM.qwen35RopeDeltasKeyID)])
        #expect(delta.ndim == 0)
        #expect(delta.item(Int32.self) == 0)
        #expect(FullPathVLM(family: .gemma4).iteratorState() == nil)
    }

    @Test
    func aBatchShapedTextPromptBecomesFlatWithNoMask() {
        let input = LMInput(text: .init(
            tokens: MLXArray([Int32(5), 6, 7]).reshaped(1, 3),
            mask: MLXArray.ones([1, 3], dtype: .int8)))
        let flat = FullPathVLM.textOnlyInput(input)
        #expect(flat.text.tokens.shape == [3])
        #expect(flat.text.tokens.asArray(Int32.self) == [5, 6, 7])
        #expect(flat.text.mask == nil)
    }

    @Test
    func aFlatPromptLosesOnlyItsMask() {
        let masked = FullPathVLM.textOnlyInput(LMInput(text: .init(
            tokens: MLXArray([Int32(5), 6]), mask: MLXArray.ones([2], dtype: .int8))))
        #expect(masked.text.tokens.asArray(Int32.self) == [5, 6])
        #expect(masked.text.mask == nil)
        let plain = FullPathVLM.textOnlyInput(LMInput(tokens: MLXArray([Int32(5), 6])))
        #expect(plain.text.tokens.asArray(Int32.self) == [5, 6])
    }

    @Test
    func aPromptWithAnImageKeepsTheProcessorsShape() {
        let input = LMInput(
            text: .init(tokens: MLXArray([Int32(5), 6, 7]).reshaped(1, 3), mask: MLXArray.ones([1, 3], dtype: .int8)),
            image: .init(pixels: MLXArray.zeros([1, 3, 2, 2])))
        #expect(FullPathVLM.carriesMedia(input))
        let kept = FullPathVLM.textOnlyInput(input)
        #expect(kept.text.tokens.shape == [1, 3])
        #expect(kept.text.mask != nil)
        #expect(kept.image != nil)
    }
}
