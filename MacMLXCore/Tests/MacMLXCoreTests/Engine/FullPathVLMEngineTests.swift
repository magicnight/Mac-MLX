// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
import os

@testable import MacMLXCore

/// The engine side of ``FullPathVLM``, on a tiny random model behind a
/// processor shaped like the VLM ones (`[1, N]` tokens and a mask, an image
/// when the request has one) and a model wrapper that records what each
/// `prepare` is handed:
/// - a text prompt is flattened before the cache hit and the history
///   boundary slice it (without that, the hit slices away the whole prompt);
/// - every iterator on a warm cache starts from the family's state, so a
///   Qwen3.5 continuation finds its rope delta (the stock, constrained and
///   boundary iterators all start one);
/// - a request with an image reuses the cache only up to its first media
///   block, never leaves an entry or a boundary snapshot behind, and never
///   speculates.
@Suite(
    "MLXSwiftEngine full-path VLM input",
    .serialized, .enabled(if: mlxMetallibIsAvailable, "needs the MLX Metal library"))
struct FullPathVLMEngineTests {

    private static let noColdTier = PromptCacheConfig(coldEnabled: false)

    private struct Collected {
        var text = ""
        var promptTokens: Int?
        var cachedPromptTokens: Int?
    }

    private func collect(_ engine: MLXSwiftEngine, _ request: GenerateRequest) async throws -> Collected {
        var out = Collected()
        for try await chunk in engine.generate(request) {
            out.text += chunk.text
            if let usage = chunk.usage {
                out.promptTokens = usage.promptTokens
                out.cachedPromptTokens = usage.cachedPromptTokens
            }
        }
        return out
    }

    private static let model = LocalModel(
        id: "tiny-vlm", displayName: "tiny-vlm",
        directory: URL(fileURLWithPath: "/nonexistent/tiny-vlm", isDirectory: true),
        sizeBytes: 0, format: .mlxVLM, quantization: nil, parameterCount: nil, architecture: nil)

    private func container(_ model: any LanguageModel) -> ModelContainer {
        ModelContainer(
            context: ModelContext(
                configuration: ModelConfiguration(directory: Self.model.directory),
                model: model,
                processor: BatchShapedProcessor(),
                tokenizer: LetterTokenizer()))
    }

    private func engine(
        family: FullPathVLM.Family, recorder: PrepareRecorder = PrepareRecorder(), withDraft: Bool = false,
        reusesPrefixBeforeMedia: Bool = true
    ) async throws -> MLXSwiftEngine {
        let target = RecordingModel(try TinyRandomLanguageModel.qwen3.make(), recorder: recorder)
        let draft = withDraft ? container(try TinyRandomLanguageModel.qwen3.make()) : nil
        let engine = MLXSwiftEngine(promptCache: Self.noColdTier)
        await engine.installForTesting(
            container: container(target), draft: draft, model: Self.model,
            fullPath: FullPathVLM(
                family: family, mediaTokenIDs: BatchShapedProcessor.mediaTokenIDs,
                reusesPrefixBeforeMedia: reusesPrefixBeforeMedia))
        return engine
    }

    /// `imageAt` puts a media block in that message. `pixels: false` leaves
    /// the block's ids in a text prompt with no image attached, the same ids
    /// an image request would have.
    private func request(
        turns: Int, imageAt: Int? = nil, pixels: Bool = true, logprobs: Bool = false, draft: Bool = false,
        kvBits: Int? = nil
    ) -> GenerateRequest {
        let picture = ImageAttachment(fileURL: URL(fileURLWithPath: "/nonexistent/cat.png"), mimeType: "image/png")
        let messages = (0..<turns).map {
            ChatMessage(
                role: $0.isMultiple(of: 2) ? .user : .assistant,
                content: ($0 == imageAt ? BatchShapedProcessor.imageMarker : "") + "turn \($0)",
                images: $0 == imageAt && pixels ? [picture] : [])
        }
        var parameters = GenerationParameters(temperature: 0, topP: 1.0, maxTokens: 3, stream: true)
        parameters.logprobs = logprobs
        parameters.kvBits = kvBits
        var request = GenerateRequest(model: "tiny-vlm", messages: messages, parameters: parameters)
        // `installForTesting` names the draft it holds `<model>-draft`, so
        // this keeps it rather than loading one.
        if draft { request.draftModelID = "tiny-vlm-draft" }
        return request
    }

    // MARK: Text prompts

    @Test(arguments: FullPathVLM.Family.allCases)
    func aLongerPromptHitsTheCacheItsPrefixLeft(_ family: FullPathVLM.Family) async throws {
        let engine = try await engine(family: family)
        let first = try await collect(engine, request(turns: 1))
        #expect(first.promptTokens == BatchShapedProcessor.tokenCount(messages: 1))
        #expect(first.cachedPromptTokens ?? 0 == 0)

        let second = try await collect(engine, request(turns: 3))
        #expect(second.promptTokens == BatchShapedProcessor.tokenCount(messages: 3), "usage counts the whole prompt")
        let reused = try #require(second.cachedPromptTokens)
        #expect(reused >= BatchShapedProcessor.tokenCount(messages: 1), "the first prompt's tokens are served from the cache")
        let record = try #require(await engine.lastPromptCacheRecord)
        #expect(record.recorded)
    }

    @Test(arguments: FullPathVLM.Family.allCases)
    func theHistoryBoundarySplitsAFlatPrompt(_ family: FullPathVLM.Family) async throws {
        let engine = try await engine(family: family)
        await engine.setForcedHistoryBoundary(BatchShapedProcessor.tokenCount(messages: 1))
        let out = try await collect(engine, request(turns: 3))
        #expect(out.promptTokens == BatchShapedProcessor.tokenCount(messages: 3), "the boundary prefill is counted")
        #expect(out.text.count == 3)
    }

    /// The stock iterator on a hit, the constrained one (logprobs) on a hit,
    /// and the generation leg after a boundary prefill all start on a warm
    /// cache; each must carry Qwen3.5's delta, and Gemma 4 none.
    @Test(arguments: FullPathVLM.Family.allCases)
    func everyWarmIteratorStartsFromTheFamilysState(_ family: FullPathVLM.Family) async throws {
        let recorder = PrepareRecorder()
        let engine = try await engine(family: family, recorder: recorder)
        _ = try await collect(engine, request(turns: 1))
        _ = try await collect(engine, request(turns: 3))
        _ = try await collect(engine, request(turns: 5, logprobs: true))
        await engine.setForcedHistoryBoundary(BatchShapedProcessor.tokenCount(messages: 6))
        _ = try await collect(engine, request(turns: 7))

        let warm = recorder.calls.filter { $0.cacheOffset > 0 }
        #expect(warm.count >= 4, "a hit, a constrained hit, and the boundary's prefill and generation legs: \(recorder.calls)")
        let wantsDelta = family == .qwen35
        #expect(warm.allSatisfy { $0.hasRopeDelta == wantsDelta }, "\(recorder.calls)")
    }

    // MARK: Prompts with an image

    /// The warm `prepare` gets the remainder in the processor's shape, its
    /// mask sliced with it (Qwen3.5's `getRopeIndex` reads it), the image
    /// still attached, and the family's state.
    @Test(arguments: FullPathVLM.Family.allCases)
    func anImageRequestReusesTheCacheUpToItsFirstBlock(_ family: FullPathVLM.Family) async throws {
        let recorder = PrepareRecorder()
        let engine = try await engine(family: family, recorder: recorder)
        _ = try await collect(engine, request(turns: 3))
        let out = try await collect(engine, request(turns: 3, imageAt: 2))
        let blockStart = BatchShapedProcessor.blockStart(imageAt: 2)
        let total = BatchShapedProcessor.tokenCount(messages: 3, blocks: 1)
        #expect(out.cachedPromptTokens == blockStart)
        #expect(out.promptTokens == total, "usage counts the whole prompt")
        #expect(await engine.lastPromptCacheRecord == nil, "an image request records no entry")

        let warm = try #require(recorder.calls.last)
        #expect(warm.cacheOffset == blockStart)
        #expect(warm.tokensShape == [1, total - blockStart])
        #expect(warm.maskShape == [1, total - blockStart])
        #expect(warm.hasImage)
        #expect(warm.hasRopeDelta == (family == .qwen35))
    }

    /// An entry whose ids run on past the block (a text prompt that spelled
    /// out the block's ids) is still reused only up to the block: past it the
    /// KV depends on pixels the ids do not name.
    @Test
    func anImageRequestNeverReusesPastItsFirstBlock() async throws {
        let engine = try await engine(family: .gemma4)
        _ = try await collect(engine, request(turns: 3, imageAt: 2, pixels: false))
        #expect(await engine.lastPromptCacheRecord?.recorded == true, "the text prompt recorded its entry")
        let out = try await collect(engine, request(turns: 3, imageAt: 2))
        #expect(out.cachedPromptTokens == BatchShapedProcessor.blockStart(imageAt: 2))
    }

    @Test
    func aRepeatedImageRequestReusesNothingPastItsBlock() async throws {
        let engine = try await engine(family: .gemma4)
        _ = try await collect(engine, request(turns: 1))
        for _ in 0..<2 {
            let out = try await collect(engine, request(turns: 1, imageAt: 0))
            #expect(
                out.cachedPromptTokens == BatchShapedProcessor.blockStart(imageAt: 0),
                "the same placeholder ids must not reuse the first image's KV")
            #expect(await engine.lastPromptCacheRecord == nil)
        }
    }

    @Test
    func aBidirectionalGemma4CheckpointRunsImageRequestsCold() async throws {
        let engine = try await engine(family: .gemma4, reusesPrefixBeforeMedia: false)
        _ = try await collect(engine, request(turns: 3))
        let image = try await collect(engine, request(turns: 3, imageAt: 2))
        #expect(image.cachedPromptTokens ?? 0 == 0)
        let text = try await collect(engine, request(turns: 3))
        #expect(text.cachedPromptTokens ?? 0 > 0, "text requests keep the cache")
    }

    /// `kv_bits` keeps a request off the cache whatever it carries (logprobs
    /// keep the tiny model off quantization; the bypass reads `kv_bits` alone).
    @Test
    func anImageRequestThatBypassesTheCacheLooksNothingUp() async throws {
        let engine = try await engine(family: .gemma4)
        _ = try await collect(engine, request(turns: 3))
        let out = try await collect(engine, request(turns: 3, imageAt: 2, logprobs: true, kvBits: 4))
        #expect(out.cachedPromptTokens ?? 0 == 0)
    }

    /// A text prompt with the image request's token ids finds nothing it
    /// left: no entry, no boundary snapshot.
    @Test
    func anImageRequestLeavesNoBoundarySnapshotBehind() async throws {
        let engine = try await engine(family: .gemma4)
        await engine.setForcedHistoryBoundary(BatchShapedProcessor.tokenCount(messages: 1))
        _ = try await collect(engine, request(turns: 3, imageAt: 0))
        await engine.setForcedHistoryBoundary(nil)
        let text = try await collect(engine, request(turns: 3, imageAt: 0, pixels: false))
        #expect(text.cachedPromptTokens ?? 0 == 0, "nothing the image request computed is served")
    }

    /// With a draft resident, a text request speculates (the speculative path
    /// leaves `lastPromptTokensHash` nil) and an image request does not.
    @Test
    func anImageRequestNeverSpeculates() async throws {
        let engine = try await engine(family: .gemma4, withDraft: true)
        _ = try await collect(engine, request(turns: 1, draft: true))
        #expect(await engine.lastPromptTokensHash == nil, "the text request took the speculative path")
        _ = try await collect(engine, request(turns: 1, imageAt: 0, draft: true))
        #expect(await engine.lastPromptTokensHash != nil, "the image request took the plain path")
        #expect(await engine.hasDraftContainer, "and left the resident draft alone")
    }

    /// An image request that names no draft leaves the resident one alone:
    /// it never speculates, so its `draftModelID` is not reconciled (a text
    /// request naming none would unload it).
    @Test
    func anImageRequestDoesNotReconcileTheDraft() async throws {
        let engine = try await engine(family: .gemma4, withDraft: true)
        _ = try await collect(engine, request(turns: 1, imageAt: 0))
        #expect(await engine.hasDraftContainer)
        _ = try await collect(engine, request(turns: 1))
        #expect(await engine.hasDraftContainer == false, "a text request naming no draft unloads it")
    }
}

/// Answers like the VLM processors do: `[1, N]` tokens and an all-ones int8
/// mask, plus a placeholder image when the request carries one. Each message
/// adds a fixed run of text tokens, so a longer conversation's prompt extends
/// a shorter one's. A message with an image, or whose content starts with
/// ``imageMarker``, opens with a media block (an opening token and
/// placeholders) whose ids do not depend on the pixels, as a real image's do
/// not.
private struct BatchShapedProcessor: UserInputProcessor {
    static let tokensPerMessage = 12
    static let opener = 60
    static let placeholder = 61
    static let mediaTokenIDs: Set<Int> = [opener, placeholder]
    static let block = [opener, placeholder, placeholder, placeholder]
    static let imageMarker = "[image] "

    static func tokenCount(messages: Int, blocks: Int = 0) -> Int {
        4 + messages * tokensPerMessage + blocks * block.count
    }

    /// Whether `message` carries an image part or text starting with
    /// ``imageMarker``.
    static func opensWithABlock(_ message: [String: any Sendable]) -> Bool {
        if let content = message["content"] as? String { return content.hasPrefix(imageMarker) }
        let parts = message["content"] as? [[String: any Sendable]] ?? []
        return parts.contains { $0["type"] as? String == "image" }
            || parts.contains { ($0["text"] as? String)?.hasPrefix(imageMarker) == true }
    }

    /// Where the block opens when the only one is in message `imageAt`.
    static func blockStart(imageAt: Int) -> Int { tokenCount(messages: imageAt) }

    func prepare(input: UserInput) async throws -> LMInput {
        // A full-path load hands its generator's dictionaries over, as a
        // real processor gets them; an image is a part of type `image`.
        guard case .messages(let messages) = input.prompt else {
            throw CocoaError(.featureUnsupported)
        }
        var text = 0
        func textIDs(_ count: Int) -> [Int32] {
            defer { text += count }
            return (text..<text + count).map { Int32($0 % 50 + 1) }
        }
        var ids = textIDs(4)
        for message in messages {
            if Self.opensWithABlock(message) {
                ids += Self.block.map(Int32.init)
            }
            ids += textIDs(Self.tokensPerMessage)
        }
        return LMInput(
            text: .init(
                tokens: MLXArray(ids).reshaped(1, ids.count),
                mask: MLXArray.ones([1, ids.count], dtype: .int8)),
            image: input.images.isEmpty ? nil : .init(pixels: MLXArray.zeros([1, 3, 2, 2])))
    }
}

/// What one `prepare` was handed.
private struct PrepareCall: Sendable, CustomStringConvertible {
    let cacheOffset: Int
    let hasRopeDelta: Bool
    let tokensShape: [Int]
    let maskShape: [Int]?
    let hasImage: Bool
    var description: String { "(offset \(cacheOffset), delta \(hasRopeDelta), tokens \(tokensShape))" }
}

private final class PrepareRecorder: Sendable {
    private let lock = OSAllocatedUnfairLock<[PrepareCall]>(initialState: [])
    func append(_ call: PrepareCall) { lock.withLock { $0.append(call) } }
    var calls: [PrepareCall] { lock.withLock { $0 } }
}

/// A text model standing in for a VLM: records the cache offset and whether
/// the state carries Qwen3.5's rope delta at each `prepare`, then runs the
/// wrapped model on the flattened tokens with the image dropped (a text
/// model cannot read pixels, and `LLMModel.prepare` hands `[1, N]` back for
/// an extra axis on top).
private final class RecordingModel: Module, LanguageModel {
    private let inner: any LanguageModel
    private let recorder: PrepareRecorder

    init(_ inner: any LanguageModel, recorder: PrepareRecorder) {
        self.inner = inner
        self.recorder = recorder
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        let key = LMOutput.Key<MLXArray>(FullPathVLM.qwen35RopeDeltasKeyID)
        recorder.append(
            PrepareCall(
                cacheOffset: cache.first?.offset ?? 0, hasRopeDelta: state?[key] != nil,
                tokensShape: input.text.tokens.shape, maskShape: input.text.mask?.shape,
                hasImage: input.image != nil))
        return try inner.prepare(
            LMInput(tokens: input.text.tokens.reshaped(-1)), cache: cache, state: state, prefill: prefill)
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
        inner(input, cache: cache, state: state)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        inner(inputs, cache: cache)
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try inner.newCache(parameters: parameters)
    }

    func cacheStatus(parameters: GenerateParameters?) throws -> KVCacheStatus {
        try inner.cacheStatus(parameters: parameters)
    }
}

/// One letter per token, no special tokens, so generation runs to its token
/// budget and the text's length counts the tokens.
private struct LetterTokenizer: Tokenizer {
    let bosToken: String? = nil
    let eosToken: String? = nil
    let unknownToken: String? = nil

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(repeating: "a", count: tokenIds.count)
    }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { "a" }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}
