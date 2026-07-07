import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM
@preconcurrency import Tokenizers

// MARK: - Sendable-box helpers

/// Lightweight unchecked-Sendable wrapper used to pass non-Sendable
/// mlx-swift-lm values (`LMInput`, `AsyncStream<TokenGeneration>`) across
/// isolation boundaries when we know the handoff is safe — we `consume`
/// them into the actor via `ModelContainer.perform(nonSendable:_:)` and
/// the actor owns them exclusively afterwards.
private struct NonSendableBox<T>: @unchecked Sendable {
    let value: T
    init(_ value: T) { self.value = value }
}

// MARK: - Tokenizer loader

/// Concrete TokenizerLoader that uses the HuggingFace swift-transformers library.
///
/// This is the manual equivalent of the #huggingFaceTokenizerLoader() macro
/// from MLXHuggingFace, inlined here so MacMLXCore does not need the macro-only
/// MLXHuggingFace product.
private struct HuggingFaceTokenizerLoader: TokenizerLoader, @unchecked Sendable {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let upstream = try await AutoTokenizer.from(modelFolder: directory)
        return TokenizerBridge(upstream)
    }
}

/// Bridge between Tokenizers.Tokenizer (swift-transformers) and MLXLMCommon.Tokenizer.
private struct TokenizerBridge: MLXLMCommon.Tokenizer, @unchecked Sendable {
    private let upstream: any Tokenizers.Tokenizer

    init(_ upstream: any Tokenizers.Tokenizer) {
        self.upstream = upstream
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        upstream.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        upstream.convertIdToToken(id)
    }

    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        do {
            return try upstream.applyChatTemplate(
                messages: messages, tools: tools, additionalContext: additionalContext)
        } catch let tokenizerError as Tokenizers.TokenizerError {
            if case .chatTemplate = tokenizerError {
                throw MLXLMCommon.TokenizerError.missingChatTemplate
            }
            throw tokenizerError
        }
    }
}

// MARK: - MLXSwiftEngine

/// In-process MLX inference engine backed by Apple's mlx-swift-lm library.
///
/// Lifecycle: `.idle` → `.loading` → `.ready` → `.generating` → `.ready` → `.idle`.
/// Any state may transition to `.error(_)`.
///
/// - Note: This is the default inference engine for macMLX. It requires Apple Silicon
///   and a local MLX model directory containing `config.json` and `.safetensors` weights.
public actor MLXSwiftEngine: InferenceEngine {

    // MARK: Protocol properties

    public let engineID: EngineID = .mlxSwift

    public private(set) var status: EngineStatus = .idle

    public private(set) var loadedModel: LocalModel?

    /// Version string including the mlx-swift-lm library tag.
    public let version: String = "mlx-swift-lm 3.31.3"

    // MARK: Private state

    /// What's currently loaded — text-only LLM (`MLXLLM`), vision-
    /// language VLM (`MLXVLM`), or nothing. Both modalities wrap a
    /// `ModelContainer`; the case discriminates which factory built
    /// it so generation can choose the right code path (LLM gets the
    /// prompt cache; VLM bypasses it for now — multimodal cache keys
    /// would need to fold image bytes into the hash, deferred to a
    /// follow-up).
    private enum LoadedSupport {
        case none
        case llm(ModelContainer)
        case vlm(ModelContainer)

        var container: ModelContainer? {
            switch self {
            case .none: return nil
            case .llm(let c): return c
            case .vlm(let c): return c
            }
        }

        var isVLM: Bool {
            if case .vlm = self { return true }
            return false
        }
    }

    private var loadedSupport: LoadedSupport = .none

    /// Two-tier prompt cache (hot dict + cold safetensors sidecar). Used
    /// by `runGeneration` to reuse KV state across successive turns on
    /// the same LLM. VLM generations bypass it.
    private let promptCacheStore: PromptCacheStore

    /// Guards the one-time `ModelOverlay.registerAll()` so macMLX-owned
    /// architectures are registered into the shared factory registry
    /// exactly once, before the first model load. `registerAll` is itself
    /// idempotent; this flag just avoids the redundant actor hop.
    private var overlayRegistered = false

    // MARK: Initialiser

    public init() {
        self.promptCacheStore = PromptCacheStore(
            root: DataRoot.macMLX("kv-cache")
        )
    }

    // MARK: InferenceEngine

    /// Load a model from its local directory into memory.
    ///
    /// - Parameter model: The ``LocalModel`` to load. `model.directory` must contain
    ///   a valid MLX model (`config.json`, `.safetensors` weights, tokenizer files).
    /// - Throws: ``EngineError/modelLoadFailed(reason:)`` if loading fails for any reason.
    public func load(_ model: LocalModel) async throws {
        status = .loading(model: model.id)

        // Teach the stock factory about macMLX-owned architectures before
        // the first load. No-op until an overlay architecture ships, but
        // wired now so `LLMModelFactory.shared.loadContainer` can resolve
        // our `model_type`s the moment one is registered. See ModelOverlay.
        if !overlayRegistered {
            await ModelOverlay.registerAll()
            overlayRegistered = true
        }

        // Preflight: catch Gemma 4 MoE checkpoints before handing off to
        // LLMModelFactory, which surfaces a cryptic "Unhandled keys"
        // error (see mlx-swift-lm#219).
        let configURL = model.directory.appending(
            path: "config.json", directoryHint: .notDirectory)
        if Self.isUnsupportedGemma4MoE(configURL: configURL) {
            let reason = "Gemma 4 Mixture-of-Experts variants (e.g. `a4b`) are not yet "
                + "supported by mlx-swift-lm 3.31.x. Tracking upstream at "
                + "https://github.com/ml-explore/mlx-swift-lm/issues/219. "
                + "Use a dense Gemma 4 checkpoint (E2B / E4B) in the meantime."
            status = .error(reason)
            loadedSupport = .none
            loadedModel = nil
            throw EngineError.modelLoadFailed(reason: reason)
        }

        do {
            let support: LoadedSupport
            switch model.format {
            case .mlx:
                let container = try await LLMModelFactory.shared.loadContainer(
                    from: model.directory,
                    using: HuggingFaceTokenizerLoader()
                )
                support = .llm(container)

            case .mlxVLM:
                let container = try await VLMModelFactory.shared.loadContainer(
                    from: model.directory,
                    using: HuggingFaceTokenizerLoader()
                )
                support = .vlm(container)

            case .gguf, .unknown:
                // Surfaced via the Models tab — these formats never
                // reach the engine in practice, but throw a clean
                // error if someone hand-constructs a `LocalModel`.
                let reason = "Unsupported model format: \(model.format.rawValue). " +
                    "MLXSwiftEngine handles `mlx` (text) and `mlxVLM` (vision-language) only."
                status = .error(reason)
                loadedSupport = .none
                loadedModel = nil
                throw EngineError.modelLoadFailed(reason: reason)
            }
            loadedSupport = support
            loadedModel = model
            status = .ready(model: model.id)
        } catch let engineError as EngineError {
            // Already shaped — preserve the typed error.
            loadedSupport = .none
            loadedModel = nil
            throw engineError
        } catch {
            let reason = error.localizedDescription
            status = .error(reason)
            loadedSupport = .none
            loadedModel = nil
            throw EngineError.modelLoadFailed(reason: reason)
        }
    }

    // MARK: Preflight

    /// Inspect the model's `config.json` for Gemma 4 MoE markers. Returns
    /// `true` when the config declares Mixture-of-Experts fields that
    /// mlx-swift-lm 3.31.x does not yet implement (see mlx-swift-lm#219).
    ///
    /// Kept internal so tests can exercise it. Any IO / JSON error is
    /// treated as "not MoE" — preflight should never hijack load errors
    /// from unrelated causes; we only want to catch the specific
    /// Gemma 4 MoE-on-3.31.x case.
    static func isUnsupportedGemma4MoE(configURL: URL) -> Bool {
        guard let data = try? Data(contentsOf: configURL),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return false
        }
        // Gemma 4 configs sometimes nest their text fields under "text_config".
        let containers: [[String: Any]] = [
            root,
            root["text_config"] as? [String: Any] ?? [:],
        ]
        let isGemma4 = containers.contains { container in
            guard let modelType = container["model_type"] as? String else { return false }
            return modelType.hasPrefix("gemma4") || modelType.hasPrefix("gemma_4")
        }
        guard isGemma4 else { return false }
        return containers.contains { container in
            if let n = container["num_experts"] as? Int, n > 0 { return true }
            if let n = container["num_local_experts"] as? Int, n > 0 { return true }
            return false
        }
    }

    /// Release the loaded model from memory.
    public func unload() async throws {
        loadedSupport = .none
        loadedModel = nil
        status = .idle
    }

    /// Apply a LoRA adapter (v0.5+) to the currently-loaded model.
    ///
    /// PEFT-format adapters are auto-converted to mlx-native format
    /// via `LoRAAdapterConverter`, with the conversion output cached
    /// at `~/.mac-mlx/adapters/.cache/<adapter-name>/` so repeat
    /// loads reuse the converted bytes. mlx-native adapters skip the
    /// converter and load directly.
    public func applyAdapter(_ adapter: LocalAdapter) async throws {
        guard let container = loadedSupport.container else {
            throw EngineError.modelNotLoaded
        }

        // Resolve the directory the LoRAContainer should read from.
        // PEFT → run the converter into a sibling cache dir; mlx-
        // native → use the adapter's own directory.
        let mlxDirectory: URL
        switch adapter.format {
        case .mlx:
            mlxDirectory = adapter.directory
        case .peft:
            mlxDirectory = try await convertedDirectory(for: adapter)
        }

        // Hand the mlx-format directory to LoRAContainer.from then
        // load it into the model. Both calls happen inside the
        // ModelContainer's actor so we serialise correctly with any
        // concurrent generation.
        do {
            try await container.perform { context in
                let loraContainer = try LoRAContainer.from(directory: mlxDirectory)
                try context.model.load(adapter: loraContainer)
            }
        } catch {
            throw EngineError.adapterApplyFailed(reason: error.localizedDescription)
        }

        await LogManager.shared.info(
            "LoRA adapter applied: \(adapter.name) (format=\(adapter.format.rawValue))",
            category: .inference
        )
    }

    /// Convert a PEFT-format adapter to the mlx-native cache layout.
    /// Cached at `~/.mac-mlx/adapters/.cache/<adapter-name>/` so repeat
    /// loads of the same adapter reuse the conversion result.
    private func convertedDirectory(for adapter: LocalAdapter) async throws -> URL {
        let cacheDir = DataRoot.macMLX("adapters/.cache")
            .appending(path: adapter.name, directoryHint: .isDirectory)
        let configURL = cacheDir.appending(path: "adapter_config.json", directoryHint: .notDirectory)
        let weightsURL = cacheDir.appending(path: "adapters.safetensors", directoryHint: .notDirectory)

        if FileManager.default.fileExists(atPath: configURL.path),
           FileManager.default.fileExists(atPath: weightsURL.path) {
            return cacheDir
        }

        do {
            try LoRAAdapterConverter.convertPEFTAdapter(
                source: adapter.directory,
                destination: cacheDir
            )
        } catch {
            throw EngineError.adapterApplyFailed(reason: "PEFT → mlx conversion failed: \(error)")
        }
        return cacheDir
    }

    /// Stream tokens for a generation request.
    ///
    /// This method is `nonisolated` so the `AsyncThrowingStream` is returned
    /// synchronously. The actual generation work runs inside a `Task` that
    /// re-enters the actor for state access.
    public nonisolated func generate(
        _ request: GenerateRequest
    ) -> AsyncThrowingStream<GenerateChunk, Error> {
        AsyncThrowingStream { continuation in
            Task { [weak self] in
                guard let self else {
                    continuation.finish()
                    return
                }
                do {
                    try await self.runGeneration(request, into: continuation)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Confirm the engine is responsive.
    ///
    /// Returns `true` always — a model need not be loaded for the engine to be healthy.
    public func healthCheck() async -> Bool {
        true
    }

    /// Whether the rendered prompt for `request` ends inside an open
    /// `<think>` block. Applies the chat template exactly as `generate`
    /// does (same `Chat.Message` mapping + `container.prepare`), decodes
    /// the resulting tokens, and inspects the tail via
    /// `MessageSegmenter.promptOpensThink`. Returns false if no model is
    /// loaded or the template can't be applied.
    public func promptOpensThinkBlock(_ request: GenerateRequest) async -> Bool {
        guard let container = loadedSupport.container else { return false }
        let chatMessages: [Chat.Message] = request.allMessages.map { msg in
            let role: Chat.Message.Role
            switch msg.role {
            case .user: role = .user
            case .assistant: role = .assistant
            case .system: role = .system
            }
            return Chat.Message(role: role, content: msg.content)
        }
        do {
            let lmInput = try await container.prepare(input: UserInput(
                chat: chatMessages,
                additionalContext: request.templateKwargs?.mapValues { $0.toSendable() }
            ))
            let ids = lmInput.text.tokens.asArray(Int32.self).map(Int.init)
            let text = await container.decode(tokens: ids)
            return MessageSegmenter.promptOpensThink(text)
        } catch {
            return false
        }
    }

    // MARK: Prompt cache management

    /// Drop both tiers of the prompt cache. Wired up to the Settings
    /// → "Clear All KV Caches" button via `EngineCoordinator`.
    public func clearPromptCache() async {
        await promptCacheStore.clearAll()
    }

    // MARK: Private generation helper

    /// Actor-isolated generation driver called from within `generate(_:)`.
    ///
    /// Flow:
    /// 1. Prepare the `LMInput` (tokenisation + chat template application).
    /// 2. Hash the full input-token sequence into a `PromptCacheKey`.
    /// 3. Look up a prior cache snapshot in `promptCacheStore`. On hit,
    ///    reuse its `[KVCache]` so the shared prefix skips prefill. On
    ///    miss, allocate a fresh cache via `model.newCache(...)`.
    /// 4. Drive the low-level `generateTokens(input:cache:...)` call so
    ///    we see raw token IDs and can build the extended key
    ///    `inputTokens + generatedTokenIDs` after the stream ends.
    /// 5. The `KVCache` protocol is class-bound — the same reference we
    ///    passed in is mutated in-place during generation, so at the
    ///    end we can save that same reference under the extended key.
    private func runGeneration(
        _ request: GenerateRequest,
        into continuation: AsyncThrowingStream<GenerateChunk, Error>.Continuation
    ) async throws {
        let support = loadedSupport
        guard let container = support.container else {
            continuation.finish(throwing: EngineError.modelNotLoaded)
            return
        }
        guard let loadedModelSnapshot = loadedModel else {
            continuation.finish(throwing: EngineError.modelNotLoaded)
            return
        }
        let isVLM = support.isVLM

        let params = request.parameters

        // Map GenerationParameters to mlx-swift-lm's GenerateParameters.
        // temperature/topP: our values are Double, MLXLLM uses Float.
        let generateParams = GenerateParameters(
            maxTokens: params.maxTokens,
            temperature: Float(params.temperature),
            topP: Float(params.topP)
        )

        // Map our ChatMessage array to MLXLMCommon Chat.Message array.
        // For VLM models, fold each message's `images` into the `Chat.Message`
        // image bag — the VLM's `UserInputProcessor` injects image tokens at
        // the right position when it builds the prompt. For LLM models we
        // drop attachments with a debug-level warning so `[image attached]`
        // stub strings don't sneak into the chat template.
        let chatMessages: [Chat.Message] = request.allMessages.map { msg in
            let role: Chat.Message.Role
            switch msg.role {
            case .user:      role = .user
            case .assistant: role = .assistant
            case .system:    role = .system
            }
            if isVLM {
                let images: [UserInput.Image] = msg.images.map { .url($0.fileURL) }
                return Chat.Message(role: role, content: msg.content, images: images)
            } else {
                if !msg.images.isEmpty {
                    Task.detached { [count = msg.images.count] in
                        await LogManager.shared.debug(
                            "Dropping \(count) image attachment(s) on text-only model — load a VLM (Qwen-VL, Gemma-3, SmolVLM, …) to use images.",
                            category: .inference
                        )
                    }
                }
                return Chat.Message(role: role, content: msg.content)
            }
        }

        // Forward per-model chat-template kwargs (v0.5.1) as
        // `additionalContext` — the tokenizer hands them to the Jinja
        // template (e.g. `enable_thinking` for Qwen3). Unwrap JSONValue
        // to the plain Sendable shape the upstream API expects.
        let userInput = UserInput(
            chat: chatMessages,
            additionalContext: request.templateKwargs?.mapValues { $0.toSendable() }
        )

        status = .generating

        defer {
            // Return to ready when generation exits (success, cancel, or error).
            if case .generating = status {
                if let model = loadedModel {
                    status = .ready(model: model.id)
                } else {
                    status = .idle
                }
            }
        }

        // Prepare input (tokenize + apply chat template) using the container's processor.
        let lmInput: LMInput
        do {
            lmInput = try await container.prepare(input: userInput)
        } catch {
            throw EngineError.modelLoadFailed(reason: error.localizedDescription)
        }

        if isVLM {
            // VLM path: bypass the prompt cache (the cache key would
            // need to fold image content hashes into the chained hash
            // — deferred to a follow-up).
            try await runVLMGeneration(
                lmInput: lmInput,
                container: container,
                generateParams: generateParams,
                into: continuation
            )
        } else {
            try await runLLMGeneration(
                lmInput: lmInput,
                container: container,
                generateParams: generateParams,
                modelID: loadedModelSnapshot.id,
                into: continuation
            )
        }
    }

    /// Text-only path: tokenise, look up the prompt cache, prefill only
    /// the new suffix, and stream tokens. Saves the extended cache back
    /// to `promptCacheStore` once the stream completes.
    private func runLLMGeneration(
        lmInput: LMInput,
        container: ModelContainer,
        generateParams: GenerateParameters,
        modelID: String,
        into continuation: AsyncThrowingStream<GenerateChunk, Error>.Continuation
    ) async throws {
        // Flat Int token array for key construction. `LMInput.text.tokens`
        // is an `MLXArray`; `asArray(Int.self)` materialises to Swift.
        let inputTokens = lmInput.text.tokens.asArray(Int.self)
        let priorKey = PromptCacheKey(modelID: modelID, tokens: inputTokens)

        // Try the store. On hit we reuse the restored cache; on miss we
        // let the iterator allocate a fresh one inside `generateTokens`.
        let priorSnapshot = await promptCacheStore.get(priorKey)
        let priorCache: [any KVCache]?
        if let snapshot = priorSnapshot {
            priorCache = snapshot.caches
            await LogManager.shared.debug(
                "Prompt cache HIT — restored \(priorKey.tokenCount) tokens (model=\(modelID))",
                category: .inference
            )
        } else {
            priorCache = nil
            await LogManager.shared.debug(
                "Prompt cache MISS — cold prefill of \(priorKey.tokenCount) tokens (model=\(modelID))",
                category: .inference
            )
        }

        // Build the working cache. When we have a prior snapshot we pass
        // that reference straight through; otherwise we ask the model to
        // allocate a fresh `[KVCache]`. We hold onto the same array so we
        // can save it after generation (KVCache is class-bound, so the
        // iterator populates our instances in place).
        //
        // `KVCache` is not `Sendable`, and `LMInput` is not `Sendable`
        // either. Route both through the `perform(nonSendable:_:)`
        // overload on `ModelContainer`, which explicitly accepts a
        // non-Sendable value by `consuming` it into the actor.
        let tokenizer = await container.tokenizer
        let priorCacheBox: PromptCacheSnapshot? = priorCache.map { PromptCacheSnapshot($0) }
        let inputBox = NonSendableBox(lmInput)

        let setup: (cache: PromptCacheSnapshot, stream: AsyncStream<TokenGeneration>) =
            try await container.perform(nonSendable: inputBox) { context, inputBox in
                let cache: [any KVCache] = priorCacheBox?.caches
                    ?? context.model.newCache(parameters: generateParams)
                let stream = try MLXLMCommon.generateTokens(
                    input: inputBox.value,
                    cache: cache,
                    parameters: generateParams,
                    context: context
                )
                return (PromptCacheSnapshot(cache), stream)
            }
        let workingCache = setup.cache.caches
        let stream = setup.stream

        var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        var generatedTokenIDs: [Int] = []
        var completionInfo: GenerateCompletionInfo?

        for await event in stream {
            switch event {
            case .token(let token):
                generatedTokenIDs.append(token)
                detokenizer.append(token: token)
                if let piece = detokenizer.next() {
                    let chunk = GenerateChunk(text: piece)
                    if case .terminated = continuation.yield(chunk) {
                        return
                    }
                }
            case .info(let info):
                completionInfo = info
            }
        }

        // Save the post-generation cache under the extended key. The
        // same `workingCache` reference has been mutated in-place by the
        // iterator, so it now reflects prompt + generated tokens.
        let finalTokens = inputTokens + generatedTokenIDs
        let newKey = PromptCacheKey(modelID: modelID, tokens: finalTokens)
        await promptCacheStore.put(
            key: newKey,
            snapshot: PromptCacheSnapshot(workingCache)
        )

        emitFinalChunk(completionInfo: completionInfo, into: continuation)
        continuation.finish()
    }

    /// Vision-language path: prepare the multimodal input (which already
    /// includes processed image embeddings via the VLM's UserInputProcessor),
    /// allocate a fresh KV cache, and stream tokens. Bypasses the prompt
    /// cache — multimodal cache keys are a follow-up.
    private func runVLMGeneration(
        lmInput: LMInput,
        container: ModelContainer,
        generateParams: GenerateParameters,
        into continuation: AsyncThrowingStream<GenerateChunk, Error>.Continuation
    ) async throws {
        let tokenizer = await container.tokenizer
        let inputBox = NonSendableBox(lmInput)

        let stream: AsyncStream<TokenGeneration> = try await container.perform(nonSendable: inputBox) { context, inputBox in
            let cache = context.model.newCache(parameters: generateParams)
            return try MLXLMCommon.generateTokens(
                input: inputBox.value,
                cache: cache,
                parameters: generateParams,
                context: context
            )
        }

        var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        var completionInfo: GenerateCompletionInfo?

        for await event in stream {
            switch event {
            case .token(let token):
                detokenizer.append(token: token)
                if let piece = detokenizer.next() {
                    let chunk = GenerateChunk(text: piece)
                    if case .terminated = continuation.yield(chunk) {
                        return
                    }
                }
            case .info(let info):
                completionInfo = info
            }
        }

        emitFinalChunk(completionInfo: completionInfo, into: continuation)
        continuation.finish()
    }

    /// Shared "final chunk" emit (usage + finish reason). Both LLM and
    /// VLM paths funnel through this so the wire-format chunk shape
    /// stays identical.
    private func emitFinalChunk(
        completionInfo: GenerateCompletionInfo?,
        into continuation: AsyncThrowingStream<GenerateChunk, Error>.Continuation
    ) {
        guard let info = completionInfo else { return }
        let finishReason: FinishReason
        switch info.stopReason {
        case .length:
            finishReason = .length
        case .stop, .cancelled:
            finishReason = .stop
        }
        let usage = TokenUsage(
            promptTokens: info.promptTokenCount,
            completionTokens: info.generationTokenCount
        )
        let finalChunk = GenerateChunk(text: "", finishReason: finishReason, usage: usage)
        continuation.yield(finalChunk)
    }
}
