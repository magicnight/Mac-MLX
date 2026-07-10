import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM

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

// Note: `HuggingFaceTokenizerLoader` and `TokenizerBridge` were promoted to
// `Engine/HuggingFaceTokenizerLoader.swift` (shared internal types) so the
// embedding engine can reuse the same swift-transformers-backed loader.

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

    /// Resident draft model container for classic per-request speculative
    /// decoding (D1, text-only), or nil when no draft model is loaded.
    /// Swapped/unloaded by `ensureDraftContainer` when a request's
    /// `draftModelID` differs from `loadedDraftModelID`. Reuses the same
    /// `LLMModelFactory` container-loading path as the primary model — a
    /// draft model is always a plain text LLM, never a VLM.
    ///
    /// - Note: known v1 tradeoff — unlike primary models in `ModelPool`,
    ///   a resident draft container has no idle/TTL-based eviction; it stays
    ///   resident until swapped, explicitly unloaded, or the target model
    ///   reloads.
    private var draftContainer: ModelContainer?

    /// Model id of the currently-loaded `draftContainer`, or nil. Tracked
    /// separately from `draftContainer` so `ensureDraftContainer` can decide
    /// keep/swap/unload by comparing ids without touching the container.
    private var loadedDraftModelID: String?

    /// Fast-fail re-entrancy guard for the speculative decoding path (D1)
    /// ONLY — the non-speculative path is unaffected. Swift actors are
    /// reentrant across `await` suspension points, so nothing about being
    /// an actor method alone prevents two overlapping calls into
    /// `runSpeculativeLLMGeneration` from interleaving on this instance
    /// (see that method's doc comment for what actually serializes callers
    /// today, and where it does not). A silent race here would let two
    /// callers both "loan out" and mutate the SAME `draftContainer`'s model
    /// concurrently — surfacing as garbled tokens or a crash, not a clean
    /// error — so this guard fails loudly instead. Set true for the
    /// duration of `runSpeculativeLLMGeneration`, reset via `defer`.
    private var speculativeGenerationInFlight = false

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

            case .gguf, .unknown, .embedder:
                // Surfaced via the Models tab — these formats never
                // reach the engine in practice, but throw a clean
                // error if someone hand-constructs a `LocalModel`.
                // Embedder models are served by `EmbeddingEngine`, not
                // this generation engine.
                let reason = "Unsupported model format: \(model.format.rawValue). " +
                    "MLXSwiftEngine handles `mlx` (text) and `mlxVLM` (vision-language) only."
                status = .error(reason)
                loadedSupport = .none
                loadedModel = nil
                throw EngineError.modelLoadFailed(reason: reason)
            }
            loadedSupport = support
            loadedModel = model
            // Symmetric with `unload()`: a freshly-loaded target model must
            // not inherit a draft container that was paired with whatever
            // model was previously loaded on this actor instance — a stale
            // draft left resident across a target swap only wastes memory
            // (the tokenizer-compatibility check in
            // `runSpeculativeLLMGeneration` already fails safely on a
            // mismatched pairing either way).
            draftContainer = nil
            loadedDraftModelID = nil
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
    ///
    /// Also releases the D1 draft container (if any) — "the loaded model
    /// and any caches" (protocol doc) reasonably includes the speculative
    /// decoding drafter, and a stale draft left resident across a target
    /// swap would only waste memory (the per-request tokenizer compatibility
    /// check in `runSpeculativeLLMGeneration` already fails safely on a
    /// mismatched pairing, but there's no reason to keep it around).
    public func unload() async throws {
        loadedSupport = .none
        loadedModel = nil
        draftContainer = nil
        loadedDraftModelID = nil
        status = .idle
    }

    /// Test seam: whether a draft container is currently resident. Internal
    /// (not `private`) so `@testable import` tests can observe D1
    /// draft-state resets (e.g. after `unload()` / a successful `load(_:)`)
    /// without needing a real model load — mirrors `isUnsupportedGemma4MoE`'s
    /// "kept internal so tests can exercise it" precedent below.
    var hasDraftContainer: Bool { draftContainer != nil }

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
            let task = Task { [weak self] in
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
            // POOL-2: propagate abandonment/cancellation of this stream's
            // iteration down into the generation Task. `onTermination`
            // fires when the consumer stops iterating (including when the
            // consuming task itself is cancelled) — without this, walking
            // away from the stream never stops the underlying token loop,
            // so GPU work burns to completion even after a Stop button or
            // a server-side stall watchdog (SRV-4) gives up on the response.
            continuation.onTermination = { @Sendable _ in
                task.cancel()
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
        // Same tool-aware mapping as generation, without image attachments —
        // this heuristic only inspects the rendered token tail.
        let chatMessages: [Chat.Message] = request.allMessages.map {
            Self.upstreamChatMessage(from: $0, images: [])
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

    // MARK: D1 — speculative decoding draft container lifecycle

    /// Action to take on `draftContainer` for a request's `draftModelID`.
    /// Pure decision function (no I/O) so the keep/swap/unload state machine
    /// is unit-testable without loading any model or touching Metal — see
    /// `ensureDraftContainer`, the only caller, for the actual load/unload.
    enum DraftContainerAction: Equatable {
        /// Nothing requested and nothing resident, OR the resident draft
        /// already matches the request — do nothing.
        case keep
        /// The request has no draft model (nil) but one is resident — unload it.
        case unload
        /// The request names a draft model that isn't currently resident
        /// (either nothing was loaded, or a DIFFERENT one was) — load it,
        /// replacing any existing draft container.
        case load(id: String)

        static func decide(currentDraftModelID: String?, requestedDraftModelID: String?) -> Self {
            switch (currentDraftModelID, requestedDraftModelID) {
            case (.none, .none):
                return .keep
            case (.some(let current), .some(let requested)) where current == requested:
                return .keep
            case (.some, .none):
                return .unload
            case (_, .some(let requested)):
                return .load(id: requested)
            }
        }
    }

    /// Ensure `draftContainer`/`loadedDraftModelID` match `requestedDraftModelID`
    /// before a generation request. A no-op when they already match (the
    /// common case: repeated requests with the same — or no — draft model).
    ///
    /// - Throws: `EngineError.modelLoadFailed` if the draft directory doesn't
    ///   exist or fails to load through the same `LLMModelFactory` path
    ///   `load(_:)` uses for the primary model. On failure, any previously
    ///   loaded draft container is dropped too — a half-applied swap must
    ///   never leave a stale, silently-mismatched draft resident.
    private func ensureDraftContainer(requestedDraftModelID: String?) async throws {
        switch DraftContainerAction.decide(
            currentDraftModelID: loadedDraftModelID,
            requestedDraftModelID: requestedDraftModelID
        ) {
        case .keep:
            return
        case .unload:
            draftContainer = nil
            loadedDraftModelID = nil
        case .load(let id):
            do {
                let container = try await LLMModelFactory.shared.loadContainer(
                    from: Self.draftModelDirectory(id: id),
                    using: HuggingFaceTokenizerLoader()
                )
                draftContainer = container
                loadedDraftModelID = id
            } catch {
                draftContainer = nil
                loadedDraftModelID = nil
                throw EngineError.modelLoadFailed(
                    reason: "Draft model '\(id)' failed to load: \(error.localizedDescription)"
                )
            }
        }
    }

    /// Default on-disk location for a draft model named by bare id —
    /// `~/.mac-mlx/models/<id>`, mirroring `SettingsManager`'s default
    /// `modelDirectory`.
    ///
    /// - Note: v1 limitation — does not honor a user-customized model
    ///   directory setting. `MLXSwiftEngine` has no dependency on
    ///   `SettingsManager`/`ModelLibraryManager` today: resolving the
    ///   PRIMARY model's id to a directory happens one layer up (server's
    ///   `ModelResolver` / GUI's `ModelPool`), which hands the engine an
    ///   already-resolved `LocalModel`. The draft model, by contrast, is
    ///   named by a bare id in the wire request with no equivalent resolver
    ///   plumbed through — revisit if draft models need the same injection.
    ///
    /// - Important: `id` arrives verbatim from the wire
    ///   (`GenerateRequest.draftModelID`, client-controlled) and is used to
    ///   build a filesystem path — a path-traversal surface if unchecked
    ///   (`"../../etc"`, a leading dot reaching into a hidden dotfile, an
    ///   embedded path separator, …). Validated twice:
    ///   1. Before construction: `id` must pass `isValidDraftModelID` — an
    ///      allowlist (`[A-Za-z0-9._-]+`, no leading `.`), which rejects
    ///      `/`, `\`, NUL, `..`, and any dotfile-style id in one rule.
    ///   2. After construction: the standardized candidate path must still
    ///      resolve under the standardized models root — defense in depth
    ///      that doesn't rely on the allowlist alone having anticipated
    ///      every trick.
    ///   Either failure throws `EngineError.invalidDraftModelID(id:)`,
    ///   carrying only the raw `id` the client sent — never the resolved/
    ///   standardized path — so a client probing for filesystem layout via
    ///   error text learns nothing beyond its own input.
    /// - Throws: `EngineError.invalidDraftModelID` when `id` fails either check.
    static func draftModelDirectory(id: String) throws -> URL {
        guard Self.isValidDraftModelID(id) else {
            throw EngineError.invalidDraftModelID(id: id)
        }
        let root = DataRoot.macMLX("models")
        let candidate = root.appending(path: id, directoryHint: .isDirectory)

        // Belt-and-braces: confirm the standardized candidate is still
        // rooted under the standardized models directory.
        // `standardizedFileURL` lexically resolves `.`/`..` components, so
        // this catches anything the allowlist above didn't anticipate.
        let rootPath = root.standardizedFileURL.path
        let candidatePath = candidate.standardizedFileURL.path
        guard candidatePath == rootPath || candidatePath.hasPrefix(rootPath + "/") else {
            throw EngineError.invalidDraftModelID(id: id)
        }
        return candidate
    }

    /// Whether `id` is safe to use as a single filesystem path component.
    /// An allowlist — not a blocklist — so anything outside plain ASCII
    /// letters/digits/`.`/`_`/`-` is rejected outright (covers `/`, `\`,
    /// NUL, and any other separator), plus an explicit leading-dot ban,
    /// which blocks `.`, `..`, and dotfile-style ids (`.hidden`) in one rule.
    static func isValidDraftModelID(_ id: String) -> Bool {
        guard !id.isEmpty, id.first != "." else { return false }
        return id.allSatisfy { char in
            char.isASCII && (char.isLetter || char.isNumber || char == "." || char == "_" || char == "-")
        }
    }

    /// Simple tokenizer-compatibility probe for speculative decoding.
    /// mlx-swift-lm's `SpeculativeTokenIterator` documents that the target
    /// and draft models "must share the same tokenizer" but does not itself
    /// verify this — a silent mismatch would misinterpret draft-proposed
    /// token ids against the target vocabulary (garbled output, not a
    /// crash). Not a full vocabulary diff, but enough to catch "wrong model
    /// family" mistakes cheaply and without touching MLX:
    ///  1. Compares the three special-token STRINGS the `Tokenizer` protocol
    ///     exposes (`bosToken`/`eosToken`/`unknownToken`).
    ///  2. Also compares their underlying numeric ids — two tokenizers can
    ///     share special-token SPELLING while assigning it a different
    ///     vocabulary id (e.g. after a vocab resize/retrain on an otherwise
    ///     similar config), which the string-only check above would miss
    ///     and which would misindex the target's embedding table just the
    ///     same as a spelling mismatch. `eosTokenId`/`unknownTokenId` are
    ///     `MLXLMCommon.Tokenizer`'s own protocol-extension-provided id
    ///     lookups (derived internally via `convertTokenToId`).
    ///     `MLXLMCommon.Tokenizer` — the type actually in scope here via
    ///     `import MLXLMCommon` — has no `bosTokenId` member (unlike
    ///     swift-transformers' distinct `Tokenizers.Tokenizer` protocol,
    ///     which does), so the BOS id is derived the same way the extension
    ///     derives the other two, via `convertTokenToId`, keeping the check
    ///     symmetric across all three special tokens.
    static func tokenizersAreCompatible(target: any Tokenizer, draft: any Tokenizer) -> Bool {
        guard target.bosToken == draft.bosToken,
              target.eosToken == draft.eosToken,
              target.unknownToken == draft.unknownToken
        else {
            return false
        }
        let targetBOSId = target.bosToken.flatMap { target.convertTokenToId($0) }
        let draftBOSId = draft.bosToken.flatMap { draft.convertTokenToId($0) }
        return targetBOSId == draftBOSId
            && target.eosTokenId == draft.eosTokenId
            && target.unknownTokenId == draft.unknownTokenId
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
            let images: [UserInput.Image]
            if isVLM {
                images = msg.images.map { .url($0.fileURL) }
            } else {
                images = []
                if !msg.images.isEmpty {
                    Task.detached { [count = msg.images.count] in
                        await LogManager.shared.debug(
                            "Dropping \(count) image attachment(s) on text-only model — load a VLM (Qwen-VL, Gemma-3, SmolVLM, …) to use images.",
                            category: .inference
                        )
                    }
                }
            }
            return Self.upstreamChatMessage(from: msg, images: images)
        }

        // Convert any OpenAI tool specs (v0.5) to the tokenizer's `[ToolSpec]`
        // shape. Non-object elements are dropped (never force-cast) with a
        // debug note so a malformed spec degrades instead of crashing.
        let toolSpecs = request.tools.map { Self.toolSpecs(from: $0) }
        if let requested = request.tools, let converted = toolSpecs,
           converted.count < requested.count {
            Task.detached { [dropped = requested.count - converted.count] in
                await LogManager.shared.debug(
                    "Dropping \(dropped) non-object tool spec(s) from GenerateRequest.tools",
                    category: .inference
                )
            }
        }

        // Forward per-model chat-template kwargs (v0.5.1) as
        // `additionalContext` — the tokenizer hands them to the Jinja
        // template (e.g. `enable_thinking` for Qwen3). Unwrap JSONValue
        // to the plain Sendable shape the upstream API expects. Tools (when
        // present) reach the template via `UserInput.tools`.
        let userInput = UserInput(
            chat: chatMessages,
            tools: toolSpecs,
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
            //
            // D1 speculative decoding is text-only — a VLM request that
            // sets `draftModelID` is not an error, just ignored (silently
            // per the field's own doc comment, but logged for visibility).
            // Known v1 tradeoff: this branch never calls `ensureDraftContainer`,
            // so a draft container resident from an earlier LLM request stays
            // resident (memory only, not correctness) through VLM generations.
            if request.draftModelID != nil {
                Task.detached {
                    await LogManager.shared.debug(
                        "Ignoring draftModelID on a VLM request — speculative decoding "
                            + "(D1) is text-only.",
                        category: .inference
                    )
                }
            }
            try await runVLMGeneration(
                lmInput: lmInput,
                container: container,
                generateParams: generateParams,
                into: continuation
            )
        } else {
            // D1: bring the draft container in line with this request
            // BEFORE dispatching — throws (rather than silently falling
            // back to non-speculative generation) if the requested draft
            // model fails to load.
            try await ensureDraftContainer(requestedDraftModelID: request.draftModelID)
            try await runLLMGeneration(
                lmInput: lmInput,
                container: container,
                generateParams: generateParams,
                modelID: loadedModelSnapshot.id,
                hasTools: request.tools?.isEmpty == false,
                numDraftTokens: request.numDraftTokens,
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
        hasTools: Bool,
        numDraftTokens: Int?,
        into continuation: AsyncThrowingStream<GenerateChunk, Error>.Continuation
    ) async throws {
        // D1: `ensureDraftContainer` (called just before this, in
        // `runGeneration`) has already reconciled `draftContainer` with the
        // request — a resident draft container IS this request's signal to
        // speculate. Fork out to the dedicated path, which bypasses the
        // prompt cache entirely (see `runSpeculativeLLMGeneration`'s doc
        // comment) rather than threading a third branch through the cache
        // logic below.
        if let draftContainer {
            try await runSpeculativeLLMGeneration(
                lmInput: lmInput,
                container: container,
                draftContainer: draftContainer,
                generateParams: generateParams,
                numDraftTokens: numDraftTokens,
                hasTools: hasTools,
                into: continuation
            )
            return
        }

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

        let setup: (
            cache: PromptCacheSnapshot,
            stream: AsyncStream<TokenGeneration>,
            toolCallFormat: ToolCallFormat?
        ) =
            try await container.perform(nonSendable: inputBox) { context, inputBox in
                let cache: [any KVCache] = priorCacheBox?.caches
                    ?? context.model.newCache(parameters: generateParams)
                let stream = try MLXLMCommon.generateTokens(
                    input: inputBox.value,
                    cache: cache,
                    parameters: generateParams,
                    context: context
                )
                // `ToolCallFormat?` is Sendable, so it rides back out of the
                // actor alongside the cache + stream.
                return (PromptCacheSnapshot(cache), stream, context.configuration.toolCallFormat)
            }
        let workingCache = setup.cache.caches
        let stream = setup.stream

        // Only route through the streaming `ToolCallProcessor` when the caller
        // actually requested tools this turn — see `makeToolProcessor` for why
        // the gate exists (else EVERY generation on a tool-capable model, incl.
        // plain non-tool chat, would have `{...}`-shaped content silently
        // stripped and misreported as `finish_reason: tool_calls`).
        let toolProcessor = Self.makeToolProcessor(
            format: setup.toolCallFormat, hasTools: hasTools)

        var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        var generatedTokenIDs: [Int] = []
        var completionInfo: GenerateCompletionInfo?

        for await event in stream {
            // POOL-2: stop promptly once the consumer has abandoned/
            // cancelled this stream (see `generate`'s `onTermination`
            // hook) instead of running to maxTokens/EOS regardless.
            if Task.isCancelled {
                continuation.finish(throwing: CancellationError())
                return
            }
            switch event {
            case .token(let token):
                generatedTokenIDs.append(token)
                detokenizer.append(token: token)
                if let piece = detokenizer.next() {
                    if let toolProcessor {
                        // Emit only the processor's display text; nil means the
                        // piece is being buffered as (potential) tool-call syntax.
                        if let display = toolProcessor.processChunk(piece), !display.isEmpty {
                            if case .terminated = continuation.yield(GenerateChunk(text: display)) {
                                return
                            }
                        }
                    } else if case .terminated = continuation.yield(GenerateChunk(text: piece)) {
                        return
                    }
                }
            case .info(let info):
                completionInfo = info
            }
        }

        // Finalise tool-call parsing: flush any residual buffered text (a
        // false-positive tool-call start that never completed) as display
        // output, then drain the fully-parsed calls.
        var drainedToolCalls: [ToolCallRequest] = []
        if let toolProcessor {
            if let residual = toolProcessor.processEOS(returnBufferedText: true), !residual.isEmpty {
                continuation.yield(GenerateChunk(text: residual))
            }
            // `drainToolCalls()` is internal to MLXLMCommon; the equivalent
            // public `toolCalls` property holds every parsed call after EOS,
            // and the processor is discarded here so there's nothing to drain.
            drainedToolCalls = toolProcessor.toolCalls.map { Self.toolCallRequest(from: $0) }
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

        emitFinalChunk(
            completionInfo: completionInfo,
            toolCalls: drainedToolCalls,
            into: continuation
        )
        continuation.finish()
    }

    /// D1: classic per-request speculative decoding, via mlx-swift-lm's
    /// public `generateTokens(input:cache:parameters:context:draftModel:draftCache:numDraftTokens:wiredMemoryTicket:)`
    /// overload (`SpeculativeTokenIterator` under the hood — the "classic"
    /// draft-model path, distinct from the MTP block-drafter path this
    /// engine does not use). Called from `runLLMGeneration` once
    /// `ensureDraftContainer` has confirmed `draftContainer` matches the
    /// request's `draftModelID`.
    ///
    /// v1 tradeoffs, documented here rather than silently absorbed elsewhere:
    ///  - **Bypasses `promptCacheStore` entirely.** `SpeculativeTokenIterator.init`
    ///    requires BOTH the target and draft caches to be trimmable
    ///    (`canTrimPromptCache`) and throws `KVCacheError` otherwise. Our
    ///    stored snapshots aren't guaranteed to satisfy that for every cache
    ///    variant this codebase might produce (e.g. a future quantized or
    ///    batch-positioned cache — see `Batching/`), so every speculative
    ///    request pays a cold prefill on both models rather than risk that
    ///    throw. `cache`/`draftCache` are omitted below, so mlx-swift-lm
    ///    allocates fresh (trimmable-by-default) caches for both models.
    ///  - **Mutually exclusive with continuous batching** — see
    ///    `GenerateRequest.draftModelID`'s doc comment. Not enforced here;
    ///    batching isn't wired to the server yet, so this is a documented
    ///    invariant rather than a runtime guard.
    private func runSpeculativeLLMGeneration(
        lmInput: LMInput,
        container: ModelContainer,
        draftContainer: ModelContainer,
        generateParams: GenerateParameters,
        numDraftTokens: Int?,
        hasTools: Bool,
        into continuation: AsyncThrowingStream<GenerateChunk, Error>.Continuation
    ) async throws {
        // Fast-fail re-entrancy guard (speculative path only — see
        // `speculativeGenerationInFlight`'s doc comment for why this exists
        // and what it protects against). Reset via `defer` on every exit:
        // success, throw, or cancellation.
        guard !speculativeGenerationInFlight else {
            throw EngineError.generationInProgress
        }
        speculativeGenerationInFlight = true
        defer { speculativeGenerationInFlight = false }

        await LogManager.shared.debug(
            "Speculative decoding (D1): deliberate prompt-cache bypass — cold prefill on "
                + "BOTH target and draft models this turn (see runSpeculativeLLMGeneration's "
                + "doc comment for why the prompt cache is skipped on this path)",
            category: .inference
        )

        let targetTokenizer = await container.tokenizer
        let draftTokenizer = await draftContainer.tokenizer
        guard Self.tokenizersAreCompatible(target: targetTokenizer, draft: draftTokenizer) else {
            throw EngineError.draftModelTokenizerMismatch(
                reason: "target and draft models must share the same tokenizer "
                    + "(bosToken/eosToken/unknownToken or their ids differ)"
            )
        }

        // Extract the draft model out of its own container's isolation so it
        // can be handed into the TARGET container's `perform` call below —
        // `generateTokens(...)` needs the draft model and the target
        // `ModelContext` in the same call. `any LanguageModel` isn't
        // `Sendable` (it wraps `MLXNN.Module`), so this crosses the
        // container boundary the same way `LMInput` already does elsewhere
        // in this file: boxed via `NonSendableBox`, which IS Sendable
        // (`@unchecked`) even though its payload isn't.
        //
        // Safety of "on loan" here does NOT come from actor isolation —
        // Swift actors are REENTRANT across `await` suspension points, so
        // two overlapping calls into an `async` method on this SAME actor
        // instance can interleave; the actor alone does not serialize them.
        // What actually keeps this safe today is caller-side serialization:
        // - Server path: `HummingbirdServer`'s `generationLocked` FIFO lock
        //   (`HummingbirdServer.swift:516`) admits only one request through
        //   `generate(_:)` at a time, so at most one caller can have a
        //   draft model "on loan" via this actor at once.
        // - GUI path: verified NOT equivalently serialized.
        //   `EngineCoordinator.generate(_:)` (macMLX/macMLX/App/EngineCoordinator.swift)
        //   has no analogous lock — it just looks up the current engine from
        //   the shared `ModelPool` and calls `engine.generate(request)`.
        //   `ChatViewModel` and `BenchmarkViewModel` each guard only their
        //   OWN re-entrancy (`isGenerating` / `runTask`), and both drive the
        //   same shared `EngineCoordinator` instance (see `AppState`), so a
        //   chat generation and a benchmark run against the same resident
        //   engine could overlap in principle. `speculativeGenerationInFlight`
        //   above is this engine's own guard against exactly that gap for
        //   the speculative path — scoped narrowly so the non-speculative
        //   path's behavior is unchanged.
        let draftModelBox: NonSendableBox<any LanguageModel> = await draftContainer.perform { context in
            NonSendableBox(context.model)
        }

        let inputBox = NonSendableBox(lmInput)

        let setup: (stream: AsyncStream<TokenGeneration>, toolCallFormat: ToolCallFormat?) =
            try await container.perform(nonSendable: inputBox) { context, inputBox in
                let stream: AsyncStream<TokenGeneration>
                // Omit `numDraftTokens` entirely (rather than substitute a
                // number we hardcode) when the request didn't set one, so a
                // `nil` genuinely defers to whatever mlx-swift-lm's own
                // default is, tracking it even if that default changes.
                if let numDraftTokens {
                    stream = try MLXLMCommon.generateTokens(
                        input: inputBox.value,
                        parameters: generateParams,
                        context: context,
                        draftModel: draftModelBox.value,
                        numDraftTokens: numDraftTokens
                    )
                } else {
                    stream = try MLXLMCommon.generateTokens(
                        input: inputBox.value,
                        parameters: generateParams,
                        context: context,
                        draftModel: draftModelBox.value
                    )
                }
                return (stream, context.configuration.toolCallFormat)
            }
        let stream = setup.stream

        // Same tool-call gating as the non-speculative path — see
        // `makeToolProcessor`: only route through the processor when the
        // caller actually requested tools this turn.
        let toolProcessor = Self.makeToolProcessor(format: setup.toolCallFormat, hasTools: hasTools)

        var detokenizer = NaiveStreamingDetokenizer(tokenizer: targetTokenizer)
        var completionInfo: GenerateCompletionInfo?

        for await event in stream {
            // POOL-2: stop promptly once the consumer has abandoned/
            // cancelled this stream, mirroring the non-speculative path.
            if Task.isCancelled {
                continuation.finish(throwing: CancellationError())
                return
            }
            switch event {
            case .token(let token):
                detokenizer.append(token: token)
                if let piece = detokenizer.next() {
                    if let toolProcessor {
                        if let display = toolProcessor.processChunk(piece), !display.isEmpty {
                            if case .terminated = continuation.yield(GenerateChunk(text: display)) {
                                return
                            }
                        }
                    } else if case .terminated = continuation.yield(GenerateChunk(text: piece)) {
                        return
                    }
                }
            case .info(let info):
                completionInfo = info
            }
        }

        var drainedToolCalls: [ToolCallRequest] = []
        if let toolProcessor {
            if let residual = toolProcessor.processEOS(returnBufferedText: true), !residual.isEmpty {
                continuation.yield(GenerateChunk(text: residual))
            }
            drainedToolCalls = toolProcessor.toolCalls.map { Self.toolCallRequest(from: $0) }
        }

        // Telemetry: pass through accepted/proposed counts when mlx-swift-lm
        // reported them on the terminal `.info` record; never fabricate the
        // field when it didn't. Known v1 tradeoff: only the terminal chunk
        // carries this — there is no incremental/streaming acceptance-rate
        // telemetry per token as the speculative loop runs.
        let speculativeDecoding = completionInfo?.speculativeDecodingTelemetry.map {
            SpeculativeDecodingUsage(
                proposedTokens: $0.draftTokenCount,
                acceptedTokens: $0.acceptedDraftTokenCount
            )
        }

        emitFinalChunk(
            completionInfo: completionInfo,
            toolCalls: drainedToolCalls,
            speculativeDecoding: speculativeDecoding,
            into: continuation
        )
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
            // POOL-2: stop promptly once the consumer has abandoned/
            // cancelled this stream (see `generate`'s `onTermination`
            // hook) instead of running to maxTokens/EOS regardless.
            if Task.isCancelled {
                continuation.finish(throwing: CancellationError())
                return
            }
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
    ///
    /// Always yields exactly one terminal chunk, even when the underlying
    /// mlx-swift-lm token stream ended WITHOUT its closing `.info` event
    /// (`completionInfo == nil`). This is reached only on a NATURAL stream
    /// end: cancellation and consumer-abandonment (`.terminated`) both
    /// `return` early from the generation loop above and never call this, so
    /// a `nil` info here means an abnormal-but-non-cancel end (e.g. the
    /// stream closed without emitting a final info record). Previously that
    /// case yielded nothing, so the server-side reasoning splitter never
    /// received a `finishReason` to `finish()` on — dropping any held-back
    /// partial `</think>` tail — and usage stayed 0. Emit a synthetic `.stop`
    /// terminal chunk with zero usage so every natural end still delivers one
    /// finish-bearing chunk.
    private func emitFinalChunk(
        completionInfo: GenerateCompletionInfo?,
        toolCalls: [ToolCallRequest] = [],
        speculativeDecoding: SpeculativeDecodingUsage? = nil,
        into continuation: AsyncThrowingStream<GenerateChunk, Error>.Continuation
    ) {
        let infoReason: FinishReason
        let usage: TokenUsage
        if let info = completionInfo {
            switch info.stopReason {
            case .length:
                infoReason = .length
            case .stop, .cancelled:
                infoReason = .stop
            }
            usage = TokenUsage(
                promptTokens: info.promptTokenCount,
                completionTokens: info.generationTokenCount
            )
        } else {
            infoReason = .stop
            usage = TokenUsage(promptTokens: 0, completionTokens: 0)
        }
        // A generation that produced tool calls finishes with `.toolCalls`
        // (OpenAI semantics), overriding the raw stop reason; usage still
        // comes from the `.info` record exactly as before.
        let finishReason: FinishReason = toolCalls.isEmpty ? infoReason : .toolCalls
        let finalChunk = GenerateChunk(
            text: "",
            finishReason: finishReason,
            usage: usage,
            toolCalls: toolCalls.isEmpty ? nil : toolCalls,
            speculativeDecoding: speculativeDecoding
        )
        continuation.yield(finalChunk)
    }

    // MARK: Tool-call helpers

    /// Decide whether `runLLMGeneration` should route pieces through a
    /// streaming `ToolCallProcessor` this turn.
    ///
    /// Gated on `hasTools` (the request actually supplied `GenerateRequest.tools`)
    /// — NOT merely on the loaded model declaring a `toolCallFormat`. Most
    /// popular models (Qwen, Llama, Mistral, …) declare a format unconditionally,
    /// so gating on format-presence alone would route every generation on those
    /// models — including plain `/v1/chat/completions` calls and ordinary GUI
    /// chat with no tools attached — through tool-call parsing. For `.json`
    /// format specifically that silently strips any bare `{"name":…,
    /// "arguments":…}`-shaped model output from the visible content while
    /// mis-tagging `finish_reason` as `tool_calls`, with no `tool_calls` payload
    /// to compensate. Requiring `hasTools` confines that buffering to requests
    /// that actually asked for tools, restoring byte-for-byte passthrough
    /// otherwise (prior, pre-tool-routing behaviour).
    static func makeToolProcessor(format: ToolCallFormat?, hasTools: Bool) -> ToolCallProcessor? {
        guard hasTools else { return nil }
        return format.map { ToolCallProcessor(format: $0) }
    }

    /// Convert macMLX `[JSONValue]` tool specs into the tokenizer's `[ToolSpec]`
    /// (`[[String: any Sendable]]`) shape. Elements that aren't JSON objects are
    /// dropped rather than force-cast — the project forbids `as!`, and a
    /// malformed spec must never crash generation. The caller compares counts
    /// to log how many were dropped.
    static func toolSpecs(from tools: [JSONValue]) -> [ToolSpec] {
        tools.compactMap { element in
            guard case .object(let object) = element else { return nil }
            return object.mapValues { $0.toSendable() }
        }
    }

    /// Convert an upstream parsed `ToolCall` into macMLX's `ToolCallRequest`.
    /// `ToolCallProcessor` normalises every drained call to a non-empty id, but
    /// we synthesise `call_<uuid>` defensively if one is ever missing.
    static func toolCallRequest(from call: ToolCall) -> ToolCallRequest {
        let id: String
        if let callID = call.id, !callID.isEmpty {
            id = callID
        } else {
            id = "call_\(UUID().uuidString)"
        }
        let arguments = call.function.arguments.mapValues { ToolValueBridge.jsonValue(from: $0) }
        return ToolCallRequest(id: id, name: call.function.name, arguments: arguments)
    }

    /// Convert a macMLX `ToolCallRequest` back into an upstream `ToolCall` so the
    /// chat template can render it as an assistant tool-call block. Uses the
    /// `[String: any Sendable]` `Function` initialiser (no `as!`).
    static func upstreamToolCall(from request: ToolCallRequest) -> ToolCall {
        ToolCall(
            function: .init(
                name: request.name,
                arguments: request.arguments.mapValues { $0.toSendable() }
            ),
            id: request.id
        )
    }

    /// Map one macMLX `ChatMessage` to an upstream `Chat.Message`, honouring the
    /// `.tool` role and assistant-issued tool calls. `images` is supplied by the
    /// caller (the VLM path folds in attachments; text and think-block paths
    /// pass `[]`).
    static func upstreamChatMessage(
        from msg: ChatMessage,
        images: [UserInput.Image]
    ) -> Chat.Message {
        // Tool-result turn → upstream tool message carrying the correlating id.
        if msg.role == .tool {
            return .tool(msg.content, id: msg.toolCallID)
        }
        // Assistant turn that issued tool calls → attach them so the template
        // reproduces the assistant's tool-call block.
        if msg.role == .assistant, let toolCalls = msg.toolCalls, !toolCalls.isEmpty {
            return .assistant(
                msg.content,
                images: images,
                toolCalls: toolCalls.map { upstreamToolCall(from: $0) }
            )
        }
        let role: Chat.Message.Role
        switch msg.role {
        case .user:      role = .user
        case .assistant: role = .assistant
        case .system:    role = .system
        case .tool:      role = .user  // unreachable — handled above
        }
        return Chat.Message(role: role, content: msg.content, images: images)
    }

    /// Test seam: run the streaming `ToolCallProcessor` over pre-detokenized
    /// text `pieces` exactly as `runLLMGeneration`'s loop does, returning the
    /// concatenated display text and the drained tool calls. Pure string
    /// processing — no MLX — so tool-call detection is unit-testable without a
    /// model or Metal.
    static func processToolCallStream(
        format: ToolCallFormat,
        pieces: [String]
    ) -> (display: String, toolCalls: [ToolCallRequest]) {
        let processor = ToolCallProcessor(format: format)
        var display = ""
        for piece in pieces {
            if let out = processor.processChunk(piece), !out.isEmpty {
                display += out
            }
        }
        if let residual = processor.processEOS(returnBufferedText: true), !residual.isEmpty {
            display += residual
        }
        // `toolCalls` (public) mirrors the internal `drainToolCalls()` for a
        // one-shot read; see `runLLMGeneration`.
        let calls = processor.toolCalls.map { toolCallRequest(from: $0) }
        return (display, calls)
    }
}
