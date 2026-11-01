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
///
/// `internal` (not `private`) so the batch-serving extension can carry a
/// non-`Sendable` `BatchDecodeSession` across `container.perform` segments (H1).
struct NonSendableBox<T>: @unchecked Sendable {
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

    /// Weight-identity fingerprint (``ModelFingerprint/compute(directory:loader:fileManager:)``)
    /// of the resident model's on-disk bytes, captured at load beside
    /// ``loadedModel``. Threaded into every cold-tier prompt-cache read/write so a
    /// restored KV snapshot is only ever reused against the exact weights it was
    /// computed from. `nil` when the model has no readable `config.json` — the
    /// cold tier treats that as "never reuse", never a wildcard match.
    private var loadedModelFingerprint: String?

    /// Version string including the mlx-swift-lm library tag.
    public let version: String = "mlx-swift-lm 3.32.3"

    // MARK: Private state

    /// What's currently loaded — text-only LLM (`MLXLLM`), vision-
    /// language VLM (`MLXVLM`), or nothing. Both modalities wrap a
    /// `ModelContainer`; the case discriminates which factory built
    /// it so generation can choose the right code path (an LLM, and a VLM
    /// served on the full path, get the prompt cache; any other VLM bypasses
    /// it — multimodal cache keys would need to fold image bytes into the
    /// hash, deferred to a follow-up). An LLM, and a full-path VLM, keep the
    /// message generator their input processor renders with, so
    /// `userInput(for:isVLM:messageGenerator:)` can build the message
    /// dictionaries itself when history carries reasoning.
    private enum LoadedSupport {
        case none
        case llm(ModelContainer, messageGenerator: any MessageGenerator)
        /// `fullPath` is non-nil when the VLM runs on the text generation
        /// path (see ``FullPathVLM``), with the generator the engine renders
        /// its prompts with (``FullPathMessageGenerator`` unless the
        /// configuration names one).
        case vlm(ModelContainer, fullPath: (FullPathVLM, messageGenerator: any MessageGenerator)?)

        var container: ModelContainer? {
            switch self {
            case .none: return nil
            case .llm(let c, _): return c
            case .vlm(let c, _): return c
            }
        }

        /// The message generator of an LLM or a full-path VLM; nil for any
        /// other VLM or nothing loaded.
        var messageGenerator: (any MessageGenerator)? {
            switch self {
            case .llm(_, let generator): return generator
            case .vlm(_, let fullPath): return fullPath?.messageGenerator
            case .none: return nil
            }
        }

        var isVLM: Bool {
            if case .vlm = self { return true }
            return false
        }

        /// The full path a VLM is served on; nil for an LLM, another VLM or
        /// nothing loaded.
        var fullPath: FullPathVLM? {
            if case .vlm(_, let fullPath) = self { return fullPath?.0 }
            return nil
        }
    }

    private var loadedSupport: LoadedSupport = .none

    /// The resident LLM's message generator (nil for a VLM, full path or not,
    /// and for no model), for the batched path's prompt build, which lives in
    /// another file. A VLM never batches.
    var loadedLLMMessageGenerator: (any MessageGenerator)? {
        if case .llm(_, let generator) = loadedSupport { return generator }
        return nil
    }

    /// Test seam: load a VLM of a family ``FullPathVLM`` knows on the full
    /// path even when the family is not in ``FullPathVLM/servedFamilies``
    /// yet, so the gated suites and the parity gate can run it before it
    /// flips.
    private(set) var forcesFullPathVLM = false

    /// Test seam for ``forcesFullPathVLM``; takes effect at the next load.
    func setForcesFullPathVLM(_ forced: Bool) {
        forcesFullPathVLM = forced
    }

    /// Test seam: load every VLM on the stock VLM path, a family in
    /// ``FullPathVLM/servedFamilies`` included, for a reference run of the
    /// decoder image requests took before the full path. Wins over
    /// ``forcesFullPathVLM``; takes effect at the next load.
    private(set) var keepsStockVLMPath = false

    /// Test seam for ``keepsStockVLMPath``.
    func setKeepsStockVLMPath(_ keeps: Bool) {
        keepsStockVLMPath = keeps
    }

    /// Test seam: whether the resident model is a VLM served on the full
    /// path.
    var servesFullPathVLM: Bool { loadedSupport.fullPath != nil }

    /// The drain the engine runs after releasing a resident model, from
    /// `unload()`, from the catches in `load()` and `ensureDraftContainer`,
    /// and from `endGenerationRun()` when one was deferred:
    /// `EngineMemory.releaseCachedBuffers` in production. A test seam, internal
    /// like `hasDraftContainer`: the SPM test job has no metallib, so the test
    /// that proves a never-loaded engine's unload stays away from MLX has to
    /// observe the call rather than let it happen.
    private var releaseCachedBuffers: @Sendable () -> Void = { EngineMemory.releaseCachedBuffers() }

    /// Test seam: replace the drain `unload()` performs (see
    /// `releaseCachedBuffers`).
    func setReleaseCachedBuffers(_ release: @escaping @Sendable () -> Void) {
        releaseCachedBuffers = release
    }

    /// Whether a drain is safe although this engine holds nothing: some model
    /// has loaded in the process, so Metal is up. `EngineMemory.hasLoadedAModel`
    /// in production, a seam so the SPM tests decide it themselves instead of
    /// inheriting it from whatever ran before them in the process.
    private var metalKnownUp: @Sendable () -> Bool = { EngineMemory.hasLoadedAModel }

    /// Test seam: replace the process-wide "a model has loaded" answer (see
    /// `metalKnownUp`).
    func setMetalKnownUp(_ known: @escaping @Sendable () -> Bool) {
        metalKnownUp = known
    }

    /// Whether this engine holds a model through any of its three references:
    /// the container, a draft, or the batch-serving copy. Each is set only
    /// after weights were evaluated, so any of them means Metal is up; and
    /// the engine is an actor, so a draft or batch probe interrupted by an
    /// unload can outlive the container it came with.
    private var holdsAModel: Bool {
        loadedSupport.container != nil || draftContainer != nil || batchServingContainer != nil
    }

    /// Runs holding this engine's containers: `runGeneration` (which waits
    /// for mlx-swift-lm's worker task before it counts itself over), the
    /// batch drive loop, `promptOpensThinkBlock` and `applyAdapter`, each
    /// bracketed by `beginGenerationRun()` and `endGenerationRun()`. An
    /// unload that lands while one runs cannot free the weights, since the
    /// run holds the container; it leaves `drainWhenIdle` set instead, and
    /// the last run to end drains (#136). The speculative path has no worker
    /// task to wait for (mlx-swift-lm 3.32.3's raw-token
    /// `generateTokens(…draftModel:)` discards it): its run ends as soon as
    /// the consumer leaves, while the iterator finishes the speculative round
    /// it is in (draft, verify, synchronize) on its own, so a drain deferred
    /// to that run's end can come before the weights are let go, and then
    /// they wait for the next drain, the next generation in the process that
    /// steps mlx-swift-lm's plain `TokenIterator` (text, constrained or
    /// vision: its decode loop clears MLX's cache at its first token, while
    /// a speculative round and a batched cohort never do), or allocator
    /// pressure.
    private var generationsInFlight = 0
    private var drainWhenIdle = false

    func beginGenerationRun() {
        generationsInFlight += 1
    }

    func endGenerationRun() {
        assert(generationsInFlight > 0, "endGenerationRun without a matching begin")
        generationsInFlight = max(0, generationsInFlight - 1)
        if generationsInFlight == 0, drainWhenIdle {
            drainWhenIdle = false
            releaseCachedBuffers()
        }
    }

    /// Drain now, or once the last generation running on this engine ends.
    /// The one rule for every drain the engine decides on: a running
    /// generation holds the model, and a drain before it lets go frees
    /// nothing.
    private func drainOrDefer() {
        if generationsInFlight > 0 {
            drainWhenIdle = true
        } else {
            releaseCachedBuffers()
        }
    }

    /// Wait for mlx-swift-lm's token worker, which holds the model until it
    /// returns: it checks for cancellation only between decode steps and
    /// synchronizes the GPU stream before it lets go. Its stream cancels it
    /// only when the stream is dropped or a consumer is cancelled while
    /// waiting on it; `runGeneration` keeps the stream alive in its frame, so
    /// a loop that leaves the stream early would leave the worker decoding
    /// to EOS or maxTokens. It is cancelled here first: a no-op when the
    /// stream ran to its end (the worker is then past its last cancellation
    /// check and its `finish()`, with nothing cancellable after them in
    /// mlx-swift-lm 3.32.x, which this depends on) or the consumer was
    /// cancelled while waiting (the stream did it), and what stops the worker
    /// otherwise. Waiting is what lets `endGenerationRun()` count the run as
    /// over only once the worker has let go of the model (#136); on the one
    /// exit that records the prompt cache, the natural end, the worker's
    /// last cache write already precedes the stream's end.
    static func awaitWorker(_ worker: Task<Void, Never>) async {
        worker.cancel()
        await worker.value
    }

    /// Test seam: hold `container` (and a draft or batch-serving reference)
    /// the way a load would, without evaluating any weights, so the SPM tests
    /// can drive `unload()` and `load()` over a resident model without Metal.
    /// Internal, like `hasDraftContainer`.
    func installForTesting(
        container: ModelContainer?, draft: ModelContainer? = nil, batch: ModelContainer? = nil,
        model: LocalModel, fullPath: FullPathVLM? = nil
    ) {
        loadedSupport = container.map { container in
            if let fullPath {
                return .vlm(container, fullPath: (fullPath, fullPath.messageGenerator))
            }
            return .llm(container, messageGenerator: DefaultMessageGenerator())
        } ?? .none
        draftContainer = draft
        loadedDraftModelID = draft == nil ? nil : "\(model.id)-draft"
        batchServingContainer = batch
        // A draft or batch copy left behind without its target has no loaded
        // model and no ready status, which is the shape the residency test
        // must see through.
        loadedModel = container == nil ? nil : model
        status = container == nil ? .idle : .ready(model: model.id)
    }

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

    /// Identity of the LoRA adapter currently layered on the resident model
    /// (Track E per-request adapter hot-swap), or nil when the base model is
    /// resident with no adapter. `reconcileAdapter` compares this to a request's
    /// `adapters` to decide keep / apply / reload. Reset to nil by every
    /// `load(_:)` / `unload()` — a fresh model container carries no adapter, and
    /// mlx-swift-lm's `LanguageModel.load(adapter:)` is additive-only (no unapply),
    /// so a base-model reload is the reliable way to shed a resident adapter.
    private var loadedAdapterID: String?

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

    /// Optional bottleneck-classifier seam (v0.7 silicon track, W2). When an
    /// observer is attached the generation paths report their prefill → decode →
    /// complete phase timeline to it; when nil — the default — every phase call
    /// site is a no-op, so generation is byte-for-byte the pre-W2 path.
    private let siliconObserver: SiliconEngineObserver?

    /// Two-tier prompt cache (hot dict + cold safetensors sidecar). Used
    /// by `runGeneration` to reuse KV state across successive turns on
    /// the same LLM or full-path VLM. Other VLM generations, and requests
    /// that carry images, bypass it.
    private let promptCacheStore: PromptCacheStore

    /// Per-model token→JSON-classification tables (Track C structured output).
    /// Built once per resident model on the first constrained request and reused
    /// thereafter, so the vocabulary-decode cost is amortized. Empty and inert
    /// for every unconstrained request.
    private let tokenVocabularyCache = TokenVocabularyCache()

    /// Guards the one-time `ModelOverlay.registerAll()` so macMLX-owned
    /// architectures are registered into the shared factory registry
    /// exactly once, before the first model load. `registerAll` is itself
    /// idempotent; this flag just avoids the redundant actor hop.
    private var overlayRegistered = false

    // MARK: Batch serving (A2d-2)
    //
    // These back the `BatchGenerationServing` seam implemented in
    // `MLXSwiftEngine+BatchGenerationServing.swift`. They are `internal` (not
    // `private`) only so that extension can reach them; nothing here is `public`.
    // A `ModelContainer`'s serial-access mutex is the model-mutex primitive the
    // batch drive loop and every other `container.perform` converge on — see the
    // extension's type doc.

    /// MLX-free coordinator for the seam's admission queue, drain epoch, and
    /// per-row cancellation. Persists across cold-swaps on this engine instance.
    let batchCoordinator = BatchServingCoordinator()

    /// Whether the RESIDENT model can be batched — set once per load by a cheap
    /// MLX-free coverage probe (dense caches + verified-`ropeOffset` architecture,
    /// LLM only). `false` routes every request to the legacy single-stream path.
    var batchServingCoverage = false

    /// The resident model's complete stop-token set (config `eosTokenIds` +
    /// tokenizer EOS + encoded `extraEOSTokens`), captured at load so batched decode
    /// stops each row exactly as the single-stream path does.
    var batchServingEOSTokenIds: Set<Int> = []

    /// The resident tokenizer's unknown-token id, treated as a stop token like the
    /// single-stream path.
    var batchServingUnknownTokenId: Int?

    /// Serial admission tokenizer for the batched path, built once per load with a
    /// DEDICATED tokenizer instance (separate from `context.tokenizer`) so concurrent
    /// admissions neither race each other nor the drive loop's detokenizer, all while
    /// staying OFF the container mutex the drive loop holds for whole cohorts (H2).
    /// Nil unless the resident model is batchable.
    var batchAdmissionProcessor: BatchAdmissionProcessor?

    /// The resident model's container, captured at load for the batched decode loop.
    /// Nil unless the resident model is batchable.
    var batchServingContainer: ModelContainer?

    /// Hard per-row token ceiling for the batched path (safety net above each row's
    /// own `maxTokens`). No lower than the server's default output budget
    /// (`HummingbirdServer.defaultMaxOutputTokens`), or a batched request that
    /// names no limit would stop short of the budget a single-stream one gets.
    /// The cost: every row of a cohort (32 by default) may now run eight times
    /// as long as at the old 4096, with no memory guard over the batch, and a
    /// model swap waits for the cohort to drain. Routing requests without a
    /// limit away from the batcher instead would mean they never batch.
    static let batchServingGlobalMaxTokens = 32_768

    /// Max decode steps the drive loop runs in ONE `container.perform` before it
    /// returns to release the model mutex, then re-acquires it to continue the same
    /// cohort (H1). This bounds how long a single-stream `container.prepare` (GUI
    /// chat, speculative, VLM) can wait behind an active batch cohort: at most one
    /// budget window, NOT the cohort's whole multi-thousand-token lifetime. 32 steps
    /// is a fraction of a second at typical decode rates — short enough that a queued
    /// single-stream request is served promptly (the vendored `AsyncMutex` is
    /// FIFO-fair), long enough to amortize the perform/mutex re-acquire cost so
    /// batched throughput stays high.
    static let batchServingStepBudget = 32

    // MARK: Initialiser

    /// - Parameters:
    ///   - siliconObserver: optional W2 bottleneck-classifier seam. Defaults to
    ///     nil, keeping the generation path unchanged; pass one to receive
    ///     per-generation phase-transition notifications.
    ///   - promptCache: prompt-cache budget for the two-tier store. Defaults to
    ///     ``PromptCacheConfig``'s safe production defaults (bounded hot bytes,
    ///     bounded cold disk, cold tier on); the GUI + CLI construction sites
    ///     override it with the user's persisted `Settings` values.
    ///
    /// The cold tier lives under ``defaultPromptCacheRoot()``.
    public init(
        siliconObserver: SiliconEngineObserver? = nil,
        promptCache: PromptCacheConfig = PromptCacheConfig()
    ) {
        self.init(
            siliconObserver: siliconObserver, promptCache: promptCache,
            promptCacheRoot: Self.defaultPromptCacheRoot())
    }

    /// Where an engine keeps its prompt cache's cold tier unless told
    /// otherwise: `~/.mac-mlx/kv-cache`, except in a test runner
    /// (``TestHost``), where every engine gets a fresh directory of its own
    /// under the temporary directory. Building a store sweeps and prunes its
    /// root and `clearPromptCache()` deletes it, so a test engine on the real
    /// root wiped the user's cache. A test that means to use the real root
    /// passes it to ``init(siliconObserver:promptCache:promptCacheRoot:)``.
    static func defaultPromptCacheRoot() -> URL {
        guard TestHost.isActive else { return DataRoot.macMLX("kv-cache") }
        return testPromptCacheParent.appending(path: UUID().uuidString, directoryHint: .isDirectory)
    }

    /// Where ``defaultPromptCacheRoot()`` puts a test engine's cold tier.
    private static let testPromptCacheParent = FileManager.default.temporaryDirectory
        .appending(path: "macmlx-test-kv-cache", directoryHint: .isDirectory)

    /// A test engine whose cold tier stayed empty removes the directory
    /// ``defaultPromptCacheRoot()`` gave it, so unit tests that build engines
    /// by the hundred leave nothing behind. Only under a test runner and
    /// only that directory; one holding files is left for the test that
    /// wrote them.
    deinit {
        guard TestHost.isActive else { return }
        let root = promptCacheStore.root
        guard root.deletingLastPathComponent().standardizedFileURL == Self.testPromptCacheParent.standardizedFileURL,
            (try? FileManager.default.contentsOfDirectory(atPath: root.path))?.isEmpty == true
        else { return }
        try? FileManager.default.removeItem(at: root)
    }

    /// Test seam: the directory this engine's cold tier lives under.
    var promptCacheRoot: URL {
        get async { await promptCacheStore.root }
    }

    /// Test seam: the same engine with its prompt cache's cold tier under
    /// `promptCacheRoot`, so a test that clears or fills the cold tier never
    /// touches the user's `~/.mac-mlx/kv-cache`.
    init(
        siliconObserver: SiliconEngineObserver? = nil,
        promptCache: PromptCacheConfig = PromptCacheConfig(),
        promptCacheRoot: URL
    ) {
        self.siliconObserver = siliconObserver
        self.promptCacheStore = PromptCacheStore(
            root: promptCacheRoot,
            maxEntries: promptCache.maxEntries,
            maxBytes: promptCache.hotBytes,
            coldCapBytes: promptCache.coldCapBytes,
            coldEnabled: promptCache.coldEnabled
        )
    }

    /// Build the phase-context config the observer receives from the applied
    /// MLX generation parameters. `kvGroupSize` / `quantizedKVStart` are surfaced
    /// only when the KV cache is actually being quantized (they are inert
    /// otherwise). `batchSize` is 1 for every single-stream generation path.
    private static func generationConfig(
        from params: GenerateParameters,
        batchSize: Int
    ) -> EngineGenerationConfig {
        let quantizing = params.kvBits != nil
        return EngineGenerationConfig(
            kvBits: params.kvBits,
            kvGroupSize: quantizing ? params.kvGroupSize : nil,
            quantizedKVStart: quantizing ? params.quantizedKVStart : nil,
            batchSize: batchSize
        )
    }

    /// Assemble the terminal `GenerationPhaseSummary` from an engine completion
    /// record. Token counts default to 0 and timings to nil when the stream
    /// ended without a completion record, so the summary never fabricates a rate.
    ///
    /// NOTE: `promptTokenCount` is the count actually processed during prefill —
    /// on a prompt-cache HIT that is the incremental suffix, NOT the full prompt
    /// (the wire `usage.promptTokens` adds `reusedPromptTokens` back). This is the
    /// right denominator for prefill tok/s (suffix tokens ÷ suffix time), but a
    /// consumer comparing it to `usage.promptTokens` will see them differ on hits.
    static func phaseSummary(
        config: EngineGenerationConfig,
        completionInfo: GenerateCompletionInfo?,
        stopTokenCounted: Bool = false
    ) -> GenerationPhaseSummary {
        GenerationPhaseSummary(
            config: config,
            promptTokenCount: completionInfo?.promptTokenCount ?? 0,
            // The loop counts the stop token it delivered (`includeStopToken`,
            // #142); the summary keeps counting the tokens that became text.
            generationTokenCount: max(0, (completionInfo?.generationTokenCount ?? 0) - (stopTokenCounted ? 1 : 0)),
            promptSeconds: completionInfo?.promptTime,
            generateSeconds: completionInfo?.generateTime
        )
    }

    // MARK: InferenceEngine

    /// Load a model from its local directory into memory.
    ///
    /// - Parameter model: The ``LocalModel`` to load. `model.directory` must contain
    ///   a valid MLX model (`config.json`, `.safetensors` weights, tokenizer files).
    /// - Throws: ``EngineError/modelLoadFailed(reason:)`` if loading fails for any reason.
    public func load(_ model: LocalModel) async throws {
        // Release whatever is resident before the new model allocates (keeping
        // the first until the second had finished allocating is the shape #130
        // found in the reranker swap). On an empty engine it resets state and
        // drains nothing. Remembered, because a load that fails after allocating (a
        // tokenizer the factory awaits only once the weights are in) leaves
        // those buffers in MLX's cache, and a drain is safe only once Metal is
        // known to be up: a model having been resident here, or any model
        // having loaded in this process (`EngineMemory.hasLoadedAModel`).
        let wasResident = holdsAModel
        try await unload()
        status = .loading(model: model.id)

        // Teach the stock factory about macMLX-owned architectures before
        // the first load. No-op until an overlay architecture ships, but
        // wired now so `LLMModelFactory.shared.loadContainer` can resolve
        // our `model_type`s the moment one is registered. See ModelOverlay.
        if !overlayRegistered {
            await ModelOverlay.registerAll()
            overlayRegistered = true
        }

        let configURL = model.directory.appending(
            path: "config.json", directoryHint: .notDirectory)
        do {
            let support: LoadedSupport
            switch model.format {
            case .mlx:
                let container = try await LLMModelFactory.shared.loadContainer(
                    from: model.directory,
                    using: HuggingFaceTokenizerLoader()
                )
                // Backfill the tool-call format upstream `ToolCallFormat.infer`
                // leaves nil for some tool-capable families (e.g. plain Qwen3),
                // so the streaming `ToolCallProcessor` actually fires for them.
                // Applied via the container's own `update` seam and gated on nil
                // ⇒ a format upstream already set is never overridden, and every
                // downstream read (streaming / non-streaming / batch / speculative)
                // sees the corrected value.
                if let fallback = Self.inferToolCallFormatFallback(configURL: configURL) {
                    await container.update { ctx in
                        if ctx.configuration.toolCallFormat == nil {
                            ctx.configuration.toolCallFormat = fallback
                        }
                    }
                }
                // The generator the factory gave the input processor, chosen
                // the same way (the configuration's, else the model's, else
                // the default): Gemma's drops system messages.
                let messageGenerator = await container.perform { context -> any MessageGenerator in
                    context.configuration.messageGenerator
                        ?? (context.model as? LLMModel)?.messageGenerator(tokenizer: context.tokenizer)
                        ?? DefaultMessageGenerator()
                }
                support = .llm(container, messageGenerator: messageGenerator)

            case .mlxVLM:
                let container = try await VLMModelFactory.shared.loadContainer(
                    from: model.directory,
                    using: HuggingFaceTokenizerLoader()
                )
                // Decided again here, from `config.json`, rather than taken
                // from `model.architecture`, which hand-built models leave
                // nil. A failure above is not retried through the LLM
                // factory, so a broken vision load is never masked as text.
                if !keepsStockVLMPath,
                   let fullPath = FullPathVLM.resolve(configURL: configURL, forced: forcesFullPathVLM) {
                    // The same backfill as the LLM branch: these requests
                    // stream through the `ToolCallProcessor` too.
                    if let fallback = Self.inferToolCallFormatFallback(configURL: configURL) {
                        await container.update { ctx in
                            if ctx.configuration.toolCallFormat == nil {
                                ctx.configuration.toolCallFormat = fallback
                            }
                        }
                    }
                    // The configuration's generator when it names one, as the
                    // factory wraps the processor in it; otherwise the full
                    // path's, which renders a message without media as the
                    // text load does. `userInput` renders through it.
                    let configured = await container.perform { context in
                        context.configuration.messageGenerator
                    }
                    support = .vlm(
                        container,
                        fullPath: (fullPath, configured ?? fullPath.messageGenerator))
                } else {
                    support = .vlm(container, fullPath: nil)
                }

            case .gguf, .unknown, .embedder, .reranker, .audioSTT, .audioTTS:
                // Surfaced via the Models tab — these formats never
                // reach the engine in practice, but throw a clean
                // error if someone hand-constructs a `LocalModel`.
                // Embedder models are served by `EmbeddingEngine`,
                // reranker (cross-encoder) models by `RerankEngine`, and
                // speech models by `AudioEngine` — not this generation
                // engine.
                let reason = "Unsupported model format: \(model.format.rawValue). " +
                    "MLXSwiftEngine handles `mlx` (text) and `mlxVLM` (vision-language) only."
                status = .error(reason)
                loadedSupport = .none
                loadedModel = nil
                throw EngineError.modelLoadFailed(reason: reason)
            }
            // Wave 2a: fingerprint the resident weights so the cold prompt-cache
            // tier can reject a restored KV snapshot whose weights changed under
            // the same directory path (a re-download / re-quantize / swap).
            // A VLM load salts it with its loader, so a cold entry written by
            // the text-path class of the same checkpoint is never restored
            // into the VLM class, and the reverse.
            let newFingerprint = ModelFingerprint.compute(
                directory: model.directory, loader: support.isVLM ? "vlm" : nil)
            EngineMemory.noteModelLoaded()
            // No clear-on-reload here for the HOT tier: `PromptCacheStore.fetchNearest`
            // now gates every hot hit on this fingerprint, so a stale entry left by a
            // swap-weights-at-same-path reload misses at the point of reuse — safe
            // regardless of the load/unload/switch sequence, which a same-id-only
            // clear here could not cover.
            loadedSupport = support
            loadedModel = model
            loadedModelFingerprint = newFingerprint
            // A2d-2: probe batch-serving coverage for the resident model once, while
            // the container's serial mutex is free (the drive loop later holds it for
            // whole cohorts). A VLM never batches, so clear the state for it.
            switch support {
            case .llm(let container, _):
                await refreshBatchServingCoverage(container: container)
            case .vlm, .none:
                clearBatchServingState()
                await batchCoordinator.resumeAdmissions()
            }
            // Symmetric with `unload()`: a freshly-loaded target model must
            // not inherit a draft container that was paired with whatever
            // model was previously loaded on this actor instance — a stale
            // draft left resident across a target swap only wastes memory
            // (the tokenizer-compatibility check in
            // `runSpeculativeLLMGeneration` already fails safely on a
            // mismatched pairing either way). The `unload()` above dropped
            // the draft; a generation that ran meanwhile may have installed
            // one, which goes the same way.
            releaseDraft()
            // A freshly-loaded model carries no LoRA adapter (Track E).
            loadedAdapterID = nil
            status = .ready(model: model.id)
        } catch let engineError as EngineError {
            // Already shaped — preserve the typed error.
            loadedSupport = .none
            loadedModel = nil
            if wasResident || metalKnownUp() { drainOrDefer() }
            throw engineError
        } catch {
            let reason = error.localizedDescription
            status = .error(reason)
            loadedSupport = .none
            loadedModel = nil
            if wasResident || metalKnownUp() { drainOrDefer() }
            throw EngineError.modelLoadFailed(reason: reason)
        }
    }

    /// macMLX-side backfill for `ModelConfiguration.toolCallFormat` when upstream
    /// `ToolCallFormat.infer` leaves it nil for a tool-capable family. Upstream
    /// maps only `qwen3_5` / `qwen3_next` (→ `.xmlFunction`), so plain Qwen3 —
    /// and Qwen2 / Qwen2.5 — load with NO format and their hermes-style
    /// `<tool_call>{JSON}</tool_call>` output (the `.json` format) is never
    /// parsed: the streaming `ToolCallProcessor` doesn't run and the tool call
    /// leaks into visible content. This returns `.json` for those Qwen LLM
    /// variants and nil for everything else; the caller applies it ONLY when the
    /// upstream format is nil, so a format upstream already set is never
    /// overridden. Read from `config.json`'s `model_type`; any IO/JSON error is
    /// treated as "no fallback".
    static func inferToolCallFormatFallback(configURL: URL) -> ToolCallFormat? {
        guard let data = try? Data(contentsOf: configURL),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let modelType = root["model_type"] as? String
        else {
            return nil
        }
        let type = modelType.lowercased()
        // Upstream already handles these (→ `.xmlFunction`); never shadow them.
        if type.hasPrefix("qwen3_5") || type.hasPrefix("qwen3_next") { return nil }
        // Qwen2 / Qwen2.5 / Qwen3 (incl. `_moe`) speak the hermes `<tool_call>` +
        // JSON body format, which is `ToolCallFormat.json`.
        if type.hasPrefix("qwen2") || type.hasPrefix("qwen3") { return .json }
        return nil
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
        // A resident model is what proves Metal is up. Decided before the state
        // is reset, because the drain below must not run for an engine that
        // never held one: a load that failed before reaching MLX, or the SPM
        // test job with no metallib.
        let releasedResident = holdsAModel
        loadedSupport = .none
        loadedModel = nil
        // No model resident ⇒ no fingerprint (Wave 2a). Harmless if left stale
        // (it is overwritten on the next load before any read), but cleared for
        // hygiene so no unload can leave a dangling weight identity behind.
        loadedModelFingerprint = nil
        draftContainer = nil
        loadedDraftModelID = nil
        // No model resident ⇒ no adapter (Track E).
        loadedAdapterID = nil
        // A2d-2: no model resident ⇒ nothing is batchable. Any in-flight cohort's
        // drive loop captured its container locally and finishes on its own; new
        // submits decline (coverage now false). The server drains the cohort
        // before reaching here on a swap; the pool's eviction and sweep skip a
        // model with live rows (POOL-3), though a run can still be finishing
        // when they land, since the pool counts a generation over when the
        // consumer's loop ends; `ModelPool.unload(_:)` (the GUI's unload) does
        // not skip at all. The deferred drain below covers both.
        clearBatchServingState()
        // The engine's references are gone, but MLX keeps the model's buffers
        // in its cache until an allocation reuses them, the cache passes its
        // limit, or a plain generation's decode loop clears it (mlx-swift-lm
        // 3.32.3's `TokenIterator` does at its first token and every 256
        // after; speculative and batched runs never do, and nothing does
        // while nothing generates): after
        // unloading a 12B 4-bit checkpoint the process still held 7,021 MB of
        // the 7,035 MB the model had used (#136). Hand them to the OS now, so
        // the pool's eviction makes room before the next model allocates
        // (POOL-5), and the idle sweep and the Models tab's unload shrink the
        // process. Only after a resident model was released, the terms the
        // embedder, reranker and audio swaps use (#133, #134), and only after
        // `clearBatchServingState()` above, which drops the last reference the
        // engine itself holds. A generation still running on the model holds
        // its own reference: then the drain waits for the last one to finish
        // (`endGenerationRun`). The next load allocates its buffers afresh
        // instead of reusing cached ones.
        if releasedResident { drainOrDefer() }
        status = .idle
    }

    /// Test seam: whether a draft container is currently resident. Internal
    /// (not `private`) so `@testable import` tests can observe D1
    /// draft-state resets (e.g. after `unload()` / a successful `load(_:)`)
    /// without needing a real model load.
    var hasDraftContainer: Bool { draftContainer != nil }

    /// Test seam: entries in the prompt cache's in-memory tier, for the gated
    /// smoke that checks a cancelled response records none.
    var promptCacheResidentCount: Int {
        get async { await promptCacheStore.residentCount }
    }

    /// Test seam: entries in the prompt cache's cold tier. A snapshot larger
    /// than the in-memory tier (Gemma 4 31B's at 1.5k tokens) is recorded
    /// only here.
    var promptCacheColdCount: Int {
        get async { await promptCacheStore.coldEntriesSnapshot().count }
    }

    /// Test seam: bytes held by the prompt cache's in-memory tier, for the
    /// gated smoke that measures what a history-boundary snapshot costs.
    var promptCacheResidentBytes: Int {
        get async { await promptCacheStore.residentBytes }
    }

    /// What the last text generation did with its cache (#142): the key's
    /// length (the stop token included when the stream ended on one), the
    /// positions the cache reported once the stream ended (0 for a layout
    /// whose caches report none, such as a `CacheList`), whether the stream
    /// ended on a stop token, and whether the entry was recorded. A test seam,
    /// reset at the start of every text generation.
    struct PromptCacheRecord: Equatable, Sendable {
        var tokenCount: Int
        var heldPositions: Int
        var endedOnStopToken: Bool
        var recorded: Bool
    }

    private(set) var lastPromptCacheRecord: PromptCacheRecord?

    /// The `hashValue` of the prompt tokens the last text generation
    /// prefilled, so a gated test can check they are the tokens it rendered
    /// (hashes compare within one process). A test seam, reset at the start
    /// of every text-path generation (an LLM's or a full-path VLM's) and left
    /// nil when it takes the speculative path. Other VLM generations don't
    /// touch it, so after one it still holds the previous value.
    private(set) var lastPromptTokensHash: Int?

    /// Whether a text generation may split its prefill at the history
    /// boundary and store the cache there (see ``HistoryBoundary``). Always on
    /// in production; a test seam, so a real-weights test can run its cold
    /// reference as one unsplit prefill.
    private(set) var historyBoundarySnapshotsEnabled = true

    /// Test seam for ``historyBoundarySnapshotsEnabled``.
    func setHistoryBoundarySnapshotsEnabled(_ enabled: Bool) {
        historyBoundarySnapshotsEnabled = enabled
    }

    /// Test seam: when set, a text generation splits its prefill at this
    /// many prompt tokens instead of the rendered history boundary (when it
    /// lies inside the part still to prefill), whatever the gain. A
    /// real-weights test uses it to replay, cold, the exact prefill a cache-
    /// warm turn ran, and expect the warm turn's output bit for bit.
    private(set) var forcedHistoryBoundary: Int?

    /// Test seam for ``forcedHistoryBoundary``.
    func setForcedHistoryBoundary(_ boundary: Int?) {
        forcedHistoryBoundary = boundary
    }

    /// Test seam: wait for the prompt cache's cold-tier writes in flight, so
    /// a test can count what a turn wrote.
    func drainPromptCacheWrites() async {
        await promptCacheStore.drainInFlight()
    }

    /// The key a finished text generation records its cache under, or nil
    /// when nothing may be recorded (#142). mlx-swift-lm's iterator steps the
    /// model on each token before returning it, and its loop breaks on a stop
    /// token without yielding it as text; with `includeStopToken` the stop
    /// token still arrives as the last `.token`, and the engine keeps it out
    /// of the text but in the key, so the key names every position the cache
    /// holds whichever way the stream ended: on a stop token, or on its
    /// budget, where the last token yielded was the last one stepped. A
    /// stream that reports `.stop` without having delivered the token (a
    /// loop that no longer delivers it) leaves a cache one position past any
    /// key this could build, so nothing is recorded; nor without a
    /// completion, nor after a cancel, nor when a token followed the stop
    /// token (a loop that no longer breaks on it: the order the cache holds
    /// is then not the order a key built here would name).
    static func promptCacheKey(
        inputTokens: [Int], generatedTokenIDs: [Int], terminalStopToken: Int?,
        tokenAfterStopToken: Bool = false, stopReason: GenerateStopReason?
    ) -> [Int]? {
        guard let stopReason, !tokenAfterStopToken else { return nil }
        switch stopReason {
        case .stop:
            guard let terminalStopToken else { return nil }
            return inputTokens + generatedTokenIDs + [terminalStopToken]
        case .length:
            return inputTokens + generatedTokenIDs + (terminalStopToken.map { [$0] } ?? [])
        case .cancelled:
            return nil
        }
    }

    /// Per-token bookkeeping of the stop token the loop delivers (#142): the
    /// first token in the stop set is the terminal one and stays out of the
    /// text; anything delivered after it, another stop token included, means
    /// a loop that no longer breaks on it, and the cache is then not recorded
    /// (`promptCacheKey`). Pure, so a test can drive it.
    struct StopTokenTracker: Equatable, Sendable {
        private(set) var terminal: Int?
        private(set) var tokenAfterTerminal = false

        /// Note `token`; true when it is a stop token to keep out of the text.
        mutating func note(_ token: Int, isStop: Bool) -> Bool {
            if terminal != nil { tokenAfterTerminal = true }
            if isStop {
                if terminal == nil { terminal = token }
                return true
            }
            return false
        }
    }

    /// The model's EOS tokens, built as mlx-swift-lm's `buildStopTokenIds` is:
    /// the configuration's EOS ids, the tokenizer's, the extra EOS tokens.
    private static func eosTokenIDs(_ context: ModelContext) -> Set<Int> {
        var ids = context.configuration.eosTokenIds
        if let eos = context.tokenizer.eosTokenId { ids.insert(eos) }
        for token in context.configuration.extraEOSTokens {
            if let id = context.tokenizer.convertTokenToId(token) { ids.insert(id) }
        }
        return ids
    }

    /// The tokens mlx-swift-lm's loop stops on: the EOS tokens plus the
    /// tokenizer's unknown token, which the loop stops on the same way.
    private static func stopTokenIDs(_ context: ModelContext) -> Set<Int> {
        var ids = eosTokenIDs(context)
        if let unknown = context.tokenizer.unknownTokenId { ids.insert(unknown) }
        return ids
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
        // Borrows the container across awaits (a large LoRA takes a while):
        // counted like a generation, see `promptOpensThinkBlock`.
        beginGenerationRun()
        defer { endGenerationRun() }

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

        // Track E: record the resident adapter's identity. Without this, a GUI /
        // direct `applyAdapter` (AppState auto-pin, EngineCoordinator) left
        // `loadedAdapterID` nil even though adapted weights are live — so a later
        // request's `reconcileAdapter` saw `current: nil` and (a) never shed the
        // adapter for `adapters: nil` (`.decide(nil, nil) == .keep`), (b) treated a
        // DIFFERENT adapter as `.applyOnly` (additive → double-stacking), and (c)
        // left `bypassPromptCache` false so base and adapted caches shared a key.
        // `applyAdapter` is only ever invoked on an adapter-free base (the GUI
        // applies right after a fresh load; `reconcileAdapter` reloads the base
        // before its `.reloadThenApply`), so recording the id here — not merging —
        // is correct, and a DIFFERENT resident adapter always routes through the
        // reload-base-then-apply path in `reconcileAdapter`.
        loadedAdapterID = adapter.name

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

    // MARK: Track E — per-request LoRA adapter hot-swap

    /// The action needed to bring the resident adapter in line with a request's
    /// `adapters`, given the currently-applied adapter identity. Pure decision
    /// function (no I/O, no MLX) so the keep/apply/reload state machine is
    /// unit-testable without loading a model — mirrors `DraftContainerAction`.
    ///
    /// mlx-swift-lm's `LanguageModel.load(adapter:)` is additive-only (there is no
    /// "unapply"), so shedding or swapping a resident adapter requires reloading
    /// the base model first — matching mlx-lm's server, which reloads on any
    /// (model, adapter) change and treats an unchanged pairing as a no-op.
    enum AdapterAction: Equatable {
        /// Resident adapter already matches (including both nil) — do nothing.
        case keep
        /// Base model is adapter-free; just apply the requested adapter.
        case applyOnly(id: String)
        /// A resident adapter must be shed with no replacement — reload the base.
        case reloadOnly
        /// A DIFFERENT adapter is requested — reload the base, then apply it.
        case reloadThenApply(id: String)

        static func decide(current: String?, requested: String?) -> Self {
            if current == requested { return .keep }
            switch (current, requested) {
            case (.none, .some(let requested)): return .applyOnly(id: requested)
            case (.some, .none): return .reloadOnly
            case (.some, .some(let requested)): return .reloadThenApply(id: requested)
            case (.none, .none): return .keep  // unreachable (== handled above)
            }
        }
    }

    /// Whether a request must skip the base-`modelID`-keyed prompt cache. Pure so
    /// the bypass rule is unit-testable without a model. Bypass when quantizing the
    /// KV cache (a `QuantizedKVCache` snapshot must not leak into a later plain
    /// request) OR when a LoRA adapter is resident (base and adapted weights share
    /// the same `modelID` key, so reusing a base-model entry for adapted weights —
    /// or vice versa — would poison the cache).
    static func shouldBypassPromptCache(kvBits: Int?, adapterID: String?) -> Bool {
        kvBits != nil || adapterID != nil
    }

    /// Reconcile the resident LoRA adapter with a request's `adapters` BEFORE
    /// generation. A no-op when they already match (the common case: no adapter,
    /// or the same adapter as the previous request). A change reloads the base
    /// model (to shed any resident adapter) and applies the new one.
    ///
    /// - Important: This can reload the model container, so callers MUST invoke it
    ///   before capturing `loadedSupport` / the container for generation.
    /// - Throws: `EngineError.adapterApplyFailed` if the named adapter can't be
    ///   resolved or applied, or a load error if the base-model reload fails.
    private func reconcileAdapter(requested: String?, baseModel: LocalModel) async throws {
        switch AdapterAction.decide(current: loadedAdapterID, requested: requested) {
        case .keep:
            return
        case .applyOnly(let id):
            let adapter = try await resolveAdapter(named: id)
            // `applyAdapter` itself now writes `loadedAdapterID = adapter.name`
            // (== `id`, by `resolveAdapter`'s name lookup) — no separate write needed.
            try await applyAdapter(adapter)
        case .reloadOnly:
            // `load(_:)` resets `loadedAdapterID` to nil.
            try await load(baseModel)
        case .reloadThenApply(let id):
            try await load(baseModel)
            let adapter = try await resolveAdapter(named: id)
            // Same as `.applyOnly` above: `applyAdapter` owns the write.
            try await applyAdapter(adapter)
        }
    }

    /// Resolve a request's `adapters` value (a bare name) to a ``LocalAdapter``
    /// under the adapters root (`~/.mac-mlx/adapters/<name>`). Reuses the
    /// draft-model id allowlist (`isValidDraftModelID`) so the name is a single,
    /// traversal-safe path component — v1 accepts a bare NAME only, not an
    /// absolute path (documented; add path support behind an explicit opt-in).
    ///
    /// - Throws: `EngineError.adapterApplyFailed` when the name is invalid or no
    ///   adapter with that name exists under the adapters directory.
    private func resolveAdapter(named name: String) async throws -> LocalAdapter {
        guard Self.isValidDraftModelID(name) else {
            throw EngineError.adapterApplyFailed(
                reason: "Invalid adapter id '\(name)' — expected a bare name "
                    + "([A-Za-z0-9._-], no leading dot) under the adapters directory.")
        }
        let root = DataRoot.macMLX("adapters")
        let scanned = (try? await AdapterStore().scan(root)) ?? []
        guard let adapter = scanned.first(where: { $0.name == name }) else {
            throw EngineError.adapterApplyFailed(
                reason: "Adapter '\(name)' not found under \(root.path).")
        }
        return adapter
    }

    /// Test seam: the adapter identity currently applied to the resident model,
    /// or nil for the base model. Internal so `@testable import` tests can observe
    /// reconcile decisions without a real model load (mirrors `hasDraftContainer`).
    var currentAdapterID: String? { loadedAdapterID }

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
                // Counted while `runGeneration`'s frame holds the container,
                // so an unload that lands meanwhile drains once it returns.
                await self.beginGenerationRun()
                do {
                    try await self.runGeneration(request, into: continuation)
                } catch {
                    continuation.finish(throwing: error)
                }
                await self.endGenerationRun()
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
    /// does (the same `userInput(for:isVLM:messageGenerator:)` — tools,
    /// reasoning and template kwargs — and `container.prepare`), decodes
    /// the last ``thinkProbeTailTokens`` tokens, and inspects them via
    /// `MessageSegmenter.promptOpensThink`. Returns false if no model is
    /// loaded or the template can't be applied.
    public func promptOpensThinkBlock(_ request: GenerateRequest) async -> Bool {
        let support = loadedSupport
        guard let container = support.container else { return false }
        // Borrows the container across an await: counted like a generation,
        // so an unload that lands meanwhile drains once this lets go (#136).
        beginGenerationRun()
        defer { endGenerationRun() }
        // Without image attachments (`isVLM: false`): this heuristic only
        // inspects the rendered token tail. A full-path load still renders
        // through its generator, as generation does.
        do {
            let lmInput = try await container.prepare(input: Self.userInput(
                for: request, isVLM: false, messageGenerator: support.messageGenerator,
                rendersThroughGenerator: support.fullPath != nil, logsDroppedInputs: false))
            let ids = lmInput.text.tokens.asArray(Int32.self).suffix(Self.thinkProbeTailTokens).map(Int.init)
            let text = await container.decode(tokenIds: ids)
            return MessageSegmenter.promptOpensThink(text)
        } catch {
            return false
        }
    }

    /// How many trailing prompt tokens `promptOpensThinkBlock` decodes. The
    /// `<think>` a model continues is opened by the generation prompt, its
    /// last few tokens (Qwen3's `<|im_start|>assistant\n<think>\n`; with
    /// thinking off, `<think>\n\n</think>\n\n` closes it there too), so
    /// a tail this long holds the last tag, without decoding a long agent
    /// prompt whole on every request. A block opened earlier and left open
    /// for longer than this would be missed; no chat template we know of
    /// renders one.
    static let thinkProbeTailTokens = 64

    /// How many tokens `request`'s prompt renders to: the `UserInput`
    /// generation builds (`userInput(for:isVLM:messageGenerator:)` — tools,
    /// reasoning, template kwargs and, on a VLM, images, whose image tokens
    /// count) through the same
    /// `container.prepare`, so it equals the `promptTokens` a cold-cache
    /// generation of the request reports. Throws `modelNotLoaded` with no
    /// model, and whatever applying the template throws.
    public func countPromptTokens(_ request: GenerateRequest) async throws -> Int {
        let support = loadedSupport
        guard let container = support.container else { throw EngineError.modelNotLoaded }
        // Borrows the container across an await: counted like a generation,
        // see `promptOpensThinkBlock`.
        beginGenerationRun()
        defer { endGenerationRun() }
        let lmInput = try await container.prepare(input: Self.userInput(
            for: request, isVLM: support.isVLM, messageGenerator: support.messageGenerator))
        return lmInput.text.tokens.size
    }

    /// Test seam for the full-path parity gate (`FullPathVLMParitySmokeTests`):
    /// the prompt `request` renders to on the resident load, as generation
    /// feeds it (``FullPathVLM/textOnlyInput(_:)`` on the full path), and the
    /// float32 logits a cold prefill of it gives for the next token, before
    /// any processor or sampler touches them. The prefill is generation's:
    /// a `TokenIterator` over a fresh cache, with the request's prefill
    /// parameters and the full path's iterator state, in one leg (no
    /// history-boundary split). `prefill` overrides the request's prefill
    /// parameters, for a read of a load's own chunking noise. Throws
    /// `modelNotLoaded` with no model.
    func coldNextTokenLogits(
        _ request: GenerateRequest, prefill: PrefillParameters? = nil
    ) async throws -> (tokens: [Int], logits: [Float]) {
        let support = loadedSupport
        guard let container = support.container else { throw EngineError.modelNotLoaded }
        beginGenerationRun()
        defer { endGenerationRun() }
        let prepared = try await container.prepare(input: Self.userInput(
            for: request, isVLM: support.isVLM, messageGenerator: support.messageGenerator,
            logsDroppedInputs: false))
        let fullPath = support.fullPath
        let input = NonSendableBox(fullPath == nil ? prepared : FullPathVLM.textOnlyInput(prepared))
        let generateParams = request.parameters.mlxGenerateParameters
        return try await container.perform(nonSendable: input) { context, input in
            let sink = NextTokenLogitsSink()
            _ = try TokenIterator(
                input: input.value, model: context.model,
                cache: try context.model.newCache(parameters: generateParams),
                state: fullPath?.iteratorState(), processor: NextTokenLogitsCapture(sink: sink),
                sampler: ArgMaxSampler(), prefill: prefill ?? generateParams.prefill, maxTokens: 1)
            return (input.value.text.tokens.asArray(Int.self), sink.logits)
        }
    }

    /// Where ``NextTokenLogitsCapture`` leaves the first logits it sees.
    private final class NextTokenLogitsSink: @unchecked Sendable {
        var logits: [Float] = []
    }

    /// Records the first logits the iterator hands its processor (the
    /// prompt's last position), in float32, and changes nothing.
    private struct NextTokenLogitsCapture: LogitProcessor {
        let sink: NextTokenLogitsSink
        mutating func prompt(_ prompt: MLXArray) {}
        func process(logits: MLXArray) -> MLXArray {
            if sink.logits.isEmpty { sink.logits = logits.asType(.float32).asArray(Float.self) }
            return logits
        }
        mutating func didSample(token: MLXArray) {}
    }

    /// The `UserInput` `request` renders from — the one place generation
    /// (`runGeneration`), `countPromptTokens`, `promptOpensThinkBlock` and the
    /// batched path build it, so all of them render one prompt.
    /// `logsDroppedInputs` is false for a second build of the same request,
    /// so its notes (dropped images, dropped tool specs) are logged once.
    ///
    /// `messageGenerator` is the loaded LLM's or full-path VLM's (nil for any
    /// other VLM). A full-path VLM's input is always that generator's message
    /// dictionaries (``FullPathMessageGenerator``), with any reasoning added.
    /// For an LLM, when an assistant message carries reasoning, the input is
    /// built as the generator's message dictionaries with that reasoning
    /// added as `reasoning_content` (see `injectingReasoning`), which the
    /// input processor passes to the template unchanged: `Chat.Message` has
    /// no reasoning slot. Otherwise it is the `.chat` input it always was, so
    /// a request without reasoning renders exactly as before. The
    /// dictionaries carry only image placeholders, so the input keeps the
    /// `.chat` input's images and videos, which the processor reads from
    /// `UserInput.images` either way. Another VLM has no generator here and
    /// never gets the reasoning.
    static func userInput(
        for request: GenerateRequest,
        isVLM: Bool,
        messageGenerator: (any MessageGenerator)?,
        rendersThroughGenerator: Bool? = nil,
        logsDroppedInputs: Bool = true
    ) -> UserInput {
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
                if logsDroppedInputs, !msg.images.isEmpty {
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
        if logsDroppedInputs, let requested = request.tools, let converted = toolSpecs,
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
        let additionalContext = request.templateKwargs?.mapValues { $0.toSendable() }
        let chatInput = UserInput(chat: chatMessages, tools: toolSpecs, additionalContext: additionalContext)

        let assistantReasoning = request.allMessages.filter { $0.role == .assistant }.map(\.reasoning)
        let hasReasoning = assistantReasoning.contains(where: { $0?.isEmpty == false })
        // A full-path VLM (the only VLM handed a generator) always renders
        // through it: from `.chat`, its processor would render with its
        // family's own generator, which lays out a message without media
        // differently from the text load (see `FullPathMessageGenerator`).
        // A caller that drops the images (`isVLM: false`) says so itself.
        let rendersThroughGenerator = rendersThroughGenerator ?? (isVLM && messageGenerator != nil)
        if let messageGenerator, hasReasoning || rendersThroughGenerator {
            // The generator's own entry point over the input the processor
            // would get, so one that reads `additionalContext` (TranslateGemma)
            // builds what it always builds; its `.messages` output then passes
            // through the processor's `generate(from:)` unchanged.
            let messages = messageGenerator.generate(from: chatInput)
            func rendered(_ messages: [[String: any Sendable]]) -> UserInput {
                UserInput(
                    messages: messages, images: chatInput.images, videos: chatInput.videos,
                    tools: toolSpecs, additionalContext: additionalContext)
            }
            if !hasReasoning { return rendered(messages) }
            if let injected = injectingReasoning(messages, assistantReasoning: assistantReasoning) {
                return rendered(injected)
            }
            if logsDroppedInputs {
                Task.detached {
                    await LogManager.shared.debug(
                        "Not rendering replayed reasoning: the message generator's assistant turns "
                            + "do not match the request's",
                        category: .inference
                    )
                }
            }
            if rendersThroughGenerator { return rendered(messages) }
        }
        return chatInput
    }

    /// `messages` — a message generator's dictionaries for a conversation —
    /// with the k-th assistant dictionary's `reasoning_content` set to the
    /// k-th entry of `assistantReasoning` (the conversation's assistant
    /// messages' reasoning, in order) when that is non-empty. The thinking
    /// templates read that key (Qwen3.x, Mellum2; Gemma 4 also reads
    /// `reasoning`) and render it only where they choose, typically for the
    /// assistant turns after the last user query (Gemma 4: only those with
    /// tool calls).
    ///
    /// A generator may drop messages (Gemma's drops system ones) but never an
    /// assistant one, so the k-th assistant dictionary is the k-th assistant
    /// message. Nil when the counts differ anyway, rather than guess.
    static func injectingReasoning(
        _ messages: [[String: any Sendable]],
        assistantReasoning: [String?]
    ) -> [[String: any Sendable]]? {
        let assistantIndices = messages.indices.filter { messages[$0]["role"] as? String == "assistant" }
        guard assistantIndices.count == assistantReasoning.count else { return nil }
        var out = messages
        for (index, reasoning) in zip(assistantIndices, assistantReasoning) {
            if let reasoning, !reasoning.isEmpty {
                out[index]["reasoning_content"] = reasoning
            }
        }
        return out
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
    ///   `load(_:)` uses for the primary model. The draft resident so far is
    ///   released before the new one loads, so a failure leaves no draft — a
    ///   half-applied swap must never leave a stale, silently-mismatched
    ///   draft resident.
    ///
    /// Internal (not `private`) so the SPM tests can drive the draft's
    /// release and a failed draft load over a weightless container.
    func ensureDraftContainer(requestedDraftModelID: String?) async throws {
        switch DraftContainerAction.decide(
            currentDraftModelID: loadedDraftModelID,
            requestedDraftModelID: requestedDraftModelID
        ) {
        case .keep:
            return
        case .unload:
            releaseDraft()
        case .load(let id):
            // The reference goes before the next one allocates, as in
            // `load()`; the drain itself waits for this run's end, since the
            // previous speculative worker is not awaited and may still hold
            // the old draft (#136).
            releaseDraft()
            do {
                let container = try await LLMModelFactory.shared.loadContainer(
                    from: Self.draftModelDirectory(id: id),
                    using: HuggingFaceTokenizerLoader()
                )
                draftContainer = container
                loadedDraftModelID = id
            } catch {
                // What a part-way failure allocated goes with a drain, under
                // the rule `load()` applies: only once Metal is known to be up.
                if holdsAModel || metalKnownUp() { drainOrDefer() }
                throw EngineError.modelLoadFailed(
                    reason: "Draft model '\(id)' failed to load: \(error.localizedDescription)"
                )
            }
        }
    }

    /// Drop the draft model and hand its buffers back (#136): a draft is a
    /// model too, and its buffers stayed in MLX's cache after every swap and
    /// drop before. Drains only when a draft was held, which is what proves
    /// Metal is up (an unload may have dropped the main model meanwhile).
    private func releaseDraft() {
        let held = draftContainer != nil
        draftContainer = nil
        loadedDraftModelID = nil
        if held { drainOrDefer() }
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

    /// Whether classic per-request speculative decoding (D1) may proceed,
    /// given a cache-trimmability precheck already run against the target
    /// and draft models. Pure decision function (no I/O, no MLX dispatch) —
    /// mirrors `DraftContainerAction.decide`'s "pure decision, real work
    /// stays in the caller" split, so the true/false fallback decision is
    /// unit-testable without loading a model or touching Metal. The actual
    /// trimmability check (`canTrimPromptCache` against a freshly-allocated
    /// probe `[KVCache]`) has to run inside `ModelContainer.perform` against
    /// a real model and can't be stubbed cheaply — this seam starts one step
    /// later, at the two resulting booleans, which IS cheap to stub.
    ///
    /// mlx-swift-lm's `SpeculativeTokenIterator.init` requires BOTH the main
    /// and draft `[KVCache]` to satisfy `canTrimPromptCache` and THROWS
    /// `KVCacheError` otherwise — hybrid/linear-attention architectures
    /// (e.g. Qwen3.5's GatedDeltaNet layers, which allocate a non-trimmable
    /// `MambaCache`) fail this unconditionally and architecture-wide, so a
    /// draft-model request against such a target must be caught by a
    /// precheck and routed to plain decoding instead of letting that throw
    /// bubble up as a request failure. See `runLLMGeneration`'s call site.
    static func canUseSpeculativeDecoding(
        targetCacheIsTrimmable: Bool, draftCacheIsTrimmable: Bool
    ) -> Bool {
        targetCacheIsTrimmable && draftCacheIsTrimmable
    }

    // MARK: Private generation helper

    /// Actor-isolated generation driver called from within `generate(_:)`.
    ///
    /// Flow:
    /// 1. Prepare the `LMInput` (tokenisation + chat template application).
    /// 2. Nearest-prefix (LCP) lookup in `promptCacheStore.fetchNearest`. On a
    ///    hit the store returns a cache already trimmed to the longest shared
    ///    prefix; we prefill only the remaining suffix. On a miss, allocate a
    ///    fresh cache via `model.newCache(...)` and prefill the whole prompt.
    /// 3. Drive the low-level `generateTokensTask(input:cache:...)` call so
    ///    we see raw token IDs and can `insert` the full sequence,
    ///    `inputTokens + generatedTokenIDs` and the stop token the loop
    ///    delivered, after the stream ends (`promptCacheKey`).
    /// 4. The `KVCache` protocol is class-bound — the same reference we
    ///    passed in is mutated in-place during generation, so at the
    ///    end we store that same reference under the full token sequence,
    ///    with the stop token appended when the stream ended on one: the
    ///    iterator had stepped it before the loop saw it (#142).
    private func runGeneration(
        _ request: GenerateRequest,
        into continuation: AsyncThrowingStream<GenerateChunk, Error>.Continuation
    ) async throws {
        guard let loadedModelSnapshot = loadedModel else {
            continuation.finish(throwing: EngineError.modelNotLoaded)
            return
        }
        // Track E: bring the resident LoRA adapter in line with this request
        // BEFORE capturing the container — `reconcileAdapter` may reload the base
        // model (the only reliable way to shed/swap an additive-only adapter), so
        // a stale `support`/`container` captured earlier would point at the old
        // container. A no-op in the common case (no adapter, or unchanged).
        do {
            try await reconcileAdapter(
                requested: request.adapters, baseModel: loadedModelSnapshot)
        } catch {
            continuation.finish(throwing: error)
            return
        }

        let support = loadedSupport
        guard let container = support.container else {
            continuation.finish(throwing: EngineError.modelNotLoaded)
            return
        }
        let isVLM = support.isVLM

        let params = request.parameters

        let generateParams = params.mlxGenerateParameters

        let userInput = Self.userInput(for: request, isVLM: isVLM, messageGenerator: support.messageGenerator)

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

        let fullPath = support.fullPath
        if isVLM, fullPath == nil {
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
            // Track E: the VLM path uses the stock parameter-driven decoder, which
            // has no custom processor/sampler seam — so logit_bias / XTC / logprobs
            // are inert here. Log rather than silently drop (kv_bits DOES apply on
            // this path, via generateParams). Documented v1 limitation.
            if params.logitBias != nil || (params.xtcProbability ?? 0) > 0 || params.logprobs {
                Task.detached {
                    await LogManager.shared.debug(
                        "Ignoring logit_bias / XTC / logprobs on a VLM request — these are "
                            + "supported on the text-only (LLM) path in this version.",
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
            // A full-path VLM's processor hands back `[1, N]` tokens and a
            // mask even for text; the text path slices `[N]`. An input with
            // media keeps its shape, and `carriesMedia` keeps it off every
            // path that slices axis 0.
            let carriesMedia = fullPath != nil && FullPathVLM.carriesMedia(lmInput)
            // D1: bring the draft container in line with this request
            // BEFORE dispatching — throws (rather than silently falling
            // back to non-speculative generation) if the requested draft
            // model fails to load. A request with images never speculates,
            // so, as on the stock VLM path, its `draftModelID` is not
            // reconciled: no draft is loaded for it, or failed on.
            if !carriesMedia {
                try await ensureDraftContainer(requestedDraftModelID: request.draftModelID)
            }
            try await runLLMGeneration(
                lmInput: fullPath == nil ? lmInput : FullPathVLM.textOnlyInput(lmInput),
                // A second copy of the same `UserInput`: `prepare` consumed the
                // first, and the history-boundary snapshot renders it again.
                // The first build already logged anything it dropped.
                historyInput: Self.userInput(
                    for: request, isVLM: false, messageGenerator: support.messageGenerator,
                    logsDroppedInputs: false),
                historyGenerator: support.messageGenerator ?? DefaultMessageGenerator(),
                container: container,
                generateParams: generateParams,
                sampling: params,
                modelID: loadedModelSnapshot.id,
                modelFingerprint: loadedModelFingerprint,
                hasTools: request.tools?.isEmpty == false,
                numDraftTokens: request.numDraftTokens,
                responseFormat: request.responseFormat,
                fullPath: fullPath,
                carriesMedia: carriesMedia,
                into: continuation
            )
        }
    }

    /// Build a raw-token stream with a hand-assembled processor/sampler chain for
    /// any request that needs one — structured-output constraints (Track C) AND
    /// the Track E per-request sampling extensions (`logit_bias`, XTC, `logprobs`).
    /// It installs the composed processor + sampler on a hand-built `TokenIterator`
    /// then runs it through the same `generateTokenTask` the plain path uses, so
    /// all downstream streaming and detokenization is byte-for-byte identical.
    ///
    /// ## Processor order (mlx-lm–aligned, constraint last)
    /// `logit_bias` → penalty (`repetition`/`presence`/`frequency`) →
    /// `JSONConstraintProcessor` mask (if constrained) → `logprobs` capture (if
    /// requested). The constraint mask is applied AFTER any bias/penalty so a bias
    /// can never resurrect a grammar-forbidden token; the logprobs capture is the
    /// outermost observer so it sees the exact distribution the sampler samples.
    ///
    /// ## Sampler
    /// The base sampler from `generateParams.sampler()`, optionally wrapped by
    /// ``XTCSampler`` when `xtcProbability > 0` and `temperature > 0`.
    ///
    /// Runs inside `ModelContainer.perform` (hence `static` + an explicit
    /// `context`) so the non-`Sendable` MLX state never leaves the actor; the
    /// captured `vocabCache`, `tokenizer`, and `logprobsSink` are `Sendable`.
    ///
    /// - Note: this initializer path carries NO KV-cache quantization parameters
    ///   (see `TokenIterator.init(input:model:cache:processor:sampler:…)`), so
    ///   `kv_bits` is inert on any request routed here — the caller logs that.
    private static func customTokenStream(
        input: LMInput,
        cache: [any KVCache],
        state: LMOutput.State?,
        generateParams: GenerateParameters,
        context: ModelContext,
        responseFormat: ResponseFormat?,
        logitBias: [Int: Float]?,
        xtcProbability: Float?,
        xtcThreshold: Float?,
        logprobsSink: LogprobsCaptureProcessor.Sink?,
        topLogprobs: Int,
        modelID: String,
        vocabCache: TokenVocabularyCache
    ) throws -> (AsyncStream<TokenGeneration>, Task<Void, Never>) {
        // logit_bias runs FIRST (mlx-lm `make_logits_processors` order), then the
        // penalty processor. `ChainedLogitProcessor` collapses them into the one
        // `inner` slot the constraint (and the plain path) expect; nil when neither
        // is present.
        let bias = logitBias.flatMap { LogitBiasProcessor(bias: $0) }
        let innerChain = ChainedLogitProcessor([bias, generateParams.processor()])

        var processor: LogitProcessor?
        if let responseFormat {
            // Complete stop-token set, mirroring upstream `buildStopTokenIds`:
            // config EOS ids ∪ the tokenizer's EOS id ∪ any extra stop tokens. The
            // constraint masks every one of these until the document is complete,
            // so the model cannot terminate mid-JSON.
            let stopTokenIDs = Self.eosTokenIDs(context)
            processor = JSONConstraintProcessor(
                format: responseFormat,
                inner: innerChain,
                cache: vocabCache,
                modelID: modelID,
                tokenizer: context.tokenizer,
                stopTokenIDs: stopTokenIDs,
                greedy: generateParams.temperature == 0
            )
        } else {
            processor = innerChain
        }

        // logprobs capture is the OUTERMOST processor so it observes the final
        // logits. v1 does not combine it with a constraint (the server rejects
        // `logprobs` + `response_format`), so here it only ever wraps the plain
        // bias/penalty chain.
        if let logprobsSink {
            processor = LogprobsCaptureProcessor(
                inner: processor, sink: logprobsSink, topN: topLogprobs)
        }

        // XTC wraps the base sampler when requested; only meaningful at
        // temperature > 0 (mlx-lm disables it under argmax).
        var sampler = generateParams.sampler()
        if let xtcProbability, xtcProbability > 0, generateParams.temperature > 0 {
            sampler = XTCSampler(
                base: sampler,
                probability: xtcProbability,
                threshold: xtcThreshold ?? 0.1,
                seed: generateParams.seed
            )
        }

        // Caveat: `maxTokens` bounds a constrained stream just like any other —
        // hitting the cap mid-document yields a valid-but-truncated JSON prefix
        // surfaced as `finish_reason: length` (not `stop`).
        let iterator = try TokenIterator(
            input: input,
            model: context.model,
            cache: cache,
            state: state,
            processor: processor,
            sampler: sampler,
            prefill: generateParams.prefill,
            maxTokens: generateParams.maxTokens
        )

        // The worker task comes back with the stream: the caller waits for it
        // after the stream ends, since it holds the model until it returns.
        // `includeStopToken`: the loop delivers the stop token it breaks on
        // as the last `.token`, so the record can name its position (#142).
        return MLXLMCommon.generateTokenTask(
            promptTokenCount: input.text.tokens.size,
            modelConfiguration: context.configuration,
            tokenizer: context.tokenizer,
            iterator: iterator,
            includeStopToken: true
        )
    }

    /// Text-only path: tokenise, look up the prompt cache, prefill only
    /// the new suffix, and stream tokens. Saves the extended cache back
    /// to `promptCacheStore` once the stream completes.
    ///
    /// When the prompt's history boundary qualifies (see ``HistoryBoundary``,
    /// rendered from `historyInput`), the prefill stops there first, a copy of
    /// the cache is stored under the history's tokens, and generation resumes
    /// from the boundary, so the next turn of an agent loop reuses everything
    /// up to it even on a cache that cannot be trimmed.
    ///
    /// `historyGenerator` is the generator the prompt was rendered with (the
    /// LLM's, or a full-path VLM's processor's), so the history render
    /// matches it.
    ///
    /// A full-path VLM (`fullPath` non-nil) runs here too: every iterator
    /// starts from `fullPath.iteratorState()`. A request whose input
    /// `carriesMedia` records nothing, takes no history boundary and never
    /// speculates: a key of token ids alone would let one image's KV answer
    /// for another, and the boundary and the draft slice a text-only `[N]`
    /// prompt. It reuses the cache only up to its first media block
    /// (``FullPathVLM/imageRequestLookupKey(_:)``), the part of the prompt
    /// whose KV does not depend on pixels.
    private func runLLMGeneration(
        lmInput: LMInput,
        historyInput: UserInput,
        historyGenerator: any MessageGenerator,
        container: ModelContainer,
        generateParams: GenerateParameters,
        sampling: GenerationParameters,
        modelID: String,
        modelFingerprint: String?,
        hasTools: Bool,
        numDraftTokens: Int?,
        responseFormat: ResponseFormat?,
        fullPath: FullPathVLM? = nil,
        carriesMedia: Bool = false,
        into continuation: AsyncThrowingStream<GenerateChunk, Error>.Continuation
    ) async throws {
        lastPromptCacheRecord = nil
        lastPromptTokensHash = nil
        // Track E — per-request sampling extensions. Any of these forces the
        // hand-assembled `customTokenStream` (constraint / logit_bias / XTC /
        // logprobs share it), which is mutually exclusive with the stock
        // parameter-driven path and with speculative decoding.
        let logitBias = sampling.logitBias
        let xtcProbability = sampling.xtcProbability
        let xtcThreshold = sampling.xtcThreshold
        let wantsLogprobs = sampling.logprobs
        let topLogprobs = sampling.topLogprobs ?? 0
        let usesXTC = (xtcProbability ?? 0) > 0 && sampling.samplingTemperature > 0
        let usesCustomPipeline = responseFormat != nil || logitBias != nil || usesXTC || wantsLogprobs
        // This flag also keeps our logit processors out of speculative decoding,
        // where mlx-swift-lm 3.32.3's `LogitProcessor.copy()` would hand the
        // draft round the default struct copy. Ours hold reference-type state
        // (`LogprobsCaptureProcessor`'s boxes, the JSON constraint table, the
        // chained composite's children) that a shallow copy would share.
        // KV-cache quantization rides the stock/speculative/VLM `TokenIterator`
        // (built from `generateParams`), which the custom pipeline bypasses — so a
        // request that combines `kv_bits` with a custom feature gets no
        // quantization. Log it rather than silently drop it (the field docs the
        // exclusion too), and bypass the prompt cache when quantizing so a
        // `QuantizedKVCache` snapshot never leaks into a later plain request.
        if sampling.kvBits != nil, usesCustomPipeline {
            await LogManager.shared.debug(
                "kv_bits is ignored when combined with logit_bias / XTC / logprobs / "
                    + "response_format — the custom decode path carries no KV-cache "
                    + "quantization parameters.",
                category: .inference
            )
        }
        // Bypass the prompt cache when quantizing the KV cache (a QuantizedKVCache
        // snapshot must not leak into a later plain request) OR when a LoRA adapter
        // is resident: `promptCacheStore` is keyed by the BASE `modelID` only, so a
        // base-model cache entry would otherwise be wrongly reused for the adapted
        // model (different weights, same id) — and vice versa. `reconcileAdapter`
        // above has already settled `loadedAdapterID` for this request.
        let bypassesForWeights = Self.shouldBypassPromptCache(
            kvBits: sampling.kvBits, adapterID: loadedAdapterID)
        let bypassPromptCache = carriesMedia || bypassesForWeights

        // Track C: a structured-output request must NOT speculate. A draft
        // model proposes tokens from its own logits, which the constraint mask
        // never sees, so accepted drafts could violate the JSON grammar. When a
        // draft container is resident AND this request is constrained, skip the
        // speculative branch and fall through to the plain (constrained) path.
        if responseFormat != nil, draftContainer != nil {
            await LogManager.shared.debug(
                "Structured output (response_format) disables speculative decoding "
                    + "for this request — the draft model cannot honor the constraint.",
                category: .inference
            )
        }

        // D1: `ensureDraftContainer` (called just before this, in
        // `runGeneration`) has already reconciled `draftContainer` with the
        // request — a resident draft container IS this request's signal to
        // speculate. Fork out to the dedicated path, which bypasses the
        // prompt cache entirely (see `runSpeculativeLLMGeneration`'s doc
        // comment) rather than threading a third branch through the cache
        // logic below. `!usesCustomPipeline` keeps a custom-feature request
        // (constraint / logit_bias / XTC / logprobs) OFF the speculative path,
        // whose `SpeculativeTokenIterator` cannot honor any of them.
        if usesCustomPipeline, draftContainer != nil, responseFormat == nil {
            // The constraint case logged its own diagnostic above; log the
            // Track E cases too so a draft_model + logit_bias/XTC/logprobs
            // request never skips speculation silently.
            await LogManager.shared.debug(
                "Custom sampling features (logit_bias/XTC/logprobs) disable "
                    + "speculative decoding for this request — the speculative "
                    + "iterator cannot honor them; features are applied on the "
                    + "plain path.",
                category: .inference
            )
        }
        if carriesMedia, draftContainer != nil {
            // The draft is a text LLM: it cannot read the image, and its
            // iterator would slice a prompt that keeps the processor's shape.
            await LogManager.shared.debug(
                "Ignoring the draft model on a request with images: speculative decoding "
                    + "is text-only.",
                category: .inference
            )
        }
        if let draftContainer, !usesCustomPipeline, !carriesMedia {
            // D1 follow-up fix: cache-trimmability precheck BEFORE
            // committing to the speculative branch. mlx-swift-lm's
            // `SpeculativeTokenIterator.init` (invoked inside
            // `runSpeculativeLLMGeneration`) requires BOTH the target and
            // draft `[KVCache]` to satisfy `canTrimPromptCache` and THROWS
            // `KVCacheError` otherwise — hybrid/linear-attention
            // architectures (e.g. Qwen3.5's GatedDeltaNet layers, which
            // allocate a non-trimmable `MambaCache`) fail this
            // unconditionally, architecture-wide, so without this precheck a
            // draft-model request against such a target bubbled that throw
            // straight out as a request failure (E2E-observed).
            //
            // Probe with a throwaway `[KVCache]` from `newCache(parameters:)`
            // on BOTH containers — lightweight: no prefill, just the
            // allocation shape, discarded immediately after the check.
            let targetCacheIsTrimmable = try await container.perform { context in
                MLXLMCommon.canTrimPromptCache(try context.model.newCache(parameters: generateParams))
            }
            let draftCacheIsTrimmable = try await draftContainer.perform { context in
                MLXLMCommon.canTrimPromptCache(try context.model.newCache(parameters: generateParams))
            }
            if Self.canUseSpeculativeDecoding(
                targetCacheIsTrimmable: targetCacheIsTrimmable,
                draftCacheIsTrimmable: draftCacheIsTrimmable
            ) {
                try await runSpeculativeLLMGeneration(
                    lmInput: lmInput,
                    container: container,
                    draftContainer: draftContainer,
                    generateParams: generateParams,
                    numDraftTokens: numDraftTokens,
                    hasTools: hasTools,
                    fullPath: fullPath,
                    into: continuation
                )
                return
            }

            // Not trimmable — fall back to plain decoding below rather than
            // let `SpeculativeTokenIterator.init` throw. Clean, non-recursive
            // control flow: `runSpeculativeLLMGeneration` is simply never
            // called on this path, and execution falls through into the same
            // prompt-cache / plain-decode logic used when no draft model was
            // requested at all — no recursive re-entry into
            // `runLLMGeneration`, no double-run.
            await LogManager.shared.warning(
                "Speculative decoding requested (draft model resident) but the "
                    + "target and/or draft model's KV cache is not trimmable — "
                    + "likely a hybrid/linear-attention architecture (e.g. "
                    + "GatedDeltaNet's MambaCache). Falling back to plain "
                    + "decoding for this request.",
                category: .inference
            )
        }

        // Flat Int token array for the cache lookup + telemetry.
        // `LMInput.text.tokens` is an `MLXArray`; `asArray(Int.self)`
        // materialises to Swift.
        let inputTokens = lmInput.text.tokens.asArray(Int.self)
        lastPromptTokensHash = inputTokens.hashValue

        // Track B — nearest-prefix (LCP) lookup. On a hit the store returns a
        // cache already TRIMMED to the shared prefix, so we prefill only the
        // remaining suffix `prompt[reused...]`; the reused KV already covers
        // positions `0..<reused`. On a miss we feed the whole prompt and let
        // the iterator allocate a fresh cache inside `generateTokensTask`. This is
        // what turns a tool-loop / multi-turn agent's per-turn full cold
        // prefill into an incremental one (only the newly-appended tokens).
        // Skip the prompt cache entirely when quantizing the KV cache: the
        // iterator mutates a `KVCacheSimple` into a `QuantizedKVCache` mid-stream,
        // and storing that back would hand a later plain request a quantized
        // snapshot it never asked for. A `kv_bits` request therefore always cold-
        // prefills and never persists its cache (mirrors the speculative path's
        // deliberate prompt-cache bypass).
        // An image request bypasses the cache for everything but this lookup,
        // and looks up only the prefix before its first media block. Its
        // remainder is sliced on the last axis below, so it must keep the
        // processor's `[1, N]` shape.
        let lookupTokens: [Int]? =
            if !bypassPromptCache {
                inputTokens
            } else if carriesMedia, !bypassesForWeights, lmInput.text.tokens.ndim == 2 {
                fullPath?.imageRequestLookupKey(inputTokens)
            } else {
                nil
            }
        let hit: PromptCacheHit? =
            if let lookupTokens {
                await promptCacheStore.fetchNearest(
                    modelID: modelID, tokens: lookupTokens, modelFingerprint: modelFingerprint)
            } else {
                nil
            }
        let priorCache: [any KVCache]?
        let promptInput: LMInput
        if let hit, carriesMedia {
            priorCache = hit.snapshot.caches
            // The processor's `[1, N]` shape, sliced on the last axis with its
            // mask (Qwen3.5's `getRopeIndex` reads it), the media unchanged:
            // every media token lies past the reused prefix.
            promptInput = LMInput(
                text: lmInput.text[0..., hit.reusedTokenCount...], image: lmInput.image,
                video: lmInput.video, audio: lmInput.audio)
            await LogManager.shared.debug(
                "Prompt cache HIT before the first media block — reused "
                    + "\(hit.reusedTokenCount)/\(inputTokens.count) tokens (model=\(modelID))",
                category: .inference
            )
        } else if let hit {
            priorCache = hit.snapshot.caches
            // Suffix-only prefill input, reconstructed from the sliced token
            // array alone: `lmInput.text.mask` is intentionally dropped here.
            // A text prompt reaches this point as `[N]` with no mask (a
            // full-path VLM's was normalized), so nothing is lost; an image
            // request takes the branch above.
            promptInput = LMInput(tokens: lmInput.text.tokens[hit.reusedTokenCount...])
            await LogManager.shared.debug(
                "Prompt cache HIT — reused \(hit.reusedTokenCount)/\(inputTokens.count) tokens, "
                    + "incremental prefill of \(inputTokens.count - hit.reusedTokenCount) "
                    + "(model=\(modelID))",
                category: .inference
            )
        } else {
            priorCache = nil
            promptInput = lmInput
            await LogManager.shared.debug(
                "Prompt cache MISS — cold prefill of \(inputTokens.count) tokens (model=\(modelID))",
                category: .inference
            )
        }

        // On a HIT the iterator only sees the suffix, so `info.promptTokenCount`
        // will report just the incremental prefill length. Carry the reused
        // count so the final usage restores the full prompt length (OpenAI
        // `prompt_tokens` semantics). MISS ⇒ 0, i.e. unchanged.
        let reusedPromptTokens = hit?.reusedTokenCount ?? 0

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
        // `promptInput` is the suffix to prefill (full prompt on a miss); the
        // history input rides along for the boundary render, nil when the
        // prompt cache is bypassed, since a snapshot would never be stored
        // (or when a test turned snapshots off).
        let inputBox = NonSendableBox(
            (
                prompt: promptInput,
                history: bypassPromptCache || !historyBoundarySnapshotsEnabled ? nil : Optional(historyInput)
            ))
        // Captured into the perform closure for a constrained / custom-sampling
        // generation. The cache is Sendable (`@unchecked` behind a lock);
        // `responseFormat`, `logitBias`, the XTC params and `modelID` are value
        // types.
        let vocabCache = tokenVocabularyCache
        // Track E: allocate the logprobs sink only when requested, and capture it
        // into the perform closure so the ``LogprobsCaptureProcessor`` fills it
        // while the engine loop below drains it (one entry per emitted token). The
        // sink is Sendable (locked internally).
        let logprobsSink = wantsLogprobs ? LogprobsCaptureProcessor.Sink() : nil
        let forcedBoundary = forcedHistoryBoundary

        let setup: (
            cache: PromptCacheSnapshot,
            stream: AsyncStream<TokenGeneration>,
            worker: Task<Void, Never>,
            toolCallFormat: ToolCallFormat?,
            stopTokenIDs: Set<Int>,
            history: HistoryBoundarySnapshot?,
            historySkip: String?,
            mediaHead: (prefilledTokens: Int, seconds: TimeInterval)?
        ) =
            try await container.perform(nonSendable: inputBox) { context, inputBox in
                let cache: [any KVCache] = try priorCacheBox?.caches
                    ?? context.model.newCache(parameters: generateParams)

                // History boundary: prefill `[reused, boundary)` into the
                // working cache, keep an evaluated copy of it, and hand the
                // generation only `[boundary, end)`. `cache` is the same
                // instances either way, so the record after the stream is
                // unchanged.
                var generationInput = inputBox.value.prompt
                var history: HistoryBoundarySnapshot?
                var historySkip: String?
                if let historyInput = inputBox.value.history {
                    let decision: HistoryBoundary
                    if let forcedBoundary {
                        decision =
                            forcedBoundary > reusedPromptTokens && forcedBoundary < inputTokens.count
                            ? .snapshot(boundary: forcedBoundary)
                            : .skip(reason: "the forced boundary \(forcedBoundary) is outside the prefill")
                    } else {
                        decision = Self.historyBoundary(
                            for: historyInput, messageGenerator: historyGenerator, cache: cache,
                            context: context, promptTokens: inputTokens,
                            reusedTokenCount: reusedPromptTokens)
                    }
                    switch decision {
                    case .snapshot(let boundary):
                        let split = boundary - reusedPromptTokens
                        let suffix = generationInput.text.tokens
                        let seconds = try Self.prefillHistory(
                            suffix[..<split], into: cache, context: context,
                            state: fullPath?.iteratorState(), prefill: generateParams.prefill)
                        let copies = cache.map { $0.copy() }
                        eval(copies.flatMap { $0.state })
                        history = HistoryBoundarySnapshot(
                            boundary: boundary, prefilledTokens: split, seconds: seconds,
                            snapshot: PromptCacheSnapshot(copies))
                        generationInput = LMInput(tokens: suffix[split...])
                    case .skip(let reason):
                        historySkip = reason
                    }
                }
                // An image request's text before its first media block and
                // after its last is prefilled as a text request's is, in
                // chunks: the processor's input runs in one shot, so an image
                // late in a long conversation, or a long agent turn after
                // one, otherwise holds the whole prompt's attention at once.
                var mediaHead: (prefilledTokens: Int, seconds: TimeInterval)?
                if let first = fullPath?.textPrefillBeforeMedia(generationInput) {
                    let seconds = try Self.prefillHistory(
                        generationInput.text.tokens[0, ..<first], into: cache, context: context,
                        state: fullPath?.iteratorState(), prefill: generateParams.prefill)
                    mediaHead = (first, seconds)
                    // A cancel that came during that prefill stops here,
                    // before the media's one-shot `prepare`.
                    try Task.checkCancellation()
                    generationInput = LMInput(
                        text: generationInput.text[0..., first...], image: generationInput.image,
                        video: generationInput.video, audio: generationInput.audio)
                }
                if let split = fullPath?.mediaPrefillSplit(generationInput) {
                    let head = LMInput(
                        text: generationInput.text[0..., ..<split], image: generationInput.image,
                        video: generationInput.video, audio: generationInput.audio)
                    let seconds = try Self.prefill(
                        head, into: cache, context: context, state: fullPath?.iteratorState(),
                        prefill: generateParams.prefill)
                    mediaHead = ((mediaHead?.prefilledTokens ?? 0) + split, (mediaHead?.seconds ?? 0) + seconds)
                    generationInput = LMInput(tokens: generationInput.text.tokens[0, split...])
                }
                let generation: (stream: AsyncStream<TokenGeneration>, worker: Task<Void, Never>)
                if usesCustomPipeline {
                    // Track C constraint AND the Track E sampling extensions
                    // (logit_bias / XTC / logprobs) all install a hand-built
                    // `TokenIterator` via `customTokenStream`, then run it through
                    // the same raw-token task the plain path uses so all downstream
                    // streaming/detokenization is byte-for-byte the same.
                    generation = try Self.customTokenStream(
                        input: generationInput,
                        cache: cache,
                        state: fullPath?.iteratorState(),
                        generateParams: generateParams,
                        context: context,
                        responseFormat: responseFormat,
                        logitBias: logitBias,
                        xtcProbability: xtcProbability,
                        xtcThreshold: xtcThreshold,
                        logprobsSink: logprobsSink,
                        topLogprobs: topLogprobs,
                        modelID: modelID,
                        vocabCache: vocabCache
                    )
                } else {
                    // Stock parameter-driven path — the ONLY LLM path that honors
                    // `kv_bits` (via `generateParams`).
                    generation = try MLXLMCommon.generateTokensTask(
                        input: generationInput,
                        cache: cache,
                        state: fullPath?.iteratorState(),
                        parameters: generateParams,
                        context: context,
                        includeStopToken: true
                    )
                }
                // `ToolCallFormat?` and the worker task are Sendable, so they
                // ride back out of the actor alongside the cache + stream.
                return (
                    PromptCacheSnapshot(cache), generation.stream, generation.worker,
                    context.configuration.toolCallFormat, Self.stopTokenIDs(context),
                    history, historySkip, mediaHead
                )
            }
        let workingCache = setup.cache.caches
        if let history = setup.history {
            // Stored now, while the generation runs: the snapshot holds the
            // prompt's first `boundary` tokens and nothing after, so it is
            // valid whatever the stream does next, a cancel included. `.user`
            // so the classified LRU drops the `.assistant` record of the turn
            // first.
            await promptCacheStore.insert(
                modelID: modelID,
                tokens: Array(inputTokens[..<history.boundary]),
                snapshot: history.snapshot,
                cacheType: .user,
                modelFingerprint: modelFingerprint
            )
            await LogManager.shared.debug(
                "Prompt cache history boundary — stored \(history.boundary)/\(inputTokens.count) tokens "
                    + "(reused \(reusedPromptTokens), prefilled \(history.prefilledTokens) in "
                    + "\(String(format: "%.3f", history.seconds)) s, "
                    + "\(PromptCacheStore.cacheBytes(history.snapshot.caches)) bytes, model=\(modelID))",
                category: .inference
            )
        } else if let reason = setup.historySkip {
            await LogManager.shared.debug(
                "Prompt cache history boundary — no snapshot: \(reason) (model=\(modelID))",
                category: .inference
            )
        }
        let stream = setup.stream
        let worker = setup.worker
        let stopTokenIDs = setup.stopTokenIDs

        // Only route through the streaming `ToolCallProcessor` when the caller
        // actually requested tools this turn — see `makeToolProcessor` for why
        // the gate exists (else EVERY generation on a tool-capable model, incl.
        // plain non-tool chat, would have `{...}`-shaped content silently
        // stripped and misreported as `finish_reason: tool_calls`).
        let toolProcessor = Self.makeToolProcessor(
            format: setup.toolCallFormat, hasTools: hasTools,
            hasResponseFormat: responseFormat != nil)

        var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        var generatedTokenIDs: [Int] = []
        // The stop token the loop delivers last (`includeStopToken`): in the
        // cache's key, never in the text, the logprobs or the counts (#142).
        // A token after it would mean a loop that no longer breaks on it; the
        // text still flows, the cache is not recorded.
        var stopTokens = StopTokenTracker()
        var completionInfo: GenerateCompletionInfo?
        // Track E: per-token logprobs accumulated since the last emitted chunk.
        // Each `.token` pops one capture entry (aligned in emission order); the
        // batch is attached to whatever chunk next carries text — a token whose
        // text is still buffered by the detokenizer keeps its logprob pending
        // until that text flushes.
        var pendingLogprobs: [TokenLogprob] = []

        // W2 phase seam: nil unless an observer is attached, so the loop below
        // is unchanged for every ordinary generation (the config is built only
        // when observed). `begin()` marks the prefill boundary — prefill runs as
        // the stream is first pulled — and the `defer` guarantees exactly one
        // terminal event even on the cancel/abandon early returns below.
        var phaseReporter = siliconObserver.map {
            GenerationPhaseReporter(
                observer: $0, config: Self.generationConfig(from: generateParams, batchSize: 1))
        }
        phaseReporter?.begin()
        defer { phaseReporter?.abortIfUnfinished() }

        // `ended`: this loop left the stream early (the generate task was
        // cancelled, or the client went away); the worker is stopped and
        // awaited below before this returns. A cancel that lands while the
        // loop waits for a token ends the stream instead, with `ended` false
        // and the task cancelled.
        var ended = false
        consume: for await event in stream {
            // POOL-2: stop promptly once the consumer has abandoned/
            // cancelled this stream (see `generate`'s `onTermination`
            // hook) instead of running to maxTokens/EOS regardless.
            if Task.isCancelled {
                continuation.finish(throwing: CancellationError())
                ended = true
                break consume
            }
            switch event {
            case .token(let token):
                if stopTokens.note(token, isStop: stopTokenIDs.contains(token)) {
                    // The loop breaks right after delivering a stop token, so
                    // it is the last token. Its capture entry goes with it.
                    if wantsLogprobs { _ = logprobsSink?.popFirst() }
                    continue consume
                }
                phaseReporter?.noteTokenGenerated()
                generatedTokenIDs.append(token)
                if wantsLogprobs, let entry = logprobsSink?.popFirst() {
                    pendingLogprobs.append(Self.tokenLogprob(from: entry, tokenizer: tokenizer))
                }
                detokenizer.append(token: token)
                if let piece = detokenizer.next() {
                    let lp: [TokenLogprob]? = pendingLogprobs.isEmpty ? nil : pendingLogprobs
                    if let toolProcessor {
                        // Emit only the processor's display text; nil means the
                        // piece is being buffered as (potential) tool-call syntax.
                        if let display = toolProcessor.processChunk(piece), !display.isEmpty {
                            if case .terminated = continuation.yield(
                                GenerateChunk(text: display, logprobs: lp)) {
                                ended = true
                                break consume
                            }
                            pendingLogprobs.removeAll()
                        }
                    } else {
                        if case .terminated = continuation.yield(
                            GenerateChunk(text: piece, logprobs: lp)) {
                            ended = true
                            break consume
                        }
                        pendingLogprobs.removeAll()
                    }
                }
            case .info(let info):
                // The boundary prefill ran before the iterator that reports
                // this; count it, so usage and the phase summary cover the
                // whole prefill as they did before the split.
                completionInfo = [setup.history.map { ($0.prefilledTokens, $0.seconds) }, setup.mediaHead]
                    .compactMap { $0 }
                    .reduce(info) { Self.completionInfo($0, addingPrefillOf: $1.0, seconds: $1.1) }
            }
        }

        // The worker holds the model until it returns, and nothing but this
        // frame stops it once the loop left early (#136): see `awaitWorker`.
        await Self.awaitWorker(worker)
        if ended { return }
        if Task.isCancelled {
            // Cancelled while the loop waited for a token: the worker may
            // have stepped a token nobody received (it had, unless the cancel
            // reached it before its first check), so the cache can hold more
            // positions than `inputTokens + generatedTokenIDs` names, and an
            // entry under that key would serve a stale position on its next
            // hit. Nothing is recorded.
            continuation.finish(throwing: CancellationError())
            return
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

        // Save the post-generation cache under the extended key. The same
        // `workingCache` reference has been mutated in-place by the iterator,
        // so it now reflects the prompt, the generated tokens, and the stop
        // token when the stream ended on one: the iterator had stepped it
        // before the loop saw it, so its position is in the cache and belongs
        // in the key (#142; see `promptCacheKey`). Skipped when quantizing the
        // KV cache (see `bypassPromptCache`) so a `QuantizedKVCache` snapshot
        // is never persisted for later plain reuse, and when this turn stored
        // a history-boundary snapshot of a cache that cannot be trimmed (see
        // `recordsTurnEntry`).
        if !bypassPromptCache {
            let key = Self.promptCacheKey(
                inputTokens: inputTokens, generatedTokenIDs: generatedTokenIDs,
                terminalStopToken: stopTokens.terminal, tokenAfterStopToken: stopTokens.tokenAfterTerminal,
                stopReason: completionInfo?.stopReason)
            let records = Self.recordsTurnEntry(
                historySnapshotStored: setup.history != nil, caches: workingCache)
            lastPromptCacheRecord = PromptCacheRecord(
                tokenCount: key?.count ?? inputTokens.count + generatedTokenIDs.count,
                heldPositions: workingCache.map(\.offset).max() ?? 0,
                endedOnStopToken: stopTokens.terminal != nil, recorded: key != nil && records)
            if let key, !records {
                await LogManager.shared.debug(
                    "Prompt cache: the cache cannot be trimmed and the history boundary was "
                        + "stored, so the turn's own entry (\(key.count) tokens) is not (model=\(modelID))",
                    category: .inference)
            } else if let key {
                await promptCacheStore.insert(
                    modelID: modelID,
                    tokens: key,
                    snapshot: PromptCacheSnapshot(workingCache),
                    modelFingerprint: modelFingerprint
                )
            } else if stopTokens.tokenAfterTerminal {
                await LogManager.shared.warning(
                    "Prompt cache: the loop delivered a token after the stop token, so the "
                        + "cache holds positions in an order no key names; nothing recorded (#142)",
                    category: .inference)
            } else if completionInfo?.stopReason == .stop {
                await LogManager.shared.warning(
                    "Prompt cache: the loop reported a stop token it did not deliver, so the "
                        + "cache holds a position no key names; nothing recorded (#142)",
                    category: .inference)
            }
        }

        if let reporter = phaseReporter {
            phaseReporter?.complete(
                summary: Self.phaseSummary(
                    config: reporter.config, completionInfo: completionInfo,
                    stopTokenCounted: stopTokens.terminal != nil))
        }

        emitFinalChunk(
            completionInfo: completionInfo,
            toolCalls: drainedToolCalls,
            reusedPromptTokens: reusedPromptTokens,
            // Any logprobs for tokens whose text never flushed as its own chunk
            // (e.g. a trailing partial-UTF8 fragment) ride out on the final chunk.
            logprobs: pendingLogprobs.isEmpty ? nil : pendingLogprobs,
            stopTokenCounted: stopTokens.terminal != nil,
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
    /// request's `draftModelID` AND the cache-trimmability precheck
    /// (`canUseSpeculativeDecoding`) has confirmed both models' freshly
    /// allocated caches are trimmable — this function assumes that precheck
    /// already passed and does not repeat it.
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
    ///    allocates fresh caches for both models. Freshness alone does NOT
    ///    guarantee trimmability, though — a hybrid/linear-attention
    ///    architecture's `MambaCache` is non-trimmable no matter how fresh
    ///    (architectural, not a snapshot-staleness issue), which is exactly
    ///    what `runLLMGeneration`'s precheck now catches before this
    ///    function is ever called.
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
        fullPath: FullPathVLM?,
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
                // A full-path VLM's `prepare` returns logits, which upstream's
                // speculative iterator would drop the first token of (see
                // `LastPromptTokenTarget`).
                var context = context
                if fullPath != nil { context.model = LastPromptTokenTarget(context.model) }
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
        // caller actually requested tools this turn. The speculative branch is
        // only ever reached with no response_format (constrained output disables
        // speculation upstream), so `hasResponseFormat` is always false here.
        let toolProcessor = Self.makeToolProcessor(
            format: setup.toolCallFormat, hasTools: hasTools, hasResponseFormat: false)

        var detokenizer = NaiveStreamingDetokenizer(tokenizer: targetTokenizer)
        var completionInfo: GenerateCompletionInfo?

        // W2 phase seam (see runLLMGeneration) — no-op unless an observer is set;
        // the `defer` guarantees exactly one terminal event on the early returns.
        var phaseReporter = siliconObserver.map {
            GenerationPhaseReporter(
                observer: $0, config: Self.generationConfig(from: generateParams, batchSize: 1))
        }
        phaseReporter?.begin()
        defer { phaseReporter?.abortIfUnfinished() }

        for await event in stream {
            // POOL-2: stop promptly once the consumer has abandoned/
            // cancelled this stream, mirroring the non-speculative path.
            if Task.isCancelled {
                continuation.finish(throwing: CancellationError())
                return
            }
            switch event {
            case .token(let token):
                phaseReporter?.noteTokenGenerated()
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

        if Task.isCancelled {
            // Cancelled while the loop waited for a token: reported as such,
            // not as a completed response (this path records no prompt cache).
            continuation.finish(throwing: CancellationError())
            return
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

        if let reporter = phaseReporter {
            phaseReporter?.complete(
                summary: Self.phaseSummary(config: reporter.config, completionInfo: completionInfo))
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

        let (stream, worker): (AsyncStream<TokenGeneration>, Task<Void, Never>) =
            try await container.perform(nonSendable: inputBox) { context, inputBox in
                let cache = try context.model.newCache(parameters: generateParams)
                return try MLXLMCommon.generateTokensTask(
                    input: inputBox.value,
                    cache: cache,
                    parameters: generateParams,
                    context: context
                )
            }

        var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        var completionInfo: GenerateCompletionInfo?

        // W2 phase seam (see runLLMGeneration) — no-op unless an observer is set;
        // the `defer` guarantees exactly one terminal event on the early returns.
        var phaseReporter = siliconObserver.map {
            GenerationPhaseReporter(
                observer: $0, config: Self.generationConfig(from: generateParams, batchSize: 1))
        }
        phaseReporter?.begin()
        defer { phaseReporter?.abortIfUnfinished() }

        var ended = false
        consume: for await event in stream {
            // POOL-2: stop promptly once the consumer has abandoned/
            // cancelled this stream (see `generate`'s `onTermination`
            // hook) instead of running to maxTokens/EOS regardless.
            if Task.isCancelled {
                continuation.finish(throwing: CancellationError())
                ended = true
                break consume
            }
            switch event {
            case .token(let token):
                phaseReporter?.noteTokenGenerated()
                detokenizer.append(token: token)
                if let piece = detokenizer.next() {
                    let chunk = GenerateChunk(text: piece)
                    if case .terminated = continuation.yield(chunk) {
                        ended = true
                        break consume
                    }
                }
            case .info(let info):
                completionInfo = info
            }
        }

        // Same as the text path: the worker holds the model until it returns,
        // and only this frame stops it once the loop left early (#136).
        await Self.awaitWorker(worker)
        if ended { return }
        if Task.isCancelled {
            // Cancelled while the loop waited for a token: reported as such,
            // not as a completed response.
            continuation.finish(throwing: CancellationError())
            return
        }

        if let reporter = phaseReporter {
            phaseReporter?.complete(
                summary: Self.phaseSummary(config: reporter.config, completionInfo: completionInfo))
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
        reusedPromptTokens: Int = 0,
        logprobs: [TokenLogprob]? = nil,
        stopTokenCounted: Bool = false,
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
            // On a prompt-cache HIT the iterator only prefilled the suffix, so
            // `info.promptTokenCount` is the incremental count; add back the
            // reused prefix to report the FULL prompt length (OpenAI
            // `prompt_tokens` semantics). `reusedPromptTokens` is 0 on a MISS
            // and on the VLM / speculative paths (which don't reuse), leaving
            // their reported counts unchanged.
            usage = TokenUsage(
                promptTokens: info.promptTokenCount + reusedPromptTokens,
                // The text path's loop counts the stop token it delivered
                // (`includeStopToken`, #142); the wire count stays the number
                // of tokens that became text, as before.
                completionTokens: max(0, info.generationTokenCount - (stopTokenCounted ? 1 : 0)),
                // The same reused count, reported on its own so a client can
                // see how much of the prompt the cache served (0 on a MISS and
                // on the paths that never consult the cache).
                cachedPromptTokens: reusedPromptTokens
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
            speculativeDecoding: speculativeDecoding,
            logprobs: logprobs
        )
        continuation.yield(finalChunk)
    }

    // MARK: Logprobs (Track E)

    /// Decode one raw capture entry into the wire ``TokenLogprob`` shape,
    /// resolving each token id to its display text and UTF-8 bytes via the
    /// resident tokenizer. `skipSpecialTokens: false` so a sampled special token
    /// still renders its spelling (OpenAI includes it verbatim).
    static func tokenLogprob(
        from entry: LogprobsCaptureProcessor.Sink.Entry,
        tokenizer: any Tokenizer
    ) -> TokenLogprob {
        func decode(_ id: Int) -> String {
            tokenizer.decode(tokenIds: [id], skipSpecialTokens: false)
        }
        let text = decode(entry.tokenID)
        let alternatives = entry.top.map { alternative -> TokenLogprob.Alternative in
            let altText = decode(alternative.id)
            return TokenLogprob.Alternative(
                token: altText,
                logprob: alternative.logprob,
                bytes: Array(altText.utf8).map(Int.init)
            )
        }
        return TokenLogprob(
            token: text,
            logprob: entry.logprob,
            bytes: Array(text.utf8).map(Int.init),
            topLogprobs: alternatives
        )
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
    static func makeToolProcessor(
        format: ToolCallFormat?, hasTools: Bool, hasResponseFormat: Bool
    ) -> ToolCallProcessor? {
        guard hasTools else { return nil }
        // A grammar-constrained (response_format) turn emits a bare JSON document
        // that cannot contain tool-call syntax anyway; running the processor over
        // it would, for a `.json`-format model, buffer the whole document and can
        // mis-drain the constrained JSON as an (empty-content) tool call. Never
        // route a constrained turn through the tool processor.
        guard !hasResponseFormat else { return nil }
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
