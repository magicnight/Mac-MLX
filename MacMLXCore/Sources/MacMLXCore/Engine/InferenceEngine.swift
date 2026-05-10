/// The contract every inference engine implementation must satisfy.
///
/// Implementations are `actor` types — concurrent calls into a single engine
/// are serialised by the actor. Coordination across multiple engines is the
/// responsibility of `EngineCoordinator` (added in Stage 3).
public protocol InferenceEngine: Actor {
    /// Stable identifier for this engine implementation.
    var engineID: EngineID { get }

    /// Current lifecycle state.
    var status: EngineStatus { get }

    /// The model currently in memory, if any.
    var loadedModel: LocalModel? { get }

    /// Version string of the underlying engine library (e.g. mlx-swift-lm tag).
    var version: String { get }

    /// Bring a model into memory. Replaces any previously loaded model.
    func load(_ model: LocalModel) async throws

    /// Apply a LoRA adapter to the currently-loaded model (v0.5+).
    ///
    /// Called after `load(_:)` to layer adapter weights on top of the
    /// base model. The protocol-extension default is a no-op so engines
    /// that don't support adapters (test stubs, future CPU/Python
    /// engines) compile unchanged. The MLX engine routes through
    /// `LoRAContainer.from(directory:)` + `LanguageModel.load(adapter:)`,
    /// auto-converting PEFT-format adapters via
    /// `LoRAAdapterConverter` when needed.
    func applyAdapter(_ adapter: LocalAdapter) async throws

    /// Release the loaded model (and any caches) from memory.
    func unload() async throws

    /// Stream tokens for a generation request.
    ///
    /// The stream finishes naturally on `.stop`, on `.length` when `maxTokens`
    /// is reached, or with an error on engine failure.
    ///
    /// Declared `nonisolated` so callers (including @MainActor SwiftUI views)
    /// can invoke it without awaiting the actor's executor; implementations
    /// must not touch actor-isolated state in this method's synchronous body
    /// (use a `Task` inside the stream's continuation if you need to).
    nonisolated func generate(_ request: GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error>

    /// Synchronously confirm the engine is responsive.
    func healthCheck() async -> Bool
}

extension InferenceEngine {
    /// Default no-op for engines that don't yet support LoRA adapters
    /// (test stubs, future CPU/Python engines, …). Throws nothing,
    /// silently leaves the model unchanged.
    public func applyAdapter(_ adapter: LocalAdapter) async throws {
        // intentional no-op
    }
}
