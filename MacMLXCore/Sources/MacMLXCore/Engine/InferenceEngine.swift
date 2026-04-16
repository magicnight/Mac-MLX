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

    /// Release the loaded model (and any caches) from memory.
    func unload() async throws

    /// Stream tokens for a generation request.
    ///
    /// The stream finishes naturally on `.stop`, on `.length` when `maxTokens`
    /// is reached, or with an error on engine failure.
    func generate(_ request: GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error>

    /// Synchronously confirm the engine is responsive.
    func healthCheck() async -> Bool
}
