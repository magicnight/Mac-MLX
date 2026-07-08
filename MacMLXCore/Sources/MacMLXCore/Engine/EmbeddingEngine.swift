import Foundation
import MLX
import MLXEmbedders
import MLXLMCommon

// MARK: - EmbeddingModel protocol

/// The contract every text-embedding engine implementation must satisfy.
///
/// A deliberate *sibling* to ``InferenceEngine`` rather than a conformer:
/// `InferenceEngine` is generation-specific (streaming tokens, KV cache, chat
/// templates) and embedders share none of that surface. Implementations are
/// `actor` types so concurrent calls into a single engine are serialised.
public protocol EmbeddingModel: Actor {
    /// The embedder currently in memory, if any.
    var loadedModel: LocalModel? { get }

    /// Bring an embedder model into memory. Replaces any previously loaded one.
    func load(_ model: LocalModel) async throws

    /// Embed a batch of texts, returning one L2-normalized vector per input,
    /// row-aligned with `texts`.
    func embed(_ texts: [String]) async throws -> [[Float]]
}

// MARK: - EmbeddingEngine

/// In-process MLX embedding engine backed by Apple's `MLXEmbedders` library.
///
/// Loads an encoder/embedding checkpoint from a local directory and produces
/// pooled, L2-normalized sentence embeddings. Powers the OpenAI-compatible
/// `/v1/embeddings` endpoint and the bi-encoder `/v1/rerank` MVP.
///
/// - Note: MVP scope — a single engine with no pool. `ModelPool` hard-binds
///   `any InferenceEngine`, so a dedicated `EmbeddingPool` is a follow-up.
public actor EmbeddingEngine: EmbeddingModel {

    /// The embedder currently in memory, if any.
    public private(set) var loadedModel: LocalModel?

    /// The loaded MLXEmbedders container (model + tokenizer + pooling), or
    /// nil before the first successful `load`.
    private var container: EmbedderModelContainer?

    public init() {}

    /// Load an embedder model from its local directory into memory.
    ///
    /// - Parameter model: The ``LocalModel`` to load. `model.directory` must
    ///   contain a valid MLX embedder (`config.json`, `.safetensors` weights,
    ///   tokenizer files).
    /// - Throws: ``EngineError/modelLoadFailed(reason:)`` if loading fails.
    public func load(_ model: LocalModel) async throws {
        do {
            let loaded = try await EmbedderModelFactory.shared.loadContainer(
                from: model.directory,
                using: HuggingFaceTokenizerLoader()
            )
            container = loaded
            loadedModel = model
        } catch {
            container = nil
            loadedModel = nil
            throw EngineError.modelLoadFailed(reason: error.localizedDescription)
        }
    }

    /// Embed a batch of texts into L2-normalized vectors.
    ///
    /// Tokenizes each input, pads the batch to its longest sequence, runs the
    /// encoder, then pools + normalizes per the model's pooling strategy. The
    /// pooled `MLXArray` is `eval()`'d and materialised to `[[Float]]` before
    /// leaving the container closure — `MLXArray` is not `Sendable`.
    public func embed(_ texts: [String]) async throws -> [[Float]] {
        guard let container else {
            throw EngineError.modelNotLoaded
        }
        if texts.isEmpty { return [] }

        return try await container.perform { (context: EmbedderModelContext) -> [[Float]] in
            let tokenizer = context.tokenizer
            let encoded = texts.map { tokenizer.encode(text: $0, addSpecialTokens: true) }

            // Pad every sequence to the batch's longest (min 16, mirroring the
            // MLXEmbedders reference) using the eos token id as the pad token.
            let padID = tokenizer.eosTokenId ?? 0
            let maxLength = encoded.reduce(into: 16) { acc, ids in
                acc = max(acc, ids.count)
            }
            let padded = stacked(
                encoded.map { ids in
                    MLXArray(ids + Array(repeating: padID, count: maxLength - ids.count))
                }
            )
            let mask = padded .!= padID
            let tokenTypes = MLXArray.zeros(like: padded)

            let output = context.model(
                padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask)
            let pooled = context.pooling(output, mask: mask, normalize: true, applyLayerNorm: true)
            // MUST eval before returning — MLXArray is not Sendable.
            pooled.eval()
            return pooled.map { $0.asArray(Float.self) }
        }
    }
}
