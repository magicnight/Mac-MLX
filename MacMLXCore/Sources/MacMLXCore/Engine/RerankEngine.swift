import Foundation
import MLXLMCommon
import MLXRerankers

// MARK: - RerankEngine

/// In-process reranker backed by mlx-swift-lm's `MLXRerankers`.
///
/// Loads a `.reranker` checkpoint through `RerankerModelFactory`, which reads
/// `config.json` and picks the implementation: an encoder cross-encoder with a
/// sequence-classification head (BERT / RoBERTa / XLM-RoBERTa, e.g.
/// `cross-encoder/ms-marco-MiniLM-L-6-v2` and `BAAI/bge-reranker-*`), a Qwen3
/// causal reranker (`Qwen3-Reranker-*`, scored by its yes/no logit margin), or
/// Jina reranker v3 (`JinaForRanking`, listwise). Every `[query, document]`
/// pair is scored jointly — the two share one forward pass — which is what makes
/// this the accuracy-first path of `/v1/rerank`; the bi-encoder cosine
/// ``EmbeddingEngine`` path remains the documented fallback for `.embedder`
/// models.
///
/// An `actor`, like ``EmbeddingEngine``: concurrent calls serialize, the
/// container is owned here, and only `Sendable` `[Double]` scores cross the
/// isolation boundary.
public actor RerankEngine {

    /// The reranker currently in memory, if any.
    public private(set) var loadedModel: LocalModel?

    /// The loaded, architecture-neutral reranker, or `nil` before the first
    /// successful `load`.
    private var container: RerankerContainer?

    public init() {}

    /// What the scores returned by ``score(query:documents:)`` mean for the
    /// loaded model: `.normalizedRelevance` (0...1) for encoder and Qwen3
    /// rerankers, `.cosineSimilarity` (-1...1) for Jina reranker v3. `nil`
    /// before a load.
    public var scoreKind: RerankScoreKind? { container?.scoreKind }

    // MARK: Loading

    /// Load a `.reranker` model from its local directory into memory.
    ///
    /// `allowUnverifiedModel` is passed as `true` on purpose. Upstream's own
    /// guard accepts a checkpoint only when its identifier contains "rerank"
    /// (or it declares `JinaForRanking`), which would refuse
    /// `cross-encoder/ms-marco-MiniLM-L-6-v2`. The decision that this directory
    /// holds a reranker is made once, by `ModelLibraryManager.upgradeFormat`,
    /// on the same evidence upstream uses plus the single-logit head check for
    /// encoders; the engine trusts that classification.
    ///
    /// - Throws: ``EngineError/modelLoadFailed(reason:)`` on any failure: an
    ///   unsupported architecture, a weight-key mismatch, a tokenizer that
    ///   fails to load, or an encoder whose classification head has no
    ///   identifiable positive class.
    public func load(_ model: LocalModel) async throws {
        do {
            let loaded = try await RerankerModelFactory.shared.loadContainer(
                from: model.directory,
                using: HuggingFaceTokenizerLoader(),
                allowUnverifiedModel: true)
            container = loaded
            loadedModel = model
            await LogManager.shared.info(
                "Reranker loaded: \(model.id) (\(Self.describe(loaded.scoreKind)))",
                category: .inference)
        } catch {
            container = nil
            loadedModel = nil
            throw EngineError.modelLoadFailed(reason: error.localizedDescription)
        }
    }

    // MARK: Scoring

    /// Score every `[query, document]` pair, returning one score per document
    /// in `documents`' order (NOT sorted). Higher = more relevant; the scale is
    /// ``scoreKind``. `MLXRerankers` micro-batches the pairs (16 pairs or
    /// 8,192 token slots per forward pass) and truncates inputs to the model's
    /// context.
    ///
    /// - Throws: ``EngineError/modelNotLoaded`` before a load; upstream's
    ///   `RerankerError` for an empty query or document, or a non-finite score.
    public func score(query: String, documents: [String]) async throws -> [Double] {
        guard let container else {
            throw EngineError.modelNotLoaded
        }
        if documents.isEmpty { return [] }
        let response = try await container.scores(query: query, documents: documents)
        // `scores` returns one result per document, in input order. Place
        // each by its declared index anyway, so a reordered response could
        // never be misattributed to the wrong document.
        var scores = [Double?](repeating: nil, count: documents.count)
        for result in response.results where scores.indices.contains(result.index) {
            scores[result.index] = result.score
        }
        return try scores.map { score in
            guard let score else {
                throw RerankerError.invalidScoreCount(
                    expected: documents.count, actual: response.results.count)
            }
            return score
        }
    }

    // MARK: Private

    private static func describe(_ kind: RerankScoreKind) -> String {
        switch kind {
        case .normalizedRelevance: "normalized relevance 0...1"
        case .cosineSimilarity: "cosine similarity"
        case .logit: "raw logit"
        }
    }
}
