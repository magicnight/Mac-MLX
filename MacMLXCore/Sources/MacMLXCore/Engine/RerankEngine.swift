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
/// Jina reranker v3 (`JinaForRanking`, listwise). The encoder and Qwen3 kinds
/// score each `[query, document]` pair jointly — the two share one forward
/// pass — and Jina scores the whole candidate list in one prompt; either way
/// the document is read in the light of the query, which is what makes this
/// the accuracy-first path of `/v1/rerank`. The bi-encoder cosine
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
    /// holds a reranker is made once, by `ModelLibraryManager.upgradeFormat`:
    /// a single-logit sequence-classification head for encoders, the
    /// `Qwen3-Reranker` name plus its yes/no logit-score declaration for
    /// Qwen3, or `JinaForRanking` without sliding-window layers — each at
    /// least as strict as upstream's check. The engine trusts that
    /// classification.
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
    /// - Throws: ``EngineError/modelNotLoaded`` before a load;
    ///   ``RerankRequestError`` when `MLXRerankers` refuses the input itself
    ///   (a blank query or document, more documents than a listwise model
    ///   takes); upstream's `RerankerError` for anything else, such as a
    ///   non-finite score.
    public func score(query: String, documents: [String]) async throws -> [Double] {
        guard let container else {
            throw EngineError.modelNotLoaded
        }
        if documents.isEmpty { return [] }
        let response: RerankResponse
        do {
            response = try await container.scores(query: query, documents: documents)
        } catch let error as RerankerError {
            throw RerankRequestError(error) ?? error
        }
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

// MARK: - RerankRequestError

/// A rerank request `MLXRerankers` refuses before it touches the model: the
/// caller's input is at fault, not the server, so `/v1/rerank` answers 400
/// rather than 500. Mirrors the request-shaped cases of upstream's
/// `RerankerError`; every other case stays an upstream error.
public enum RerankRequestError: LocalizedError, Equatable, Sendable {
    case emptyQuery
    case emptyDocument(index: Int)
    case tooManyDocuments(actual: Int, maximum: Int)
    case inputTooLong(actual: Int, maximum: Int)

    init?(_ error: RerankerError) {
        switch error {
        case .emptyQuery: self = .emptyQuery
        case .emptyDocument(let index): self = .emptyDocument(index: index)
        case .tooManyDocuments(let actual, let maximum):
            self = .tooManyDocuments(actual: actual, maximum: maximum)
        case .inputTooLong(let actual, let maximum):
            self = .inputTooLong(actual: actual, maximum: maximum)
        default: return nil
        }
    }

    public var errorDescription: String? {
        switch self {
        case .emptyQuery:
            "query must not be empty"
        case .emptyDocument(let index):
            "documents[\(index)] must not be empty"
        case .tooManyDocuments(let actual, let maximum):
            "\(actual) documents sent, but this reranker scores at most \(maximum) per request"
        case .inputTooLong(let actual, let maximum):
            "input of \(actual) tokens exceeds the model's limit of \(maximum)"
        }
    }
}
