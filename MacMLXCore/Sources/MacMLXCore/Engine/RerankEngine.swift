import Foundation
import MLX
import MLXEmbedders
import MLXLMCommon

// MARK: - RerankEngine

/// In-process MLX cross-encoder reranker backed by ``CrossEncoderModel``.
///
/// Loads a `.reranker` checkpoint (a BERT + `*ForSequenceClassification`
/// head, e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) and scores each
/// `[query, document]` pair JOINTLY in one forward pass, returning a raw
/// relevance logit per document. Powers the accuracy-first path of
/// `/v1/rerank`; the bi-encoder cosine `EmbeddingEngine` path remains the
/// documented fallback for `.embedder` models.
///
/// A deliberate sibling to ``EmbeddingEngine`` — like it, an `actor` (so
/// concurrent calls serialize) that owns a non-`Sendable` MLX model and
/// tokenizer and only ever hands `Sendable` `[Float]` scores back across the
/// isolation boundary (every `MLXArray` is `eval()`'d and materialized inside
/// the actor before returning).
///
/// - Note: Architecture-faithful but NOT validated against a real checkpoint
///   in this environment — see the deferred-validation notes on
///   ``CrossEncoderModel``.
public actor RerankEngine {

    /// The reranker currently in memory, if any.
    public private(set) var loadedModel: LocalModel?

    /// The loaded cross-encoder, or `nil` before the first successful `load`.
    /// Non-`Sendable`, but actor-isolated so that's safe.
    private var model: CrossEncoderModel?

    /// The paired tokenizer (WordPiece for BERT). `Tokenizer` is `Sendable`.
    private var tokenizer: (any MLXLMCommon.Tokenizer)?

    /// `[CLS]` id, resolved once at load (segment boundaries need it per pair).
    private var clsId: Int = 0
    /// `[SEP]` id — separates query from document AND terminates the sequence.
    private var sepId: Int = 0
    /// `[PAD]` id used to right-pad a batch to its longest row (masked out).
    private var padId: Int = 0
    /// Truncation ceiling — the checkpoint's `max_position_embeddings`
    /// (assumed 512 per standard HF BERT when absent).
    private var maxLength: Int = 512

    public init() {}

    // MARK: Loading

    /// Load a `.reranker` model from its local directory into memory.
    ///
    /// Reads `config.json` (assumed a standard HF
    /// `BertForSequenceClassification` config — `model_type` `bert`,
    /// `hidden_size`, `max_position_embeddings`), builds a
    /// ``CrossEncoderModel``, and loads its weights through the SAME strict
    /// loader (`loadWeights` → `verify: [.all]`) the LLMs/embedders use, then
    /// loads the tokenizer and resolves the `[CLS]`/`[SEP]`/`[PAD]` ids.
    ///
    /// - Throws: ``EngineError/modelLoadFailed(reason:)`` on any failure
    ///   (missing/mis-shaped config, weight-key mismatch under `verify:[.all]`,
    ///   or a tokenizer without `[CLS]`/`[SEP]`).
    public func load(_ model: LocalModel) async throws {
        let dir = model.directory
        let cross: CrossEncoderModel
        let loadedTokenizer: any MLXLMCommon.Tokenizer
        let resolvedMaxLength: Int
        do {
            let configURL = dir.appendingPathComponent("config.json")
            let data = try Data(contentsOf: configURL)
            // Assumed per standard HF, unverified in this environment: a
            // BertForSequenceClassification config. `BertConfiguration` drives
            // the encoder; `hidden_size` sizes the classifier head separately
            // (it isn't reachable off `BertConfiguration` from this module).
            let bertConfig = try JSONDecoder().decode(BertConfiguration.self, from: data)
            let fields = try JSONDecoder().decode(RerankerConfigFields.self, from: data)
            resolvedMaxLength = fields.maxPositionEmbeddings ?? 512  // HF BERT default
            cross = CrossEncoderModel(
                bertConfiguration: bertConfig,
                hiddenSize: fields.hiddenSize ?? 768)  // HF BERT-base default
            try loadWeights(modelDirectory: dir, model: cross)
            loadedTokenizer = try await HuggingFaceTokenizerLoader().load(from: dir)
        } catch {
            reset()
            throw EngineError.modelLoadFailed(reason: error.localizedDescription)
        }

        // A cross-encoder pair is [CLS] query [SEP] document [SEP]; both special
        // tokens are load-bearing. Missing one means this isn't a BERT-family
        // reranker (the only kind this first cut supports) — fail loudly rather
        // than tokenize wrongly. Thrown outside the do so it isn't re-wrapped.
        guard let cls = loadedTokenizer.convertTokenToId("[CLS]"),
              let sep = loadedTokenizer.convertTokenToId("[SEP]")
        else {
            reset()
            throw EngineError.modelLoadFailed(
                reason: "Reranker tokenizer is missing [CLS]/[SEP]; only BERT-family "
                    + "cross-encoders (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2) are "
                    + "supported in this first cut.")
        }

        self.model = cross
        self.tokenizer = loadedTokenizer
        self.clsId = cls
        self.sepId = sep
        self.padId = loadedTokenizer.convertTokenToId("[PAD]") ?? 0
        self.maxLength = resolvedMaxLength
        self.loadedModel = model
    }

    // MARK: Scoring

    /// Score every `[query, document]` pair jointly, returning one raw
    /// relevance logit per document (row-aligned with `documents`, NOT
    /// sorted). Higher = more relevant. The endpoint ranks by these and
    /// exposes `sigmoid(logit)` as the bounded `relevance_score`.
    ///
    /// Batches all documents into a single padded forward pass (mirroring
    /// ``EmbeddingEngine/embed(_:)``), differing in the one way that makes it
    /// a cross-encoder: real per-pair `token_type_ids` (0 over the query span,
    /// 1 over the document span) instead of the all-zero types an embedder
    /// uses.
    public func score(query: String, documents: [String]) async throws -> [Float] {
        guard let model, let tokenizer else {
            throw EngineError.modelNotLoaded
        }
        if documents.isEmpty { return [] }

        // Assemble each pair's ids + segment ids independently (WordPiece
        // segments tokenize independently, then concat — see `buildPair`).
        let pairs = documents.map { document in
            Self.buildPair(
                query: query, document: document,
                clsId: clsId, sepId: sepId, maxLength: maxLength,
                encode: { tokenizer.encode(text: $0, addSpecialTokens: false) })
        }

        // Pad the batch to its longest row. Padded positions are masked, so
        // their token / segment ids are arbitrary; only the mask matters.
        let batchLength = pairs.reduce(0) { Swift.max($0, $1.ids.count) }
        let idRows = pairs.map { pair in
            MLXArray(pair.ids + Array(repeating: padId, count: batchLength - pair.ids.count))
        }
        let typeRows = pairs.map { pair in
            MLXArray(
                pair.tokenTypeIds + Array(repeating: 0, count: batchLength - pair.tokenTypeIds.count))
        }
        let maskRows = pairs.map { pair in
            MLXArray(
                Array(repeating: 1, count: pair.ids.count)
                    + Array(repeating: 0, count: batchLength - pair.ids.count))
        }

        let ids = stacked(idRows)            // [N, L]
        let tokenTypes = stacked(typeRows)   // [N, L]
        let mask = stacked(maskRows)         // [N, L]

        let logits = model(ids, tokenTypeIds: tokenTypes, attentionMask: mask)  // [N]
        // MUST eval + materialize before returning — MLXArray is not Sendable.
        logits.eval()
        return logits.asArray(Float.self)
    }

    // MARK: Pair assembly (pure, testable)

    /// Assemble one BERT cross-encoder pair: `[CLS] query [SEP] document [SEP]`
    /// with matching segment ids (0 over `[CLS] query [SEP]`, 1 over
    /// `document [SEP]`), truncated to `maxLength`.
    ///
    /// Kept a `nonisolated static` PURE function (no actor / model / real
    /// tokenizer) so the parity-critical assembly is unit-testable with a stub
    /// `encode`. Correctness rests on faithfully reproducing HF WordPiece pair
    /// encoding:
    /// - Each segment is tokenized WITHOUT special tokens, then concatenated —
    ///   BERT WordPiece is context-free, so `encode(q) + encode(d)` equals
    ///   encoding them jointly. `[CLS]`/`[SEP]` are inserted here.
    /// - `token_type_ids` are 0 for `[CLS] query [SEP]` and 1 for
    ///   `document [SEP]` — the segment signal a cross-encoder learns the
    ///   query/document roles from.
    /// - Truncation follows HF's default `truncation="longest_first"` with
    ///   `truncation_side="right"`: reserve the 3 special tokens, then trim the
    ///   last token of whichever segment is currently longer until the pair
    ///   fits; ties trim the document (the second sequence), matching
    ///   transformers. `maxLength` is assumed 512 per standard HF BERT when the
    ///   checkpoint doesn't say otherwise.
    ///
    /// - Parameter encode: Tokenizes a segment to ids WITHOUT special tokens
    ///   (the caller wires `tokenizer.encode(text:addSpecialTokens:false)`).
    /// - Returns: `ids` and same-length `tokenTypeIds` for one pair.
    static func buildPair(
        query: String,
        document: String,
        clsId: Int,
        sepId: Int,
        maxLength: Int,
        encode: (String) -> [Int]
    ) -> (ids: [Int], tokenTypeIds: [Int]) {
        var queryIds = encode(query)
        var documentIds = encode(document)

        // Content budget after reserving [CLS] + [SEP] + [SEP].
        let budget = Swift.max(0, maxLength - 3)
        while queryIds.count + documentIds.count > budget {
            if queryIds.count > documentIds.count {
                queryIds.removeLast()
            } else if !documentIds.isEmpty {
                documentIds.removeLast()
            } else if !queryIds.isEmpty {
                queryIds.removeLast()
            } else {
                break
            }
        }

        let ids = [clsId] + queryIds + [sepId] + documentIds + [sepId]
        let tokenTypeIds =
            Array(repeating: 0, count: 1 + queryIds.count + 1)  // [CLS] query [SEP]
            + Array(repeating: 1, count: documentIds.count + 1)  // document [SEP]
        return (ids: ids, tokenTypeIds: tokenTypeIds)
    }

    // MARK: Private

    /// Drop all resident state back to "no model loaded".
    private func reset() {
        model = nil
        tokenizer = nil
        loadedModel = nil
        clsId = 0
        sepId = 0
        padId = 0
        maxLength = 512
    }
}

/// The two `config.json` scalars ``RerankEngine`` needs beyond what
/// `BertConfiguration` exposes: the classifier's input width and the
/// truncation ceiling. Both optional — defaults (768 / 512) are the standard
/// HF BERT-base values, applied when a checkpoint omits them.
private struct RerankerConfigFields: Decodable {
    let hiddenSize: Int?
    let maxPositionEmbeddings: Int?

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case maxPositionEmbeddings = "max_position_embeddings"
    }
}
