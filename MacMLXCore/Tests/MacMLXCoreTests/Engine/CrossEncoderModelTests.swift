import Testing

@testable import MacMLXCore

/// Ungated tests for ``CrossEncoderModel``'s weight-key sanitizer — the
/// load-bearing, most breakage-prone part of the port. `loadWeights` finishes
/// with `update(parameters:, verify: [.all])`, which demands an EXACT key
/// bijection between the module's parameter tree and the sanitized checkpoint;
/// a single wrong rename aborts the load. These validate the rename table with
/// PURE STRINGS via `CrossEncoderModel.sanitizedKey` — NO model instance, NO
/// `MLXArray`, NO Metal, so they run under bare `swift test`.
///
/// The complementary check — that these target keys equal the CONSTRUCTED
/// module's actual parameters — needs an MLX model instance (Metal in this
/// environment) and is deferred to the real-checkpoint pass, together with the
/// forward-shape smoke and numeric parity. The target keys below are derived
/// from `MLXEmbedders.BertModel`'s `@ModuleInfo` keys (`query_proj`, `ln1`,
/// `linear1`, `pooler`, …), verified by reading its source.
@Suite("CrossEncoderModel sanitize")
struct CrossEncoderModelTests {

    @Test
    func renamesEachBertSubmoduleKeyKeepingTheBertPrefix() {
        let cases: [(hf: String, expected: String)] = [
            // Attention projections: `.self.{q,k,v}.` → `.{q,k,v}_proj.`
            (
                "bert.encoder.layer.0.attention.self.query.weight",
                "bert.encoder.layers.0.attention.query_proj.weight"
            ),
            (
                "bert.encoder.layer.3.attention.self.key.bias",
                "bert.encoder.layers.3.attention.key_proj.bias"
            ),
            (
                "bert.encoder.layer.0.attention.self.value.weight",
                "bert.encoder.layers.0.attention.value_proj.weight"
            ),
            // Attention output dense + its LayerNorm → out_proj / ln1
            (
                "bert.encoder.layer.0.attention.output.dense.weight",
                "bert.encoder.layers.0.attention.out_proj.weight"
            ),
            (
                "bert.encoder.layer.0.attention.output.LayerNorm.bias",
                "bert.encoder.layers.0.ln1.bias"
            ),
            // FFN: intermediate/output dense → linear1/linear2; output LN → ln2
            (
                "bert.encoder.layer.0.intermediate.dense.weight",
                "bert.encoder.layers.0.linear1.weight"
            ),
            (
                "bert.encoder.layer.0.output.dense.bias",
                "bert.encoder.layers.0.linear2.bias"
            ),
            (
                "bert.encoder.layer.0.output.LayerNorm.weight",
                "bert.encoder.layers.0.ln2.weight"
            ),
            // Embeddings LayerNorm → norm; pooler.dense → pooler
            ("bert.embeddings.LayerNorm.weight", "bert.embeddings.norm.weight"),
            ("bert.pooler.dense.bias", "bert.pooler.bias"),
        ]
        for c in cases {
            #expect(CrossEncoderModel.sanitizedKey(c.hf) == c.expected)
        }
    }

    @Test
    func passesEmbeddingAndClassifierKeysThroughUntouched() {
        let untouched = [
            "bert.embeddings.word_embeddings.weight",
            "bert.embeddings.position_embeddings.weight",
            "bert.embeddings.token_type_embeddings.weight",
            "classifier.weight",
            "classifier.bias",
        ]
        for key in untouched {
            #expect(CrossEncoderModel.sanitizedKey(key) == key)
        }
    }

    @Test
    func dropsThePositionIdsBuffer() {
        #expect(CrossEncoderModel.sanitizedKey("bert.embeddings.position_ids") == nil)
    }

    /// A different BERT-family reranker export may persist this default-segment
    /// buffer (the target `ms-marco-MiniLM-L-6-v2` checkpoint does not) — real
    /// per-pair segment ids are always supplied explicitly by `RerankEngine`,
    /// so an unused persisted buffer must be dropped the same way
    /// `position_ids` is, or it becomes an unexpected key under `verify: [.all]`.
    @Test
    func dropsTheTokenTypeIdsBuffer() {
        #expect(CrossEncoderModel.sanitizedKey("bert.embeddings.token_type_ids") == nil)
    }

    /// The full contract: a standard-HF `BertForSequenceClassification`
    /// checkpoint key set (1 layer) maps EXACTLY onto the expected module
    /// parameter key set — nothing missing, nothing extra, both the
    /// `position_ids` AND `token_type_ids` buffers dropped. This is the
    /// `verify: [.all]` bijection, string-checked. (Assumed per standard HF,
    /// unverified against a real download here.)
    @Test
    func fullCheckpointKeySetMapsOntoTheParameterTree() {
        let hfKeys = [
            "bert.embeddings.word_embeddings.weight",
            "bert.embeddings.position_embeddings.weight",
            "bert.embeddings.token_type_embeddings.weight",
            "bert.embeddings.LayerNorm.weight",
            "bert.embeddings.LayerNorm.bias",
            "bert.embeddings.position_ids",  // buffer → dropped
            "bert.embeddings.token_type_ids",  // buffer → dropped
            "bert.encoder.layer.0.attention.self.query.weight",
            "bert.encoder.layer.0.attention.self.query.bias",
            "bert.encoder.layer.0.attention.self.key.weight",
            "bert.encoder.layer.0.attention.self.key.bias",
            "bert.encoder.layer.0.attention.self.value.weight",
            "bert.encoder.layer.0.attention.self.value.bias",
            "bert.encoder.layer.0.attention.output.dense.weight",
            "bert.encoder.layer.0.attention.output.dense.bias",
            "bert.encoder.layer.0.attention.output.LayerNorm.weight",
            "bert.encoder.layer.0.attention.output.LayerNorm.bias",
            "bert.encoder.layer.0.intermediate.dense.weight",
            "bert.encoder.layer.0.intermediate.dense.bias",
            "bert.encoder.layer.0.output.dense.weight",
            "bert.encoder.layer.0.output.dense.bias",
            "bert.encoder.layer.0.output.LayerNorm.weight",
            "bert.encoder.layer.0.output.LayerNorm.bias",
            "bert.pooler.dense.weight",
            "bert.pooler.dense.bias",
            "classifier.weight",
            "classifier.bias",
        ]
        // Expected module parameter keys (from BertModel's @ModuleInfo layout).
        let expected: Set<String> = [
            "bert.embeddings.word_embeddings.weight",
            "bert.embeddings.position_embeddings.weight",
            "bert.embeddings.token_type_embeddings.weight",
            "bert.embeddings.norm.weight",
            "bert.embeddings.norm.bias",
            "bert.encoder.layers.0.attention.query_proj.weight",
            "bert.encoder.layers.0.attention.query_proj.bias",
            "bert.encoder.layers.0.attention.key_proj.weight",
            "bert.encoder.layers.0.attention.key_proj.bias",
            "bert.encoder.layers.0.attention.value_proj.weight",
            "bert.encoder.layers.0.attention.value_proj.bias",
            "bert.encoder.layers.0.attention.out_proj.weight",
            "bert.encoder.layers.0.attention.out_proj.bias",
            "bert.encoder.layers.0.ln1.weight",
            "bert.encoder.layers.0.ln1.bias",
            "bert.encoder.layers.0.linear1.weight",
            "bert.encoder.layers.0.linear1.bias",
            "bert.encoder.layers.0.linear2.weight",
            "bert.encoder.layers.0.linear2.bias",
            "bert.encoder.layers.0.ln2.weight",
            "bert.encoder.layers.0.ln2.bias",
            "bert.pooler.weight",
            "bert.pooler.bias",
            "classifier.weight",
            "classifier.bias",
        ]
        let mapped = Set(hfKeys.compactMap { CrossEncoderModel.sanitizedKey($0) })
        #expect(mapped == expected)
    }
}
