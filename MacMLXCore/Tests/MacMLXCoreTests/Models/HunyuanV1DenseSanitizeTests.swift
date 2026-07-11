import Testing
import Foundation
import MLX
@testable import MacMLXCore

/// `HunyuanV1DenseModel.sanitize` weight-rewrite contract. Ports the Python
/// `Model.sanitize`: `lm_head.weight` is dropped only when the embeddings are
/// tied (the shipped checkpoints tie); everything else passes through. Building
/// the model and its `MLXArray` weights touches the MLX runtime, so the suite
/// gates on the metallib exactly like the other MLX-backed suites (runs under
/// xcodebuild; skipped under bare `swift test`).
@Suite("HunyuanV1Dense sanitize", .enabled(if: mlxMetallibIsAvailable, "Requires default.metallib (run under xcodebuild)"))
struct HunyuanV1DenseSanitizeTests {

    /// A tiny model with the given tie flag — cheap, lazy construction.
    private func tinyModel(tieWordEmbeddings: Bool) throws -> HunyuanV1DenseModel {
        let json = """
        {
          "model_type": "hunyuan_v1_dense",
          "vocab_size": 16,
          "hidden_size": 8,
          "num_hidden_layers": 1,
          "intermediate_size": 16,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 4,
          "tie_word_embeddings": \(tieWordEmbeddings)
        }
        """
        let config = try JSONDecoder().decode(
            HunyuanV1DenseConfiguration.self, from: Data(json.utf8))
        return HunyuanV1DenseModel(config)
    }

    /// Tied embeddings (the shipped checkpoints): `lm_head.weight` must be dropped,
    /// other keys survive.
    @Test("tied embeddings drop lm_head.weight, keep the rest")
    func tiedDropsLmHead() throws {
        let model = try tinyModel(tieWordEmbeddings: true)

        var weights: [String: MLXArray] = [
            "lm_head.weight": MLXArray([Float(1)]),
            "model.embed_tokens.weight": MLXArray([Float(2)]),
            "model.norm.weight": MLXArray([Float(3)]),
        ]
        let out = model.sanitize(weights: weights)

        #expect(out["lm_head.weight"] == nil)
        #expect(out["model.embed_tokens.weight"] != nil)
        #expect(out["model.norm.weight"] != nil)
        weights.removeAll()
    }

    /// Untied embeddings: `lm_head.weight` is preserved and every key passes
    /// through untouched.
    @Test("untied embeddings keep lm_head.weight; pure passthrough")
    func untiedKeepsLmHead() throws {
        let model = try tinyModel(tieWordEmbeddings: false)

        let weights: [String: MLXArray] = [
            "lm_head.weight": MLXArray([Float(1)]),
            "model.embed_tokens.weight": MLXArray([Float(2)]),
            "model.layers.0.self_attn.q_proj.weight": MLXArray([Float(4)]),
        ]
        let out = model.sanitize(weights: weights)

        #expect(out["lm_head.weight"] != nil)
        #expect(out["model.embed_tokens.weight"] != nil)
        #expect(out["model.layers.0.self_attn.q_proj.weight"] != nil)
        #expect(out.count == weights.count)
    }
}
