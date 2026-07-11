import Testing
import Foundation
import MLX
@testable import MacMLXCore

/// `SeedOssModel.sanitize` weight-rewrite contract. Ports the Python
/// `Model.sanitize`: `lm_head.weight` is dropped only when the embeddings are
/// tied; everything else passes through. Constructing the model and its
/// `MLXArray` weights touches the MLX runtime, so the suite gates on the metallib
/// exactly like the other MLX-backed suites (runs under xcodebuild; skipped under
/// bare `swift test`).
@Suite("SeedOss sanitize", .enabled(if: mlxMetallibIsAvailable, "Requires default.metallib (run under xcodebuild)"))
struct SeedOssSanitizeTests {

    /// A tiny SeedOss with the given tie flag — cheap, lazy construction.
    private func tinyModel(tieWordEmbeddings: Bool) throws -> SeedOssModel {
        let json = """
        {
          "model_type": "seed_oss",
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
        let config = try JSONDecoder().decode(SeedOssConfiguration.self, from: Data(json.utf8))
        return SeedOssModel(config)
    }

    /// Tied embeddings: `lm_head.weight` must be dropped, other keys survive.
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

    /// Untied embeddings (the real checkpoint): `lm_head.weight` is preserved and
    /// every key passes through untouched.
    @Test("untied embeddings keep lm_head.weight; pure passthrough")
    func untiedKeepsLmHead() throws {
        let model = try tinyModel(tieWordEmbeddings: false)

        let weights: [String: MLXArray] = [
            "lm_head.weight": MLXArray([Float(1)]),
            "model.embed_tokens.weight": MLXArray([Float(2)]),
            "model.layers.0.self_attn.q_proj.bias": MLXArray([Float(4)]),
        ]
        let out = model.sanitize(weights: weights)

        #expect(out["lm_head.weight"] != nil)
        #expect(out["model.embed_tokens.weight"] != nil)
        #expect(out["model.layers.0.self_attn.q_proj.bias"] != nil)
        #expect(out.count == weights.count)
    }
}
