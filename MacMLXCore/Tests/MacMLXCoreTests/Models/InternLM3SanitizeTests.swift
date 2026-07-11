import Testing
import Foundation
import MLX
@testable import MacMLXCore

/// `InternLM3Model.sanitize` weight-rewrite contract. Ports the Python
/// `Model.sanitize`: keys containing `attention.rope.inv_freq` (InternLM2-legacy
/// precomputed rotary buffers some checkpoints carry) are dropped; everything else
/// passes through untouched. Unlike Hunyuan, there is NO tie-driven `lm_head` drop.
/// Building the model and its `MLXArray` weights touches the MLX runtime, so the
/// suite gates on the metallib exactly like the other MLX-backed suites (runs under
/// xcodebuild; skipped under bare `swift test`).
@Suite("InternLM3 sanitize", .enabled(if: mlxMetallibIsAvailable, "Requires default.metallib (run under xcodebuild)"))
struct InternLM3SanitizeTests {

    /// A tiny model — cheap, lazy construction.
    private func tinyModel() throws -> InternLM3Model {
        let json = """
        {
          "model_type": "internlm3",
          "vocab_size": 16,
          "hidden_size": 8,
          "num_hidden_layers": 1,
          "intermediate_size": 16,
          "num_attention_heads": 2,
          "num_key_value_heads": 1
        }
        """
        let config = try JSONDecoder().decode(
            InternLM3Configuration.self, from: Data(json.utf8))
        return InternLM3Model(config)
    }

    /// The InternLM2-legacy rotary buffer key is filtered; every other key survives.
    @Test("drops attention.rope.inv_freq, keeps the rest")
    func dropsInvFreq() throws {
        let model = try tinyModel()

        var weights: [String: MLXArray] = [
            "model.layers.0.attention.rope.inv_freq": MLXArray([Float(1)]),
            "model.embed_tokens.weight": MLXArray([Float(2)]),
            "model.layers.0.self_attn.q_proj.weight": MLXArray([Float(3)]),
            "lm_head.weight": MLXArray([Float(4)]),
        ]
        let out = model.sanitize(weights: weights)

        #expect(out["model.layers.0.attention.rope.inv_freq"] == nil)
        #expect(out["model.embed_tokens.weight"] != nil)
        #expect(out["model.layers.0.self_attn.q_proj.weight"] != nil)
        // No tie-driven lm_head drop — sanitize only filters inv_freq.
        #expect(out["lm_head.weight"] != nil)
        #expect(out.count == weights.count - 1)
        weights.removeAll()
    }

    /// With no legacy rotary buffers present, sanitize is a pure passthrough.
    @Test("no inv_freq keys: pure passthrough")
    func passthroughWhenAbsent() throws {
        let model = try tinyModel()

        let weights: [String: MLXArray] = [
            "model.embed_tokens.weight": MLXArray([Float(1)]),
            "model.norm.weight": MLXArray([Float(2)]),
            "lm_head.weight": MLXArray([Float(3)]),
        ]
        let out = model.sanitize(weights: weights)

        #expect(out.count == weights.count)
        #expect(out["model.embed_tokens.weight"] != nil)
        #expect(out["model.norm.weight"] != nil)
        #expect(out["lm_head.weight"] != nil)
    }
}
