import Testing
import Foundation
import MLX
@testable import MacMLXCore

/// `Mellum2Model.sanitize` weight-rewrite edge cases. The INCOMPLETE-expert-set
/// path is pure dictionary logic (no stacking, no eval), but constructing the
/// model and its `MLXArray` weights still touches the MLX runtime, so the suite
/// gates on the metallib exactly like every other MLX-backed suite (runs under
/// xcodebuild; skipped under bare `swift test`). A tiny config keeps construction
/// cheap.
@Suite("Mellum2 sanitize", .enabled(if: mlxMetallibIsAvailable, "Requires default.metallib (run under xcodebuild)"))
struct Mellum2SanitizeTests {

    /// A tiny Mellum2: 1 layer, 2 experts, tiny dims — cheap, lazy construction.
    private func tinyModel() throws -> Mellum2Model {
        let json = """
        {
          "model_type": "mellum",
          "vocab_size": 16,
          "hidden_size": 8,
          "num_hidden_layers": 1,
          "intermediate_size": 8,
          "num_attention_heads": 1,
          "num_key_value_heads": 1,
          "head_dim": 8,
          "num_experts": 2,
          "num_experts_per_tok": 1,
          "moe_intermediate_size": 8,
          "tie_word_embeddings": false
        }
        """
        let config = try JSONDecoder().decode(Mellum2Configuration.self, from: Data(json.utf8))
        return Mellum2Model(config)
    }

    /// A partial expert set (one expert's tensors missing for a projection) must
    /// leave EVERY per-expert tensor untouched — never remove experts 0..<k while
    /// stacking, then discard the result on the first gap. Regression for the
    /// destructive `removeValue`-as-you-go loop that lost already-removed experts.
    @Test("incomplete expert set leaves per-expert weights intact, no switch_mlp")
    func incompleteExpertSetIsNonDestructive() throws {
        let model = try tinyModel()

        // Expert 0 present for all three projections; expert 1 missing entirely →
        // the set is incomplete for every projection.
        let present = [
            "model.layers.0.mlp.experts.0.up_proj.weight",
            "model.layers.0.mlp.experts.0.down_proj.weight",
            "model.layers.0.mlp.experts.0.gate_proj.weight",
        ]
        var weights: [String: MLXArray] = [:]
        for key in present { weights[key] = MLXArray([Float(1)]) }

        let out = model.sanitize(weights: weights)

        // Every original per-expert key survives (nothing lost)...
        for key in present {
            #expect(out[key] != nil, "sanitize dropped \(key) on a partial expert set")
        }
        // ...and no stacked bank was fabricated from the incomplete set.
        #expect(out["model.layers.0.mlp.switch_mlp.up_proj.weight"] == nil)
        #expect(out["model.layers.0.mlp.switch_mlp.down_proj.weight"] == nil)
        #expect(out["model.layers.0.mlp.switch_mlp.gate_proj.weight"] == nil)
    }

    /// Already-stacked checkpoints (no per-expert `experts.*` keys) short-circuit
    /// untouched — the common case for the shipped MLX conversion.
    @Test("pre-stacked weights short-circuit unchanged")
    func preStackedShortCircuits() throws {
        let model = try tinyModel()
        var weights: [String: MLXArray] = [
            "model.layers.0.mlp.switch_mlp.up_proj.weight": MLXArray([Float(1)]),
        ]
        weights["model.embed_tokens.weight"] = MLXArray([Float(2)])

        let out = model.sanitize(weights: weights)
        #expect(out["model.layers.0.mlp.switch_mlp.up_proj.weight"] != nil)
        #expect(out["model.embed_tokens.weight"] != nil)
    }
}
