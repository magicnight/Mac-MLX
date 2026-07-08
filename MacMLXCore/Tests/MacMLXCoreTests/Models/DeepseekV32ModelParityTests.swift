import XCTest
import MLX
import MLXNN
import MLXLMCommon
@testable import MacMLXCore

/// Numerical parity: our Swift `DeepseekV32Model` (full assembly — token
/// embedding, decoder layers, final norm, `lm_head`) must reproduce the
/// Python mlx-lm reference's logits given identical weights + inputs.
///
/// S4 target — the model end-to-end. The tiny 2-layer fixture is built with
/// `first_k_dense_replace = 1` + `n_routed_experts = 4`, so layer 0 is DENSE
/// and layer 1 is MoE: the run exercises the `DeepseekV32DecoderLayer`
/// dense/MoE selection predicate top to bottom. Sequence length 4 ==
/// `index_topk`, so the DSA indexer short-circuits to dense attention (its
/// sparse/decode parity is proven by the S2.2 / S2.3 fixtures).
///
/// The fixture stores the *module* (post-sanitize) layout — layer 1's experts
/// are already stacked into `switch_mlp` — so this test loads weights directly
/// and does NOT exercise `sanitize`; that has its own round-trip test below.
///
/// Fixture `Fixtures/model_fixture.safetensors` captured offline from mlx-lm
/// 0.31.3 (+ the PR #1431 rope patch) — see `docs/reference/capture_model.py`.
/// Python never enters macMLX.
final class DeepseekV32ModelParityTests: XCTestCase {

    /// Must match `capture_model.py` (tiny 2-layer config: dense layer 0 +
    /// MoE layer 1).
    private func fixtureConfig() throws -> DeepseekV32Configuration {
        let json = """
        {
          "model_type": "deepseek_v32",
          "vocab_size": 32,
          "hidden_size": 16,
          "num_attention_heads": 2,
          "q_lora_rank": 12,
          "qk_nope_head_dim": 4,
          "qk_rope_head_dim": 4,
          "kv_lora_rank": 6,
          "v_head_dim": 4,
          "index_head_dim": 8,
          "index_n_heads": 2,
          "index_topk": 4,
          "intermediate_size": 32,
          "moe_intermediate_size": 8,
          "n_routed_experts": 4,
          "n_shared_experts": 1,
          "num_experts_per_tok": 2,
          "n_group": 1,
          "topk_group": 1,
          "norm_topk_prob": true,
          "routed_scaling_factor": 2.5,
          "topk_method": "noaux_tc",
          "scoring_func": "sigmoid",
          "num_hidden_layers": 2,
          "first_k_dense_replace": 1,
          "moe_layer_freq": 1,
          "rms_norm_eps": 1e-6,
          "max_position_embeddings": 64,
          "rope_theta": 10000
        }
        """
        return try JSONDecoder().decode(DeepseekV32Configuration.self, from: Data(json.utf8))
    }

    /// THE correctness gate: load the captured fixture, run the full Swift
    /// model with identical weights + inputs (default cache → prefill), and
    /// assert the logits match the Python reference within 1e-4. Exercises
    /// embedding, the dense + MoE decoder layers via the predicate, the causal
    /// mask sourced from `cache[0][0]`, the final norm, and `lm_head`.
    func testFullModelMatchesPythonReference() throws {
        try requireMLXRuntimeOrSkip()

        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "model_fixture", withExtension: "safetensors",
                subdirectory: "Fixtures"),
            "model_fixture not found in test bundle")
        let arrays = try MLX.loadArrays(url: url)

        let config = try fixtureConfig()
        let model = DeepseekV32Model(config)

        var params: [String: MLXArray] = [:]
        for (key, value) in arrays where key != "x" && key != "expected_output" {
            params[key] = value
        }
        try model.update(
            parameters: ModuleParameters.unflattened(params), verify: [.noUnusedKeys])

        let x = try XCTUnwrap(arrays["x"])  // int32 token ids [1, 4]
        let expected = try XCTUnwrap(arrays["expected_output"])

        let out = model(x)  // default cache == nil → prefill, offset 0
        out.eval()

        XCTAssertEqual(out.shape, expected.shape, "full-model output shape mismatch")
        let close = allClose(out, expected, rtol: 1e-4, atol: 1e-4)
        XCTAssertTrue(
            close.item(Bool.self),
            "full-model logits diverge from the Python reference")
    }

    // MARK: - sanitize round-trip

    /// Deterministic host-side array (no eval): the reshape/stack/swap ops in
    /// `sanitize` only need shape metadata, so the round-trip runs Metal-free.
    private func arr(_ shape: [Int]) -> MLXArray {
        let n = shape.reduce(1, *)
        return MLXArray((0 ..< n).map { Float($0) }, shape)
    }

    /// Config for the sanitize round-trip: 1 layer, 4 routed experts, and the
    /// attention dims that make the `kv_b_proj` split shapes work out.
    private func sanitizeConfig() throws -> DeepseekV32Configuration {
        let json = """
        {
          "model_type": "deepseek_v32",
          "vocab_size": 32,
          "hidden_size": 16,
          "num_attention_heads": 2,
          "qk_nope_head_dim": 4,
          "v_head_dim": 4,
          "kv_lora_rank": 6,
          "moe_intermediate_size": 8,
          "n_routed_experts": 4,
          "num_hidden_layers": 1,
          "first_k_dense_replace": 0
        }
        """
        return try JSONDecoder().decode(DeepseekV32Configuration.self, from: Data(json.utf8))
    }

    /// `sanitize` must: stack per-expert MoE projections into `switch_mlp`,
    /// split `kv_b_proj.weight` into `embed_q` / `unembed_out`, drop the
    /// per-expert / `kv_b_proj` / MTP (`layers.99`) / fp8 keys, and pass
    /// unrelated keys through untouched. Pure dict transform — no Metal (shape
    /// metadata only), so it runs even under bare `swift test`.
    func testSanitizeRoundTrip() throws {
        let config = try sanitizeConfig()
        let model = DeepseekV32Model(config)

        var weights: [String: MLXArray] = [:]
        // Per-expert MoE projections (to be stacked → switch_mlp).
        for e in 0 ..< 4 {
            let p = "model.layers.0.mlp.experts.\(e)"
            weights["\(p).gate_proj.weight"] = arr([8, 16])
            weights["\(p).up_proj.weight"] = arr([8, 16])
            weights["\(p).down_proj.weight"] = arr([16, 8])
        }
        // Absorbed-MLA kv_b_proj (to be split → embed_q / unembed_out).
        // [num_heads * (qk_nope + v_head), kv_lora] = [2 * 8, 6] = [16, 6].
        weights["model.layers.0.self_attn.kv_b_proj.weight"] = arr([16, 6])
        // Multi-token-prediction layer (index 99 ≥ num_hidden_layers) → drop.
        weights["model.layers.99.self_attn.q_a_proj.weight"] = arr([12, 16])
        // fp8 block-scale key → drop defensively.
        weights["model.layers.0.mlp.gate_proj.weight_scale_inv"] = arr([1])
        // Unrelated key → pass through untouched.
        weights["model.embed_tokens.weight"] = arr([32, 16])

        let out = model.sanitize(weights: weights)

        // Expert stacking: switch_mlp banks appear with [n_experts, out, in].
        XCTAssertEqual(
            out["model.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape, [4, 8, 16])
        XCTAssertEqual(
            out["model.layers.0.mlp.switch_mlp.up_proj.weight"]?.shape, [4, 8, 16])
        XCTAssertEqual(
            out["model.layers.0.mlp.switch_mlp.down_proj.weight"]?.shape, [4, 16, 8])
        // Per-expert keys are consumed.
        for e in 0 ..< 4 {
            XCTAssertNil(out["model.layers.0.mlp.experts.\(e).gate_proj.weight"])
            XCTAssertNil(out["model.layers.0.mlp.experts.\(e).up_proj.weight"])
            XCTAssertNil(out["model.layers.0.mlp.experts.\(e).down_proj.weight"])
        }

        // kv_b_proj split into per-head embed_q (kᵀ) and unembed_out (v).
        XCTAssertEqual(
            out["model.layers.0.self_attn.embed_q.weight"]?.shape, [2, 6, 4])
        XCTAssertEqual(
            out["model.layers.0.self_attn.unembed_out.weight"]?.shape, [2, 4, 6])
        XCTAssertNil(out["model.layers.0.self_attn.kv_b_proj.weight"])

        // Dropped: MTP layer + fp8 scale.
        XCTAssertNil(out["model.layers.99.self_attn.q_a_proj.weight"])
        XCTAssertNil(out["model.layers.0.mlp.gate_proj.weight_scale_inv"])

        // Passthrough survives.
        XCTAssertEqual(out["model.embed_tokens.weight"]?.shape, [32, 16])
    }
}
