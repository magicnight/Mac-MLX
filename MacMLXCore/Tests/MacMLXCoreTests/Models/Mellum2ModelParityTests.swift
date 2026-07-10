import XCTest
import MLX
import MLXNN
import MLXLMCommon
@testable import MacMLXCore

/// Numerical parity: our Swift `Mellum2Model` (full assembly — token embedding,
/// decoder layers, final norm, `lm_head`) must reproduce the Python mlx-lm
/// reference's logits given identical weights + inputs.
///
/// The tiny 4-layer fixture uses `layer_types = [sliding, sliding, full, full]`
/// with `sliding_window (3) < seq_len (6)`, so the run exercises: the token
/// embedding, both RoPE families (default on sliding, YaRN on full), both mask
/// families (windowed vs full causal — built internally from the per-layer
/// caches), the 6-expert MoE on every layer, the residual wiring, the final
/// norm, and the untied `lm_head`. If this passes at 1e-4 the whole port is
/// numerically correct end-to-end.
///
/// The fixture stores the *module* (post-sanitize) layout — experts already
/// stacked into `switch_mlp` — so this loads weights directly and does NOT
/// exercise `sanitize` (that has its own round-trip test below).
///
/// Fixture `Fixtures/mellum2_model_fixture.safetensors` captured offline from
/// mlx-lm @ e476a22 (mellum.py) — see `docs/reference/capture_mellum2.py`.
final class Mellum2ModelParityTests: XCTestCase {

    /// Must match `capture_mellum2.py` (the tiny 4-layer mixed-attention MoE).
    private func fixtureConfig() throws -> Mellum2Configuration {
        let json = """
        {
          "model_type": "mellum",
          "vocab_size": 40,
          "hidden_size": 32,
          "num_hidden_layers": 4,
          "intermediate_size": 48,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 16,
          "num_experts": 6,
          "num_experts_per_tok": 2,
          "moe_intermediate_size": 24,
          "rms_norm_eps": 1e-6,
          "tie_word_embeddings": false,
          "max_position_embeddings": 64,
          "norm_topk_prob": true,
          "sliding_window": 3,
          "layer_types": [
            "sliding_attention", "sliding_attention",
            "full_attention", "full_attention"
          ],
          "rope_parameters": {
            "full_attention": {
              "rope_type": "yarn",
              "rope_theta": 500000.0,
              "factor": 16.0,
              "original_max_position_embeddings": 8192,
              "beta_fast": 32.0,
              "beta_slow": 1.0,
              "attention_factor": 1.2772588722239782
            },
            "sliding_attention": {
              "rope_type": "default",
              "rope_theta": 500000.0
            }
          }
        }
        """
        return try JSONDecoder().decode(Mellum2Configuration.self, from: Data(json.utf8))
    }

    func testFullModelMatchesPythonReference() throws {
        try requireTrustworthyMetalOrSkip()

        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "mellum2_model_fixture", withExtension: "safetensors",
                subdirectory: "Fixtures"),
            "mellum2_model_fixture not found in test bundle")
        let arrays = try MLX.loadArrays(url: url)

        let config = try fixtureConfig()
        let model = Mellum2Model(config)

        var params: [String: MLXArray] = [:]
        for (key, value) in arrays where key != "x" && key != "expected_output" {
            params[key] = value
        }
        try model.update(
            parameters: ModuleParameters.unflattened(params), verify: [.noUnusedKeys])

        let x = try XCTUnwrap(arrays["x"])  // int32 token ids [1, 6]
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

    /// Deterministic host-side array — the stack/reshape ops in `sanitize` only
    /// need shape metadata.
    private func arr(_ shape: [Int]) -> MLXArray {
        let n = shape.reduce(1, *)
        return MLXArray((0 ..< n).map { Float($0) }, shape)
    }

    private func sanitizeConfig() throws -> Mellum2Configuration {
        let json = """
        {
          "model_type": "mellum",
          "vocab_size": 40,
          "hidden_size": 32,
          "num_hidden_layers": 1,
          "num_experts": 6,
          "moe_intermediate_size": 24
        }
        """
        return try JSONDecoder().decode(Mellum2Configuration.self, from: Data(json.utf8))
    }

    /// `sanitize` must stack per-expert MoE projections into the `switch_mlp`
    /// bank `[num_experts, out, in]`, consume the per-expert keys, and pass
    /// unrelated keys through untouched.
    func testSanitizeStacksExperts() throws {
        try requireMLXRuntimeOrSkip()

        let config = try sanitizeConfig()
        let model = Mellum2Model(config)

        var weights: [String: MLXArray] = [:]
        for e in 0 ..< 6 {
            let p = "model.layers.0.mlp.experts.\(e)"
            weights["\(p).gate_proj.weight"] = arr([24, 32])
            weights["\(p).up_proj.weight"] = arr([24, 32])
            weights["\(p).down_proj.weight"] = arr([32, 24])
        }
        weights["model.embed_tokens.weight"] = arr([40, 32])  // passthrough

        let out = model.sanitize(weights: weights)

        XCTAssertEqual(
            out["model.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape, [6, 24, 32])
        XCTAssertEqual(
            out["model.layers.0.mlp.switch_mlp.up_proj.weight"]?.shape, [6, 24, 32])
        XCTAssertEqual(
            out["model.layers.0.mlp.switch_mlp.down_proj.weight"]?.shape, [6, 32, 24])
        for e in 0 ..< 6 {
            XCTAssertNil(out["model.layers.0.mlp.experts.\(e).gate_proj.weight"])
        }
        XCTAssertEqual(out["model.embed_tokens.weight"]?.shape, [40, 32])
    }

    /// A pre-stacked (module-layout) checkpoint must short-circuit `sanitize`
    /// untouched — the shipped 4-bit MLX conversion already carries `switch_mlp`.
    func testSanitizePreStackedShortCircuits() throws {
        try requireMLXRuntimeOrSkip()

        let config = try sanitizeConfig()
        let model = Mellum2Model(config)

        var weights: [String: MLXArray] = [
            "model.layers.0.mlp.switch_mlp.gate_proj.weight": arr([6, 24, 32]),
            "model.layers.0.mlp.gate.weight": arr([6, 32]),
        ]
        weights["lm_head.weight"] = arr([40, 32])

        let out = model.sanitize(weights: weights)

        XCTAssertEqual(
            out["model.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape, [6, 24, 32])
        // Untied model keeps lm_head.
        XCTAssertEqual(out["lm_head.weight"]?.shape, [40, 32])
    }
}
