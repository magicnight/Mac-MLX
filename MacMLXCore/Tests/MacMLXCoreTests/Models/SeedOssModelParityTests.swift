import XCTest
import MLX
import MLXNN
import MLXLMCommon
@testable import MacMLXCore

/// Numerical parity: our Swift `SeedOssModel` (full assembly — token embedding,
/// decoder layers, final norm, untied `lm_head`) must reproduce the Python mlx-lm
/// reference's logits given identical weights + inputs, across TWO bias regimes:
///
///  • `seed_oss_model`        — the real-checkpoint shape: attention_bias=true,
///    attention_out_bias=false, mlp_bias=false.
///  • `seed_oss_model_allbias`— every bias on (attention/out/mlp), so the o_proj
///    and MLP bias paths are exercised end-to-end too.
///
/// Each tiny 2-layer fixture exercises: the token embedding, the "default" RoPE
/// (theta 1e7), the full causal mask, GQA (4 heads / 2 kv), SwiGLU MLP, residual
/// wiring, final norm, and the untied `lm_head`. Passing both at 1e-4 proves the
/// port is numerically correct end-to-end with the bias switches independently
/// wired.
///
/// Fixtures captured offline from mlx-lm 0.31.3 (`seed_oss.py`) — see
/// `docs/reference/capture_seed_oss.py`.
final class SeedOssModelParityTests: XCTestCase {

    /// Must match `capture_seed_oss.py` (tiny 2-layer dense GQA).
    private func fixtureConfig(
        attentionBias: Bool, attentionOutBias: Bool, mlpBias: Bool
    ) throws -> SeedOssConfiguration {
        let json = """
        {
          "model_type": "seed_oss",
          "vocab_size": 40,
          "hidden_size": 32,
          "num_hidden_layers": 2,
          "intermediate_size": 48,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 16,
          "rms_norm_eps": 1e-6,
          "max_position_embeddings": 64,
          "attention_bias": \(attentionBias),
          "attention_out_bias": \(attentionOutBias),
          "mlp_bias": \(mlpBias),
          "rope_theta": 10000000.0,
          "rope_scaling": {"rope_type": "default"},
          "tie_word_embeddings": false
        }
        """
        return try JSONDecoder().decode(SeedOssConfiguration.self, from: Data(json.utf8))
    }

    private func runParity(
        fixture: String, attentionBias: Bool, attentionOutBias: Bool, mlpBias: Bool
    ) throws {
        try requireTrustworthyMetalOrSkip()

        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: fixture, withExtension: "safetensors",
                subdirectory: "Fixtures"),
            "\(fixture) not found in test bundle")
        let arrays = try MLX.loadArrays(url: url)

        let config = try fixtureConfig(
            attentionBias: attentionBias, attentionOutBias: attentionOutBias,
            mlpBias: mlpBias)
        let model = SeedOssModel(config)

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

        XCTAssertEqual(out.shape, expected.shape, "\(fixture): output shape mismatch")
        let close = allClose(out, expected, rtol: 1e-4, atol: 1e-4)
        XCTAssertTrue(
            close.item(Bool.self),
            "\(fixture): full-model logits diverge from the Python reference")
    }

    /// Real-checkpoint bias shape: q/k/v biased, o and MLP unbiased.
    func testFullModelRealBiasMatchesPythonReference() throws {
        try runParity(
            fixture: "seed_oss_model_fixture",
            attentionBias: true, attentionOutBias: false, mlpBias: false)
    }

    /// Every bias on — exercises the o_proj and MLP bias paths end-to-end.
    func testFullModelAllBiasMatchesPythonReference() throws {
        try runParity(
            fixture: "seed_oss_model_allbias_fixture",
            attentionBias: true, attentionOutBias: true, mlpBias: true)
    }
}
