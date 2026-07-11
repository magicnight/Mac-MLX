import XCTest
import MLX
import MLXNN
import MLXLMCommon
@testable import MacMLXCore

/// Numerical parity: our Swift `HunyuanV1DenseModel` (full assembly — token
/// embedding, decoder layers with post-RoPE q/k norm + DynamicNTKAlpha RoPE,
/// final norm, tied/untied head) must reproduce the Python mlx-lm reference's
/// logits given identical weights + inputs, across TWO adversarial regimes whose
/// EVERY switch is inverted:
///
///  • `hunyuan_v1_dense_realistic` — the shipped-checkpoint shape: use_qk_norm=
///    true, attention_bias=false, tie_word_embeddings=true, DynamicNTK RoPE with
///    alpha=1000, and an EXPLICIT head_dim (16) that is NOT hidden/heads (8) —
///    so the o_proj shape and the head_dim scale are both exercised off the
///    non-default path (explicit head_dim 16 != hidden/heads = 8).
///  • `hunyuan_v1_dense_inverse`  — use_qk_norm=false, attention_bias=true (q/k/v
///    AND o biased), tie_word_embeddings=false (untied lm_head), rope_scaling
///    OMITTED (alpha fallback → 1.0) and head_dim OMITTED (fallback → hidden/heads
///    = 8).
///
/// Because the two configs disagree on every switch, any switch read backwards
/// (q/k norm before RoPE instead of after; alpha ignored; head_dim fallback wrong;
/// bias or tie flipped) diverges on at least one fixture at 1e-4, or fails to load
/// its weights. Each tiny 2-layer, 2-sequence fixture exercises the token
/// embedding, the DynamicNTKAlpha RoPE, the full causal mask, GQA (4 heads / 2
/// kv), optional post-RoPE q/k RMSNorm, SwiGLU MLP, residual wiring, final norm,
/// and the tied/untied head end-to-end.
///
/// Fixtures captured offline from mlx-lm 0.31.3 (`hunyuan_v1_dense.py`) — see
/// `docs/reference/capture_hunyuan_v1_dense.py`. Python never enters macMLX.
final class HunyuanV1DenseModelParityTests: XCTestCase {

    /// Realistic config — must match `capture_hunyuan_v1_dense.py`'s realistic
    /// fixture (explicit head_dim 16, q/k norm on, no bias, tied, dynamic alpha).
    private func realisticConfig() throws -> HunyuanV1DenseConfiguration {
        let json = """
        {
          "model_type": "hunyuan_v1_dense",
          "vocab_size": 64,
          "hidden_size": 32,
          "num_hidden_layers": 2,
          "intermediate_size": 48,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 16,
          "rms_norm_eps": 1e-5,
          "max_position_embeddings": 128,
          "rope_theta": 10000.0,
          "attention_bias": false,
          "use_qk_norm": true,
          "tie_word_embeddings": true,
          "rope_scaling": {"type": "dynamic", "alpha": 1000.0, "factor": 1.0}
        }
        """
        return try JSONDecoder().decode(HunyuanV1DenseConfiguration.self, from: Data(json.utf8))
    }

    /// Inverse config — must match the inverse fixture (head_dim OMITTED → 16,
    /// q/k norm off, attention bias on, untied, rope_scaling OMITTED → alpha 1.0).
    private func inverseConfig() throws -> HunyuanV1DenseConfiguration {
        let json = """
        {
          "model_type": "hunyuan_v1_dense",
          "vocab_size": 64,
          "hidden_size": 32,
          "num_hidden_layers": 2,
          "intermediate_size": 48,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "rms_norm_eps": 1e-5,
          "max_position_embeddings": 128,
          "rope_theta": 10000.0,
          "attention_bias": true,
          "use_qk_norm": false,
          "tie_word_embeddings": false
        }
        """
        return try JSONDecoder().decode(HunyuanV1DenseConfiguration.self, from: Data(json.utf8))
    }

    private func runParity(fixture: String, config: HunyuanV1DenseConfiguration) throws {
        try requireTrustworthyMetalOrSkip()

        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: fixture, withExtension: "safetensors",
                subdirectory: "Fixtures"),
            "\(fixture) not found in test bundle")
        let arrays = try MLX.loadArrays(url: url)

        let model = HunyuanV1DenseModel(config)

        var params: [String: MLXArray] = [:]
        for (key, value) in arrays where key != "x" && key != "expected_output" {
            params[key] = value
        }
        try model.update(
            parameters: ModuleParameters.unflattened(params), verify: [.noUnusedKeys])

        let x = try XCTUnwrap(arrays["x"])  // int32 token ids [2, 6]
        let expected = try XCTUnwrap(arrays["expected_output"])

        let out = model(x)  // default cache == nil → prefill, offset 0
        out.eval()

        XCTAssertEqual(out.shape, expected.shape, "\(fixture): output shape mismatch")
        let close = allClose(out, expected, rtol: 1e-4, atol: 1e-4)
        XCTAssertTrue(
            close.item(Bool.self),
            "\(fixture): full-model logits diverge from the Python reference")
    }

    /// Shipped-checkpoint shape: q/k norm on, no bias, tied, dynamic alpha=1000,
    /// explicit head_dim != hidden/heads.
    func testRealisticMatchesPythonReference() throws {
        try runParity(fixture: "hunyuan_v1_dense_realistic_fixture", config: realisticConfig())
    }

    /// Inverse shape: q/k norm off, attention bias on, untied, alpha + head_dim
    /// fallbacks.
    func testInverseMatchesPythonReference() throws {
        try runParity(fixture: "hunyuan_v1_dense_inverse_fixture", config: inverseConfig())
    }
}
