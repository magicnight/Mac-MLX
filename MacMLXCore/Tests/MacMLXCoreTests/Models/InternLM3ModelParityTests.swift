import XCTest
import MLX
import MLXNN
import MLXLMCommon
@testable import MacMLXCore

/// Numerical parity: our Swift `InternLM3Model` (full assembly — token embedding,
/// dense GQA decoder layers with the corrected DynamicNTK RoPE, SwiGLU MLP, final
/// RMSNorm, and the untied `lm_head` / tied-embedding head) must reproduce the
/// mlx-lm reference's logits given identical weights + inputs, across TWO adversarial
/// regimes whose EVERY switch is inverted AND whose RoPE path pins all three
/// corrected upstream defects.
///
/// The reference is a MINIMALLY-PATCHED mlx-lm (`internlm3.py` with only the three
/// defective RoPE lines corrected — see `docs/reference/capture_internlm3.py`), NOT
/// stock mlx-lm, because the port deliberately implements the reference
/// `modeling_internlm3.py` semantics, not the upstream bugs.
///
///  • `internlm3_dynamic_active` — UNTIED, qkv_bias=false, bias=false, rope_scaling
///    {rope_type: dynamic, factor: 4.0}, max_position 6 < seq len 8 → the NTK base
///    fires in prefill, pinning all three corrections: factor 4 enters the base
///    (defect B, not the buggy 2.0), the sequence length is the SEQUENCE axis 8 and
///    not the heads axis 4 (defect C — reading heads=4 < 6 would NOT fire the base),
///    and positions are not doubled (defect A, scale 1.0). The config also carries a
///    bogus `head_dim: 999` decoy that must be ignored (head_dim is always
///    hidden/heads).
///  • `internlm3_inverse_plain` — TIED (no `lm_head` key), qkv_bias=true (q/k/v/o all
///    biased), bias=true (MLP gate/up/down all biased), no rope_scaling (plain RoPE,
///    position scale 1.0 — pins defect A on the no-scaling path), max_position 128
///    (dynamic base never fires).
///
/// Because the two configs disagree on every switch, any one read backwards diverges
/// or fails to load its weights. Fixtures captured offline from patched mlx-lm 0.31.3
/// — see `docs/reference/capture_internlm3.py`. Python never enters macMLX.
final class InternLM3ModelParityTests: XCTestCase {

    /// Dynamic-active config — must match the `internlm3_dynamic_active` fixture
    /// (untied, no biases, dynamic scaling factor 4, max_position 6 < seq 8, bogus
    /// `head_dim: 999` decoy).
    private func dynamicActiveConfig() throws -> InternLM3Configuration {
        let json = """
        {
          "model_type": "internlm3",
          "vocab_size": 64,
          "hidden_size": 32,
          "num_hidden_layers": 2,
          "intermediate_size": 48,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "rms_norm_eps": 1e-5,
          "rope_theta": 10000.0,
          "bias": false,
          "qkv_bias": false,
          "tie_word_embeddings": false,
          "max_position_embeddings": 6,
          "head_dim": 999,
          "rope_scaling": { "rope_type": "dynamic", "factor": 4.0 }
        }
        """
        return try JSONDecoder().decode(InternLM3Configuration.self, from: Data(json.utf8))
    }

    /// Inverse-plain config — must match the `internlm3_inverse_plain` fixture (tied,
    /// qkv_bias ON, bias ON, no rope_scaling, max_position 128).
    private func inversePlainConfig() throws -> InternLM3Configuration {
        let json = """
        {
          "model_type": "internlm3",
          "vocab_size": 64,
          "hidden_size": 32,
          "num_hidden_layers": 2,
          "intermediate_size": 48,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "rms_norm_eps": 1e-5,
          "rope_theta": 10000.0,
          "bias": true,
          "qkv_bias": true,
          "tie_word_embeddings": true,
          "max_position_embeddings": 128
        }
        """
        return try JSONDecoder().decode(InternLM3Configuration.self, from: Data(json.utf8))
    }

    private func runParity(fixture: String, config: InternLM3Configuration) throws {
        try requireTrustworthyMetalOrSkip()

        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: fixture, withExtension: "safetensors",
                subdirectory: "Fixtures"),
            "\(fixture) not found in test bundle")
        let arrays = try MLX.loadArrays(url: url)

        let model = InternLM3Model(config)

        var params: [String: MLXArray] = [:]
        for (key, value) in arrays where key != "x" && key != "expected_output" {
            params[key] = value
        }
        try model.update(
            parameters: ModuleParameters.unflattened(params), verify: [.noUnusedKeys])

        let x = try XCTUnwrap(arrays["x"])  // int32 token ids [2, 8]
        let expected = try XCTUnwrap(arrays["expected_output"])

        let out = model(x)  // default cache == nil → prefill, offset 0
        out.eval()

        XCTAssertEqual(out.shape, expected.shape, "\(fixture): output shape mismatch")
        let close = allClose(out, expected, rtol: 1e-4, atol: 1e-4)
        XCTAssertTrue(
            close.item(Bool.self),
            "\(fixture): full-model logits diverge from the patched Python reference")
    }

    /// Untied, no biases, dynamic NTK base active in prefill (pins defects A/B/C).
    func testDynamicActiveMatchesPythonReference() throws {
        try runParity(fixture: "internlm3_dynamic_active_fixture", config: dynamicActiveConfig())
    }

    /// Tied, qkv_bias + bias ON, plain RoPE (pins defect A on the no-scaling path).
    func testInversePlainMatchesPythonReference() throws {
        try runParity(fixture: "internlm3_inverse_plain_fixture", config: inversePlainConfig())
    }
}
