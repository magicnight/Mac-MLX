import XCTest
import MLX
import MLXNN
import MLXLMCommon
@testable import MacMLXCore

/// Numerical parity: our Swift `Mellum2Attention` must reproduce the Python
/// mlx-lm reference for BOTH attention families:
///  • sliding-window layers — plain (default) RoPE + a windowed causal mask,
///  • full-attention layers — YaRN RoPE + a full causal mask.
///
/// The two share weights + input but differ in RoPE (chosen by `layer_type`)
/// and mask, so this proves `mellum2Rope` picks the right family and the
/// windowed vs full mask both flow correctly through `attentionWithCacheUpdate`.
/// The captured outputs are identical only at position 0 (self-attention, RoPE
/// identity) and diverge from there — see `capture_mellum2.py`.
///
/// Fixtures captured offline from mlx-lm @ e476a22 (mellum.py); Python never
/// enters macMLX. The bool mask is stored as uint8 and reloaded as `!= 0`.
final class Mellum2AttentionParityTests: XCTestCase {

    /// Must match `capture_mellum2.py`: hidden 32, 4 heads / 2 kv-heads,
    /// head-dim 16, sliding_window 3, YaRN on full-attention layers. Layer 0 is
    /// sliding, layer 2 is full (`layer_types`).
    private func fixtureConfig() throws -> Mellum2Configuration {
        let json = """
        {
          "model_type": "mellum",
          "hidden_size": 32,
          "num_hidden_layers": 4,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 16,
          "rms_norm_eps": 1e-6,
          "max_position_embeddings": 64,
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

    private let weightKeys = [
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
        "q_norm.weight", "k_norm.weight",
    ]

    /// Load a fixture, build `Mellum2Attention` at `layerIdx`, feed the captured
    /// bool mask, and assert 1e-4 parity against the reference output.
    private func runParity(fixture: String, layerIdx: Int) throws {
        try requireTrustworthyMetalOrSkip()

        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: fixture, withExtension: "safetensors",
                subdirectory: "Fixtures"),
            "\(fixture) not found in test bundle")
        let arrays = try MLX.loadArrays(url: url)

        let config = try fixtureConfig()
        let attn = Mellum2Attention(config, layerIdx: layerIdx)

        var params: [String: MLXArray] = [:]
        for key in weightKeys {
            params[key] = try XCTUnwrap(arrays[key], "missing weight \(key) in \(fixture)")
        }
        try attn.update(
            parameters: ModuleParameters.unflattened(params), verify: [.noUnusedKeys])

        let x = try XCTUnwrap(arrays["x"])
        let maskU8 = try XCTUnwrap(arrays["mask"])
        let expected = try XCTUnwrap(arrays["expected_output"])

        // uint8 -> bool: attend where != 0 (same semantics as the Python mask).
        let boolMask = maskU8 .!= MLXArray(UInt8(0))
        let out = attn(x, mask: .array(boolMask), cache: nil)
        out.eval()

        XCTAssertEqual(out.shape, expected.shape, "\(fixture): output shape mismatch")
        let close = allClose(out, expected, rtol: 1e-4, atol: 1e-4)
        XCTAssertTrue(
            close.item(Bool.self),
            "\(fixture): attention output diverges from the Python reference")
    }

    /// Sliding-window layer (layer 0): default RoPE + windowed causal mask.
    func testSlidingAttentionMatchesPythonReference() throws {
        try runParity(fixture: "mellum2_attention_sliding_fixture", layerIdx: 0)
    }

    /// Full-attention layer (layer 2): YaRN RoPE + full causal mask.
    func testFullAttentionMatchesPythonReference() throws {
        try runParity(fixture: "mellum2_attention_full_fixture", layerIdx: 2)
    }
}
