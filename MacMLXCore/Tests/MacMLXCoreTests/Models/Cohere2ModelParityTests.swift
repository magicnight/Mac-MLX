import XCTest
import MLX
import MLXNN
import MLXLMCommon
@testable import MacMLXCore

/// Numerical parity: our Swift `Cohere2Model` (full assembly — token embedding,
/// decoder layers with the PARALLEL residual block, interleaved sliding-window /
/// global attention, traditional RoPE on sliding layers + NoPE on global layers,
/// final LayerNorm, tied head scaled by `logit_scale`) must reproduce the Python
/// mlx-lm reference's logits given identical weights + inputs, across TWO
/// adversarial regimes whose EVERY switch is inverted:
///
///  • `cohere2_realistic` — the shipped-checkpoint shape: attention_bias=false,
///    layer_norm_bias=false, logit_scale=0.25 (the checkpoint's value, NOT the
///    0.0625 dataclass default), sliding_window_pattern=4, sliding_window=4. With
///    4 layers this gives L0/L1/L2 sliding (+RoPE) and L3 global (NoPE).
///  • `cohere2_inverse`  — attention_bias=true (q/k/v AND o biased),
///    layer_norm_bias=true (LayerNorm carries a bias), logit_scale OMITTED
///    (fallback → 0.0625), sliding_window_pattern=2, sliding_window=5. With 4
///    layers this gives L0/L2 sliding (+RoPE) and L1/L3 global (NoPE).
///
/// CRITICAL: both fixtures use seq_len (8) > sliding_window (4 / 5), so the
/// windowed sliding mask genuinely differs from the full-causal mask — a port that
/// mis-sources the two masks, or applies RoPE on the wrong (global) layers, or
/// drops the NoPE distinction, diverges on at least one fixture at 1e-4. Because
/// the two configs disagree on every switch (bias, ln_bias, logit_scale, pattern,
/// window), any switch read backwards diverges or fails to load its weights. Each
/// tiny 4-layer, 2-sequence fixture exercises the token embedding, the interleaved
/// attention (both mask families, both RoPE/NoPE paths), the parallel residual
/// wiring, the SwiGLU MLP, the final LayerNorm, and the tied + logit-scaled head
/// end-to-end.
///
/// Fixtures captured offline from mlx-lm 0.31.3 (`cohere2.py`) — see
/// `docs/reference/capture_cohere2.py`. Python never enters macMLX.
final class Cohere2ModelParityTests: XCTestCase {

    /// Realistic config — must match `capture_cohere2.py`'s realistic fixture
    /// (no bias, no ln_bias, logit_scale 0.25, pattern 4, window 4).
    private func realisticConfig() throws -> Cohere2Configuration {
        let json = """
        {
          "model_type": "cohere2",
          "vocab_size": 64,
          "hidden_size": 32,
          "num_hidden_layers": 4,
          "intermediate_size": 48,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 8,
          "layer_norm_eps": 1e-5,
          "rope_theta": 50000.0,
          "logit_scale": 0.25,
          "attention_bias": false,
          "layer_norm_bias": false,
          "sliding_window": 4,
          "sliding_window_pattern": 4
        }
        """
        return try JSONDecoder().decode(Cohere2Configuration.self, from: Data(json.utf8))
    }

    /// Inverse config — must match the inverse fixture (attention bias ON, ln_bias
    /// ON, logit_scale OMITTED → 0.0625 fallback, pattern 2, window 5).
    private func inverseConfig() throws -> Cohere2Configuration {
        let json = """
        {
          "model_type": "cohere2",
          "vocab_size": 64,
          "hidden_size": 32,
          "num_hidden_layers": 4,
          "intermediate_size": 48,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 8,
          "layer_norm_eps": 1e-5,
          "rope_theta": 50000.0,
          "attention_bias": true,
          "layer_norm_bias": true,
          "sliding_window": 5,
          "sliding_window_pattern": 2
        }
        """
        return try JSONDecoder().decode(Cohere2Configuration.self, from: Data(json.utf8))
    }

    private func runParity(fixture: String, config: Cohere2Configuration) throws {
        try requireTrustworthyMetalOrSkip()

        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: fixture, withExtension: "safetensors",
                subdirectory: "Fixtures"),
            "\(fixture) not found in test bundle")
        let arrays = try MLX.loadArrays(url: url)

        let model = Cohere2Model(config)

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
            "\(fixture): full-model logits diverge from the Python reference")
    }

    /// Shipped-checkpoint shape: no bias, no ln_bias, logit_scale 0.25, pattern 4.
    func testRealisticMatchesPythonReference() throws {
        try runParity(fixture: "cohere2_realistic_fixture", config: realisticConfig())
    }

    /// Inverse shape: attention bias on, ln_bias on, logit_scale fallback, pattern 2.
    func testInverseMatchesPythonReference() throws {
        try runParity(fixture: "cohere2_inverse_fixture", config: inverseConfig())
    }
}
