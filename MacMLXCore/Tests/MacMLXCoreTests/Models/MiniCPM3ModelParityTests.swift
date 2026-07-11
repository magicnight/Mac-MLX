import XCTest
import MLX
import MLXNN
import MLXLMCommon
@testable import MacMLXCore

/// Numerical parity: our Swift `MiniCPM3Model` (full assembly — token embedding
/// scaled by `scale_emb`, decoder layers with NON-ABSORBED MLA and muP depth
/// scaling on both residual branches, final RMSNorm, and the muP head — untied
/// `lm_head` on the `÷(hidden/dim_model_base)` hidden, or the tied embedding
/// matrix without that division) must reproduce the Python mlx-lm reference's
/// logits given identical weights + inputs, across TWO adversarial regimes whose
/// EVERY switch/scaling is inverted:
///
///  • `minicpm3_realistic` — UNTIED (head ÷ hidden/dim_base = 32/8 = 4),
///    attention_bias=false, scale_emb=12, scale_depth=1.4, longrope with
///    long_factor [1.2,1.7] and a DELIBERATELY-wrong short_factor [9.9,9.9]
///    (MiniCPM3 ignores short_factor — a port that uses it diverges), and
///    max_position == original_max → mscale 1.0.
///  • `minicpm3_inverse`  — TIED (NO head division; no `lm_head` key),
///    attention_bias=true (q_a/kv_a/o biased; q_b/kv_b still bias-free),
///    scale_emb=3, scale_depth=0.7, long_factor [1.5,1.1], max_position 256 >
///    original_max 64 → mscale = √(1+ln(4)/ln(64)) ≈ 1.1547 (pins the non-trivial
///    mscale formula), and dim_model_base=16 which must NOT be consumed on the tied
///    path.
///
/// Because the two configs disagree on every switch/scaling, any one read backwards
/// diverges or fails to load its weights. Each tiny 2-layer, 2-sequence fixture
/// exercises the token embedding + scale_emb, the full MLA data flow (q/kv low-rank
/// projections, the nope/rope split, the single-head RoPE key broadcast to all
/// heads, the distinct value head dim, the q_head_dim softmax scale), the muP depth
/// scaling on both branches, the SwiGLU MLP, the final RMSNorm, and the muP head
/// (both the untied `÷` path and the tied no-`÷` path) end-to-end.
///
/// Fixtures captured offline from mlx-lm 0.31.3 (`minicpm3.py`) — see
/// `docs/reference/capture_minicpm3.py`. Python never enters macMLX.
final class MiniCPM3ModelParityTests: XCTestCase {

    /// Realistic config — must match `capture_minicpm3.py`'s realistic fixture
    /// (untied, no attention bias, scale_emb 12, scale_depth 1.4, mscale 1.0).
    private func realisticConfig() throws -> MiniCPM3Configuration {
        let json = """
        {
          "model_type": "minicpm3",
          "vocab_size": 64,
          "hidden_size": 32,
          "dim_model_base": 8,
          "num_hidden_layers": 2,
          "intermediate_size": 48,
          "num_attention_heads": 4,
          "num_key_value_heads": 4,
          "q_lora_rank": 16,
          "kv_lora_rank": 12,
          "qk_nope_head_dim": 8,
          "qk_rope_head_dim": 4,
          "scale_emb": 12,
          "scale_depth": 1.4,
          "rms_norm_eps": 1e-5,
          "rope_theta": 10000.0,
          "attention_bias": false,
          "tie_word_embeddings": false,
          "max_position_embeddings": 64,
          "rope_scaling": {
            "type": "longrope",
            "long_factor": [1.2, 1.7],
            "short_factor": [9.9, 9.9],
            "original_max_position_embeddings": 64
          }
        }
        """
        return try JSONDecoder().decode(MiniCPM3Configuration.self, from: Data(json.utf8))
    }

    /// Inverse config — must match the inverse fixture (tied, attention bias ON,
    /// scale_emb 3, scale_depth 0.7, mscale ≈ 1.1547, dim_model_base 16 unused,
    /// rms_norm_eps 3e-4 — the LAYER norms move off 1e-5 while the internal q_a/kv_a
    /// norms stay at the mlx 1e-5 default, pinning the eps-source split; see the
    /// EPS SOURCE SPLIT note in `capture_minicpm3.py`).
    private func inverseConfig() throws -> MiniCPM3Configuration {
        let json = """
        {
          "model_type": "minicpm3",
          "vocab_size": 64,
          "hidden_size": 32,
          "dim_model_base": 16,
          "num_hidden_layers": 2,
          "intermediate_size": 48,
          "num_attention_heads": 4,
          "num_key_value_heads": 4,
          "q_lora_rank": 16,
          "kv_lora_rank": 12,
          "qk_nope_head_dim": 8,
          "qk_rope_head_dim": 4,
          "scale_emb": 3,
          "scale_depth": 0.7,
          "rms_norm_eps": 3e-4,
          "rope_theta": 10000.0,
          "attention_bias": true,
          "tie_word_embeddings": true,
          "max_position_embeddings": 256,
          "rope_scaling": {
            "type": "longrope",
            "long_factor": [1.5, 1.1],
            "short_factor": [9.9, 9.9],
            "original_max_position_embeddings": 64
          }
        }
        """
        return try JSONDecoder().decode(MiniCPM3Configuration.self, from: Data(json.utf8))
    }

    private func runParity(fixture: String, config: MiniCPM3Configuration) throws {
        try requireTrustworthyMetalOrSkip()

        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: fixture, withExtension: "safetensors",
                subdirectory: "Fixtures"),
            "\(fixture) not found in test bundle")
        let arrays = try MLX.loadArrays(url: url)

        let model = MiniCPM3Model(config)

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

    /// Untied shape: head ÷ (hidden/dim_base), no attention bias, scale_emb 12,
    /// scale_depth 1.4, mscale 1.0.
    func testRealisticMatchesPythonReference() throws {
        try runParity(fixture: "minicpm3_realistic_fixture", config: realisticConfig())
    }

    /// Tied shape: no head division, attention bias on, scale_emb 3, scale_depth
    /// 0.7, non-trivial mscale.
    func testInverseMatchesPythonReference() throws {
        try runParity(fixture: "minicpm3_inverse_fixture", config: inverseConfig())
    }
}
