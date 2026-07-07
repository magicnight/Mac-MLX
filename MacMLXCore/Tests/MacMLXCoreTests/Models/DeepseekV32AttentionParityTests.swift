import XCTest
import MLX
import MLXNN
import MLXLMCommon
@testable import MacMLXCore

/// Numerical parity: our Swift `DeepseekV32Attention` must reproduce the
/// Python mlx-lm reference's output tensor given identical weights +
/// inputs.
///
/// S2.1 target — the **prefill** absorbed-MLA core in isolation. The
/// fixture uses `s = 3 <= index_topk = 4`, so the indexer short-circuits
/// to `nil` and no sparsification happens: this validates the
/// q_a/q_b + kv_a_proj + RoPE + embed_q/unembed_out + pe_scores-as-mask +
/// o_proj math, not the top-k gather/scatter (S2.2 / S2.3).
///
/// Fixture `Fixtures/attn_prefill_fixture.safetensors` captured offline
/// from mlx-lm 0.31.3 (+ the PR #1431 rope patch) — see
/// `docs/reference/capture_attention_prefill.py`. Python never enters
/// macMLX.
final class DeepseekV32AttentionParityTests: XCTestCase {

    /// Must match `capture_attention_prefill.py`.
    private func fixtureConfig() throws -> DeepseekV32Configuration {
        let json = """
        {
          "model_type": "deepseek_v32",
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
          "max_position_embeddings": 64,
          "rope_theta": 10000
        }
        """
        return try JSONDecoder().decode(DeepseekV32Configuration.self, from: Data(json.utf8))
    }

    func testPrefillMatchesPythonReference() throws {
        try requireMLXRuntimeOrSkip()

        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "attn_prefill_fixture", withExtension: "safetensors",
                subdirectory: "Fixtures"),
            "attention prefill fixture not found in test bundle")
        let arrays = try MLX.loadArrays(url: url)

        let config = try fixtureConfig()
        let attn = DeepseekV32Attention(config)

        // Load the captured weights (nested keys unflatten onto the
        // submodules: indexer.wq_b.weight → indexer.wqB.weight, etc.).
        let weightKeys = [
            "q_a_proj.weight", "q_a_layernorm.weight", "q_b_proj.weight",
            "kv_a_proj_with_mqa.weight", "kv_a_layernorm.weight",
            "embed_q.weight", "unembed_out.weight", "o_proj.weight",
            "indexer.wq_b.weight", "indexer.wk.weight",
            "indexer.k_norm.weight", "indexer.k_norm.bias",
            "indexer.weights_proj.weight",
        ]
        var params: [String: MLXArray] = [:]
        for key in weightKeys {
            params[key] = try XCTUnwrap(arrays[key], "missing weight \(key) in fixture")
        }
        try attn.update(
            parameters: ModuleParameters.unflattened(params), verify: [.noUnusedKeys])

        let x = try XCTUnwrap(arrays["x"])
        let mask = try XCTUnwrap(arrays["mask"]).asType(.bool)  // uint8 → bool
        let expected = try XCTUnwrap(arrays["expected_output"])

        let out = attn(x, mask: mask, cache: nil)
        out.eval()

        XCTAssertEqual(out.shape, expected.shape, "output shape mismatch")
        let close = allClose(out, expected, rtol: 1e-4, atol: 1e-4)
        XCTAssertTrue(
            close.item(Bool.self),
            "attention prefill output diverges from the Python reference")
    }
}
