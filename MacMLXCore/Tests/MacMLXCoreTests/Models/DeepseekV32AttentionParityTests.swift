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

    /// Load a captured attention fixture, run the Swift attention with the
    /// identical weights + inputs, and assert the output matches the Python
    /// reference. Shared by the prefill (S2.1) and sparse-prefill (S2.2)
    /// cases — they differ only in the fixture (s ≤ vs > index_topk).
    private func assertAttentionParity(fixture: String, _ label: String) throws {
        try requireMLXRuntimeOrSkip()

        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: fixture, withExtension: "safetensors", subdirectory: "Fixtures"),
            "\(fixture) not found in test bundle")
        let arrays = try MLX.loadArrays(url: url)

        let config = try fixtureConfig()
        let attn = DeepseekV32Attention(config)

        // Nested keys unflatten onto submodules (indexer.wq_b.weight →
        // indexer.wqB.weight, etc.).
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

        XCTAssertEqual(out.shape, expected.shape, "\(label): output shape mismatch")
        let close = allClose(out, expected, rtol: 1e-4, atol: 1e-4)
        XCTAssertTrue(close.item(Bool.self), "\(label) diverges from the Python reference")
    }

    /// S2.1: prefill, `s <= index_topk` → indexer short-circuits, dense.
    func testPrefillMatchesPythonReference() throws {
        try assertAttentionParity(fixture: "attn_prefill_fixture", "attention prefill")
    }

    /// S2.2: prefill, `s > index_topk` → indexer returns top-k, driving the
    /// sparse-mask scatter branch (`put_along_axis` + AND with the causal
    /// mask). Exercises the path S2.1 short-circuited past.
    func testSparsePrefillMatchesPythonReference() throws {
        try assertAttentionParity(fixture: "attn_sparse_fixture", "attention sparse prefill")
    }

    /// S2.3: decode step with a primed `CacheList`. Prefill 6 tokens to fill
    /// both sub-caches (main MLA KV + indexer), then decode 1 token — total
    /// 7 > index_topk 4, so the indexer returns a real top-k and the `L == 1`
    /// `take_along_axis` gather branch runs (the path S2.1/S2.2 never hit).
    /// Reproduces the exact two-stage cache state, then compares the decode
    /// output. Fixture `attn_decode_fixture.safetensors` — see
    /// `docs/reference/capture_attention_decode.py`.
    func testDecodeStepMatchesPythonReference() throws {
        try requireMLXRuntimeOrSkip()

        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "attn_decode_fixture", withExtension: "safetensors",
                subdirectory: "Fixtures"),
            "attn_decode_fixture not found in test bundle")
        let arrays = try MLX.loadArrays(url: url)

        let config = try fixtureConfig()
        let attn = DeepseekV32Attention(config)

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

        // One CacheList holds the layer's two sub-caches (main KV + indexer).
        let cache = CacheList(KVCacheSimple(), KVCacheSimple())

        // Stage 1: prefill — primes both sub-caches with 6 tokens.
        let xPrefill = try XCTUnwrap(arrays["x_prefill"])
        let maskPrefill = try XCTUnwrap(arrays["mask_prefill"]).asType(.bool)
        _ = attn(xPrefill, mask: maskPrefill, cache: cache)

        // Stage 2: decode one token (mask nil — attend the cached top-k).
        let xDecode = try XCTUnwrap(arrays["x_decode"])
        let out = attn(xDecode, mask: nil, cache: cache)
        out.eval()

        let expected = try XCTUnwrap(arrays["expected_decode"])
        XCTAssertEqual(out.shape, expected.shape, "decode output shape mismatch")
        let close = allClose(out, expected, rtol: 1e-4, atol: 1e-4)
        XCTAssertTrue(
            close.item(Bool.self),
            "attention decode-step output diverges from the Python reference")
    }
}
