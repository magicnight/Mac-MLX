import XCTest
import MLX
import MLXNN
import MLXLMCommon
@testable import MacMLXCore

/// Numerical parity: our Swift `DeepseekV32DecoderLayer` (dense) must
/// reproduce the Python mlx-lm reference's output tensor given identical
/// weights + inputs.
///
/// S2.4 target — the **dense** decoder block: pre-norm absorbed-MLA
/// attention + residual, then pre-norm SwiGLU MLP + residual. The fixture
/// prefills `s = 6 > index_topk = 4`, so the indexer is live inside the
/// attention (the attention's own per-branch parity is proven separately
/// by `DeepseekV32AttentionParityTests`). This test adds the residual
/// wiring, the two `config.rms_norm_eps` layer norms, and — since the block
/// runs the MLP — the dense `DeepseekV32MLP` on top. The routed-expert MoE
/// branch is S3.
///
/// Fixture `Fixtures/decoder_layer_fixture.safetensors` captured offline
/// from mlx-lm 0.31.3 (+ the PR #1431 rope patch) — see
/// `docs/reference/capture_decoder_layer.py`. Python never enters macMLX.
final class DeepseekV32DecoderLayerParityTests: XCTestCase {

    /// Must match `capture_decoder_layer.py` (same tiny attention config
    /// PLUS `intermediate_size` for the MLP and `rms_norm_eps` for the
    /// block norms).
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
          "intermediate_size": 32,
          "rms_norm_eps": 1e-6,
          "max_position_embeddings": 64,
          "rope_theta": 10000
        }
        """
        return try JSONDecoder().decode(DeepseekV32Configuration.self, from: Data(json.utf8))
    }

    /// Flattened weight keys the fixture carries. Nested keys unflatten onto
    /// submodules (`self_attn.indexer.wq_b.weight` → the indexer's `wqB`,
    /// `mlp.gate_proj.weight` → the MLP's `gateProj`, etc.).
    private let weightKeys = [
        "self_attn.q_a_proj.weight", "self_attn.q_a_layernorm.weight",
        "self_attn.q_b_proj.weight", "self_attn.kv_a_proj_with_mqa.weight",
        "self_attn.kv_a_layernorm.weight", "self_attn.embed_q.weight",
        "self_attn.unembed_out.weight", "self_attn.o_proj.weight",
        "self_attn.indexer.wq_b.weight", "self_attn.indexer.wk.weight",
        "self_attn.indexer.k_norm.weight", "self_attn.indexer.k_norm.bias",
        "self_attn.indexer.weights_proj.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
        "input_layernorm.weight", "post_attention_layernorm.weight",
    ]

    /// THE correctness gate: load the captured fixture, run the Swift dense
    /// decoder layer with identical weights + inputs, and assert the output
    /// matches the Python reference within 1e-4. Exercises attention +
    /// residuals + both layer norms + the dense MLP end-to-end.
    func testDenseDecoderLayerMatchesPythonReference() throws {
        try requireTrustworthyMetalOrSkip()

        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "decoder_layer_fixture", withExtension: "safetensors",
                subdirectory: "Fixtures"),
            "decoder_layer_fixture not found in test bundle")
        let arrays = try MLX.loadArrays(url: url)

        let config = try fixtureConfig()
        // layerIdx 0 with `n_routed_experts` nil in the fixture config → the
        // predicate stays dense (still exercises the dense MLP branch).
        let layer = DeepseekV32DecoderLayer(config, layerIdx: 0)

        var params: [String: MLXArray] = [:]
        for key in weightKeys {
            params[key] = try XCTUnwrap(arrays[key], "missing weight \(key) in fixture")
        }
        try layer.update(
            parameters: ModuleParameters.unflattened(params), verify: [.noUnusedKeys])

        let x = try XCTUnwrap(arrays["x"])
        let mask = try XCTUnwrap(arrays["mask"]).asType(.bool)  // uint8 → bool
        let expected = try XCTUnwrap(arrays["expected_output"])

        // Fresh per-layer cache (main MLA KV + indexer), prefill offset 0 —
        // reproduces the reference's `CacheList(KVCache(), KVCache())`.
        let out = layer(x, mask: mask, cache: CacheList(KVCacheSimple(), KVCacheSimple()))
        out.eval()

        XCTAssertEqual(out.shape, expected.shape, "decoder-layer output shape mismatch")
        let close = allClose(out, expected, rtol: 1e-4, atol: 1e-4)
        XCTAssertTrue(
            close.item(Bool.self),
            "dense decoder-layer output diverges from the Python reference")
    }

    /// Non-Metal structure check for the cache/kv-head helpers: one
    /// `CacheList` of two `KVCacheSimple` sub-caches per layer, and a
    /// per-layer KV-head count vector of the right length. Runs even under
    /// bare `swift test` (no MLX ops).
    func testCacheAndKVHeadHelpers() throws {
        let caches = deepseekV32NewCache(layerCount: 3)
        XCTAssertEqual(caches.count, 3, "one CacheList per layer")
        for cache in caches {
            // Two sub-caches: main MLA KV + DSA indexer.
            XCTAssertTrue(cache[0] is KVCacheSimple, "sub-cache 0 should be KVCacheSimple")
            XCTAssertTrue(cache[1] is KVCacheSimple, "sub-cache 1 should be KVCacheSimple")
        }

        let config = try fixtureConfig()
        let heads = deepseekV32KVHeads(config)
        XCTAssertEqual(
            heads.count, config.numHiddenLayers, "one KV-head count per layer")
        XCTAssertTrue(
            heads.allSatisfy { $0 == config.numKeyValueHeads },
            "every layer uses num_key_value_heads")
    }
}
