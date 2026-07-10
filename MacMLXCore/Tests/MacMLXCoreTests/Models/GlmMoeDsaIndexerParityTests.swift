import XCTest
import MLX
import MLXNN
import MLXLMCommon
@testable import MacMLXCore

/// Numerical parity for the ONE component GLM-DSA changes: the lightning
/// indexer's RoPE runs interleaved (`indexer_rope_interleave = true`, GLM-DSA's
/// `ModelArgs` default) versus DeepSeek V3.2's non-interleaved default. Given
/// identical weights + inputs, our `DeepseekV32Indexer` — driven by a GLM-DSA
/// config — must select the same top-k key indices as the Python mlx-lm
/// reference on the interleaved path.
///
/// Fixture `Fixtures/glm_moe_dsa_indexer_parity_fixture.safetensors` was
/// captured from mlx-lm's `deepseek_v32.Indexer` with
/// `indexer_rope_interleave=True` (see
/// `docs/reference/capture_glm_moe_dsa_indexer.py`; Python never enters macMLX
/// — it's an offline capture). It is the interleaved sibling of
/// `indexer_parity_fixture.safetensors` (which pins the `False` path). The
/// test decodes a plain `DeepseekV32Configuration` directly (with
/// `indexer_rope_interleave: true` set by hand) — it does NOT go through
/// `GlmMoeDsaConfiguration`; see the Scope note below for the deliberate
/// separation of concerns.
///
/// argPartition's within-partition order is unstable, so both sides sort the
/// top-k along the key axis before comparing — the *set* of selected keys is the
/// invariant.
///
/// Scope: this pins the **numeric interleave difference only** — the config
/// reduces to interleave `true` + a plain (unscaled) RoPE base, exactly the
/// capture's `indexer_rope_interleave=True`, `rope_scaling=None` setup. The
/// GLM-DSA adapter's own job (deriving `rope_theta` / `rope_scaling` from
/// `rope_parameters`, defaulting interleave to `true`) is pinned separately by
/// `GlmMoeDsaConfigurationTests`, so the two concerns don't entangle.
final class GlmMoeDsaIndexerParityTests: XCTestCase {

    /// Must match `capture_glm_moe_dsa_indexer.py`: the effective V3.2 config a
    /// `glm_moe_dsa` config reduces to for the indexer — `indexer_rope_interleave`
    /// true (GLM-DSA's default), plain rope base (no scaling).
    private func fixtureConfig() throws -> DeepseekV32Configuration {
        let json = """
        {
          "model_type": "deepseek_v32",
          "hidden_size": 16,
          "q_lora_rank": 12,
          "index_head_dim": 8,
          "index_n_heads": 2,
          "index_topk": 4,
          "qk_rope_head_dim": 4,
          "max_position_embeddings": 64,
          "rope_theta": 10000,
          "indexer_rope_interleave": true
        }
        """
        return try JSONDecoder().decode(DeepseekV32Configuration.self, from: Data(json.utf8))
    }

    func testMatchesPythonInterleavedTopKSelection() throws {
        try requireTrustworthyMetalOrSkip()

        let config = try fixtureConfig()
        // The GLM-DSA difference this fixture exists to pin.
        XCTAssertTrue(
            config.indexerRopeInterleave,
            "GLM-DSA config must default indexer_rope_interleave to true")

        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "glm_moe_dsa_indexer_parity_fixture",
                withExtension: "safetensors",
                subdirectory: "Fixtures"),
            "GLM-DSA parity fixture not found in test bundle")
        let arrays = try MLX.loadArrays(url: url)

        let indexer = DeepseekV32Indexer(config)

        // Load the captured weights (only the parameter keys — x / qr /
        // expected_* are test data, not parameters).
        let weightKeys = [
            "wq_b.weight", "wk.weight",
            "k_norm.weight", "k_norm.bias",
            "weights_proj.weight",
        ]
        var params: [String: MLXArray] = [:]
        for k in weightKeys {
            params[k] = try XCTUnwrap(arrays[k], "missing weight \(k) in fixture")
        }
        try indexer.update(
            parameters: ModuleParameters.unflattened(params), verify: [.noUnusedKeys])

        let x = try XCTUnwrap(arrays["x"])
        let qr = try XCTUnwrap(arrays["qr"])
        let expected = try XCTUnwrap(arrays["expected_topk_sorted"])  // [1,1,S,topk] int32

        let out = try XCTUnwrap(
            indexer(x, qr, nil),
            "indexer returned nil but the fixture has s > index_topk")

        let gotSorted = sorted(out.asType(.int32), axis: -1)
        gotSorted.eval()

        XCTAssertEqual(gotSorted.shape, expected.shape, "top-k shape mismatch")

        let gotFlat = gotSorted.asArray(Int32.self)
        let expFlat = expected.asArray(Int32.self)
        XCTAssertEqual(
            gotFlat, expFlat,
            "Swift indexer (interleaved) selected different top-k keys than the Python reference")
    }
}
