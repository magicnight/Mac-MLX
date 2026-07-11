import XCTest
import MLX
import MLXNN
import MLXLMCommon
@testable import MacMLXCore

/// Numerical parity: our Swift `SeedOssAttention` must reproduce the Python
/// mlx-lm reference for BOTH asymmetric-bias configurations, which together
/// triangulate the two INDEPENDENT attention bias switches:
///
///  • `seed_oss_attention_qkv_bias` — attention_bias=true, attention_out_bias=
///    false: q/k/v carry a bias, o_proj does NOT (the real checkpoint's shape).
///  • `seed_oss_attention_o_bias`   — attention_bias=false, attention_out_bias=
///    true: q/k/v have no bias, o_proj does.
///
/// If `o_proj` were wired to `attention_bias` (the Llama single-switch bug), the
/// first fixture would wrongly add an o bias and the second would wrongly drop
/// it — either diverges at 1e-4. A single all-on/all-off pair could not tell the
/// two switches apart; the asymmetry is the point.
///
/// Fixtures captured offline from mlx-lm 0.31.3 (`seed_oss.py`) — Python never
/// enters macMLX. The bool mask is stored as uint8 and reloaded as `!= 0`.
final class SeedOssAttentionParityTests: XCTestCase {

    /// Must match `capture_seed_oss.py`: hidden 32, 4 heads / 2 kv-heads,
    /// head-dim 16, rope_theta 1e7, `rope_scaling {"rope_type": "default"}`.
    private func fixtureConfig(attentionBias: Bool, attentionOutBias: Bool) throws
        -> SeedOssConfiguration
    {
        let json = """
        {
          "model_type": "seed_oss",
          "hidden_size": 32,
          "num_hidden_layers": 2,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 16,
          "rms_norm_eps": 1e-6,
          "max_position_embeddings": 64,
          "attention_bias": \(attentionBias),
          "attention_out_bias": \(attentionOutBias),
          "rope_theta": 10000000.0,
          "rope_scaling": {"rope_type": "default"}
        }
        """
        return try JSONDecoder().decode(SeedOssConfiguration.self, from: Data(json.utf8))
    }

    /// Load a fixture, build `SeedOssAttention` with the matching bias switches,
    /// feed the captured bool mask, and assert 1e-4 parity. All non-control keys
    /// (weights + whatever bias tensors the config carries) load directly, so the
    /// optional biases are handled by the fixture itself.
    private func runParity(
        fixture: String, attentionBias: Bool, attentionOutBias: Bool
    ) throws {
        try requireTrustworthyMetalOrSkip()

        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: fixture, withExtension: "safetensors",
                subdirectory: "Fixtures"),
            "\(fixture) not found in test bundle")
        let arrays = try MLX.loadArrays(url: url)

        let config = try fixtureConfig(
            attentionBias: attentionBias, attentionOutBias: attentionOutBias)
        let attn = SeedOssAttention(config)

        var params: [String: MLXArray] = [:]
        for (key, value) in arrays where key != "x" && key != "mask" && key != "expected_output" {
            params[key] = value
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

    /// q/k/v biased, o_proj NOT (real checkpoint shape).
    func testQkvBiasedOutputUnbiasedMatchesPythonReference() throws {
        try runParity(
            fixture: "seed_oss_attention_qkv_bias_fixture",
            attentionBias: true, attentionOutBias: false)
    }

    /// q/k/v unbiased, o_proj biased (inverse — proves the switches are independent).
    func testQkvUnbiasedOutputBiasedMatchesPythonReference() throws {
        try runParity(
            fixture: "seed_oss_attention_o_bias_fixture",
            attentionBias: false, attentionOutBias: true)
    }
}
