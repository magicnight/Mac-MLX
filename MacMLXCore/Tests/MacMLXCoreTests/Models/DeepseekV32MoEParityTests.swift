import XCTest
import MLX
import MLXNN
import MLXLMCommon
@testable import MacMLXCore

/// Numerical parity: our Swift `DeepseekV32MoE` (routed + shared experts)
/// must reproduce the Python mlx-lm reference's output tensor given identical
/// weights + inputs.
///
/// S3 target — the sparse MoE block: a `SwitchGLU` bank of `n_routed_experts`
/// gated by `DeepseekV32MoEGate` (the `noaux_tc` sigmoid router), plus one
/// always-on shared expert. The router has five subtleties versus stock
/// DeepSeek-V3 (final-weight gather from the bias-free sigmoid, unconditional
/// `routed_scaling_factor`, the `n_group == 1` skip, no `+1e-20` epsilon,
/// float32 cast); the fixture uses a *non-zero* `e_score_correction_bias` so
/// the gather-from-orig-scores subtlety is genuinely exercised.
///
/// Fixture `Fixtures/moe_fixture.safetensors` captured offline from mlx-lm
/// 0.31.3 — see `docs/reference/capture_moe.py`. Python never enters macMLX.
final class DeepseekV32MoEParityTests: XCTestCase {

    /// Must match `capture_moe.py` (tiny MoE config: 4 routed + 1 shared
    /// expert, top-2, single group).
    private func fixtureConfig() throws -> DeepseekV32Configuration {
        let json = """
        {
          "model_type": "deepseek_v32",
          "hidden_size": 16,
          "moe_intermediate_size": 8,
          "n_routed_experts": 4,
          "n_shared_experts": 1,
          "num_experts_per_tok": 2,
          "n_group": 1,
          "topk_group": 1,
          "norm_topk_prob": true,
          "routed_scaling_factor": 2.5,
          "topk_method": "noaux_tc",
          "scoring_func": "sigmoid"
        }
        """
        return try JSONDecoder().decode(DeepseekV32Configuration.self, from: Data(json.utf8))
    }

    /// Flattened weight keys the fixture carries. Nested keys unflatten onto
    /// submodules (`gate.e_score_correction_bias` → the gate's bias,
    /// `switch_mlp.gate_proj.weight` → the stacked expert bank, etc.).
    private let weightKeys = [
        "gate.weight", "gate.e_score_correction_bias",
        "switch_mlp.gate_proj.weight", "switch_mlp.up_proj.weight",
        "switch_mlp.down_proj.weight",
        "shared_experts.gate_proj.weight", "shared_experts.up_proj.weight",
        "shared_experts.down_proj.weight",
    ]

    /// THE correctness gate: load the captured fixture, run the Swift MoE with
    /// identical weights + inputs, and assert the output matches the Python
    /// reference within 1e-4. Exercises the router (all five V3.2 fixes), the
    /// routed `SwitchGLU` bank, and the shared expert end-to-end.
    func testMoEMatchesPythonReference() throws {
        try requireMLXRuntimeOrSkip()

        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "moe_fixture", withExtension: "safetensors",
                subdirectory: "Fixtures"),
            "moe_fixture not found in test bundle")
        let arrays = try MLX.loadArrays(url: url)

        let config = try fixtureConfig()
        let moe = DeepseekV32MoE(config)

        var params: [String: MLXArray] = [:]
        for key in weightKeys {
            params[key] = try XCTUnwrap(arrays[key], "missing weight \(key) in fixture")
        }
        try moe.update(
            parameters: ModuleParameters.unflattened(params), verify: [.noUnusedKeys])

        let x = try XCTUnwrap(arrays["x"])
        let expected = try XCTUnwrap(arrays["expected_output"])

        let out = moe(x)
        out.eval()

        XCTAssertEqual(out.shape, expected.shape, "MoE output shape mismatch")
        let close = allClose(out, expected, rtol: 1e-4, atol: 1e-4)
        XCTAssertTrue(
            close.item(Bool.self),
            "MoE output diverges from the Python reference")
    }
}
