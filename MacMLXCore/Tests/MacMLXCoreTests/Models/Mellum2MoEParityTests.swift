import XCTest
import MLX
import MLXNN
import MLXLMCommon
@testable import MacMLXCore

/// Numerical parity: our Swift `Mellum2SparseMoeBlock` must reproduce the
/// Python mlx-lm reference's output given identical weights + inputs.
///
/// Target — the sparse MoE router + `SwitchGLU` expert bank: softmax over all
/// experts, top-k selection, top-k renormalization (`norm_topk_prob = true`),
/// and the weighted expert sum. This is the trickiest single component, so it
/// gets its own isolated fixture.
///
/// Fixture `Fixtures/mellum2_moe_fixture.safetensors` captured offline from
/// mlx-lm @ e476a22 (mellum.py) — see `docs/reference/capture_mellum2.py`.
/// Python never enters macMLX.
final class Mellum2MoEParityTests: XCTestCase {

    /// Must match `capture_mellum2.py` (tiny MoE: 6 experts, top-2,
    /// hidden 32, moe-intermediate 24).
    private func fixtureConfig() throws -> Mellum2Configuration {
        let json = """
        {
          "model_type": "mellum",
          "hidden_size": 32,
          "num_experts": 6,
          "num_experts_per_tok": 2,
          "moe_intermediate_size": 24,
          "norm_topk_prob": true
        }
        """
        return try JSONDecoder().decode(Mellum2Configuration.self, from: Data(json.utf8))
    }

    private let weightKeys = [
        "gate.weight",
        "switch_mlp.gate_proj.weight",
        "switch_mlp.up_proj.weight",
        "switch_mlp.down_proj.weight",
    ]

    func testMoEMatchesPythonReference() throws {
        try requireTrustworthyMetalOrSkip()

        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "mellum2_moe_fixture", withExtension: "safetensors",
                subdirectory: "Fixtures"),
            "mellum2_moe_fixture not found in test bundle")
        let arrays = try MLX.loadArrays(url: url)

        let config = try fixtureConfig()
        let moe = Mellum2SparseMoeBlock(config)

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
