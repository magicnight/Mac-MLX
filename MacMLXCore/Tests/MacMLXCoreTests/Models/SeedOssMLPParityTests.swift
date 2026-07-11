import XCTest
import MLX
import MLXNN
import MLXLMCommon
@testable import MacMLXCore

/// Numerical parity: our Swift `SeedOssMLP` (SwiGLU: `down(silu(gate(x))*up(x))`)
/// must reproduce the Python mlx-lm reference with and without `mlp_bias`, so the
/// bias wiring on all three projections is pinned.
///
/// Fixtures captured offline from mlx-lm 0.31.3 (`seed_oss.py`).
final class SeedOssMLPParityTests: XCTestCase {

    /// Must match `capture_seed_oss.py`: hidden 32, intermediate 48.
    private func fixtureConfig(mlpBias: Bool) throws -> SeedOssConfiguration {
        let json = """
        {
          "model_type": "seed_oss",
          "hidden_size": 32,
          "intermediate_size": 48,
          "mlp_bias": \(mlpBias)
        }
        """
        return try JSONDecoder().decode(SeedOssConfiguration.self, from: Data(json.utf8))
    }

    private func runParity(fixture: String, mlpBias: Bool) throws {
        try requireTrustworthyMetalOrSkip()

        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: fixture, withExtension: "safetensors",
                subdirectory: "Fixtures"),
            "\(fixture) not found in test bundle")
        let arrays = try MLX.loadArrays(url: url)

        let config = try fixtureConfig(mlpBias: mlpBias)
        let mlp = SeedOssMLP(config)

        var params: [String: MLXArray] = [:]
        for (key, value) in arrays where key != "x" && key != "expected_output" {
            params[key] = value
        }
        try mlp.update(
            parameters: ModuleParameters.unflattened(params), verify: [.noUnusedKeys])

        let x = try XCTUnwrap(arrays["x"])
        let expected = try XCTUnwrap(arrays["expected_output"])

        let out = mlp(x)
        out.eval()

        XCTAssertEqual(out.shape, expected.shape, "\(fixture): output shape mismatch")
        let close = allClose(out, expected, rtol: 1e-4, atol: 1e-4)
        XCTAssertTrue(
            close.item(Bool.self),
            "\(fixture): MLP output diverges from the Python reference")
    }

    func testMLPWithBiasMatchesPythonReference() throws {
        try runParity(fixture: "seed_oss_mlp_bias_fixture", mlpBias: true)
    }

    func testMLPWithoutBiasMatchesPythonReference() throws {
        try runParity(fixture: "seed_oss_mlp_nobias_fixture", mlpBias: false)
    }
}
