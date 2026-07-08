import XCTest
import MLX
import MLXNN
import MLXLMCommon
@testable import MacMLXCore

/// Numerical parity: our Swift `DeepseekV32Indexer` must select the same
/// top-k key indices as the Python mlx-lm reference given identical
/// weights + inputs.
///
/// Fixture `Fixtures/indexer_parity_fixture.safetensors` was captured
/// from mlx-lm 0.31.3's `deepseek_v32.Indexer` (see
/// `scratchpad/capture_indexer.py`; Python never enters macMLX — it's an
/// offline capture). It holds the deterministic weights, the inputs, and
/// the expected sorted top-k indices.
///
/// argPartition's within-partition order is unstable, so both sides sort
/// the top-k along the key axis before comparing — the *set* of selected
/// keys is the invariant.
final class DeepseekV32IndexerParityTests: XCTestCase {

    /// Must match `capture_indexer.py`.
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
          "rope_theta": 10000
        }
        """
        return try JSONDecoder().decode(DeepseekV32Configuration.self, from: Data(json.utf8))
    }

    func testMatchesPythonReferenceTopKSelection() throws {
        try requireTrustworthyMetalOrSkip()

        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "indexer_parity_fixture",
                withExtension: "safetensors",
                subdirectory: "Fixtures"),
            "parity fixture not found in test bundle")
        let arrays = try MLX.loadArrays(url: url)

        let config = try fixtureConfig()
        let indexer = DeepseekV32Indexer(config)

        // Load the captured weights into the module. Only the parameter
        // keys — x / qr / expected_* are test data, not parameters.
        let weightKeys = [
            "wq_b.weight", "wk.weight",
            "k_norm.weight", "k_norm.bias",
            "weights_proj.weight",
        ]
        var params: [String: MLXArray] = [:]
        for k in weightKeys {
            params[k] = try XCTUnwrap(arrays[k], "missing weight \(k) in fixture")
        }
        try indexer.update(parameters: ModuleParameters.unflattened(params), verify: [.noUnusedKeys])

        let x = try XCTUnwrap(arrays["x"])
        let qr = try XCTUnwrap(arrays["qr"])
        let expected = try XCTUnwrap(arrays["expected_topk_sorted"])  // [1,1,S,topk] int32

        let out = try XCTUnwrap(
            indexer(x, qr, nil),
            "indexer returned nil but the fixture has s > index_topk")

        // Canonicalize order, then compare index-for-index.
        let gotSorted = sorted(out.asType(.int32), axis: -1)
        gotSorted.eval()

        XCTAssertEqual(gotSorted.shape, expected.shape, "top-k shape mismatch")

        let gotFlat = gotSorted.asArray(Int32.self)
        let expFlat = expected.asArray(Int32.self)
        XCTAssertEqual(gotFlat, expFlat, "Swift indexer selected different top-k keys than the Python reference")
    }
}
