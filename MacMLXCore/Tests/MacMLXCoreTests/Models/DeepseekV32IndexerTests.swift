import XCTest
import MLX
import MLXLMCommon
@testable import MacMLXCore

/// Structural tests for the DSA lightning indexer. MLX-backed ⇒
/// Metal-gated (skip under `swift test`, run under `xcodebuild`).
///
/// Numerical parity against the Python reference (fixed weights +
/// input → expected top-k indices) is the S1.5 follow-up: it needs
/// reference values captured from an offline Python run. These tests
/// pin the shape contract + the short-circuit behavior, which are
/// enough to catch structural regressions.
final class DeepseekV32IndexerTests: XCTestCase {

    /// A tiny config where `index_topk` is small enough to exercise both
    /// the short-circuit (s ≤ topk) and the sparse (s > topk) paths.
    private func tinyConfig(indexTopK: Int) throws -> DeepseekV32Configuration {
        let json = """
        {
          "model_type": "deepseek_v32",
          "hidden_size": 32,
          "q_lora_rank": 16,
          "index_head_dim": 8,
          "index_n_heads": 2,
          "index_topk": \(indexTopK),
          "qk_rope_head_dim": 8,
          "max_position_embeddings": 64,
          "rope_theta": 10000
        }
        """
        return try JSONDecoder().decode(DeepseekV32Configuration.self, from: Data(json.utf8))
    }

    func testShortCircuitReturnsNilWhenContextFitsInTopK() throws {
        try requireMLXRuntimeOrSkip()
        let config = try tinyConfig(indexTopK: 8)
        let indexer = DeepseekV32Indexer(config)

        // s = 4 ≤ index_topk = 8 ⇒ no sparsification, expect nil.
        let b = 1, s = 4
        let x = MLXRandom.normal([b, s, config.hiddenSize])
        let qr = MLXRandom.normal([b, s, config.qLoraRank])
        let result = indexer(x, qr, nil)
        XCTAssertNil(result, "indexer must return nil when s <= index_topk")
    }

    func testReturnsTopKIndicesWhenContextExceedsTopK() throws {
        try requireMLXRuntimeOrSkip()
        let config = try tinyConfig(indexTopK: 4)
        let indexer = DeepseekV32Indexer(config)

        // s = 10 > index_topk = 4 ⇒ expect indices of shape [b, 1, s, 4].
        let b = 1, s = 10
        let x = MLXRandom.normal([b, s, config.hiddenSize])
        let qr = MLXRandom.normal([b, s, config.qLoraRank])
        let result = indexer(x, qr, nil)

        let indices = try XCTUnwrap(result, "indexer must return indices when s > index_topk")
        indices.eval()
        XCTAssertEqual(indices.shape, [b, 1, s, config.indexTopK])
    }
}
