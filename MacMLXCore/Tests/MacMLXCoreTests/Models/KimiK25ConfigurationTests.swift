import Testing
import Foundation
import MLXLLM
@testable import MacMLXCore

/// Kimi K2.5 (`kimi_k25`) config-nesting tests.
///
/// The model itself is BLOCKED on an upstream access level (see
/// `Models/KimiK25.swift`), but the config half is complete and ready: the
/// DeepSeek V3 text tower is nested under `text_config`. These tests prove the
/// nesting extracts a valid `DeepseekV3Configuration`. Because that type's
/// fields are `internal` to MLXLLM, the extracted config is re-encoded (it is
/// `Codable`) and probed as JSON.
@Suite("KimiK25 config nesting")
struct KimiK25ConfigurationTests {

    /// A faithful slice of the real `mlx-community/Kimi-K2.5` config.json: the
    /// full DeepSeek V3 args live under `text_config`.
    private static let realKimiConfig = """
    {
      "model_type": "kimi_k25",
      "tie_word_embeddings": false,
      "use_unified_vision_chunk": true,
      "text_config": {
        "model_type": "kimi_k2",
        "vocab_size": 163840,
        "hidden_size": 7168,
        "intermediate_size": 18432,
        "moe_intermediate_size": 2048,
        "num_hidden_layers": 61,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "n_routed_experts": 384,
        "n_shared_experts": 1,
        "num_experts_per_tok": 8,
        "n_group": 1,
        "topk_group": 1,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "qk_nope_head_dim": 128,
        "norm_topk_prob": true,
        "moe_layer_freq": 1,
        "first_k_dense_replace": 1,
        "max_position_embeddings": 262144,
        "rms_norm_eps": 1e-05,
        "rope_theta": 50000.0,
        "routed_scaling_factor": 2.827,
        "attention_bias": false,
        "rope_scaling": { "type": "yarn", "factor": 64.0, "original_max_position_embeddings": 4096 }
      }
    }
    """

    private struct DeepseekV3Probe: Decodable {
        let hiddenSize: Int
        let numHiddenLayers: Int
        let vocabSize: Int
        let kvLoraRank: Int
        let qLoraRank: Int
        let ropeTheta: Double
        let nRoutedExperts: Int?

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case numHiddenLayers = "num_hidden_layers"
            case vocabSize = "vocab_size"
            case kvLoraRank = "kv_lora_rank"
            case qLoraRank = "q_lora_rank"
            case ropeTheta = "rope_theta"
            case nRoutedExperts = "n_routed_experts"
        }
    }

    @Test
    func extractsDeepseekV3TextConfig() throws {
        let cfg = try JSONDecoder().decode(
            KimiK25Configuration.self, from: Data(Self.realKimiConfig.utf8))

        // Round-trip the extracted text config to observe its (internal) fields.
        let reencoded = try JSONEncoder().encode(cfg.textConfig)
        let probe = try JSONDecoder().decode(DeepseekV3Probe.self, from: reencoded)

        #expect(probe.hiddenSize == 7168)
        #expect(probe.numHiddenLayers == 61)
        #expect(probe.vocabSize == 163840)
        #expect(probe.kvLoraRank == 512)
        #expect(probe.qLoraRank == 1536)
        #expect(probe.ropeTheta == 50000)
        #expect(probe.nRoutedExperts == 384)
    }

    @Test
    func requiresTextConfig() {
        // Without the `text_config` wrapper there is nothing to decode.
        let json = #"{ "model_type": "kimi_k25" }"#
        #expect(throws: (any Error).self) {
            _ = try JSONDecoder().decode(KimiK25Configuration.self, from: Data(json.utf8))
        }
    }
}
