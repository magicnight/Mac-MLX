import Testing
import Foundation
@testable import MacMLXCore

@Suite("DeepseekV32Configuration")
struct DeepseekV32ConfigurationTests {

    @Test
    func decodesRealConfigFields() throws {
        // A representative slice of a real deepseek_v32 config.json.
        let json = """
        {
          "model_type": "deepseek_v32",
          "vocab_size": 129280,
          "hidden_size": 4096,
          "index_head_dim": 128,
          "index_n_heads": 64,
          "index_topk": 2048,
          "moe_intermediate_size": 2048,
          "num_hidden_layers": 43,
          "num_attention_heads": 64,
          "num_key_value_heads": 1,
          "n_routed_experts": 256,
          "n_shared_experts": 1,
          "num_experts_per_tok": 6,
          "kv_lora_rank": 512,
          "q_lora_rank": 1536,
          "qk_rope_head_dim": 64,
          "v_head_dim": 128,
          "qk_nope_head_dim": 128,
          "topk_method": "noaux_tc",
          "scoring_func": "sigmoid",
          "routed_scaling_factor": 1.5,
          "first_k_dense_replace": 3,
          "rope_theta": 10000,
          "rope_scaling": { "type": "yarn", "factor": 16 }
        }
        """
        let cfg = try JSONDecoder().decode(DeepseekV32Configuration.self, from: Data(json.utf8))
        #expect(cfg.vocabSize == 129280)
        #expect(cfg.indexHeadDim == 128)
        #expect(cfg.indexNHeads == 64)
        #expect(cfg.indexTopK == 2048)
        #expect(cfg.numHiddenLayers == 43)
        #expect(cfg.nRoutedExperts == 256)
        #expect(cfg.nSharedExperts == 1)
        #expect(cfg.numExpertsPerTok == 6)
        #expect(cfg.routedScalingFactor == 1.5)
        #expect(cfg.firstKDenseReplace == 3)
        #expect(cfg.ropeScaling?["factor"]?.asFloat() == 16)
    }

    @Test
    func fillsDefaultsForMissingKeys() throws {
        // Minimal config — everything else should fall back to the
        // Python ModelArgs defaults.
        let json = #"{ "model_type": "deepseek_v32" }"#
        let cfg = try JSONDecoder().decode(DeepseekV32Configuration.self, from: Data(json.utf8))
        #expect(cfg.vocabSize == 102400)
        #expect(cfg.hiddenSize == 4096)
        #expect(cfg.indexTopK == 2048)
        #expect(cfg.numExpertsPerTok == 1)
        #expect(cfg.nRoutedExperts == nil)
        #expect(cfg.nSharedExperts == nil)
        #expect(cfg.scoringFunc == "sigmoid")
        #expect(cfg.topkMethod == "noaux_tc")
    }
}
