import Testing
import Foundation
@testable import MacMLXCore

/// GLM-DSA (`glm_moe_dsa`) difference-surface tests.
///
/// GLM-DSA reuses `DeepseekV32Model` unchanged (`class Model(DSV32Model)`), so
/// the difference is entirely in the config defaults GLM-DSA's `ModelArgs`
/// applies on top of DeepSeek V3.2. These tests use the real `glm_moe_dsa`
/// config schema (from the `glm-moe-dsa` tiny-random checkpoints) and assert on
/// the produced `DeepseekV32Configuration` (which is macMLX-owned, so its fields
/// are readable here).
@Suite("GlmMoeDsa config adaptation")
struct GlmMoeDsaConfigurationTests {

    /// A faithful slice of the real `glm_moe_dsa` config.json: RoPE lives in
    /// `rope_parameters` (no top-level `rope_theta` / `rope_scaling`), and
    /// `indexer_rope_interleave` is present and true.
    private static let realGlmDsaConfig = """
    {
      "model_type": "glm_moe_dsa",
      "hidden_size": 8,
      "index_head_dim": 128,
      "index_n_heads": 4,
      "index_topk": 2048,
      "intermediate_size": 32,
      "moe_intermediate_size": 32,
      "num_hidden_layers": 2,
      "num_attention_heads": 4,
      "num_key_value_heads": 4,
      "n_routed_experts": 256,
      "n_shared_experts": 1,
      "num_experts_per_tok": 8,
      "kv_lora_rank": 512,
      "q_lora_rank": 32,
      "qk_nope_head_dim": 192,
      "qk_rope_head_dim": 64,
      "v_head_dim": 256,
      "routed_scaling_factor": 2.5,
      "first_k_dense_replace": 1,
      "rms_norm_eps": 1e-05,
      "indexer_rope_interleave": true,
      "rope_parameters": { "rope_theta": 1000000, "rope_type": "default" }
    }
    """

    @Test
    func decodesRealConfigAndDerivesRopeFromRopeParameters() throws {
        let cfg = try JSONDecoder().decode(
            GlmMoeDsaConfiguration.self, from: Data(Self.realGlmDsaConfig.utf8))
        let base = cfg.base

        // Difference (1): GLM-DSA runs the indexer RoPE interleaved.
        #expect(base.indexerRopeInterleave == true)

        // Difference (2): rope_theta / rope_scaling come from rope_parameters,
        // NOT the top-level keys (which are absent). Base V3.2 would have left
        // ropeTheta at its 10000 default.
        #expect(base.ropeTheta == 1_000_000)
        #expect(base.ropeScaling?["rope_theta"]?.asFloat() == 1_000_000)
        #expect(base.ropeScaling != nil)

        // Base fields pass straight through the V3.2 decoder.
        #expect(base.indexHeadDim == 128)
        #expect(base.indexNHeads == 4)
        #expect(base.indexTopK == 2048)
        #expect(base.nRoutedExperts == 256)
        #expect(base.nSharedExperts == 1)
        #expect(base.qkNopeHeadDim == 192)
        #expect(base.vHeadDim == 256)
        #expect(base.routedScalingFactor == 2.5)
    }

    @Test
    func defaultsInterleaveTrueWhenAbsent() throws {
        // GLM-DSA's ModelArgs defaults indexer_rope_interleave to true, unlike
        // base DeepSeek V3.2 (false). A config omitting the key must get true.
        let json = """
        {
          "model_type": "glm_moe_dsa",
          "rope_parameters": { "rope_theta": 5000 }
        }
        """
        let cfg = try JSONDecoder().decode(GlmMoeDsaConfiguration.self, from: Data(json.utf8))
        #expect(cfg.base.indexerRopeInterleave == true)
        #expect(cfg.base.ropeTheta == 5000)
    }

    @Test
    func honorsExplicitInterleaveFalse() throws {
        let json = """
        {
          "model_type": "glm_moe_dsa",
          "indexer_rope_interleave": false,
          "rope_parameters": { "rope_theta": 5000 }
        }
        """
        let cfg = try JSONDecoder().decode(GlmMoeDsaConfiguration.self, from: Data(json.utf8))
        #expect(cfg.base.indexerRopeInterleave == false)
    }

    @Test
    func rejectsGlm52IndexShareConfigs() {
        // GLM-5.2 IndexShare markers must be refused loudly (upstream mlx-lm
        // #1410 pending), not loaded onto the single-indexer path.
        let markers = ["indexer_types", "index_topk_freq", "index_skip_topk_offset"]
        for marker in markers {
            let json = """
            {
              "model_type": "glm_moe_dsa",
              "\(marker)": [1, 2],
              "rope_parameters": { "rope_theta": 5000 }
            }
            """
            #expect(throws: ModelOverlayError.glmDsaIndexShareUnsupported) {
                _ = try JSONDecoder().decode(
                    GlmMoeDsaConfiguration.self, from: Data(json.utf8))
            }
        }
    }
}
