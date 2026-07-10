import Testing
import Foundation
import MLXLLM
@testable import MacMLXCore

/// Solar-Open-100B (`solar_open`) difference-surface tests.
///
/// Solar reuses upstream `GLM4MoEModel` unchanged, so the ONLY thing that can go
/// wrong is config parsing: `solar_open.ModelArgs` defaults a handful of fields
/// that `GLM4MoEConfiguration` decodes as required, and Solar's real config.json
/// omits exactly those. `SolarOpen.patchedConfigData` injects the Python
/// defaults. These tests use the real `upstage/Solar-Open-100B` config schema.
@Suite("SolarOpen config injection")
struct SolarOpenConfigurationTests {

    /// A faithful slice of the real `upstage/Solar-Open-100B` config.json.
    /// Note what is ABSENT: `n_group`, `topk_group`, `use_qk_norm`,
    /// `attention_bias` — the fields `solar_open.ModelArgs` defaults and
    /// `GLM4MoEConfiguration` requires.
    private static let realSolarConfig = """
    {
      "model_type": "solar_open",
      "partial_rotary_factor": 1.0,
      "hidden_size": 4096,
      "num_hidden_layers": 48,
      "num_attention_heads": 64,
      "head_dim": 128,
      "num_key_value_heads": 8,
      "vocab_size": 196608,
      "intermediate_size": 10240,
      "moe_intermediate_size": 1280,
      "rms_norm_eps": 1e-05,
      "rope_theta": 1000000,
      "max_position_embeddings": 131072,
      "n_routed_experts": 128,
      "n_shared_experts": 1,
      "norm_topk_prob": true,
      "routed_scaling_factor": 1.0,
      "num_experts_per_tok": 8,
      "tie_word_embeddings": false,
      "first_k_dense_replace": 0,
      "rope_scaling": { "type": "yarn", "factor": 2.0, "original_max_position_embeddings": 65536 }
    }
    """

    /// All-optional probe so it decodes from any subset of a GLM4MoE config
    /// (`GLM4MoEConfiguration`'s own fields are `internal` to MLXLLM and can't be
    /// read from the test module).
    private struct InjectionProbe: Decodable {
        let modelType: String?
        let nGroup: Int?
        let topkGroup: Int?
        let useQkNorm: Bool?
        let attentionBias: Bool?
        let scoringFunc: String?
        let topkMethod: String?
        let hiddenSize: Int?
        let ropeTheta: Double?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case nGroup = "n_group"
            case topkGroup = "topk_group"
            case useQkNorm = "use_qk_norm"
            case attentionBias = "attention_bias"
            case scoringFunc = "scoring_func"
            case topkMethod = "topk_method"
            case hiddenSize = "hidden_size"
            case ropeTheta = "rope_theta"
        }
    }

    @Test
    func injectsSolarOpenDefaultsForMissingKeys() throws {
        let patched = try SolarOpen.patchedConfigData(Data(Self.realSolarConfig.utf8))
        let probe = try JSONDecoder().decode(InjectionProbe.self, from: patched)

        // Injected (absent from Solar's config, required by GLM4MoEConfiguration).
        #expect(probe.nGroup == 1)
        #expect(probe.topkGroup == 1)
        #expect(probe.useQkNorm == false)
        #expect(probe.attentionBias == false)
        #expect(probe.scoringFunc == "sigmoid")
        #expect(probe.topkMethod == "noaux_tc")

        // Real values are preserved untouched.
        #expect(probe.modelType == "solar_open")
        #expect(probe.hiddenSize == 4096)
        #expect(probe.ropeTheta == 1_000_000)
    }

    @Test
    func patchedConfigDecodesAsGLM4MoEButRawDoesNot() throws {
        // The raw Solar config is missing fields GLM4MoEConfiguration requires.
        #expect(throws: (any Error).self) {
            _ = try JSONDecoder().decode(
                GLM4MoEConfiguration.self, from: Data(Self.realSolarConfig.utf8))
        }

        // After injecting solar_open's defaults it decodes cleanly — this is the
        // whole point of the mapping. (Throws here fail the test.)
        let patched = try SolarOpen.patchedConfigData(Data(Self.realSolarConfig.utf8))
        _ = try JSONDecoder().decode(GLM4MoEConfiguration.self, from: patched)
    }

    @Test
    func explicitValueIsNeverOverwritten() throws {
        // A config that DOES set n_group / topk_group must keep them; only the
        // still-absent defaults (attention_bias) get injected.
        let explicit = #"{ "model_type": "solar_open", "n_group": 4, "topk_group": 3 }"#
        let patched = try SolarOpen.patchedConfigData(Data(explicit.utf8))
        let probe = try JSONDecoder().decode(InjectionProbe.self, from: patched)

        #expect(probe.nGroup == 4)
        #expect(probe.topkGroup == 3)
        #expect(probe.attentionBias == false)
    }

    @Test
    func rejectsNonObjectConfig() {
        #expect(throws: ModelOverlayError.solarOpenMalformedConfig) {
            _ = try SolarOpen.patchedConfigData(Data("[1, 2, 3]".utf8))
        }
    }
}
