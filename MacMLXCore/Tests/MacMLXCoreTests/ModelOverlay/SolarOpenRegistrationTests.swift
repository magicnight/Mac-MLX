import Testing
import Foundation
import MLXLLM
import MLXLMCommon
@testable import MacMLXCore

/// Proves `ModelOverlay.registerAll()` wires `model_type: solar_open` into the
/// stock `LLMTypeRegistry.shared` and resolves it to upstream `GLM4MoEModel`.
/// Mirrors `DeepseekV32RegistrationTests`.
///
/// `.serialized` because it touches the process-global registry; assertions are
/// post-conditions that hold regardless of cross-suite ordering.
@Suite("SolarOpen overlay registration", .serialized)
struct SolarOpenRegistrationTests {

    private static let modelType = "solar_open"

    @Test
    func registerAllRegistersSolarOpen() async {
        await ModelOverlay.registerAll()
        #expect(await LLMTypeRegistry.shared.contains(Self.modelType) == true)
    }

    @Test
    func registerAllIsIdempotent() async {
        await ModelOverlay.registerAll()
        await ModelOverlay.registerAll()
        #expect(await LLMTypeRegistry.shared.contains(Self.modelType) == true)
    }

    @Test(.enabled(if: mlxMetallibIsAvailable, "Requires default.metallib (run under xcodebuild)"))
    func factoryInstantiatesGLM4MoEFromSolarConfig() async throws {
        await ModelOverlay.registerAll()

        // A tiny Solar config that OMITS n_group / topk_group / use_qk_norm /
        // attention_bias exactly like the real one; the overlay's
        // `patchedConfigData` must inject them so the GLM4MoE decode succeeds.
        let configJSON = """
        {
          "model_type": "solar_open",
          "vocab_size": 40,
          "hidden_size": 32,
          "intermediate_size": 48,
          "moe_intermediate_size": 16,
          "num_hidden_layers": 2,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 8,
          "n_routed_experts": 4,
          "n_shared_experts": 1,
          "num_experts_per_tok": 2,
          "routed_scaling_factor": 1.0,
          "norm_topk_prob": true,
          "first_k_dense_replace": 0,
          "max_position_embeddings": 128,
          "rms_norm_eps": 1e-5,
          "rope_theta": 10000,
          "tie_word_embeddings": false,
          "partial_rotary_factor": 1.0
        }
        """
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(configJSON.utf8),
            modelType: Self.modelType
        )
        #expect(model is GLM4MoEModel)
    }
}
