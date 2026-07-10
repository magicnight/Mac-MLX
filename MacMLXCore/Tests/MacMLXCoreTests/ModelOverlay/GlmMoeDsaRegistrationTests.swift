import Testing
import Foundation
import MLXLLM
import MLXLMCommon
@testable import MacMLXCore

/// Proves `ModelOverlay.registerAll()` wires `model_type: glm_moe_dsa` into the
/// stock `LLMTypeRegistry.shared` and resolves it to macMLX's `DeepseekV32Model`
/// (GLM-DSA is `class Model(DSV32Model)`). Also pins the GLM-5.2 IndexShare
/// rejection through the factory path. Mirrors `DeepseekV32RegistrationTests`.
@Suite("GlmMoeDsa overlay registration", .serialized)
struct GlmMoeDsaRegistrationTests {

    private static let modelType = "glm_moe_dsa"

    @Test
    func registerAllRegistersGlmMoeDsa() async {
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
    func factoryInstantiatesDeepseekV32FromGlmDsaConfig() async throws {
        await ModelOverlay.registerAll()

        let configJSON = """
        {
          "model_type": "glm_moe_dsa",
          "vocab_size": 32,
          "hidden_size": 16,
          "num_hidden_layers": 1,
          "num_attention_heads": 2,
          "num_key_value_heads": 2,
          "rope_parameters": { "rope_theta": 10000, "rope_type": "default" }
        }
        """
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(configJSON.utf8),
            modelType: Self.modelType
        )
        #expect(model is DeepseekV32Model)
    }

    @Test
    func factoryRejectsGlm52IndexShare() async {
        await ModelOverlay.registerAll()

        // A glm_moe_dsa config carrying GLM-5.2's `indexer_types` schedule must
        // be refused at decode time (before any weights/Metal), so this runs
        // ungated.
        let configJSON = """
        {
          "model_type": "glm_moe_dsa",
          "indexer_types": ["full", "shared"],
          "rope_parameters": { "rope_theta": 10000 }
        }
        """
        do {
            _ = try await LLMTypeRegistry.shared.createModel(
                configuration: Data(configJSON.utf8),
                modelType: Self.modelType
            )
            Issue.record("expected glmDsaIndexShareUnsupported to be thrown")
        } catch let error as ModelOverlayError {
            #expect(error == .glmDsaIndexShareUnsupported)
        } catch {
            Issue.record("unexpected error: \(error)")
        }
    }
}
