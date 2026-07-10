import Testing
import Foundation
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
@testable import MacMLXCore

/// Proves `ModelOverlay.registerAll()` wires the pure-Swift Mellum 2
/// architecture into the stock `LLMTypeRegistry.shared`, so the factory
/// resolves `config.json`'s `model_type: mellum` to `Mellum2Model` with no
/// fork. Mirrors `DeepseekV32RegistrationTests`.
///
/// `.serialized` because it touches the process-global registry; the
/// assertions are post-conditions (`contains == true` after `registerAll`)
/// that hold regardless of cross-suite ordering.
@Suite("Mellum2 overlay registration", .serialized)
struct Mellum2RegistrationTests {

    private static let modelType = "mellum"

    @Test
    func registerAllRegistersMellum2() async {
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
    func factoryInstantiatesMellum2FromConfig() async throws {
        await ModelOverlay.registerAll()

        // A minimal `config.json` slice; `createModel` is exactly what
        // `LLMModelFactory` calls to turn (config data, model_type) into a
        // `LanguageModel`, so driving it proves the resolution path reaches our
        // creator and decodes via `JSONDecoder.json5()`.
        let configJSON = """
        {
          "model_type": "mellum",
          "vocab_size": 40,
          "hidden_size": 32,
          "num_hidden_layers": 2,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 16,
          "num_experts": 6,
          "num_experts_per_tok": 2,
          "moe_intermediate_size": 24,
          "sliding_window": 3,
          "layer_types": ["sliding_attention", "full_attention"],
          "rope_parameters": {
            "full_attention": {"rope_type": "yarn", "rope_theta": 500000.0, "factor": 16.0, "original_max_position_embeddings": 8192, "beta_fast": 32.0, "beta_slow": 1.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": 500000.0}
          }
        }
        """
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(configJSON.utf8),
            modelType: Self.modelType
        )
        #expect(model is Mellum2Model)
    }
}
