import Testing
import Foundation
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
@testable import MacMLXCore

/// Table-driven registration parity for every HAPPY-PATH overlay architecture:
/// each `model_type` that `ModelOverlay.registerAll()` wires into the stock
/// `LLMTypeRegistry.shared` and that resolves — through the exact factory path —
/// to a concrete `LLMModel`.
///
/// This consolidates what were four near-identical per-model files
/// (DeepseekV32 / GlmMoeDsa / Mellum2 / SolarOpen registration) into one
/// parameterized suite: a new model becomes a single row in `cases`, not a new
/// file. Each row is `(model_type, minimal config.json, type marker)`; the two
/// parameterized tests then prove, for every row:
///   1. `registerAll` registers the type (and is idempotent — double-called),
///   2. the factory instantiates the expected concrete model from a config.
///
/// The KimiK25 registration (a loud-block contract — the creator throws rather
/// than instantiating) is semantically different and stays in its own file. The
/// generic register-from-outside mechanism and the unknown-type-still-throws
/// negative case live in `ModelOverlaySpikeTests`.
///
/// `.serialized` because it touches the process-global registry; the assertions
/// are post-conditions (`contains == true` after `registerAll`) that hold
/// regardless of cross-suite ordering.
@Suite("Overlay registration", .serialized)
struct OverlayRegistrationTests {

    /// One happy-path overlay registration. `matches` is a type marker rather
    /// than a stored metatype so DeepseekV32 (shared by `deepseek_v32` and
    /// `glm_moe_dsa`), Mellum2, GLM4MoE (Solar), and SeedOss all express their
    /// expectation uniformly.
    struct Case: Sendable, CustomTestStringConvertible {
        let modelType: String
        let configJSON: String
        let matches: @Sendable (Any) -> Bool
        var testDescription: String { modelType }
    }

    static let cases: [Case] = [
        // DeepSeek V3.2 → macMLX `DeepseekV32Model`.
        Case(
            modelType: "deepseek_v32",
            configJSON: """
            {
              "model_type": "deepseek_v32",
              "vocab_size": 32,
              "hidden_size": 16,
              "num_hidden_layers": 1,
              "num_attention_heads": 2,
              "num_key_value_heads": 2
            }
            """,
            matches: { $0 is DeepseekV32Model }),

        // GLM-DSA (GLM-5.1) → macMLX `DeepseekV32Model` (`class Model(DSV32Model)`).
        Case(
            modelType: "glm_moe_dsa",
            configJSON: """
            {
              "model_type": "glm_moe_dsa",
              "vocab_size": 32,
              "hidden_size": 16,
              "num_hidden_layers": 1,
              "num_attention_heads": 2,
              "num_key_value_heads": 2,
              "rope_parameters": { "rope_theta": 10000, "rope_type": "default" }
            }
            """,
            matches: { $0 is DeepseekV32Model }),

        // Mellum 2 → macMLX `Mellum2Model`.
        Case(
            modelType: "mellum",
            configJSON: """
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
            """,
            matches: { $0 is Mellum2Model }),

        // Solar-Open-100B → upstream `GLM4MoEModel` (config re-skin).
        Case(
            modelType: "solar_open",
            configJSON: """
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
            """,
            matches: { $0 is GLM4MoEModel }),

        // Seed-OSS → macMLX `SeedOssModel`.
        Case(
            modelType: "seed_oss",
            configJSON: """
            {
              "model_type": "seed_oss",
              "vocab_size": 40,
              "hidden_size": 32,
              "num_hidden_layers": 2,
              "intermediate_size": 48,
              "num_attention_heads": 4,
              "num_key_value_heads": 2,
              "head_dim": 16,
              "attention_bias": true,
              "attention_out_bias": false,
              "rope_theta": 10000000.0,
              "rope_scaling": {"rope_type": "default"}
            }
            """,
            matches: { $0 is SeedOssModel }),

        // Hunyuan V1 Dense → macMLX `HunyuanV1DenseModel`.
        Case(
            modelType: "hunyuan_v1_dense",
            configJSON: """
            {
              "model_type": "hunyuan_v1_dense",
              "vocab_size": 40,
              "hidden_size": 32,
              "num_hidden_layers": 2,
              "intermediate_size": 48,
              "num_attention_heads": 4,
              "num_key_value_heads": 2,
              "head_dim": 16,
              "attention_bias": false,
              "use_qk_norm": true,
              "tie_word_embeddings": true,
              "rope_theta": 10000.0,
              "rope_scaling": {"type": "dynamic", "alpha": 1000.0, "factor": 1.0}
            }
            """,
            matches: { $0 is HunyuanV1DenseModel }),
    ]

    /// `registerAll` registers each `model_type` — the exact query the factory
    /// uses in a pre-download support check — and is idempotent (the engine may
    /// invoke it more than once). Ungated: this never touches the creator/MLX.
    @Test(arguments: cases)
    func registersAndIsIdempotent(_ modelCase: Case) async {
        await ModelOverlay.registerAll()
        await ModelOverlay.registerAll()  // double-call must not throw or corrupt state
        #expect(await LLMTypeRegistry.shared.contains(modelCase.modelType) == true)
    }

    /// The factory resolves each `model_type` to its expected concrete model.
    /// `createModel` is exactly what `LLMModelFactory` calls to turn (config
    /// data, model_type) into a `LanguageModel`, so driving it proves the
    /// resolution path reaches our creator and decodes via `JSONDecoder.json5()`.
    ///
    /// Gated: instantiation builds real MLXArrays (Linear / Embedding inits),
    /// fatal under bare `swift test` without the metallib. The registration test
    /// above never invokes the creator, so it stays ungated.
    @Test(.enabled(if: mlxMetallibIsAvailable, "Requires default.metallib (run under xcodebuild)"), arguments: cases)
    func factoryInstantiatesFromConfig(_ modelCase: Case) async throws {
        await ModelOverlay.registerAll()
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(modelCase.configJSON.utf8),
            modelType: modelCase.modelType
        )
        #expect(modelCase.matches(model))
    }

    /// Model-specific negative contract preserved from the old GlmMoeDsa suite: a
    /// `glm_moe_dsa` config carrying GLM-5.2's `indexer_types` schedule must be
    /// refused at decode time (before any weights/Metal), so this runs ungated.
    @Test
    func glmMoeDsaRejectsGlm52IndexShare() async {
        await ModelOverlay.registerAll()

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
                modelType: "glm_moe_dsa"
            )
            Issue.record("expected glmDsaIndexShareUnsupported to be thrown")
        } catch let error as ModelOverlayError {
            #expect(error == .glmDsaIndexShareUnsupported)
        } catch {
            Issue.record("unexpected error: \(error)")
        }
    }
}
