import Testing
import Foundation
import MLXLLM
import MLXLMCommon
@testable import MacMLXCore

/// Kimi K2.5 (`model_type: kimi_k25`) is registered but BLOCKED: upstream
/// `DeepseekV3Model.init` is `internal`, so the overlay cannot construct the
/// text core to wrap (see `Models/KimiK25.swift`). The creator decodes the
/// config (proving that half is ready) and then fails loudly. These tests pin
/// that contract — a precise, actionable error rather than a silent or generic
/// failure — so the day upstream unblocks it, the expectation flips here.
@Suite("KimiK25 overlay registration", .serialized)
struct KimiK25RegistrationTests {

    private static let modelType = "kimi_k25"

    /// A valid Kimi config: `kimi_k25` wrapping a complete DeepSeek V3
    /// `text_config`. The creator must get past the decode and then throw.
    private static let kimiConfigJSON = """
    {
      "model_type": "kimi_k25",
      "tie_word_embeddings": false,
      "text_config": {
        "model_type": "kimi_k2",
        "vocab_size": 40,
        "hidden_size": 32,
        "intermediate_size": 48,
        "moe_intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "kv_lora_rank": 16,
        "q_lora_rank": 16,
        "qk_rope_head_dim": 8,
        "v_head_dim": 8,
        "qk_nope_head_dim": 8,
        "norm_topk_prob": true,
        "moe_layer_freq": 1,
        "first_k_dense_replace": 1,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-5,
        "rope_theta": 50000.0,
        "routed_scaling_factor": 2.827,
        "attention_bias": false
      }
    }
    """

    @Test
    func registerAllRegistersKimiK25() async {
        await ModelOverlay.registerAll()
        #expect(await LLMTypeRegistry.shared.contains(Self.modelType) == true)
    }

    @Test
    func registerAllIsIdempotent() async {
        await ModelOverlay.registerAll()
        await ModelOverlay.registerAll()
        #expect(await LLMTypeRegistry.shared.contains(Self.modelType) == true)
    }

    @Test
    func createModelFailsLoudlyWhileBlocked() async {
        await ModelOverlay.registerAll()

        // Ungated: the creator decodes the (valid) config then throws before it
        // would ever touch MLX. If this ever stops throwing, the upstream
        // `DeepseekV3Model.init` access level has changed and the Kimi wrapper
        // should be implemented — update this test accordingly.
        do {
            _ = try await LLMTypeRegistry.shared.createModel(
                configuration: Data(Self.kimiConfigJSON.utf8),
                modelType: Self.modelType
            )
            Issue.record("expected kimiK25RequiresPublicDeepseekV3Init to be thrown")
        } catch let error as ModelOverlayError {
            #expect(error == .kimiK25RequiresPublicDeepseekV3Init)
        } catch {
            Issue.record("unexpected error: \(error)")
        }
    }
}
