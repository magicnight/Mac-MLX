// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing

@testable import MacMLXCore

/// A Gemma 4 checkpoint with its vision tower now loads through the VLM
/// factory as MLXVLM's `Gemma4` instead of MLXLLM's `Gemma4Model`, and keeps
/// accepting LoRA adapters. An adapter names its weights by their path from
/// the model's root and its targets by their names inside each layer
/// `loraLayers` returns, so both classes must return the same layers at the
/// same paths with the same children, or an adapter trained against one
/// would not apply to the other.
///
/// Built from one tiny configuration both classes decode. The weights are
/// never evaluated.
@Suite("Gemma 4 LoRA layers", .enabled(if: mlxMetallibIsAvailable, "needs the MLX Metal library"))
struct Gemma4LoRALayerPathTests {

    private static let config = Data(#"""
        {
          "model_type": "gemma4", "vocab_size": 64,
          "image_token_id": 60, "boi_token_id": 61, "eoi_token_id": 62,
          "vision_soft_tokens_per_image": 4,
          "text_config": {
            "model_type": "gemma4_text", "hidden_size": 32, "num_hidden_layers": 2,
            "intermediate_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6, "vocab_size": 64, "head_dim": 8, "global_head_dim": 8,
            "num_global_key_value_heads": 2, "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0, "vocab_size_per_layer_input": 64,
            "use_double_wide_mlp": false, "sliding_window": 512,
            "layer_types": ["sliding_attention", "full_attention"]
          },
          "vision_config": {
            "model_type": "gemma4_vision", "num_hidden_layers": 1, "hidden_size": 16,
            "intermediate_size": 32, "num_attention_heads": 2, "num_key_value_heads": 2,
            "head_dim": 8, "patch_size": 4, "position_embedding_size": 64,
            "default_output_length": 4, "pooling_kernel_size": 2
          }
        }
        """#.utf8)

    @Test
    func theVisionClassAdaptsTheSameLayersAsTheTextClass() throws {
        let text = MLXLLM.Gemma4Model(
            try JSONDecoder().decode(MLXLLM.Gemma4Configuration.self, from: Self.config))
        let vision = MLXVLM.Gemma4(
            try JSONDecoder().decode(MLXVLM.Gemma4Configuration.self, from: Self.config))

        let textPaths = paths(of: text.loraLayers, in: text)
        #expect(textPaths == ["language_model.model.layers.0", "language_model.model.layers.1"])
        #expect(paths(of: vision.loraLayers, in: vision) == textPaths)

        #expect(text.loraLayers.count == vision.loraLayers.count)
        for (textLayer, visionLayer) in zip(text.loraLayers, vision.loraLayers) {
            #expect(linearNames(in: visionLayer) == linearNames(in: textLayer))
            #expect(!linearNames(in: textLayer).isEmpty)
        }
    }

    /// Where each of `layers` sits in `root`, by identity; "missing" for one
    /// that is not a submodule of it.
    private func paths(of layers: [Module], in root: Module) -> [String] {
        let named = root.namedModules()
        return layers.map { layer in named.first { $0.1 === layer }?.0 ?? "missing" }
    }

    /// The names of the `Linear` modules inside `layer`, which is what an
    /// adapter's `keys` (`self_attn.q_proj`, `mlp.down_proj`, ...) resolve
    /// against.
    private func linearNames(in layer: Module) -> [String] {
        layer.namedModules().filter { $0.1 is Linear }.map(\.0).sorted()
    }
}
