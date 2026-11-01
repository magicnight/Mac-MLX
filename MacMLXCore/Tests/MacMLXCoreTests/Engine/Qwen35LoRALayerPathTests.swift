// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing

@testable import MacMLXCore

/// A Qwen3.5/3.6 checkpoint with its vision tower now loads through the VLM
/// factory as MLXVLM's `Qwen35` instead of MLXLLM's `Qwen35Model`, and keeps
/// accepting LoRA adapters. An adapter names its weights by their path from
/// the model's root and its targets by their names inside each layer
/// `loraLayers` returns, so both classes must return the same layers at the
/// same paths with the same children, or an adapter trained against one
/// would not apply to the other.
///
/// Built from one tiny configuration both classes decode, with three
/// linear-attention layers and one full-attention layer, as
/// `full_attention_interval` 4 lays them out. The weights are never
/// evaluated.
@Suite("Qwen3.5 LoRA layers", .enabled(if: mlxMetallibIsAvailable, "needs the MLX Metal library"))
struct Qwen35LoRALayerPathTests {

    private static let config = Data(#"""
        {
          "model_type": "qwen3_5", "vocab_size": 64,
          "text_config": {
            "model_type": "qwen3_5_text", "hidden_size": 32, "num_hidden_layers": 4,
            "intermediate_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 8, "rms_norm_eps": 1e-6, "vocab_size": 64,
            "linear_num_value_heads": 4, "linear_num_key_heads": 2,
            "linear_key_head_dim": 8, "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4,
            "full_attention_interval": 4, "tie_word_embeddings": false,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000000,
                                "partial_rotary_factor": 1.0, "mrope_section": [2, 1, 1]}
          },
          "vision_config": {
            "model_type": "qwen3_5", "depth": 1, "hidden_size": 16, "intermediate_size": 32,
            "out_hidden_size": 32, "num_heads": 2, "patch_size": 4, "spatial_merge_size": 2,
            "temporal_patch_size": 2, "num_position_embeddings": 16
          }
        }
        """#.utf8)

    @Test
    func theVisionClassAdaptsTheSameLayersAsTheTextClass() throws {
        let text = MLXLLM.Qwen35Model(
            try JSONDecoder().decode(MLXLLM.Qwen35Configuration.self, from: Self.config))
        let vision = MLXVLM.Qwen35(
            try JSONDecoder().decode(MLXVLM.Qwen35Configuration.self, from: Self.config))

        let textPaths = paths(of: text.loraLayers, in: text)
        #expect(textPaths == (0..<4).map { "language_model.model.layers.\($0)" })
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
