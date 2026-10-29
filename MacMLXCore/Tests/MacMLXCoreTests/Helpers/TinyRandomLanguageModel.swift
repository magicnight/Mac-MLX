// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// Tiny random-weight instances of the `MLXLLM` text architectures the batched
/// decode path has to reason about, built from a minimal configuration JSON so a
/// test can run a real forward pass without a checkpoint on disk.
///
/// Every instance is 2 layers, hidden size 32, 4 query / 2 KV heads of width 8,
/// a 64-token vocabulary, float32, and a fixed seed, so two calls build the same
/// weights. Every layer is full attention (Gemma 3's `sliding_window_pattern` is
/// 1, Gemma 4's `layer_types` are all `full_attention`, no KV-shared layers), so
/// `newCache` returns only plain caches and the dense-cache gate accepts it:
/// a sliding-window layer would get a `RotatingKVCache`, which the batched path
/// refuses before the mask matters.
///
/// MLX-backed: callers must gate on `requireMLXRuntimeOrSkip()` first.
enum TinyRandomLanguageModel: String, CaseIterable {
    case gemma
    case gemma2
    case gemma3Text
    case gemma4Text
    case qwen2
    case qwen3
    case llama

    /// Vocabulary size shared by every instance; token ids must stay below it.
    static let vocabularySize = 64

    /// Build the model with freshly seeded random weights, evaluated so later
    /// forwards all read the same materialised parameters.
    func make(seed: UInt64 = 7) throws -> any LanguageModel {
        MLXRandom.seed(seed)
        let model: any LanguageModel & Module
        switch self {
        case .gemma:
            model = GemmaModel(try decode(GemmaConfiguration.self, Self.common + #"""
                , "model_type": "gemma", "head_dim": 8
                """#))
        case .gemma2:
            model = Gemma2Model(try decode(Gemma2Configuration.self, Self.common + #"""
                , "head_dim": 8, "attn_logit_softcapping": 50.0,
                "final_logit_softcapping": 30.0, "query_pre_attn_scalar": 8
                """#))
        case .gemma3Text:
            model = Gemma3TextModel(try decode(Gemma3TextConfiguration.self, Self.common + #"""
                , "model_type": "gemma3_text", "head_dim": 8, "query_pre_attn_scalar": 8,
                "sliding_window": 512, "sliding_window_pattern": 1
                """#))
        case .gemma4Text:
            model = Gemma4TextModel(try decode(Gemma4TextConfiguration.self, Self.common + #"""
                , "model_type": "gemma4_text", "head_dim": 8, "global_head_dim": 8,
                "num_global_key_value_heads": 2, "num_kv_shared_layers": 0,
                "hidden_size_per_layer_input": 0, "vocab_size_per_layer_input": 64,
                "use_double_wide_mlp": false, "sliding_window": 512,
                "layer_types": ["full_attention", "full_attention"]
                """#))
        case .qwen2:
            model = Qwen2Model(try decode(Qwen2Configuration.self, Self.common))
        case .qwen3:
            model = Qwen3Model(try decode(Qwen3Configuration.self, Self.common + #"""
                , "head_dim": 8
                """#))
        case .llama:
            model = LlamaModel(try decode(LlamaConfiguration.self, Self.common + #"""
                , "head_dim": 8
                """#))
        }
        eval(model.parameters().flattened().map { $0.1 })
        return model
    }

    /// Keys every configuration shares; the per-architecture fragments above
    /// append to it before the closing brace.
    private static let common = """
        { "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 64,
        "num_attention_heads": 4, "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
        "vocab_size": \(vocabularySize)
        """

    private func decode<Configuration: Decodable>(
        _ type: Configuration.Type, _ body: String
    ) throws -> Configuration {
        try JSONDecoder().decode(type, from: Data((body + "}").utf8))
    }
}
