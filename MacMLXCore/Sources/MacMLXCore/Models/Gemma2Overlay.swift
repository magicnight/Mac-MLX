// Copyright © 2024 Apple Inc.
//
// Derived from mlx-swift-lm `Libraries/MLXLLM/Models/Gemma2.swift`
// (https://github.com/ml-explore/mlx-swift-lm, MIT License), revision
// 3b339ad6e3b3f44c8121ecff5131c7fd55e075e6 (3.32.3). Modifications for
// macMLX: the attention mask (see below).

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// Gemma 2 (`model_type: gemma2`) — macMLX's corrected copy of upstream's
// `Gemma2Model`, registered into the stock factory by `ModelOverlay` so it
// REPLACES upstream's `gemma2` entry (the `HunyuanV1Dense.swift` precedent).
//
// Why it exists. Upstream `Gemma2ModelInner` builds its mask with the
// `[KVCache]?` overload of `createAttentionMask`, which returns a BOOLEAN causal
// mask (or nil at one token) and never asks the cache, and `Gemma2Attention`
// then ADDS that mask to the soft-capped scores (`scores + mask`). A boolean
// added to floats is +1 on visible positions and +0 on future ones, so a
// multi-token prefill is not causal at all: every prompt token attends to every
// later one. Single-token decode is unaffected (the mask is nil there), which
// is why the model still produces plausible text. And because the mask never
// comes from the cache, `BatchKVCache`'s left-pad mask is ignored in a ragged
// batch, the reason upstream's type is off ``BatchModelAllowlist``.
//
// The fix, matching Python mlx-lm's `gemma2.py`: the mask comes from the first
// layer's cache through `makeAttentionMask(n:cache:returnArray: true)` (which is
// `cache.makeMask`, so `BatchKVCache`, `RotatingKVCache` and plain caches each
// build their own), and the attention applies it to the soft-capped scores with
// `which(mask, scores, dtype min)` for a boolean mask, `scores + mask` for an
// additive one, and a causal mask built from the cache offset for `.causal`.
//
// Everything else is upstream's, line for line: module and parameter keys, the
// configuration decode, the tied embeddings, attention / final logit
// soft-capping, `query_pre_attn_scalar`, the Gemma RMSNorms, the exact-erf
// `gelu` in the MLP, no `sanitize`, and the `KVCacheDimensionProvider` cache
// (one plain attention cache per layer, honouring `maxKVSize`). Like upstream
// and like Python mlx-lm, Gemma 2's alternating 4096-token sliding window is
// not modelled: every layer attends over the whole cache. `Gemma2OverlayTests`
// pins the key set against upstream's and the decode-path logits against
// upstream's on shared weights, so the mask is the only difference.
//
// Retire this file (and its `ModelOverlay` registration) once upstream's
// `Gemma2Model` takes its mask from the cache; `Gemma2OverlayTests`'s
// upstream canary fails when that happens.

// MARK: - Configuration

/// `config.json` schema for `model_type: gemma2`, decoded exactly as upstream's
/// `Gemma2Configuration` (whose stored properties are internal to MLXLLM, so the
/// overlay cannot reuse it): the same required keys, and defaults only for
/// `rope_theta` and `rope_traditional`.
public struct Gemma2OverlayConfiguration: Codable, Sendable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var headDimensions: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var ropeTheta: Float = 10_000
    var ropeTraditional: Bool = false
    var attnLogitSoftcapping: Float = 50.0
    var finalLogitSoftcapping: Float = 30.0
    var queryPreAttnScalar: Float = 144.0

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDimensions = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case attnLogitSoftcapping = "attn_logit_softcapping"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case queryPreAttnScalar = "query_pre_attn_scalar"
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.headDimensions = try container.decode(Int.self, forKey: .headDimensions)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.ropeTheta =
            try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        self.ropeTraditional =
            try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        self.attnLogitSoftcapping = try container.decode(
            Float.self, forKey: .attnLogitSoftcapping)
        self.finalLogitSoftcapping = try container.decode(
            Float.self, forKey: .finalLogitSoftcapping)
        self.queryPreAttnScalar = try container.decode(
            Float.self, forKey: .queryPreAttnScalar)
    }
}

// MARK: - Attention

final class Gemma2OverlayAttention: Module {
    let scale: Float
    let logitSoftCap: Float
    let headDim: Int
    let nHeads: Int
    let nKVHeads: Int
    let repeats: Int

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPE

    init(_ args: Gemma2OverlayConfiguration) {
        let dim = args.hiddenSize
        self.nHeads = args.attentionHeads
        self.nKVHeads = args.kvHeads
        self.repeats = args.attentionHeads / args.kvHeads
        self.headDim = args.headDimensions

        self.scale = 1.0 / pow(Float(args.queryPreAttnScalar), 0.5)

        self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        self._wk.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._wv.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: false)
        self.logitSoftCap = args.attnLogitSoftcapping
        self.rope = RoPE(
            dimensions: headDim, traditional: args.ropeTraditional, base: args.ropeTheta)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)
        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }
        let keyLength = keys.dim(2)

        queries = queries * self.scale

        if repeats > 1 {
            queries = queries.reshaped([B, nKVHeads, repeats, L, headDim])
            keys = expandedDimensions(keys, axes: [2])
            values = expandedDimensions(values, axes: [2])
        }

        var scores = matmul(queries, keys.swappedAxes(-1, -2))
        scores = tanh(scores / logitSoftCap) * logitSoftCap

        // The fix (upstream adds the mask unconditionally, so a boolean mask
        // becomes +1/+0). A boolean mask selects; an additive one is added.
        let maskArray: MLXArray?
        switch mask {
        case .none:
            maskArray = nil
        case .causal:
            maskArray = createCausalMask(n: L, offset: keyLength - L)
        default:
            maskArray = mask.mask
        }
        if var maskArray {
            // A per-row mask (`BatchKVCache`: [B, 1, L, S]) must line up with
            // the [B, nKVHeads, repeats, L, S] scores of the grouped layout;
            // without the extra axis B would broadcast against nKVHeads.
            if repeats > 1, maskArray.ndim == 4 {
                maskArray = expandedDimensions(maskArray, axis: 2)
            }
            if maskArray.dtype == .bool {
                scores = which(maskArray, scores, MLXArray.maskFill(for: scores.dtype))
            } else {
                scores = scores + maskArray
            }
        }

        scores = softmax(scores, axis: -1, precise: true)
        var output = matmul(scores, values)
        if repeats > 1 {
            output = output.reshaped([B, nHeads, L, headDim])
        }
        output = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return wo(output)
    }
}

// MARK: - MLP

final class Gemma2OverlayMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(gelu(gate(x)) * up(x))
    }
}

// MARK: - Transformer block

final class Gemma2OverlayTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Gemma2OverlayAttention
    let mlp: Gemma2OverlayMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma.RMSNorm

    init(_ args: Gemma2OverlayConfiguration) {
        self._attention.wrappedValue = Gemma2OverlayAttention(args)
        self.mlp = Gemma2OverlayMLP(
            dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        self._inputLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + postAttentionLayerNorm(r)
        r = mlp(preFeedforwardLayerNorm(h))
        return h + postFeedforwardLayerNorm(r)
    }
}

// MARK: - Inner model

final class Gemma2OverlayModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [Gemma2OverlayTransformerBlock]
    let norm: Gemma.RMSNorm

    let hiddenScale: Float

    init(_ args: Gemma2OverlayConfiguration) {
        precondition(args.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.hiddenScale = pow(Float(args.hiddenSize), 0.5)

        self.layers = (0 ..< args.hiddenLayers).map { _ in
            Gemma2OverlayTransformerBlock(args)
        }
        self.norm = Gemma.RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        h = h * hiddenScale

        // Python: `create_attention_mask(h, cache[0], return_array=True)`. Asking
        // the cache is what lets `BatchKVCache` mask each row's left padding.
        let mask = makeAttentionMask(n: h.dim(1), cache: cache?.first, returnArray: true)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

// MARK: - Model

final class Gemma2OverlayModel: Module, LLMModel, KVCacheDimensionProvider {
    let vocabularySize: Int
    let kvHeads: [Int]

    let model: Gemma2OverlayModelInner
    let logitSoftCap: Float

    init(_ args: Gemma2OverlayConfiguration) {
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.model = Gemma2OverlayModelInner(args)
        self.logitSoftCap = args.finalLogitSoftcapping
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        out = model.embedTokens.asLinear(out)
        out = tanh(out / logitSoftCap) * logitSoftCap
        return out
    }

    func messageGenerator(tokenizer: any Tokenizer) -> any MessageGenerator {
        NoSystemMessageGenerator()
    }

    var loraLayers: [Module] {
        model.layers
    }
}
