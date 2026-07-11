import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// Hunyuan V1 Dense (Tencent Hunyuan dense line — 0.5B/1.8B/4B/7B) — pure-Swift
// port (macMLX overlay architecture).
//
// A macMLX-owned model architecture (`model_type: hunyuan_v1_dense`) registered
// into the stock mlx-swift-lm factory via `ModelOverlay` (no fork), following the
// `SeedOss.swift` / `Mellum2.swift` precedent. Upstream mlx-swift-lm 3.31.4 has
// no `hunyuan_v1_dense` type.
//
// Hunyuan V1 Dense is a standard dense Llama-family decoder — GQA + SwiGLU MLP,
// RMSNorm — with exactly two architecture-specific twists:
//   • per-head q/k RMSNorm applied AFTER RoPE (gated by `use_qk_norm`). This is
//     the OPPOSITE order to the Qwen3 lineage (`Mellum2`), where q/k norm runs
//     BEFORE RoPE — see the ordering note in `HunyuanV1DenseAttention`.
//   • a `DynamicNTKAlphaRoPE` whose base is pre-scaled by
//     `alpha ** (head_dim / (head_dim - 2))`, consuming ONLY `rope_scaling.alpha`
//     — realized as a base-scaled plain RoPE by `hunyuanV1DenseRope`.
// Everything else is conventional: a SINGLE `attention_bias` drives q/k/v/o (not
// Seed-OSS's three independent switches; the MLP is always bias-free), an
// explicit `head_dim` (falling back to `hidden_size / num_attention_heads`), and
// `tie_word_embeddings` (the shipped checkpoints tie).
//
// Translated from Apple's Python mlx-lm reference
// (`mlx_lm/models/hunyuan_v1_dense.py`, 0.31.3). End-to-end numerical parity is
// proven at 1e-4 against fixtures captured by
// `docs/reference/capture_hunyuan_v1_dense.py` (see `HunyuanV1Dense*ParityTests`),
// with two adversarial configs whose every switch is inverted.
//
// The building blocks (`initializeRope`, `applyRotaryPosition`, `RoPELayer`,
// `createAttentionMask`, `attentionWithCacheUpdate`, `silu`) are all stock
// mlx-swift-lm public API — this file only wires them into Hunyuan's topology.

// MARK: - Configuration

/// `config.json` schema for `model_type: hunyuan_v1_dense`. Field defaults mirror
/// the Python `ModelArgs` dataclass (`rope_theta 10000`, `max_position_embeddings
/// 32768`, `attention_bias false`, `use_qk_norm true`, `tie_word_embeddings
/// false`), while the structural dimensions fall back to the shipped
/// Hunyuan-1.8B-Instruct values so a partial config still decodes.
///
/// INTENTIONAL DIVERGENCE (permissive decode, documented fallbacks — the SeedOss
/// precedent): the Python `__post_init__` raises when `rope_scaling` is present
/// but lacks the `{alpha, factor, type}` keys. We do NOT replicate that raise —
/// `rope_scaling` is decoded permissively and only `alpha` is consumed (see
/// `scalingAlpha`); a missing dict, or one without `alpha`, yields `alpha = 1.0`
/// (a plain RoPE), and `factor`/`beta_fast`/`beta_slow`/`mscale`/… are ignored
/// exactly as the upstream `DynamicNTKAlphaRoPE` ignores them. Likewise
/// `rms_norm_eps` — required in the Python dataclass — decodes permissively,
/// defaulting to the shipped checkpoints' `1e-5`.
public struct HunyuanV1DenseConfiguration: Codable, Sendable {
    public var modelType: String = "hunyuan_v1_dense"
    public var vocabSize: Int = 120818
    public var hiddenSize: Int = 2048
    public var numHiddenLayers: Int = 32
    public var intermediateSize: Int = 6144
    public var numAttentionHeads: Int = 16
    public var numKeyValueHeads: Int = 4
    public var rmsNormEps: Float = 1e-5
    public var ropeTheta: Float = 10000
    public var maxPositionEmbeddings: Int = 32768
    public var attentionBias: Bool = false
    public var useQkNorm: Bool = true
    public var ropeScaling: [String: StringOrNumber]? = nil
    public var tieWordEmbeddings: Bool = false
    /// Optional in `config.json` (Python defaults it to `None`); when absent the
    /// resolved head dim is `hidden_size / num_attention_heads` (see
    /// ``resolvedHeadDim``).
    public var headDim: Int? = nil

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionBias = "attention_bias"
        case useQkNorm = "use_qk_norm"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case headDim = "head_dim"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func d<T: Decodable>(_ k: CodingKeys, _ fallback: T) throws -> T {
            try c.decodeIfPresent(T.self, forKey: k) ?? fallback
        }
        self.modelType = try d(.modelType, "hunyuan_v1_dense")
        self.vocabSize = try d(.vocabSize, 120818)
        self.hiddenSize = try d(.hiddenSize, 2048)
        self.numHiddenLayers = try d(.numHiddenLayers, 32)
        self.intermediateSize = try d(.intermediateSize, 6144)
        self.numAttentionHeads = try d(.numAttentionHeads, 16)
        self.numKeyValueHeads = try d(.numKeyValueHeads, 4)
        self.rmsNormEps = try d(.rmsNormEps, 1e-5)
        self.ropeTheta = try d(.ropeTheta, 10000)
        self.maxPositionEmbeddings = try d(.maxPositionEmbeddings, 32768)
        self.attentionBias = try d(.attentionBias, false)
        self.useQkNorm = try d(.useQkNorm, true)
        self.ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.tieWordEmbeddings = try d(.tieWordEmbeddings, false)
        // Left nil when absent so `resolvedHeadDim` applies the hidden/heads
        // fallback (Python's `head_dim if head_dim is not None else ...`).
        self.headDim = try c.decodeIfPresent(Int.self, forKey: .headDim)
    }

    /// The attention head dim actually used: the explicit `head_dim` when given,
    /// otherwise `hidden_size / num_attention_heads`. Ports the Python
    /// `Attention` head-dim resolution; the o_proj shape
    /// (`n_heads * head_dim -> hidden`) differs from `hidden` when the explicit
    /// value is not `hidden / heads`.
    public var resolvedHeadDim: Int {
        headDim ?? (hiddenSize / numAttentionHeads)
    }

    /// The DynamicNTK scaling factor consumed by the RoPE: `rope_scaling.alpha`
    /// when present, else `1.0` (a plain RoPE — base unchanged). Only `alpha` is
    /// read; the upstream `DynamicNTKAlphaRoPE` ignores every other rope_scaling
    /// field.
    public var scalingAlpha: Float {
        if let ropeScaling, let alpha = ropeScaling["alpha"]?.asFloat() {
            return alpha
        }
        return 1.0
    }
}

// MARK: - RoPE (DynamicNTKAlphaRoPE)

/// Builds Hunyuan's `DynamicNTKAlphaRoPE` as the stock `default` RoPE with a
/// PRE-SCALED base. The upstream module scales the RoPE base by
/// `alpha ** (head_dim / (head_dim - 2))` — consuming ONLY `rope_scaling.alpha`
/// (default 1.0 → base unchanged) — and then applies a plain, non-traditional
/// RoPE at scale 1.0. That is exactly `initializeRope(..., scalingConfig: nil)`
/// at `base = ropeTheta * alpha**(head_dim/(head_dim-2))`, routing to the same
/// stock `RoPE` kernel the 1e-4-parity-proven SeedOss port uses.
///
/// WHY NOT explicit `freqs`: the upstream Python literally precomputes
/// `base'^(2i/dims)` and calls `mx.fast.rope(base: None, freqs: …)` (the
/// `Llama3RoPE`/`YarnRoPE` shape). A verbatim Swift port of that is mathematically
/// identical but empirically drifts ~1e-3 from the reference, because mlx-swift's
/// graph `pow` that materializes the frequency table is less precise than the
/// RoPE kernel's own internal frequency computation. Letting the kernel derive
/// the frequencies from `base'` (the plain-RoPE form here) matches at 1e-4. The
/// `Double` base math mirrors the Python reference's scalar precision before it
/// rounds into the float32 graph.
func hunyuanV1DenseRope(_ config: HunyuanV1DenseConfiguration) -> RoPELayer {
    let headDim = config.resolvedHeadDim
    let effectiveBase = Float(
        Double(config.ropeTheta)
            * Foundation.pow(Double(config.scalingAlpha), Double(headDim) / Double(headDim - 2)))
    return initializeRope(
        dims: headDim,
        base: effectiveBase,
        traditional: false,
        scalingConfig: nil,
        maxPositionEmbeddings: config.maxPositionEmbeddings)
}

// MARK: - Attention (GQA, single bias switch, optional post-RoPE q/k norm)

/// Grouped-query attention. Two Hunyuan-specific details:
///   1. a SINGLE `attention_bias` biases all four projections (q/k/v AND o) —
///      unlike Seed-OSS's independent q/k/v vs o switches.
///   2. optional per-head q/k RMSNorm (`use_qk_norm`) applied AFTER RoPE.
///
/// ORDERING — DO NOT "align with Qwen3": the q/k norm here runs on the ALREADY
/// ROPE'D, transposed `[B, H, L, D]` tensors (Python: `rope` then
/// `query_layernorm`/`key_layernorm` then `cache.update`). The Qwen3 lineage
/// (`Mellum2`) does the opposite — norm the `[B, L, H, D]` tensors BEFORE RoPE.
/// Swapping the two (norm-before-RoPE) silently breaks parity, so the fixtures
/// pin the post-RoPE order.
///
/// `scale` uses the RESOLVED head dim (`head_dim ** -0.5`), which is NOT
/// `hidden / heads` when an explicit `head_dim` is set. Cache update + SDPA go
/// through the stock `attentionWithCacheUpdate` router. Translated from the
/// Python `Attention`.
final class HunyuanV1DenseAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    // Present only when `use_qk_norm` — nil layers contribute no parameters, so a
    // checkpoint without q/k norm weights loads cleanly (the Python module simply
    // does not create them). Keys match the Python `query_layernorm` /
    // `key_layernorm` (NOT the Qwen3 `q_norm` / `k_norm`).
    @ModuleInfo(key: "query_layernorm") var queryLayerNorm: RMSNorm?
    @ModuleInfo(key: "key_layernorm") var keyLayerNorm: RMSNorm?
    let rope: RoPELayer

    init(_ config: HunyuanV1DenseConfiguration) {
        let dim = config.hiddenSize
        let headDim = config.resolvedHeadDim
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.scale = pow(Float(headDim), -0.5)

        // One switch drives every projection (q/k/v AND o).
        let bias = config.attentionBias
        self._wq.wrappedValue = Linear(dim, numHeads * headDim, bias: bias)
        self._wk.wrappedValue = Linear(dim, numKVHeads * headDim, bias: bias)
        self._wv.wrappedValue = Linear(dim, numKVHeads * headDim, bias: bias)
        self._wo.wrappedValue = Linear(numHeads * headDim, dim, bias: bias)

        if config.useQkNorm {
            self._queryLayerNorm.wrappedValue = RMSNorm(
                dimensions: headDim, eps: config.rmsNormEps)
            self._keyLayerNorm.wrappedValue = RMSNorm(
                dimensions: headDim, eps: config.rmsNormEps)
        }

        self.rope = hunyuanV1DenseRope(config)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries.reshaped(b, l, numHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(b, l, numKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(b, l, numKVHeads, -1).transposed(0, 2, 1, 3)

        // RoPE first, THEN q/k norm (Hunyuan order — see the type doc).
        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        if let queryLayerNorm, let keyLayerNorm {
            queries = queryLayerNorm(queries)
            keys = keyLayerNorm(keys)
        }

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(b, l, -1)

        return wo(output)
    }
}

// MARK: - MLP (SwiGLU, always bias-free)

/// The dense SwiGLU feed-forward: `down(silu(gate(x)) * up(x))`. Every projection
/// is bias-free (the upstream `MLP` never reads a bias flag). Translated from the
/// Python `MLP`.
final class HunyuanV1DenseMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ config: HunyuanV1DenseConfiguration) {
        let dim = config.hiddenSize
        let hidden = config.intermediateSize
        self._gate.wrappedValue = Linear(dim, hidden, bias: false)
        self._down.wrappedValue = Linear(hidden, dim, bias: false)
        self._up.wrappedValue = Linear(dim, hidden, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

// MARK: - Decoder layer

/// One transformer block: pre-norm attention + residual, then pre-norm MLP +
/// residual. Translated from the Python `TransformerBlock`.
final class HunyuanV1DenseDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: HunyuanV1DenseAttention
    @ModuleInfo(key: "mlp") var mlp: HunyuanV1DenseMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: HunyuanV1DenseConfiguration) {
        self._selfAttn.wrappedValue = HunyuanV1DenseAttention(config)
        self._mlp.wrappedValue = HunyuanV1DenseMLP(config)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        return h + mlp(postAttentionLayerNorm(h))
    }
}

// MARK: - Model (inner: embed → layers → norm)

/// The transformer stack. A single full-causal mask is built from the first
/// layer's cache and shared by every (uniform, dense) layer. Translated from the
/// Python `HunyuanV1DenseModel`.
final class HunyuanV1DenseModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [HunyuanV1DenseDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: HunyuanV1DenseConfiguration) {
        precondition(config.vocabSize > 0)
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map { _ in
            HunyuanV1DenseDecoderLayer(config)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

// MARK: - Model (outer: LLMModel entry point)

/// The Hunyuan V1 Dense language model: the inner transformer plus the language
/// head. Conforms to `LLMModel` so the stock `LLMModelFactory` (via `ModelOverlay`
/// registration) loads and runs it. Being a uniform dense model, it inherits the
/// default `KVCacheDimensionProvider` cache (uniform `KVCacheSimple`) — no
/// `newCache` override. When `tie_word_embeddings` (the shipped checkpoints), the
/// logits are the embedding matrix applied as a linear; otherwise an untied
/// `lm_head`. Translated from the Python `Model`.
final class HunyuanV1DenseModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    let config: HunyuanV1DenseConfiguration
    let vocabularySize: Int
    var kvHeads: [Int]
    var model: HunyuanV1DenseModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: HunyuanV1DenseConfiguration) {
        self.config = config
        self.vocabularySize = config.vocabSize
        self.kvHeads = Array(
            repeating: config.numKeyValueHeads, count: config.numHiddenLayers)
        self.model = HunyuanV1DenseModelInner(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        } else {
            return model.embedTokens.asLinear(out)
        }
    }

    /// Ports the Python `Model.sanitize`: drop `lm_head.weight` when the
    /// embeddings are tied (the untied checkpoints keep it). The checkpoint
    /// carries no `rotary_emb.inv_freq` buffers, so — unlike Llama — no
    /// rotary-key filtering is needed; everything else passes through untouched.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        if config.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }
        return weights
    }

    var loraLayers: [Module] {
        model.layers.map { $0 }
    }
}
