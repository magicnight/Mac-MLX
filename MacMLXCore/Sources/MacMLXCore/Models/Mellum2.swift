import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// Mellum 2 (12B-A2.5B) — pure-Swift port (macMLX overlay architecture).
//
// A macMLX-owned model architecture (`model_type: mellum`) registered into the
// stock mlx-swift-lm factory via `ModelOverlay` (no fork), following the
// `DeepseekV32.swift` precedent. Upstream mlx-swift-lm has no Mellum type.
//
// Mellum is a Qwen3-lineage sparse-MoE decoder:
//   • Attention with per-head `q_norm`/`k_norm` (Qwen3 style).
//   • A 64-expert `SwitchGLU` MoE on every layer (Qwen3-MoE style router:
//     softmax → top-k → optional renorm → weighted expert sum).
//   • 28 layers alternating 3 sliding-window + 1 full-attention (`layer_types`).
//   • Two RoPE families keyed by layer type (`rope_parameters`): full-attention
//     uses YaRN (factor 16), sliding uses plain RoPE — both at theta 500000.
//   • Mixed KV cache: full layers use `KVCacheSimple`, sliding layers use
//     `RotatingKVCache(maxSize: sliding_window)` — mirrors Python `make_cache`.
//
// Translated from Apple's Python mlx-lm reference (`mlx_lm/models/mellum.py`,
// PR #1339), preserved conceptually here. Component + end-to-end numerical
// parity is proven at 1e-4 against fixtures captured by
// `docs/reference/capture_mellum2.py` (see `Mellum2*ParityTests`).
//
// The building blocks (`Attention`, MoE router, `SwitchGLU`, `initializeRope`
// YaRN, `RotatingKVCache`, windowed `createAttentionMask`, `weightedExpertSum`,
// `attentionWithCacheUpdate`) are all stock mlx-swift-lm public API — this file
// only wires them into Mellum's specific topology.

// MARK: - Configuration

/// `config.json` schema for `model_type: mellum`. Field defaults mirror the
/// Python `ModelArgs` dataclass so partial configs decode. `rope_parameters`
/// is a nested map keyed by layer type (`full_attention` / `sliding_attention`),
/// each carrying its own RoPE settings (`rope_type`, `rope_theta`, and — for
/// YaRN — `factor`, `original_max_position_embeddings`, `beta_fast`,
/// `beta_slow`).
public struct Mellum2Configuration: Codable, Sendable {
    public var vocabSize: Int = 98304
    public var hiddenSize: Int = 2304
    public var numHiddenLayers: Int = 28
    public var intermediateSize: Int = 7168
    public var numAttentionHeads: Int = 32
    public var numExperts: Int = 64
    public var numExpertsPerTok: Int = 8
    public var moeIntermediateSize: Int = 896
    public var rmsNormEps: Float = 1e-6
    public var numKeyValueHeads: Int = 4
    public var headDim: Int = 128
    public var tieWordEmbeddings: Bool = false
    public var maxPositionEmbeddings: Int = 131072
    public var normTopkProb: Bool = true
    public var slidingWindow: Int = 1024
    public var layerTypes: [String] = []
    public var ropeParameters: [String: [String: StringOrNumber]] = [:]

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
        case normTopkProb = "norm_topk_prob"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func d<T: Decodable>(_ k: CodingKeys, _ fallback: T) throws -> T {
            try c.decodeIfPresent(T.self, forKey: k) ?? fallback
        }
        self.vocabSize = try d(.vocabSize, 98304)
        self.hiddenSize = try d(.hiddenSize, 2304)
        self.numHiddenLayers = try d(.numHiddenLayers, 28)
        self.intermediateSize = try d(.intermediateSize, 7168)
        self.numAttentionHeads = try d(.numAttentionHeads, 32)
        self.numExperts = try d(.numExperts, 64)
        self.numExpertsPerTok = try d(.numExpertsPerTok, 8)
        self.moeIntermediateSize = try d(.moeIntermediateSize, 896)
        self.rmsNormEps = try d(.rmsNormEps, 1e-6)
        self.numKeyValueHeads = try d(.numKeyValueHeads, 4)
        self.headDim = try d(.headDim, 128)
        self.tieWordEmbeddings = try d(.tieWordEmbeddings, false)
        self.maxPositionEmbeddings = try d(.maxPositionEmbeddings, 131072)
        self.normTopkProb = try d(.normTopkProb, true)
        self.slidingWindow = try d(.slidingWindow, 1024)
        self.layerTypes = try d(.layerTypes, [])
        self.ropeParameters = try d(.ropeParameters, [:])
    }
}

// MARK: - Per-layer-type RoPE

/// Build the RoPE for one layer, keyed by its `layer_type`. Ports the Python
/// `_rope_for`: `default`/`linear` layer types get a plain `RoPE`; every other
/// type (Mellum uses `yarn` on full-attention layers) routes through
/// `initializeRope` with the layer's `rope_parameters` as the scaling config.
///
/// - Note: Mellum's `attention_factor` key is present in the checkpoint but,
///   exactly as in mlx-lm's `initialize_rope`, it is NOT read here — only
///   `factor` / `original_max_position_embeddings` / `beta_fast` / `beta_slow`
///   (and the absent `mscale` / `mscale_all_dim`, defaulting to 1 / 0) drive
///   YaRN. For `factor = 16` this yields mscale `0.1·ln(16)+1 = 1.2772…`, which
///   equals the checkpoint's `attention_factor` — so both sides agree.
func mellum2Rope(_ config: Mellum2Configuration, layerType: String) -> RoPELayer {
    let params = config.ropeParameters[layerType] ?? [:]
    let base = params["rope_theta"]?.asFloat() ?? 500000.0
    var ropeType = "default"
    if case .string(let s)? = params["rope_type"] {
        ropeType = s
    }

    if ropeType == "default" || ropeType == "linear" {
        // Matches Python `initialize_rope(head_dim, base=base, traditional=False)`
        // — no scaling config, plain (non-traditional) RoPE at scale 1.0.
        return initializeRope(
            dims: config.headDim, base: base, traditional: false,
            scalingConfig: nil, maxPositionEmbeddings: nil)
    }

    // Non-default (YaRN et al.): pass the layer's rope_parameters as the scaling
    // config, with an explicit `type` (mirrors Python `scaling_config["type"]`).
    var scalingConfig = params
    scalingConfig["type"] = .string(ropeType)
    return initializeRope(
        dims: config.headDim, base: base, traditional: false,
        scalingConfig: scalingConfig,
        maxPositionEmbeddings: config.maxPositionEmbeddings)
}

// MARK: - Attention (Qwen3-style with q/k norm)

/// Grouped-query attention with per-head `q_norm` / `k_norm` (Qwen3 lineage).
/// The only Mellum-specific twist is the RoPE: it is chosen per layer type by
/// `mellum2Rope`, so a full-attention layer runs YaRN while a sliding layer
/// runs plain RoPE. Cache update + SDPA go through the stock
/// `attentionWithCacheUpdate` router (handles regular and quantized caches).
/// Translated from the Python `Attention`.
final class Mellum2Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    let rope: RoPELayer

    init(_ config: Mellum2Configuration, layerIdx: Int) {
        let dim = config.hiddenSize
        let headDim = config.headDim
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, numHeads * headDim, bias: false)
        self._wk.wrappedValue = Linear(dim, numKVHeads * headDim, bias: false)
        self._wv.wrappedValue = Linear(dim, numKVHeads * headDim, bias: false)
        self._wo.wrappedValue = Linear(numHeads * headDim, dim, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        let layerType = layerIdx < config.layerTypes.count
            ? config.layerTypes[layerIdx] : "full_attention"
        self.rope = mellum2Rope(config, layerType: layerType)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        // Per-head q/k RMSNorm before the transpose (Qwen3 style).
        queries = qNorm(queries.reshaped(b, l, numHeads, -1)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(b, l, numKVHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(b, l, numKVHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(b, l, -1)

        return wo(output)
    }
}

// MARK: - Sparse MoE block

/// The routed sparse feed-forward on every Mellum layer: a `SwitchGLU` bank of
/// `num_experts` gated by a linear router. Ports the Python
/// `MellumSparseMoeBlock`: softmax over all experts, top-k selection, optional
/// top-k renormalization, then the weighted expert sum.
///
/// IMPORTANT — do NOT "simplify" the router to the sibling `Qwen3MoE` pattern
/// (`argPartition(-rawGates, kth: k-1)[..<k]`): top-k here is deliberately
/// taken over the SOFTMAX gates with the same `kth = N - k` / `[N-k:]` call
/// shape as Python's `argpartition(gates, kth=-k)[..., -k:]`. Identical kernel
/// + identical arguments is what guarantees tie-identical expert selection;
/// selecting on the raw logits picks a different expert under an exact gate
/// tie (caught by the parity fixture during the port).
final class Mellum2SparseMoeBlock: Module, UnaryLayer {
    let topK: Int
    let normTopkProb: Bool

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    init(_ config: Mellum2Configuration) {
        self.topK = config.numExpertsPerTok
        self.normTopkProb = config.normTopkProb
        self._gate.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gates = gate(x)
        let softGates = softmax(gates, axis: -1, precise: true)

        // Mirror the Python `MellumSparseMoeBlock` exactly: top-k over the
        // *softmax* gates (`argpartition(gates, kth=-k)[..., -k:]`), then gather
        // and (optionally) renormalize those scores. Selecting on the softmax
        // (rather than the raw logits) keeps tie-breaking identical to the
        // reference — the two are equivalent when there is no exact tie, but a
        // synthetic tie would otherwise pick a different expert.
        let k = topK
        let expertAxis = softGates.dim(-1)
        let inds = argPartition(softGates, kth: expertAxis - k, axis: -1)[
            .ellipsis, (expertAxis - k)...]
        var scores = takeAlong(softGates, inds, axis: -1)
        if normTopkProb {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }

        let y = switchMLP(x, inds)
        return weightedExpertSum(y, scores)
    }
}

// MARK: - Decoder layer

/// One transformer block: pre-norm attention + residual, then pre-norm MoE +
/// residual. Every Mellum layer is a MoE layer. Translated from the Python
/// `MellumDecoderLayer`. The `mask` (full-causal or windowed) is chosen by the
/// model per this layer's type.
final class Mellum2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Mellum2Attention
    @ModuleInfo(key: "mlp") var mlp: Mellum2SparseMoeBlock
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: Mellum2Configuration, layerIdx: Int) {
        self._selfAttn.wrappedValue = Mellum2Attention(config, layerIdx: layerIdx)
        self._mlp.wrappedValue = Mellum2SparseMoeBlock(config)
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

/// The transformer stack. Builds two attention masks per forward — a full
/// causal mask (from the first full-attention layer's cache) and a windowed
/// mask (from the first sliding layer's cache, `windowSize = sliding_window`) —
/// then hands each decoder layer the mask matching its `layer_type`. Translated
/// from the Python `MellumModel`.
final class Mellum2ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [Mellum2DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let layerTypes: [String]
    let slidingWindow: Int
    let firstFull: Int?
    let firstSliding: Int?

    init(_ config: Mellum2Configuration) {
        precondition(config.vocabSize > 0)
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map {
            Mellum2DecoderLayer(config, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        self.layerTypes = config.layerTypes
        self.slidingWindow = config.slidingWindow
        self.firstFull = config.layerTypes.firstIndex(of: "full_attention")
        self.firstSliding = config.layerTypes.firstIndex(of: "sliding_attention")
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)

        // Full-attention mask sourced from the first full layer's cache; the
        // sliding mask from the first sliding layer's cache with the window.
        // Python: `create_attention_mask(h, cache[first_full])` and
        // `create_attention_mask(h, cache[first_sliding], window_size=…)`.
        let fullMask = createAttentionMask(
            h: h, cache: firstFull.flatMap { cache?[$0] })
        let slidingMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let firstSliding {
            slidingMask = createAttentionMask(
                h: h, cache: cache?[firstSliding], windowSize: slidingWindow)
        } else {
            slidingMask = fullMask
        }

        for (i, layer) in layers.enumerated() {
            let isFull = i < layerTypes.count && layerTypes[i] == "full_attention"
            h = layer(h, mask: isFull ? fullMask : slidingMask, cache: cache?[i])
        }

        return norm(h)
    }
}

// MARK: - Model (outer: LLMModel entry point)

/// The Mellum language model: the inner transformer plus the (untied) `lm_head`
/// projection to vocabulary logits. Conforms to `LLMModel` so the stock
/// `LLMModelFactory` (via `ModelOverlay` registration) loads and runs it.
///
/// Mellum needs a **mixed per-layer cache** — full-attention layers use a plain
/// `KVCacheSimple`, sliding layers use a `RotatingKVCache(maxSize:
/// sliding_window)` — so `newCache` is overridden rather than using the
/// `KVCacheDimensionProvider` default. Mirrors the Python `make_cache`.
final class Mellum2Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    let config: Mellum2Configuration
    let vocabularySize: Int
    var kvHeads: [Int]
    var model: Mellum2ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: Mellum2Configuration) {
        self.config = config
        self.vocabularySize = config.vocabSize
        self.kvHeads = Array(
            repeating: config.numKeyValueHeads, count: config.numHiddenLayers)
        self.model = Mellum2ModelInner(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    /// Mixed per-layer cache: `RotatingKVCache` for sliding layers, plain
    /// `KVCacheSimple` for full-attention layers. Do NOT fall back to the
    /// `KVCacheDimensionProvider` default (which is uniform `KVCacheSimple`).
    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        config.layerTypes.map { type in
            type == "full_attention"
                ? KVCacheSimple()
                : RotatingKVCache(maxSize: config.slidingWindow)
        }
    }

    /// Rewrite raw checkpoint weights into the module's parameter layout. Ports
    /// the Python `Model.sanitize`: drop `lm_head` when tied, and stack the
    /// per-expert MoE projections (`experts.{e}.{proj}`) into the `switch_mlp`
    /// bank `[num_experts, out, in]`. Pre-stacked checkpoints (e.g. the shipped
    /// 4-bit MLX conversion) already carry `switch_mlp.*` and short-circuit.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights

        if config.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        // Already in module layout (switch_mlp stacked) → nothing to stack.
        if weights["model.layers.0.mlp.experts.0.up_proj.weight"] == nil {
            return weights
        }

        for l in 0 ..< config.numHiddenLayers {
            let prefix = "model.layers.\(l)"
            for n in ["up_proj", "down_proj", "gate_proj"] {
                guard weights["\(prefix).mlp.experts.0.\(n).weight"] != nil else { continue }
                // Gather all expert keys NON-DESTRUCTIVELY first: only stack (and only
                // then remove the per-expert tensors) when the whole set is present. The
                // previous version `removeValue`d as it went and broke on the first gap,
                // so a partial set silently lost experts 0..<k — no switch_mlp bank AND
                // the source tensors already gone.
                let keys = (0 ..< config.numExperts).map {
                    "\(prefix).mlp.experts.\($0).\(n).weight"
                }
                let joined = keys.compactMap { weights[$0] }
                guard joined.count == config.numExperts else { continue }
                for key in keys { weights.removeValue(forKey: key) }
                weights["\(prefix).mlp.switch_mlp.\(n).weight"] = stacked(joined)
            }
        }

        return weights
    }

    var loraLayers: [Module] {
        model.layers.map { $0 }
    }
}
