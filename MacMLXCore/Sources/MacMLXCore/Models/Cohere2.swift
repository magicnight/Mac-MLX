import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// Cohere2 (Cohere Command R7B — `command-r7b-12-2024`) — pure-Swift port
// (macMLX overlay architecture).
//
// A macMLX-owned model architecture (`model_type: cohere2`) registered into the
// stock mlx-swift-lm factory via `ModelOverlay` (no fork), following the
// `Mellum2.swift` / `HunyuanV1Dense.swift` precedent. Upstream mlx-swift-lm ships
// `Cohere` (Command R v1, `Cohere.swift`) but has NO `cohere2` type.
//
// Cohere2 is the Cohere-family decoder (parallel residual block, LayerNorm, tied
// embeddings, logit scaling) plus the Command R7B interleaved-attention twist:
//   • PARALLEL residual block: attention AND MLP both read the SAME
//     `input_layernorm(x)`, and the block output is `attn + mlp + x`. There is
//     NO `post_attention_layernorm` — each layer carries exactly one norm.
//   • `LayerNorm` (NOT RMSNorm), with a `layer_norm_bias` switch (shipped: false).
//   • Interleaved sliding-window / global attention on a `sliding_window_pattern`
//     (shipped: 4): three sliding-window layers then one global layer, repeating.
//   • RoPE is `traditional=true` (GPT-J interleaved — `position_embedding_type:
//     rope_gptj`) and is applied ONLY on the sliding-window layers; the global
//     layers get NO positional encoding at all (NoPE).
//   • Mixed KV cache: sliding layers use `RotatingKVCache(maxSize: sliding_window,
//     keep: 0)`, global layers use a plain `KVCacheSimple` — mirrors Python
//     `make_cache`.
//   • Embeddings are ALWAYS tied (the checkpoint carries no `lm_head`); the logits
//     are the embedding matrix applied as a linear, scaled by `logit_scale`.
//
// Translated from Apple's Python mlx-lm reference (`mlx_lm/models/cohere2.py`,
// 0.31.3). End-to-end numerical parity is proven at 1e-4 against fixtures captured
// by `docs/reference/capture_cohere2.py` (see `Cohere2ModelParityTests`), with two
// adversarial configs whose every switch (attention_bias, layer_norm_bias,
// logit_scale, sliding_window_pattern, sliding_window) is inverted, and whose
// sequence length exceeds the sliding window so the windowed mask genuinely
// differs from the full-causal mask.
//
// The building blocks (`RoPE`, `LayerNorm`, `applyRotaryPosition`,
// `createAttentionMask` window variant, `attentionWithCacheUpdate`,
// `RotatingKVCache`, `KVCacheSimple`, `silu`) are all stock mlx-swift-lm / MLXNN
// public API — this file only wires them into Cohere2's topology.

// MARK: - Configuration

/// `config.json` schema for `model_type: cohere2`. Field defaults mirror the
/// Python `ModelArgs` dataclass (`hidden_size 4096`, `head_dim 128`,
/// `num_hidden_layers 32`, `rope_theta 50000`, `logit_scale 0.0625`,
/// `attention_bias false`, `layer_norm_bias false`, `sliding_window 4096`,
/// `sliding_window_pattern 4`) so a partial config still decodes.
///
/// INTENTIONAL DIVERGENCE (permissive decode — the SeedOss / Hunyuan precedent):
/// every field decodes with `decodeIfPresent ?? fallback`. Note the shipped 4-bit
/// checkpoint OVERRIDES `logit_scale` to `0.25` (the dataclass default is
/// `0.0625`), so the value is always read from config rather than assumed. The
/// upstream `ModelArgs` also carries only the fields below; Cohere2 `config.json`
/// additionally ships `layer_switch`, `order_of_interleaved_layers`, `rotary_pct`,
/// `use_parallel_block`, `use_embedding_sharing`, etc., which the Python reference
/// never reads — they are ignored here identically.
public struct Cohere2Configuration: Codable, Sendable {
    public var modelType: String = "cohere2"
    public var hiddenSize: Int = 4096
    public var headDim: Int = 128
    public var numHiddenLayers: Int = 32
    public var intermediateSize: Int = 14336
    public var numAttentionHeads: Int = 32
    public var numKeyValueHeads: Int = 8
    public var ropeTheta: Float = 50000.0
    public var vocabSize: Int = 256000
    public var layerNormEps: Float = 1e-5
    public var logitScale: Float = 0.0625
    public var attentionBias: Bool = false
    public var layerNormBias: Bool = false
    public var slidingWindow: Int = 4096
    public var slidingWindowPattern: Int = 4

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case headDim = "head_dim"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case vocabSize = "vocab_size"
        case layerNormEps = "layer_norm_eps"
        case logitScale = "logit_scale"
        case attentionBias = "attention_bias"
        case layerNormBias = "layer_norm_bias"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func d<T: Decodable>(_ k: CodingKeys, _ fallback: T) throws -> T {
            try c.decodeIfPresent(T.self, forKey: k) ?? fallback
        }
        self.modelType = try d(.modelType, "cohere2")
        self.hiddenSize = try d(.hiddenSize, 4096)
        self.headDim = try d(.headDim, 128)
        self.numHiddenLayers = try d(.numHiddenLayers, 32)
        self.intermediateSize = try d(.intermediateSize, 14336)
        self.numAttentionHeads = try d(.numAttentionHeads, 32)
        self.numKeyValueHeads = try d(.numKeyValueHeads, 8)
        self.ropeTheta = try d(.ropeTheta, 50000.0)
        self.vocabSize = try d(.vocabSize, 256000)
        self.layerNormEps = try d(.layerNormEps, 1e-5)
        self.logitScale = try d(.logitScale, 0.0625)
        self.attentionBias = try d(.attentionBias, false)
        self.layerNormBias = try d(.layerNormBias, false)
        self.slidingWindow = try d(.slidingWindow, 4096)
        self.slidingWindowPattern = try d(.slidingWindowPattern, 4)
    }
}

// MARK: - Attention (GQA, single bias switch, interleaved sliding-window / NoPE)

/// Grouped-query attention. Two Cohere2-specific details:
///   1. RoPE is `traditional=true` (GPT-J interleaved) and applied ONLY when this
///      is a sliding-window layer. On global layers `rope` is nil and NO
///      positional encoding is applied at all (NoPE) — matching the Python
///      `if self.use_sliding_window:` guard around the rope call.
///   2. a SINGLE `attention_bias` drives all four projections (q/k/v AND o).
///
/// `use_sliding_window = (layer_idx + 1) % sliding_window_pattern != 0` — the
/// complement of the model's global-layer test `layer_idx % pattern == pattern-1`.
/// `scale` uses `head_dim ** -0.5` (head_dim is an explicit config field, default
/// 128, NOT `hidden / heads`). Cache update + SDPA go through the stock
/// `attentionWithCacheUpdate` router. Translated from the Python `Attention`.
final class Cohere2Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let scale: Float
    let useSlidingWindow: Bool

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    // Present only on sliding-window layers. `traditional: true` (GPT-J
    // interleaved) is the Cohere convention — the OPPOSITE of the Llama lineage
    // (non-traditional). nil on global layers, which are NoPE.
    let rope: RoPE?

    init(_ config: Cohere2Configuration, layerIdx: Int) {
        let dim = config.hiddenSize
        let headDim = config.headDim
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.scale = pow(Float(headDim), -0.5)

        // Upstream raises when head_dim * n_heads != hidden_size. The later
        // `reshaped(b, l, numHeads, -1)` would silently mis-shape on a mismatch
        // instead of failing, so stay loud with a precondition (do NOT relax this
        // to a permissive fallback — the other config fields default permissively,
        // but this invariant must hold for the reshape to be correct).
        precondition(
            headDim * numHeads == dim,
            "hidden_size (\(dim)) must equal head_dim (\(headDim)) * num_attention_heads (\(numHeads))")

        // One switch drives every projection (q/k/v AND o).
        let bias = config.attentionBias
        self._wq.wrappedValue = Linear(dim, numHeads * headDim, bias: bias)
        self._wk.wrappedValue = Linear(dim, numKVHeads * headDim, bias: bias)
        self._wv.wrappedValue = Linear(dim, numKVHeads * headDim, bias: bias)
        self._wo.wrappedValue = Linear(numHeads * headDim, dim, bias: bias)

        // Sliding-window layers apply RoPE; global layers are NoPE.
        self.useSlidingWindow = (layerIdx + 1) % config.slidingWindowPattern != 0
        self.rope =
            useSlidingWindow
            ? RoPE(dimensions: headDim, traditional: true, base: config.ropeTheta)
            : nil
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

        // RoPE only on sliding-window layers (global layers are NoPE — the Python
        // `if self.use_sliding_window:` guard). Offset comes from the layer's own
        // cache (rotating on sliding layers), so the rotation tracks the cache.
        if let rope {
            let offset = cache?.ropeOffset
            queries = applyRotaryPosition(rope, to: queries, offset: offset)
            keys = applyRotaryPosition(rope, to: keys, offset: offset)
        }

        // On the upstream fp16 -> fp32 SDPA cast: the Python reference casts
        // `queries` to float32 when they are float16 (a fused-mask precision
        // workaround the upstream marks as provisional / maybe-removable). The stock
        // `attentionWithCacheUpdate` router used by every macMLX overlay port
        // (Mellum2, Hunyuan, Seed-OSS — all real-fp16 checkpoints) does not
        // replicate that cast and their real-weights smokes still generate
        // coherent text; the fp32 parity fixtures are unaffected either way. We
        // follow the repo convention and route through the stock helper unchanged.
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
/// is bias-free (the upstream `MLP` never reads a bias flag). Identical to the
/// stock Cohere v1 / Hunyuan MLP. Translated from the Python `MLP` (`swiglu`).
final class Cohere2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ config: Cohere2Configuration) {
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

// MARK: - Decoder layer (PARALLEL residual)

/// One transformer block. Cohere's defining feature is the PARALLEL residual:
/// attention and MLP BOTH consume the same `input_layernorm(x)`, and the block
/// output is `attn + mlp + x`. There is NO `post_attention_layernorm` — each layer
/// has exactly one norm.
///
/// DO NOT rewrite as a serial pre-norm block (`h = x + attn(norm(x)); h +
/// mlp(norm2(h))`) — that is a different architecture (Llama/Hunyuan) and breaks
/// parity. Translated from the Python `TransformerBlock`.
final class Cohere2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Cohere2Attention
    @ModuleInfo(key: "mlp") var mlp: Cohere2MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: LayerNorm

    init(_ config: Cohere2Configuration, layerIdx: Int) {
        self._selfAttn.wrappedValue = Cohere2Attention(config, layerIdx: layerIdx)
        self._mlp.wrappedValue = Cohere2MLP(config)
        // LayerNorm (not RMSNorm); `bias` gated by `layer_norm_bias` (shipped
        // false). `affine: true` always keeps the trainable weight.
        self._inputLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps,
            affine: true, bias: config.layerNormBias)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = inputLayerNorm(x)
        let attnH = selfAttn(h, mask: mask, cache: cache)
        let ffH = mlp(h)
        return attnH + ffH + x
    }
}

// MARK: - Model (inner: embed → layers → norm)

/// The transformer stack. Builds two attention masks per forward — a full causal
/// mask (from the first GLOBAL layer's cache) and a windowed mask (from the first
/// SLIDING layer's cache, `windowSize = sliding_window`) — then hands each decoder
/// layer the mask matching its interleave position. Translated from the Python
/// `CohereModel`.
final class Cohere2ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [Cohere2DecoderLayer]
    @ModuleInfo(key: "norm") var norm: LayerNorm

    let slidingWindow: Int
    let slidingWindowPattern: Int
    let firstGlobal: Int?
    let firstSliding: Int?

    init(_ config: Cohere2Configuration) {
        precondition(config.vocabSize > 0)
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map {
            Cohere2DecoderLayer(config, layerIdx: $0)
        }
        self._norm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps,
            affine: true, bias: config.layerNormBias)

        self.slidingWindow = config.slidingWindow
        self.slidingWindowPattern = config.slidingWindowPattern
        // A layer is GLOBAL when `i % pattern == pattern - 1`. Python sources the
        // full mask from `cache[pattern - 1]` (the first global layer) and the
        // windowed mask from `cache[0]` (the first sliding layer). Compute those
        // indices robustly (Mellum2's approach) so a degenerate config with no
        // global — or no sliding — layer never indexes out of range.
        self.firstGlobal = (0 ..< config.numHiddenLayers).first {
            $0 % config.slidingWindowPattern == config.slidingWindowPattern - 1
        }
        self.firstSliding = (0 ..< config.numHiddenLayers).first {
            $0 % config.slidingWindowPattern != config.slidingWindowPattern - 1
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)

        // Dual mask, exactly as Python:
        //   full_mask = create_attention_mask(h, cache[pattern - 1])
        //   swa_mask  = create_attention_mask(h, cache[0], window_size=window)
        let fullMask = createAttentionMask(h: h, cache: firstGlobal.flatMap { cache?[$0] })
        let slidingMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let firstSliding {
            slidingMask = createAttentionMask(
                h: h, cache: cache?[firstSliding], windowSize: slidingWindow)
        } else {
            slidingMask = fullMask
        }

        for (i, layer) in layers.enumerated() {
            let isGlobal = i % slidingWindowPattern == slidingWindowPattern - 1
            h = layer(h, mask: isGlobal ? fullMask : slidingMask, cache: cache?[i])
        }

        return norm(h)
    }
}

// MARK: - Model (outer: LLMModel entry point)

/// The Cohere2 language model: the inner transformer plus the tied language head.
/// Conforms to `LLMModel` so the stock `LLMModelFactory` (via `ModelOverlay`
/// registration) loads and runs it.
///
/// Two Cohere-family specifics:
///   • Embeddings are ALWAYS tied — the checkpoint carries no `lm_head` and the
///     upstream `Model` has no untied path — so logits come from
///     `embed_tokens.asLinear`, then scaled by `logit_scale`. No `sanitize`
///     override is needed (nothing to rewrite; the stock passthrough is correct).
///   • A MIXED per-layer cache — global layers use `KVCacheSimple`, sliding layers
///     use `RotatingKVCache(maxSize: sliding_window, keep: 0)` — so `newCache` is
///     overridden rather than using the `KVCacheDimensionProvider` default
///     (uniform `KVCacheSimple`). Mirrors the Python `make_cache`.
final class Cohere2Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    let config: Cohere2Configuration
    let vocabularySize: Int
    var kvHeads: [Int]
    var model: Cohere2ModelInner
    let logitScale: Float

    init(_ config: Cohere2Configuration) {
        self.config = config
        self.vocabularySize = config.vocabSize
        self.kvHeads = Array(
            repeating: config.numKeyValueHeads, count: config.numHiddenLayers)
        self.model = Cohere2ModelInner(config)
        self.logitScale = config.logitScale
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var out = model(inputs, cache: cache)
        // Tied embeddings (always) + logit scaling.
        out = model.embedTokens.asLinear(out)
        out = out * logitScale
        return out
    }

    /// Mixed per-layer cache (Python `make_cache`): `RotatingKVCache(maxSize:
    /// sliding_window, keep: 0)` for sliding layers, plain `KVCacheSimple` for
    /// global layers. Do NOT fall back to the `KVCacheDimensionProvider` default
    /// (uniform `KVCacheSimple`) — the sliding layers need the rotating cache.
    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< config.numHiddenLayers).map { i in
            let isGlobal = i % config.slidingWindowPattern == config.slidingWindowPattern - 1
            return isGlobal
                ? KVCacheSimple()
                : RotatingKVCache(maxSize: config.slidingWindow, keep: 0)
        }
    }

    var loraLayers: [Module] {
        model.layers.map { $0 }
    }
}
