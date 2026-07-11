import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// Seed-OSS (ByteDance Seed-OSS-36B) — pure-Swift port (macMLX overlay architecture).
//
// A macMLX-owned model architecture (`model_type: seed_oss`) registered into the
// stock mlx-swift-lm factory via `ModelOverlay` (no fork), following the
// `Mellum2.swift` / `DeepseekV32.swift` precedent. Upstream mlx-swift-lm 3.31.4
// has no `seed_oss` type.
//
// Seed-OSS is a standard dense Llama-family decoder — GQA + SwiGLU MLP, one RoPE
// family, RMSNorm — with exactly three architecture-specific twists, all bias
// switches:
//   • `attention_bias`      → q/k/v_proj bias (Llama's single attention_bias),
//   • `attention_out_bias`  → o_proj bias, a SEPARATE switch (this is the twist:
//     Llama drives o_proj off its one `attention_bias`; Seed-OSS gives o its own),
//   • `mlp_bias`            → gate/up/down_proj bias.
// There is NO q/k RMSNorm (unlike the Qwen3-lineage Mellum2). The real
// checkpoint (`mlx-community/Seed-OSS-36B-Instruct-4bit`) sets attention_bias
// true, attention_out_bias false, mlp_bias false — so q/k/v carry a bias but o
// does not, which is exactly the asymmetry the parity fixtures pin.
//
// Translated from Apple's Python mlx-lm reference (`mlx_lm/models/seed_oss.py`).
// Component + end-to-end numerical parity is proven at 1e-4 against fixtures
// captured by `docs/reference/capture_seed_oss.py` (see `SeedOss*ParityTests`),
// with adversarial asymmetric-bias configs that triangulate the two independent
// attention bias switches.
//
// The building blocks (`initializeRope`, `applyRotaryPosition`, `RoPELayer`,
// `createAttentionMask`, `attentionWithCacheUpdate`, `silu`) are all stock
// mlx-swift-lm public API — this file only wires them into Seed-OSS's topology.

// MARK: - Configuration

/// `config.json` schema for `model_type: seed_oss`. Field defaults mirror the
/// Python `ModelArgs` dataclass: the three bias switches and the RoPE knobs
/// default exactly as Python does (all biases `false`, `rope_theta 10000`,
/// `tie_word_embeddings true`), while the structural dimensions fall back to the
/// shipped Seed-OSS-36B-Instruct values so a partial config still decodes.
///
/// Unlike `MLXLLM.LlamaConfiguration`, this decoder does NOT validate
/// `rope_scaling` (Llama rejects any `rope_scaling` lacking `factor` and only
/// permits linear/dynamic/llama3). Seed-OSS ships `rope_scaling:
/// {"rope_type": "default"}` — no `factor` — which Llama's decoder would reject,
/// so the dict is decoded permissively and handed to `initializeRope`, whose
/// `"default"` branch yields a plain RoPE (scale 1.0), matching Python's
/// `initialize_rope`.
public struct SeedOssConfiguration: Codable, Sendable {
    public var modelType: String = "seed_oss"
    public var hiddenSize: Int = 5120
    public var numHiddenLayers: Int = 64
    public var intermediateSize: Int = 27648
    public var numAttentionHeads: Int = 80
    public var rmsNormEps: Float = 1e-6
    public var vocabSize: Int = 155136
    public var numKeyValueHeads: Int = 8
    public var headDim: Int = 128
    public var maxPositionEmbeddings: Int? = 524288
    public var attentionBias: Bool = false
    public var attentionOutBias: Bool = false
    public var mlpBias: Bool = false
    public var ropeTheta: Float = 10000
    public var ropeTraditional: Bool = false
    public var ropeScaling: [String: StringOrNumber]? = nil
    public var tieWordEmbeddings: Bool = true

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionBias = "attention_bias"
        case attentionOutBias = "attention_out_bias"
        case mlpBias = "mlp_bias"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func d<T: Decodable>(_ k: CodingKeys, _ fallback: T) throws -> T {
            try c.decodeIfPresent(T.self, forKey: k) ?? fallback
        }
        self.modelType = try d(.modelType, "seed_oss")
        self.hiddenSize = try d(.hiddenSize, 5120)
        self.numHiddenLayers = try d(.numHiddenLayers, 64)
        self.intermediateSize = try d(.intermediateSize, 27648)
        self.numAttentionHeads = try d(.numAttentionHeads, 80)
        self.rmsNormEps = try d(.rmsNormEps, 1e-6)
        self.vocabSize = try d(.vocabSize, 155136)
        self.numKeyValueHeads = try d(.numKeyValueHeads, 8)
        self.headDim = try d(.headDim, 128)
        // Python's ModelArgs defaults this to None; 524288 is the real
        // checkpoint's value. Inert for `rope_type: "default"` (unused by
        // plain RoPE) — revisit if a scaling type that consumes it appears.
        self.maxPositionEmbeddings = try c.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings) ?? 524288
        self.attentionBias = try d(.attentionBias, false)
        self.attentionOutBias = try d(.attentionOutBias, false)
        self.mlpBias = try d(.mlpBias, false)
        self.ropeTheta = try d(.ropeTheta, 10000)
        self.ropeTraditional = try d(.ropeTraditional, false)
        self.ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.tieWordEmbeddings = try d(.tieWordEmbeddings, true)
    }
}

// MARK: - Attention (plain GQA, no q/k norm, two independent bias switches)

/// Grouped-query attention. The Seed-OSS-specific detail: the output projection
/// `o_proj` is biased by `attention_out_bias`, a switch INDEPENDENT of the
/// `attention_bias` that biases q/k/v — do not collapse them (that is the bug
/// the asymmetric-bias parity fixtures catch). No per-head q/k RMSNorm. RoPE is
/// a single family built from `rope_scaling` via the stock `initializeRope`
/// (Seed-OSS's `{"rope_type": "default"}` → plain RoPE at scale 1.0). Cache
/// update + SDPA go through the stock `attentionWithCacheUpdate` router.
/// Translated from the Python `Attention`.
final class SeedOssAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    let rope: RoPELayer

    init(_ config: SeedOssConfiguration) {
        let dim = config.hiddenSize
        let headDim = config.headDim
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.scale = pow(Float(headDim), -0.5)

        // input_bias = attention_bias (q/k/v); output_bias = attention_out_bias (o).
        let inputBias = config.attentionBias
        let outputBias = config.attentionOutBias
        self._wq.wrappedValue = Linear(dim, numHeads * headDim, bias: inputBias)
        self._wk.wrappedValue = Linear(dim, numKVHeads * headDim, bias: inputBias)
        self._wv.wrappedValue = Linear(dim, numKVHeads * headDim, bias: inputBias)
        self._wo.wrappedValue = Linear(numHeads * headDim, dim, bias: outputBias)

        self.rope = initializeRope(
            dims: headDim,
            base: config.ropeTheta,
            traditional: config.ropeTraditional,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
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

// MARK: - MLP (SwiGLU, optional bias)

/// The dense SwiGLU feed-forward: `down(silu(gate(x)) * up(x))`, with every
/// projection biased by `mlp_bias`. Translated from the Python `MLP`.
final class SeedOssMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ config: SeedOssConfiguration) {
        let dim = config.hiddenSize
        let hidden = config.intermediateSize
        let bias = config.mlpBias
        self._gate.wrappedValue = Linear(dim, hidden, bias: bias)
        self._down.wrappedValue = Linear(hidden, dim, bias: bias)
        self._up.wrappedValue = Linear(dim, hidden, bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

// MARK: - Decoder layer

/// One transformer block: pre-norm attention + residual, then pre-norm MLP +
/// residual. Translated from the Python `TransformerBlock`.
final class SeedOssDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SeedOssAttention
    @ModuleInfo(key: "mlp") var mlp: SeedOssMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: SeedOssConfiguration) {
        self._selfAttn.wrappedValue = SeedOssAttention(config)
        self._mlp.wrappedValue = SeedOssMLP(config)
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
/// Python `SeedModel`.
final class SeedOssModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [SeedOssDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: SeedOssConfiguration) {
        precondition(config.vocabSize > 0)
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map { _ in
            SeedOssDecoderLayer(config)
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

/// The Seed-OSS language model: the inner transformer plus the (untied) `lm_head`
/// projection to vocabulary logits. Conforms to `LLMModel` so the stock
/// `LLMModelFactory` (via `ModelOverlay` registration) loads and runs it. Being
/// a uniform dense model, it inherits the default `KVCacheDimensionProvider`
/// cache (uniform `KVCacheSimple`) — no `newCache` override (unlike Mellum2's
/// mixed sliding/full cache). Translated from the Python `Model`.
final class SeedOssModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    let config: SeedOssConfiguration
    let vocabularySize: Int
    var kvHeads: [Int]
    var model: SeedOssModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: SeedOssConfiguration) {
        self.config = config
        self.vocabularySize = config.vocabSize
        self.kvHeads = Array(
            repeating: config.numKeyValueHeads, count: config.numHiddenLayers)
        self.model = SeedOssModelInner(config)
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
    /// embeddings are tied (the untied real checkpoint keeps it). The checkpoint
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
