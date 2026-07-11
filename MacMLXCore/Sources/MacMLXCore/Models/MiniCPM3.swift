import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// MiniCPM3 (OpenBMB MiniCPM3-4B) — pure-Swift port (macMLX overlay architecture).
//
// A macMLX-owned model architecture (`model_type: minicpm3`) registered into the
// stock mlx-swift-lm factory via `ModelOverlay` (no fork), following the
// `HunyuanV1Dense.swift` / `Cohere2.swift` precedent. Upstream mlx-swift-lm has no
// `minicpm3` type.
//
// MiniCPM3 is a Llama-family decoder with TWO defining departures:
//   • Multi-head Latent Attention (MLA), in the NON-ABSORBED (materialized) form.
//     Unlike DeepSeek V3.2's absorbed MLA (`DeepseekV32.swift`, which keeps K/V in
//     the latent space and uses per-head `embed_q`/`unembed_out`), MiniCPM3
//     materializes full multi-head Q/K/V from low-rank latents via `q_b_proj` /
//     `kv_b_proj`, then runs a plain GQA-shaped SDPA. Queries/keys carry a
//     nope+rope split (64+32), the single-head RoPE key `k_pe` is broadcast to
//     every head, and V has its OWN head dim (`hidden/heads` = 64, distinct from
//     the 96-wide Q/K). See `MiniCPM3Attention`.
//   • muP (maximal-update parametrization) — THREE numerical scalings that are each
//     a parity chokepoint: (a) the token embedding is multiplied by `scale_emb`;
//     (b) every layer scales BOTH residual branches by `scale_depth / √num_layers`;
//     (c) when the head is UNTIED, the hidden state is divided by
//     `hidden_size / dim_model_base` before `lm_head`. See `MiniCPM3DecoderLayer`
//     and `MiniCPM3Model`.
//
// RoPE is `SuScaledRoPE` (longrope) — the SAME stock `SuScaledRoPE` module Phi3
// uses. Its `_freqs = long_factor[] * base^(2i/d)` (a per-dimension array, so it
// CANNOT be folded into a single base the way `HunyuanV1Dense`'s alpha scaling was)
// is computed with a graph `pow` on BOTH the Python and Swift sides — the two are
// structurally identical, so parity holds at 1e-4. Note MiniCPM3 ALWAYS uses
// `long_factor` (short_factor is decoded but ignored, mirroring the upstream
// `SuScaledRoPE`), and the mscale is `1.0` when `max_position == original_max`
// (the shipped config) — the fixtures deliberately drive a `max > original` config
// to pin the non-trivial mscale formula.
//
// Translated from Apple's Python mlx-lm reference (`mlx_lm/models/minicpm3.py`,
// 0.31.3). End-to-end numerical parity is proven at 1e-4 against fixtures captured
// by `docs/reference/capture_minicpm3.py` (see `MiniCPM3ModelParityTests`), with
// two adversarial configs whose every switch/scaling is inverted (attention_bias,
// tie_word_embeddings, scale_emb, scale_depth, and the longrope mscale path).
//
// The building blocks (`SuScaledRoPE`, `applyRotaryPosition`, `RMSNorm`,
// `createAttentionMask`, `attentionWithCacheUpdate`, `KVCacheSimple`, `silu`) are
// all stock mlx-swift-lm / MLXNN public API — this file only wires them into
// MiniCPM3's topology.

// MARK: - Configuration

/// `config.json` schema for `model_type: minicpm3`. Field defaults mirror the
/// Python `ModelArgs` dataclass where it HAS defaults (`rope_theta 1000000.0`,
/// `attention_bias false`, `rope_traditional false`, `tie_word_embeddings false`),
/// while the structural dimensions the Python dataclass leaves required fall back
/// to the shipped MiniCPM3-4B values so a partial config still decodes.
///
/// INTENTIONAL DIVERGENCE (permissive decode — the SeedOss / Hunyuan / Cohere2
/// precedent): every field decodes with `decodeIfPresent ?? fallback`.
///
/// TWO defaults that are easy to get wrong and are pinned by the decode tests:
///   • `rope_theta` defaults to `1_000_000.0` (the Python dataclass default), NOT
///     the Llama-lineage `10_000`. The shipped `config.json` OMITS `rope_theta`, so
///     the default is what actually runs.
///   • `tie_word_embeddings` defaults to `false` → the shipped checkpoint is
///     UNTIED (it carries an `lm_head`, and the head input is divided by
///     `hidden_size / dim_model_base` — see `MiniCPM3Model`).
public struct MiniCPM3Configuration: Codable, Sendable {
    public var modelType: String = "minicpm3"
    public var hiddenSize: Int = 2560
    public var dimModelBase: Int = 256
    public var numHiddenLayers: Int = 62
    public var intermediateSize: Int = 6400
    public var numAttentionHeads: Int = 40
    public var rmsNormEps: Float = 1e-5
    public var vocabSize: Int = 73448
    public var numKeyValueHeads: Int = 40
    public var qLoraRank: Int = 768
    public var qkNopeHeadDim: Int = 64
    public var qkRopeHeadDim: Int = 32
    public var kvLoraRank: Int = 256
    public var scaleDepth: Float = 1.4
    public var scaleEmb: Float = 12
    public var maxPositionEmbeddings: Int = 32768
    public var attentionBias: Bool = false
    public var ropeTheta: Float = 1_000_000.0
    public var ropeTraditional: Bool = false
    public var ropeScaling: [String: StringOrNumber]? = nil
    public var tieWordEmbeddings: Bool = false

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case dimModelBase = "dim_model_base"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case numKeyValueHeads = "num_key_value_heads"
        case qLoraRank = "q_lora_rank"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case kvLoraRank = "kv_lora_rank"
        case scaleDepth = "scale_depth"
        case scaleEmb = "scale_emb"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionBias = "attention_bias"
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
        self.modelType = try d(.modelType, "minicpm3")
        self.hiddenSize = try d(.hiddenSize, 2560)
        self.dimModelBase = try d(.dimModelBase, 256)
        self.numHiddenLayers = try d(.numHiddenLayers, 62)
        self.intermediateSize = try d(.intermediateSize, 6400)
        self.numAttentionHeads = try d(.numAttentionHeads, 40)
        self.rmsNormEps = try d(.rmsNormEps, 1e-5)
        self.vocabSize = try d(.vocabSize, 73448)
        self.numKeyValueHeads = try d(.numKeyValueHeads, 40)
        self.qLoraRank = try d(.qLoraRank, 768)
        self.qkNopeHeadDim = try d(.qkNopeHeadDim, 64)
        self.qkRopeHeadDim = try d(.qkRopeHeadDim, 32)
        self.kvLoraRank = try d(.kvLoraRank, 256)
        self.scaleDepth = try d(.scaleDepth, 1.4)
        self.scaleEmb = try d(.scaleEmb, 12)
        self.maxPositionEmbeddings = try d(.maxPositionEmbeddings, 32768)
        self.attentionBias = try d(.attentionBias, false)
        // Dataclass default is 1_000_000.0 — NOT the Llama 10_000. The shipped
        // config omits the key, so this default is what runs.
        self.ropeTheta = try d(.ropeTheta, 1_000_000.0)
        self.ropeTraditional = try d(.ropeTraditional, false)
        self.ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.tieWordEmbeddings = try d(.tieWordEmbeddings, false)
    }

    /// The value head dim: `hidden_size / num_attention_heads` (64 in the shipped
    /// config). DISTINCT from the 96-wide Q/K head dim — V has no rope part.
    public var vHeadDim: Int {
        hiddenSize / numAttentionHeads
    }

    /// The Q/K head dim used by the attention scale and the Q reshape:
    /// `qk_nope_head_dim + qk_rope_head_dim` (96 in the shipped config).
    public var qHeadDim: Int {
        qkNopeHeadDim + qkRopeHeadDim
    }

    /// longrope `long_factor` — a per-dimension array of length `qk_rope_head_dim/2`.
    /// Mirrors the Python `rope_scaling.get("long_factor", 1.0)`: when absent, a
    /// scalar `[1.0]` (broadcast over the frequency table). `short_factor` is NOT
    /// consumed — the upstream `SuScaledRoPE` ignores it (MiniCPM3 ALWAYS applies
    /// `long_factor`), so it is intentionally not surfaced here.
    public var ropeLongFactor: [Float] {
        ropeScaling?["long_factor"]?.asFloats() ?? [1.0]
    }

    /// longrope `original_max_position_embeddings` — the pivot in the mscale
    /// formula. Mirrors the Python `.get("original_max_position_embeddings", 4096)`.
    public var ropeOriginalMaxPositionEmbeddings: Int {
        ropeScaling?["original_max_position_embeddings"]?.asInt() ?? 4096
    }
}

// MARK: - RoPE (SuScaledRoPE / longrope)

/// Builds MiniCPM3's longrope as the stock `SuScaledRoPE` — the exact module Phi3
/// uses. The Python `Attention` constructs `SuScaledRoPE` UNCONDITIONALLY (it never
/// dispatches on `rope_scaling["type"]`), so this constructs it directly rather
/// than routing through `initializeRope` (which would key off the type field and
/// `fatalError` on a missing `long_factor`). The `.get(default)` fallbacks
/// (`original_max → 4096`, `long_factor → [1.0]`) mirror the Python reference.
///
/// WHY THE STOCK MODULE MATCHES AT 1e-4 (unlike Hunyuan): both the Python and the
/// Swift `SuScaledRoPE` build `_freqs = long_factor[] * base^(2i/d)` with a graph
/// `pow` over the SAME arange, then call the rope kernel with explicit `freqs` and
/// `base: nil`. The two frequency tables are computed by structurally identical
/// graphs, so there is no graph-vs-kernel precision gap to drift on (the gap that
/// forced `HunyuanV1Dense` to fold its scaling into a plain-RoPE base instead).
func miniCPM3Rope(_ config: MiniCPM3Configuration) -> SuScaledRoPE {
    SuScaledRoPE(
        dimensions: config.qkRopeHeadDim,
        base: config.ropeTheta,
        maxPositionEmbeddings: config.maxPositionEmbeddings,
        originalMaxPositionEmbeddings: config.ropeOriginalMaxPositionEmbeddings,
        longFactor: config.ropeLongFactor)
}

// MARK: - Attention (non-absorbed Multi-head Latent Attention)

/// MiniCPM3 Multi-head Latent Attention, MATERIALIZED (non-absorbed) form.
///
/// Data flow (Python `Attention.__call__`):
///   1. `q = q_b_proj(q_a_layernorm(q_a_proj(x)))` → reshape `[B, H, L, qHeadDim]`,
///      split into `q_nope [.., 64]` and `q_pe [.., 32]`.
///   2. `kv_a_proj_with_mqa(x)` → split into `compressed_kv [B, L, 256]` and a
///      SINGLE-head `k_pe [B, L, 32]` (reshaped to `[B, 1, L, 32]`).
///   3. `kv = kv_b_proj(kv_a_layernorm(compressed_kv))` → reshape `[B, H, L, .]`,
///      split into `k_nope [.., 64]` and `values [.., 64]`.
///   4. RoPE applies ONLY to `q_pe` / `k_pe`; `k_pe` is then BROADCAST to all H
///      heads and concatenated with `k_nope` → `keys [B, H, L, 96]`. Queries are
///      `concat([q_nope, q_pe]) [B, H, L, 96]`.
///   5. SDPA at `scale = qHeadDim ** -0.5` (= 96^-0.5, NOT vHeadDim^-0.5) →
///      `[B, H, L, 64]` → o_proj.
///
/// THREE dims, THREE values (a classic MLA parity trap): `qHeadDim` (96) drives the
/// query reshape AND the softmax scale; `vHeadDim` (64) drives V and the o_proj
/// input; `qk_nope_head_dim` (64) is the split point. Getting the scale from
/// vHeadDim, or the reshape from the wrong dim, breaks parity.
///
/// BIAS ASYMMETRY: `attention_bias` gates ONLY `q_a_proj`, `kv_a_proj_with_mqa`,
/// and `o_proj`. `q_b_proj` and `kv_b_proj` are ALWAYS bias-free (the upstream
/// module never passes a bias flag to them) — the fixtures pin this asymmetry.
///
/// The `q_a_layernorm` / `kv_a_layernorm` are RMSNorms constructed WITHOUT an eps
/// argument upstream, so they use the mlx `nn.RMSNorm` library default `1e-5`
/// (which happens to equal the shipped `rms_norm_eps`, but is sourced from the
/// library default here to match the upstream semantics, NOT from `config`).
final class MiniCPM3Attention: Module {
    let numHeads: Int
    let qkNopeHeadDim: Int
    let qkRopeHeadDim: Int
    let qHeadDim: Int
    let vHeadDim: Int
    let kvLoraRank: Int
    let scale: Float

    @ModuleInfo(key: "q_a_proj") var qAProj: Linear
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    let rope: SuScaledRoPE

    init(_ config: MiniCPM3Configuration) {
        self.numHeads = config.numAttentionHeads
        self.qkNopeHeadDim = config.qkNopeHeadDim
        self.qkRopeHeadDim = config.qkRopeHeadDim
        self.qHeadDim = config.qHeadDim
        self.vHeadDim = config.vHeadDim
        self.kvLoraRank = config.kvLoraRank
        // softmax scale uses the FULL q/k head dim (nope + rope = 96), NOT vHeadDim.
        self.scale = pow(Float(config.qHeadDim), -0.5)

        // `attention_bias` gates ONLY these three projections; q_b/kv_b are always
        // bias-free (mirrors the upstream module's fixed bias=False on q_b/kv_b).
        let bias = config.attentionBias
        self._qAProj.wrappedValue = Linear(config.hiddenSize, config.qLoraRank, bias: bias)
        // No eps arg upstream → mlx `nn.RMSNorm` default 1e-5 (NOT config.rmsNormEps).
        self._qALayerNorm.wrappedValue = RMSNorm(dimensions: config.qLoraRank, eps: 1e-5)
        self._qBProj.wrappedValue = Linear(
            config.qLoraRank, config.numAttentionHeads * config.qHeadDim, bias: false)
        self._kvAProjWithMqa.wrappedValue = Linear(
            config.hiddenSize, config.kvLoraRank + config.qkRopeHeadDim, bias: bias)
        self._kvALayerNorm.wrappedValue = RMSNorm(dimensions: config.kvLoraRank, eps: 1e-5)
        // kv_b_proj emits, per head, the nope key part PLUS the value part.
        self._kvBProj.wrappedValue = Linear(
            config.kvLoraRank,
            config.numAttentionHeads * (config.qkNopeHeadDim + config.vHeadDim),
            bias: false)
        self._oProj.wrappedValue = Linear(
            config.numAttentionHeads * config.vHeadDim, config.hiddenSize, bias: bias)
        self.rope = miniCPM3Rope(config)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))

        // Q latent → materialized multi-head Q, split into nope / rope parts.
        var q = qBProj(qALayerNorm(qAProj(x)))
        q = q.reshaped(b, l, numHeads, qHeadDim).transposed(0, 2, 1, 3)  // [B, H, L, 96]
        let qSplit = split(q, indices: [qkNopeHeadDim], axis: -1)
        let qNope = qSplit[0]  // [B, H, L, 64]
        var qPe = qSplit[1]  // [B, H, L, 32]

        // KV latent → single-head RoPE key `k_pe`, plus the compressed latent that
        // materializes k_nope / values.
        let compressed = kvAProjWithMqa(x)
        let kvSplit = split(compressed, indices: [kvLoraRank], axis: -1)
        let compressedKv = kvSplit[0]  // [B, L, kvLora]
        var kPe = kvSplit[1].reshaped(b, l, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)  // [B, 1, L, 32]

        var kv = kvBProj(kvALayerNorm(compressedKv))
        kv = kv.reshaped(b, l, numHeads, qkNopeHeadDim + vHeadDim).transposed(0, 2, 1, 3)  // [B,H,L,128]
        let kvParts = split(kv, indices: [qkNopeHeadDim], axis: -1)
        let kNope = kvParts[0]  // [B, H, L, 64]
        let values = kvParts[1]  // [B, H, L, 64]

        // RoPE on the rope parts only (offset from the layer's own cache).
        let offset = cache?.ropeOffset
        qPe = applyRotaryPosition(rope, to: qPe, offset: offset)
        kPe = applyRotaryPosition(rope, to: kPe, offset: offset)

        // Broadcast the single-head RoPE key to all heads, then assemble the full
        // 96-wide Q/K. Values keep their own 64 head dim.
        let kPeBroadcast = broadcast(kPe, to: [b, numHeads, l, qkRopeHeadDim])
        let queries = concatenated([qNope, qPe], axis: -1)  // [B, H, L, 96]
        let keys = concatenated([kNope, kPeBroadcast], axis: -1)  // [B, H, L, 96]

        // Cache stores the full 96-wide keys and 64-wide values (KVCacheSimple
        // tracks the key/value head dims independently, so the mismatch is fine).
        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(b, l, -1)  // [B, L, H * vHeadDim]

        return oProj(output)
    }
}

// MARK: - MLP (SwiGLU, always bias-free)

/// The dense SwiGLU feed-forward: `down(silu(gate(x)) * up(x))`. Every projection
/// is bias-free. Translated from the Python `MLP` (`swiglu(gate(x), up(x))`).
final class MiniCPM3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ config: MiniCPM3Configuration) {
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

// MARK: - Decoder layer (muP depth scaling)

/// One transformer block: pre-norm attention + residual, then pre-norm MLP +
/// residual — BUT each residual branch is scaled by the muP depth factor
/// `scale_depth / √num_hidden_layers`. Translated from the Python
/// `DecoderLayer`:
///
/// ```
/// r = self_attn(input_layernorm(x), mask, cache)
/// h = x + r * (scale_depth / num_hidden_layers**0.5)
/// r = mlp(post_attention_layernorm(h))
/// out = h + r * (scale_depth / num_hidden_layers**0.5)
/// ```
///
/// BOTH branches carry the SAME `depthScale` — dropping it from either branch, or
/// applying it to only one, breaks parity. The two layer norms use
/// `config.rms_norm_eps` (distinct from the attention's INTERNAL q/kv-latent norms,
/// which use the mlx `nn.RMSNorm` default 1e-5).
final class MiniCPM3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MiniCPM3Attention
    @ModuleInfo(key: "mlp") var mlp: MiniCPM3MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    let depthScale: Float

    init(_ config: MiniCPM3Configuration) {
        self._selfAttn.wrappedValue = MiniCPM3Attention(config)
        self._mlp.wrappedValue = MiniCPM3MLP(config)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.depthScale = config.scaleDepth / pow(Float(config.numHiddenLayers), 0.5)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r * depthScale
        let r2 = mlp(postAttentionLayerNorm(h))
        return h + r2 * depthScale
    }
}

// MARK: - Model (inner: embed × scale_emb → layers → norm)

/// The transformer stack. The token embedding is multiplied by `scale_emb` (the
/// first of MiniCPM3's three muP scalings), then a single full-causal mask (from
/// the first layer's cache) is shared by every uniform layer. Translated from the
/// Python `MiniCPM3Model`.
final class MiniCPM3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [MiniCPM3DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let scaleEmb: Float

    init(_ config: MiniCPM3Configuration) {
        precondition(config.vocabSize > 0)
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map { _ in
            MiniCPM3DecoderLayer(config)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.scaleEmb = config.scaleEmb
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        // muP scaling #1: embedding × scale_emb.
        var h = embedTokens(inputs) * scaleEmb

        // Python `create_attention_mask(h, cache)` reads the FIRST layer's cache
        // offset; `cache?.first` is the equivalent single-cache source.
        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

// MARK: - Model (outer: LLMModel entry point)

/// The MiniCPM3 language model: the inner transformer plus the language head.
/// Conforms to `LLMModel` so the stock `LLMModelFactory` (via `ModelOverlay`
/// registration) loads and runs it. Being a uniform dense model, it inherits the
/// default `KVCacheDimensionProvider` cache (uniform `KVCacheSimple`) — no
/// `newCache` override.
///
/// muP scaling #3 (the head): when UNTIED (the shipped checkpoint), the hidden
/// state is DIVIDED by `hidden_size / dim_model_base` before `lm_head`. When TIED,
/// the logits are the embedding matrix applied as a linear WITHOUT that division —
/// the `/(hidden/dim_base)` factor is untied-only. Translated from the Python
/// `Model`. Upstream MiniCPM3 has NO `sanitize` (untied is the norm and the
/// checkpoint carries a plain `lm_head`), so none is added here; a tied checkpoint
/// simply must not ship an `lm_head` key.
final class MiniCPM3Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    let config: MiniCPM3Configuration
    let vocabularySize: Int
    let kvHeads: [Int]
    let model: MiniCPM3ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let headScaleDivisor: Float

    init(_ config: MiniCPM3Configuration) {
        self.config = config
        self.vocabularySize = config.vocabSize
        self.kvHeads = Array(
            repeating: config.numKeyValueHeads, count: config.numHiddenLayers)
        self.model = MiniCPM3ModelInner(config)
        // muP head divisor (untied path only): hidden_size / dim_model_base.
        self.headScaleDivisor = Float(config.hiddenSize) / Float(config.dimModelBase)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            // muP scaling #3: divide hidden by (hidden/dim_base) BEFORE the head.
            return lmHead(out / headScaleDivisor)
        } else {
            // Tied path: no muP head division (matches the upstream `else` branch).
            return model.embedTokens.asLinear(out)
        }
    }

    var loraLayers: [Module] {
        model.layers.map { $0 }
    }
}
