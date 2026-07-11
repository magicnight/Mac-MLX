import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// InternLM3 (Shanghai AI Lab InternLM3-8B-Instruct) — pure-Swift port (macMLX
// overlay architecture).
//
// A macMLX-owned model architecture (`model_type: internlm3`) registered into the
// stock mlx-swift-lm factory via `ModelOverlay` (no fork), following the
// `HunyuanV1Dense.swift` / `Cohere2.swift` / `MiniCPM3.swift` precedent. Upstream
// mlx-swift-lm has no `internlm3` type.
//
// InternLM3 is a conventional dense Llama-family decoder — GQA (an aggressive 16:1
// ratio: 32 query heads, 2 KV heads), SwiGLU MLP, RMSNorm, standard serial pre-norm
// blocks, standard causal mask — with exactly two architecture-specific quirks, and
// one deliberate correction of upstream defects:
//
//   • TWO independent bias switches, BOTH misleadingly named (pinned in the type
//     docs below): `qkv_bias` drives q/k/v AND o_proj (four projections), while
//     `bias` drives the MLP's gate/up/down (three projections). Neither name
//     matches the set of projections it actually gates.
//   • `head_dim` is ALWAYS `hidden_size / num_attention_heads` — the config's
//     explicit `head_dim` field is NOT consumed (mirroring upstream, which never
//     reads it; for the shipped 8B config `4096 / 32 == 128` happens to equal the
//     field, so the difference is invisible on real weights but pinned by fixtures).
//   • A DynamicNTK-scaling RoPE that INTENTIONALLY DIVERGES from mlx-lm's
//     `internlm3.py`, whose implementation carries four verified defects (see
//     `InternLM3DynamicNTKRoPE`). This port implements the corrected semantics
//     aligned with the reference `modeling_internlm3.py`; the parity fixtures are
//     captured from a MINIMALLY-PATCHED mlx-lm (`docs/reference/capture_internlm3.py`).
//
// Translated from Apple's Python mlx-lm reference (`mlx_lm/models/internlm3.py`,
// 0.31.3), CORRECTED per the four RoPE defects. End-to-end numerical parity is
// proven at 1e-4 against fixtures captured by `docs/reference/capture_internlm3.py`
// (see `InternLM3ModelParityTests`), with two adversarial configs whose every switch
// is inverted and whose RoPE path pins the corrections (A/B/C numerically; D — the
// linear-type static base — by the decode tests).
//
// The building blocks (`applyRotaryPosition`, `RoPELayer`, `MLXFast.RoPE`,
// `createAttentionMask`, `attentionWithCacheUpdate`, `silu`, `RMSNorm`) are all
// stock mlx-swift-lm / MLXNN public API — this file only wires them into InternLM3's
// topology (plus the bespoke corrected RoPE).

// MARK: - Configuration

/// `config.json` schema for `model_type: internlm3`. Field defaults mirror the
/// Python `ModelArgs` dataclass where it HAS defaults (`bias false`, `qkv_bias
/// false`, `max_position_embeddings 32768`, `rope_theta 10000`, `rope_traditional
/// false`, `tie_word_embeddings false`), while the structural dimensions the Python
/// dataclass leaves required fall back to the shipped InternLM3-8B-Instruct values
/// so a partial config still decodes.
///
/// INTENTIONAL DIVERGENCE (permissive decode — the SeedOss / Hunyuan / Cohere2 /
/// MiniCPM3 precedent): every field decodes with `decodeIfPresent ?? fallback`, and
/// unlike the Python `__post_init__` we do NOT raise when `rope_scaling` is present
/// but malformed (missing `factor`/`rope_type`, or a `rope_type` outside
/// {linear, dynamic}). Such a config is handled permissively as "no scaling" (plain
/// RoPE) — see ``ropePositionScale`` / ``ropeDynamicFactor``.
///
/// TWO defaults worth pinning (both covered by the decode tests):
///   • `rope_theta` defaults to the Python dataclass `10_000`. The shipped 8B config
///     OVERRIDES it to `50_000_000` (5e7), so the default is only a fallback — but
///     it is mirrored from the dataclass for consistency with the other ports.
///   • `num_key_value_heads`, when ABSENT, defaults to `num_attention_heads` (the
///     Python `__post_init__` rule) — NOT to any fixed constant.
public struct InternLM3Configuration: Codable, Sendable {
    public var modelType: String = "internlm3"
    public var hiddenSize: Int = 4096
    public var numHiddenLayers: Int = 48
    public var intermediateSize: Int = 10240
    public var numAttentionHeads: Int = 32
    public var rmsNormEps: Float = 1e-5
    public var vocabSize: Int = 128512
    /// Gates the MLP's gate/up/down projections (NOT the attention). Misleadingly
    /// named — see ``InternLM3MLP``. Python `ModelArgs.bias` default `false`.
    public var bias: Bool = false
    /// Gates the attention's q/k/v AND o projections (four of them). Misleadingly
    /// named — see ``InternLM3Attention``. Python `ModelArgs.qkv_bias` default
    /// `false`.
    public var qkvBias: Bool = false
    public var maxPositionEmbeddings: Int = 32768
    public var numKeyValueHeads: Int = 32
    /// Python dataclass default `10_000`; the shipped 8B config overrides to 5e7.
    public var ropeTheta: Float = 10000
    public var ropeTraditional: Bool = false
    public var ropeScaling: [String: StringOrNumber]? = nil
    public var tieWordEmbeddings: Bool = false

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case bias
        case qkvBias = "qkv_bias"
        case maxPositionEmbeddings = "max_position_embeddings"
        case numKeyValueHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        // NOTE: `head_dim` is deliberately absent — the field is NOT consumed (see
        // ``headDim``), mirroring the upstream module which never reads it.
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func d<T: Decodable>(_ k: CodingKeys, _ fallback: T) throws -> T {
            try c.decodeIfPresent(T.self, forKey: k) ?? fallback
        }
        self.modelType = try d(.modelType, "internlm3")
        self.hiddenSize = try d(.hiddenSize, 4096)
        self.numHiddenLayers = try d(.numHiddenLayers, 48)
        self.intermediateSize = try d(.intermediateSize, 10240)
        self.numAttentionHeads = try d(.numAttentionHeads, 32)
        self.rmsNormEps = try d(.rmsNormEps, 1e-5)
        self.vocabSize = try d(.vocabSize, 128512)
        self.bias = try d(.bias, false)
        self.qkvBias = try d(.qkvBias, false)
        self.maxPositionEmbeddings = try d(.maxPositionEmbeddings, 32768)
        // Python `__post_init__`: `num_key_value_heads` defaults to
        // `num_attention_heads` when absent — NOT a fixed constant.
        self.numKeyValueHeads =
            try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? self.numAttentionHeads
        self.ropeTheta = try d(.ropeTheta, 10000)
        self.ropeTraditional = try d(.ropeTraditional, false)
        self.ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.tieWordEmbeddings = try d(.tieWordEmbeddings, false)
    }

    /// The attention head dim actually used: ALWAYS `hidden_size /
    /// num_attention_heads`. The config's explicit `head_dim` field is NOT consumed
    /// (upstream never reads it) — a config carrying a bogus `head_dim` is ignored,
    /// which the decode tests pin.
    public var headDim: Int {
        hiddenSize / numAttentionHeads
    }

    /// The `rope_scaling.rope_type` string (lowercased), or `nil` when `rope_scaling`
    /// is absent / carries no string type. Only `"linear"` and `"dynamic"` are
    /// acted on; any other value is treated as no scaling (see ``ropePositionScale``
    /// / ``ropeDynamicFactor``).
    public var ropeScalingType: String? {
        guard case .string(let type)? = ropeScaling?["rope_type"] else { return nil }
        return type.lowercased()
    }

    /// The `rope_scaling.factor`, or `nil` when absent / non-numeric.
    public var ropeScalingFactor: Float? {
        ropeScaling?["factor"]?.asFloat()
    }

    /// The POSITION scale handed to the RoPE kernel:
    ///   • `linear` (with a non-zero factor) → `1 / factor` (positions compressed —
    ///     the ONLY upstream branch that was already correct);
    ///   • `dynamic`, no scaling, an unknown type, or a missing/zero factor → `1.0`
    ///     (positions unscaled).
    ///
    /// This is the fix for upstream defect **A**: mlx-lm's `internlm3.py` hard-codes
    /// this to `2.0` for every non-linear case (dynamic AND no-scaling), silently
    /// doubling every position. The reference `modeling_internlm3.py` never scales
    /// positions outside the linear branch.
    public var ropePositionScale: Float {
        if ropeScalingType == "linear", let factor = ropeScalingFactor, factor != 0 {
            return 1 / factor
        }
        return 1.0
    }

    /// The DynamicNTK factor that drives the per-call base recomputation, or `nil`
    /// when the base is static (`ropeTheta`). Present ONLY for `rope_type ==
    /// "dynamic"` with a numeric factor; `linear`, no scaling, and unknown types all
    /// keep a static base.
    ///
    /// This carries the fix for upstream defect **B**: mlx-lm's NTK base formula uses
    /// its (buggy `2.0`) position scale in place of the config `factor`, so the
    /// config `factor` (6.0 for the 8B) is NEVER consumed. Here the config factor is
    /// the sole driver of the dynamic base.
    public var ropeDynamicFactor: Float? {
        if ropeScalingType == "dynamic", let factor = ropeScalingFactor {
            return factor
        }
        return nil
    }

    /// The RoPE base for a given sequence length. For the `dynamic` type, once the
    /// sequence length exceeds `max_position_embeddings` the base is grown by the
    /// NTK formula `ropeTheta * ((factor * seqLen / maxPos) - (factor - 1)) **
    /// (d / (d - 2))` — with `factor` from config (defect **B**) and `seqLen` from
    /// the SEQUENCE axis (defect **C**, enforced by the callers). Otherwise (and for
    /// every non-dynamic path) the base is the static `ropeTheta`.
    ///
    /// The scalar arithmetic is done in `Double` (then rounded to `Float`) to match
    /// the Python reference's scalar precision before it enters the float32 kernel —
    /// the same discipline `HunyuanV1Dense` uses for its base scaling.
    public func ropeBase(sequenceLength: Int) -> Float {
        guard let factor = ropeDynamicFactor, sequenceLength > maxPositionEmbeddings else {
            return ropeTheta
        }
        let d = Double(headDim)
        let f = Double(factor)
        let scaled = (f * Double(sequenceLength) / Double(maxPositionEmbeddings)) - (f - 1)
        return Float(Double(ropeTheta) * Foundation.pow(scaled, d / (d - 2)))
    }
}

// MARK: - RoPE (corrected DynamicNTK scaling)

/// InternLM3's rotary positional encoding with Dynamic NTK scaling — a bespoke
/// module because it INTENTIONALLY DIVERGES from mlx-lm's `internlm3.py`
/// `DynamicNTKScalingRoPE`, aligning instead with the reference
/// `modeling_internlm3.py`.
///
/// THREE VERIFIED UPSTREAM DEFECTS this port corrects (all confirmed against
/// InternLM's official HuggingFace `modeling_internlm3.py`; upstream mlx-lm `main`
/// still carries them — an issue is pending, see the header of
/// `docs/reference/capture_internlm3.py`):
///
///   A. **Position scale hard-coded to `2.0`.** Upstream sets `rope_scale = 1/factor`
///      only for the `linear` type and `2.0` for everything else (dynamic AND no
///      scaling), then passes that as `mx.fast.rope(scale=…)` — silently DOUBLING
///      every position on the common path. The reference never scales positions
///      outside the linear branch. Fixed by ``InternLM3Configuration/ropePositionScale``
///      (`1.0` for dynamic / none, `1/factor` for linear).
///   B. **The config `factor` is never consumed.** Upstream's NTK base formula uses
///      `self.scale` (the buggy `2.0`) where the config `factor` (6.0 for the 8B)
///      belongs, so the configured scaling factor has NO effect. Fixed by
///      ``InternLM3Configuration/ropeDynamicFactor`` feeding
///      ``InternLM3Configuration/ropeBase(sequenceLength:)``.
///   C. **Sequence length read off the heads axis.** Upstream computes `seq_len =
///      x.shape[1] + offset`, but at the call site `x` is `[B, H, L, D]`, so
///      `shape[1]` is the HEAD count, not the sequence length. Fixed here by reading
///      the sequence axis (`x.dim(ndim - 2)`).
///
/// On the real 8B (seq ≤ `max_position_embeddings` = 32768) the dynamic branch never
/// engages, so the corrected RoPE is a plain rotary encoding at `base = 5e7`, scale
/// `1.0` — differing from the buggy upstream only by not doubling positions. The
/// dynamic base is recomputed per call (defect B/C corrected); the common
/// `seqLen ≤ maxPos` path takes the static-base fast route through ``ropeBase``.
///
/// Conforms to `RoPELayer` (`OffsetLayer & ArrayOffsetLayer`) so it plugs into the
/// stock `applyRotaryPosition` helper the same way every other overlay port's RoPE
/// does.
final class InternLM3DynamicNTKRoPE: Module, RoPELayer {
    /// Held for ``InternLM3Configuration/ropeBase(sequenceLength:)`` and the
    /// dims / position-scale / traditional flags — this module is InternLM3-specific,
    /// so carrying its config is idiomatic and keeps a single source of truth for the
    /// corrected RoPE semantics.
    let config: InternLM3Configuration

    init(_ config: InternLM3Configuration) {
        self.config = config
        super.init()
    }

    /// Scalar-offset path (the standard scalar `KVCache` route): the sequence length
    /// for the NTK threshold is the SEQUENCE axis (`ndim - 2`) plus the offset —
    /// defect **C** corrected (upstream read the heads axis).
    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        let seqLen = x.dim(x.ndim - 2) + offset
        return MLXFast.RoPE(
            x,
            dimensions: config.headDim,
            traditional: config.ropeTraditional,
            base: config.ropeBase(sequenceLength: seqLen),
            scale: config.ropePositionScale,
            offset: offset)
    }

    /// Per-sequence (batched) offset path — required by `ArrayOffsetLayer`. The NTK
    /// threshold needs a scalar sequence length, so the local chunk length (the
    /// sequence axis) is combined with the batch-max offset (`offset.max()`) to match
    /// the scalar path's `seqLen = local + offset` semantics. `max` is the correct
    /// reduction because the dynamic base is a SINGLE value shared across the whole
    /// batch: driving the threshold off the furthest-advanced sequence guarantees the
    /// recomputed base is large enough that NO batch member's positions overrun (a
    /// smaller per-member offset would only under-scale the base, never over-scale it).
    ///
    /// InternLM3 is NOT on the batched-decode allowlist, so this path is not exercised
    /// in practice today (the model runs on a scalar `KVCacheSimple`, whose `ropeOffset`
    /// is always scalar); when InternLM3 is added to that allowlist, this path must
    /// gain its own batched parity fixture.
    func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        let seqLen = x.dim(x.ndim - 2) + offset.max().item(Int.self)
        return MLXFast.RoPE(
            x,
            dimensions: config.headDim,
            traditional: config.ropeTraditional,
            base: config.ropeBase(sequenceLength: seqLen),
            scale: config.ropePositionScale,
            offset: offset)
    }
}

// MARK: - Attention (GQA, qkv_bias drives q/k/v AND o)

/// Grouped-query attention. The InternLM3-specific detail is the misleading bias
/// name: `qkv_bias` drives ALL FOUR projections — q, k, v AND o_proj — despite
/// naming only q/k/v. (Upstream literally passes `bias=qkv_bias` to `o_proj`; the
/// MLP's separate `bias` switch is the one that gates gate/up/down.)
///
/// `head_dim` is `hidden_size / num_attention_heads` (the config `head_dim` field is
/// NOT read — see ``InternLM3Configuration/headDim``); `scale` is `head_dim ** -0.5`.
/// RoPE is the corrected ``InternLM3DynamicNTKRoPE`` applied to the transposed
/// `[B, H, L, D]` queries/keys. Cache update + SDPA go through the stock
/// `attentionWithCacheUpdate` router. Translated from the Python `Attention`.
final class InternLM3Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    let rope: InternLM3DynamicNTKRoPE

    init(_ config: InternLM3Configuration) {
        let dim = config.hiddenSize
        let headDim = config.headDim
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.scale = pow(Float(headDim), -0.5)

        // `qkv_bias` gates q/k/v AND o (the misleading-name trap): upstream passes
        // `bias=qkv_bias` to all four, including o_proj.
        let bias = config.qkvBias
        self._wq.wrappedValue = Linear(dim, numHeads * headDim, bias: bias)
        self._wk.wrappedValue = Linear(dim, numKVHeads * headDim, bias: bias)
        self._wv.wrappedValue = Linear(dim, numKVHeads * headDim, bias: bias)
        self._wo.wrappedValue = Linear(numHeads * headDim, dim, bias: bias)

        self.rope = InternLM3DynamicNTKRoPE(config)
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

        // The corrected RoPE reads the sequence length off the SEQUENCE axis of the
        // now `[B, H, L, D]` tensors (defect C corrected) — the offset comes from the
        // layer's own cache.
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

// MARK: - MLP (SwiGLU, `bias` drives gate/up/down)

/// The dense SwiGLU feed-forward: `down(silu(gate(x)) * up(x))`. The InternLM3
/// twist is the OTHER misleading bias name: the `bias` switch (distinct from the
/// attention's `qkv_bias`) drives ALL THREE MLP projections — gate, up AND down.
/// Upstream constructs `MLP(dim, hidden, bias)` and passes that flag to every
/// `nn.Linear`. Translated from the Python `MLP` (`swiglu(gate(x), up(x))`).
final class InternLM3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ config: InternLM3Configuration) {
        let dim = config.hiddenSize
        let hidden = config.intermediateSize
        // `bias` gates gate/up/down (the second misleading-name trap).
        let bias = config.bias
        self._gate.wrappedValue = Linear(dim, hidden, bias: bias)
        self._down.wrappedValue = Linear(hidden, dim, bias: bias)
        self._up.wrappedValue = Linear(dim, hidden, bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

// MARK: - Decoder layer (standard serial pre-norm)

/// One transformer block: pre-norm attention + residual, then pre-norm MLP +
/// residual — the conventional Llama/Hunyuan serial block (NOT Cohere's parallel
/// residual). Translated from the Python `TransformerBlock`.
final class InternLM3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: InternLM3Attention
    @ModuleInfo(key: "mlp") var mlp: InternLM3MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: InternLM3Configuration) {
        self._selfAttn.wrappedValue = InternLM3Attention(config)
        self._mlp.wrappedValue = InternLM3MLP(config)
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

/// The transformer stack. A single full-causal mask is built from the first layer's
/// cache and shared by every (uniform, dense) layer. Translated from the Python
/// `InternLM2Model` (the class the upstream `internlm3.py` reuses verbatim).
final class InternLM3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [InternLM3DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: InternLM3Configuration) {
        precondition(config.vocabSize > 0)
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map { _ in
            InternLM3DecoderLayer(config)
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

/// The InternLM3 language model: the inner transformer plus the language head.
/// Conforms to `LLMModel` so the stock `LLMModelFactory` (via `ModelOverlay`
/// registration) loads and runs it. Being a uniform dense model, it inherits the
/// default `KVCacheDimensionProvider` cache (uniform `KVCacheSimple`) — no `newCache`
/// override. When `tie_word_embeddings` the logits are the embedding matrix applied
/// as a linear; otherwise an untied `lm_head` (the shipped 8B checkpoint is untied).
/// Translated from the Python `Model`.
final class InternLM3Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    let config: InternLM3Configuration
    let vocabularySize: Int
    let kvHeads: [Int]
    let model: InternLM3ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: InternLM3Configuration) {
        self.config = config
        self.vocabularySize = config.vocabSize
        self.kvHeads = Array(
            repeating: config.numKeyValueHeads, count: config.numHiddenLayers)
        self.model = InternLM3ModelInner(config)
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

    /// Ports the Python `Model.sanitize`: drop the InternLM2-legacy precomputed
    /// rotary frequencies (`…attention.rope.inv_freq`) that some checkpoints carry
    /// as non-parameter buffers; everything else passes through untouched. (Unlike
    /// Hunyuan, there is NO tie-driven `lm_head` drop — upstream never removes it.)
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { !$0.key.contains("attention.rope.inv_freq") }
    }

    var loraLayers: [Module] {
        model.layers.map { $0 }
    }
}
