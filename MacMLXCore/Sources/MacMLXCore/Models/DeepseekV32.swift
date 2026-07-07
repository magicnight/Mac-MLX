import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// DeepSeek V3.2 — pure-Swift port (macMLX overlay architecture).
//
// First macMLX-owned model architecture, registered into the stock
// mlx-swift-lm factory via `ModelOverlay` (no fork). V3.2 introduces
// DSA sparse attention (the "lightning indexer") and is the foundation
// for a later DeepSeek V4 port.
//
// Translated from Apple's Python mlx-lm reference, preserved at
// `docs/reference/deepseek_v32_mlx_lm_reference.py`. See the porting
// plan at `docs/superpowers/plans/2026-05-11-deepseek-v32-port.md`.
//
// Build status: Configuration + MultiLinear + Indexer (S1). Attention,
// MoE, model assembly, and registration land in follow-up sessions
// (each gated on component-level numerical parity).

// MARK: - Configuration

/// `config.json` schema for `model_type: deepseek_v32`. Field defaults
/// mirror the Python `ModelArgs` dataclass so partial configs decode.
public struct DeepseekV32Configuration: Codable, Sendable {
    public var vocabSize: Int = 102400
    public var hiddenSize: Int = 4096
    public var indexHeadDim: Int = 128
    public var indexNHeads: Int = 64
    public var indexTopK: Int = 2048
    public var intermediateSize: Int = 11008
    public var moeIntermediateSize: Int = 1407
    public var numHiddenLayers: Int = 30
    public var numAttentionHeads: Int = 32
    public var numKeyValueHeads: Int = 32
    public var nSharedExperts: Int? = nil
    public var nRoutedExperts: Int? = nil
    public var routedScalingFactor: Float = 1.0
    public var kvLoraRank: Int = 512
    public var qLoraRank: Int = 1536
    public var qkRopeHeadDim: Int = 64
    public var vHeadDim: Int = 128
    public var qkNopeHeadDim: Int = 128
    public var topkMethod: String = "noaux_tc"
    public var scoringFunc: String = "sigmoid"
    public var normTopkProb: Bool = true
    public var nGroup: Int = 1
    public var topkGroup: Int = 1
    public var numExpertsPerTok: Int = 1
    public var moeLayerFreq: Int = 1
    public var firstKDenseReplace: Int = 0
    public var maxPositionEmbeddings: Int = 2048
    public var rmsNormEps: Float = 1e-6
    public var ropeTheta: Float = 10000.0
    public var ropeScaling: [String: StringOrNumber]? = nil
    public var attentionBias: Bool = false
    public var indexerRopeInterleave: Bool = false

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case indexHeadDim = "index_head_dim"
        case indexNHeads = "index_n_heads"
        case indexTopK = "index_topk"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case topkMethod = "topk_method"
        case scoringFunc = "scoring_func"
        case normTopkProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeLayerFreq = "moe_layer_freq"
        case firstKDenseReplace = "first_k_dense_replace"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
        case indexerRopeInterleave = "indexer_rope_interleave"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func d<T: Decodable>(_ k: CodingKeys, _ fallback: T) throws -> T {
            try c.decodeIfPresent(T.self, forKey: k) ?? fallback
        }
        self.vocabSize = try d(.vocabSize, 102400)
        self.hiddenSize = try d(.hiddenSize, 4096)
        self.indexHeadDim = try d(.indexHeadDim, 128)
        self.indexNHeads = try d(.indexNHeads, 64)
        self.indexTopK = try d(.indexTopK, 2048)
        self.intermediateSize = try d(.intermediateSize, 11008)
        self.moeIntermediateSize = try d(.moeIntermediateSize, 1407)
        self.numHiddenLayers = try d(.numHiddenLayers, 30)
        self.numAttentionHeads = try d(.numAttentionHeads, 32)
        self.numKeyValueHeads = try d(.numKeyValueHeads, 32)
        self.nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts)
        self.nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts)
        self.routedScalingFactor = try d(.routedScalingFactor, 1.0)
        self.kvLoraRank = try d(.kvLoraRank, 512)
        self.qLoraRank = try d(.qLoraRank, 1536)
        self.qkRopeHeadDim = try d(.qkRopeHeadDim, 64)
        self.vHeadDim = try d(.vHeadDim, 128)
        self.qkNopeHeadDim = try d(.qkNopeHeadDim, 128)
        self.topkMethod = try d(.topkMethod, "noaux_tc")
        self.scoringFunc = try d(.scoringFunc, "sigmoid")
        self.normTopkProb = try d(.normTopkProb, true)
        self.nGroup = try d(.nGroup, 1)
        self.topkGroup = try d(.topkGroup, 1)
        self.numExpertsPerTok = try d(.numExpertsPerTok, 1)
        self.moeLayerFreq = try d(.moeLayerFreq, 1)
        self.firstKDenseReplace = try d(.firstKDenseReplace, 0)
        self.maxPositionEmbeddings = try d(.maxPositionEmbeddings, 2048)
        self.rmsNormEps = try d(.rmsNormEps, 1e-6)
        self.ropeTheta = try d(.ropeTheta, 10000.0)
        self.ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.attentionBias = try d(.attentionBias, false)
        self.indexerRopeInterleave = try d(.indexerRopeInterleave, false)
    }
}

// MARK: - MultiLinear

/// Batched per-head linear (`[..., H, in] → [..., H, out]`), one weight
/// matrix per head. Copied from mlx-swift-lm's internal `MultiLinear`
/// (it isn't `public`, so the overlay can't import it). Used by the
/// absorbed-MLA `embed_q` / `unembed_out` projections in the attention.
final class DeepseekMultiLinear: Module, Quantizable {
    let inputDims: Int
    let outputDims: Int
    let numHeads: Int

    @ParameterInfo(key: "weight") var weight: MLXArray

    init(inputDims: Int, outputDims: Int, numHeads: Int) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numHeads = numHeads
        let scale = sqrt(1.0 / Float(inputDims))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [numHeads, outputDims, inputDims])
        super.init()
    }

    /// Per-head projection. `transpose == true` (the default) maps
    /// `inputDims → outputDims` (`x @ weightᵀ`); `transpose == false`
    /// maps `outputDims → inputDims` (`x @ weight`). The absorbed-MLA
    /// attention uses both directions: `embed_q` forward (true) and
    /// the prefill `embed_q`/`unembed_out` reverse (false). Mirrors
    /// mlx-lm's `mla.MultiLinear.__call__(x, transpose=True)`.
    func callAsFunction(_ x: MLXArray, transpose: Bool = true) -> MLXArray {
        transpose ? x.matmul(weight.swappedAxes(-1, -2)) : x.matmul(weight)
    }

    func toQuantized(groupSize: Int, bits: Int, mode: QuantizationMode) -> Module {
        // Overlay only targets pre-dequantized (bf16) weights for now;
        // quantized MultiLinear is a follow-up. Return self unquantized.
        self
    }
}

// MARK: - Indexer (DSA lightning indexer)

/// The novel piece of V3.2: a lightweight attention "indexer" that
/// scores every key against the query and returns the indices of the
/// top-`index_topk` keys, so the main attention only attends to a
/// sparse subset. Translated from the Python `Indexer` class.
///
/// - Note: The KV-cache integration (incremental decode) lands with the
///   attention module. This implements the prefill path (`cache == nil`,
///   offset 0). The short-circuit — when the context length is ≤
///   `indexTopK`, every key is kept, so the indexer returns `nil` and
///   the caller falls back to dense attention — is honored here.
final class DeepseekV32Indexer: Module {
    let nHeads: Int
    let headDim: Int
    let indexTopK: Int
    let softmaxScale: Float

    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "k_norm") var kNorm: LayerNorm
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear
    let rope: RoPELayer

    init(_ config: DeepseekV32Configuration) {
        self.nHeads = config.indexNHeads
        self.headDim = config.indexHeadDim
        self.indexTopK = config.indexTopK
        self.softmaxScale = pow(Float(config.indexHeadDim), -0.5)

        self._wqB.wrappedValue = Linear(
            config.qLoraRank, config.indexNHeads * config.indexHeadDim, bias: false)
        self._wk.wrappedValue = Linear(config.hiddenSize, config.indexHeadDim, bias: false)
        self._kNorm.wrappedValue = LayerNorm(dimensions: config.indexHeadDim)
        self._weightsProj.wrappedValue = Linear(
            config.hiddenSize, config.indexNHeads, bias: false)
        // RoPE interleave follows `indexer_rope_interleave` (default false
        // — non-interleaved). Stock mlx-lm 0.31.3 hardcoded traditional=true
        // here (a GLM5-era regression); upstream PR #1431 (2026-06-24)
        // restored the config-driven value because the hardcoded mode
        // silently degrades long-sequence quality. Our parity fixture is
        // captured against 0.31.3 + that one-line patch — see
        // docs/reference/capture_indexer.py. The main attention's rope
        // (S2) is unaffected: upstream keeps traditional=true there.
        self.rope = initializeRope(
            dims: config.qkRopeHeadDim,
            base: config.ropeTheta,
            traditional: config.indexerRopeInterleave,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    /// Returns the top-k key indices to attend to, shape
    /// `[b, 1, s, indexTopK]`, or `nil` when the context is short enough
    /// (`s <= indexTopK`) that no sparsification is needed.
    ///
    /// - Parameters:
    ///   - x: hidden states `[b, s, hidden]`
    ///   - qr: the query LoRA latent `[b, s, qLoraRank]` (shared with the
    ///     main attention's `q_a_layernorm(q_a_proj(x))`)
    ///   - mask: optional additive/boolean attention mask `[b, 1, s, s]`
    func callAsFunction(
        _ x: MLXArray, _ qr: MLXArray, _ mask: MLXArray?, cache: KVCache? = nil
    ) -> MLXArray? {
        let b = x.dim(0)
        let s = x.dim(1)

        var q = wqB(qr)
        q = q.reshaped(b, s, nHeads, headDim).swappedAxes(1, 2)  // [b, nHeads, s, headDim]
        var k = wk(x)
        k = kNorm(k)
        k = k.reshaped(b, 1, s, headDim)  // [b, 1, s, headDim]

        let offset = cache?.ropeOffset
        q = applyRotaryPosition(rope, to: q, offset: offset)
        k = applyRotaryPosition(rope, to: k, offset: offset)

        // Accumulate keys across decode steps. The indexer only needs
        // keys, so values are an empty (head_dim 0) placeholder — mirrors
        // the Python `cache.update_and_fetch(k, zeros([b,1,s,0]))`.
        if let cache {
            (k, _) = cache.update(keys: k, values: zeros([b, 1, s, 0]))
        }

        // Short-circuit: nothing to prune when every key (including any
        // cached from prior decode steps) already fits in top-k.
        if k.dim(2) <= indexTopK { return nil }

        var scores = matmul(q, k.swappedAxes(-1, -2))  // [b, nHeads, s, s]
        scores = maximum(scores, 0)
        var weights = weightsProj(x) * (pow(Float(nHeads), -0.5) * softmaxScale)  // [b, s, nHeads]
        weights = weights.swappedAxes(-1, -2).expandedDimensions(axis: -1)  // [b, nHeads, s, 1]
        scores = scores * weights
        scores = scores.sum(axis: 1, keepDims: true)  // [b, 1, s, s]
        if let mask {
            scores = MLX.where(mask, scores, MLXArray(-Float.infinity))
        }

        // Indices of the top `indexTopK` scores along the key axis.
        let part = argPartition(scores, kth: scores.dim(-1) - indexTopK, axis: -1)
        return part[.ellipsis, (part.dim(-1) - indexTopK)...]
    }
}

// MARK: - Attention (absorbed MLA + DSA sparse mask)

/// DeepSeek V3.2 Multi-head Latent Attention with the indexer's sparse
/// mask folded in.
///
/// "Absorbed" form: instead of materializing full K/V via `kv_b_proj`
/// (as stock DeepSeek-V3 does), it keeps K/V in the `kv_lora_rank`
/// latent space and uses the per-head `embed_q` / `unembed_out`
/// projections, computing the RoPE contribution (`peScores`) separately
/// and feeding it to SDPA as an additive mask — so the full attention
/// score is `scale·(qNope·kᵀ) + peScores` in one softmax.
///
/// Two forward branches:
/// - **prefill** (`L > 1`): `k = embed_q(kvLatent, transpose:false)`,
///   `v = unembed_out(kvLatent)`; when the context exceeds `index_topk`
///   the indexer scatters a sparse boolean mask.
/// - **decode** (`L == 1`): `qNope = embed_q(qNope)`, `k = v = kvLatent`;
///   the indexer gathers the top-k latent rows and the SDPA output is
///   projected back through `unembed_out`.
///
/// Translated from the Python `DeepseekV32Attention`. Parity is proven
/// per-branch: `DeepseekV32AttentionParityTests` covers prefill (S2.1);
/// sparse-prefill (S2.2) and decode+cache (S2.3) follow.
final class DeepseekV32Attention: Module {
    let numHeads: Int
    let qkNopeHeadDim: Int
    let qkRopeHeadDim: Int
    let qHeadDim: Int
    let kvLoraRank: Int
    let vHeadDim: Int
    let scale: Float

    @ModuleInfo(key: "q_a_proj") var qAProj: Linear
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "embed_q") var embedQ: DeepseekMultiLinear
    @ModuleInfo(key: "unembed_out") var unembedOut: DeepseekMultiLinear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "indexer") var indexer: DeepseekV32Indexer
    let rope: RoPELayer

    init(_ config: DeepseekV32Configuration) {
        self.numHeads = config.numAttentionHeads
        self.qkNopeHeadDim = config.qkNopeHeadDim
        self.qkRopeHeadDim = config.qkRopeHeadDim
        self.qHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim
        self.kvLoraRank = config.kvLoraRank
        self.vHeadDim = config.vHeadDim

        var scale = pow(Float(config.qkNopeHeadDim + config.qkRopeHeadDim), -0.5)
        // Optional YaRN mscale — matches the Python `mscale_all_dim` branch.
        if let ropeScaling = config.ropeScaling,
            let mscaleAllDim = ropeScaling["mscale_all_dim"]?.asFloat(), mscaleAllDim != 0,
            let factor = ropeScaling["factor"]?.asFloat(), factor > 1
        {
            let s = 0.1 * mscaleAllDim * log(factor) + 1.0
            scale = scale * s * s
        }
        self.scale = scale

        self._qAProj.wrappedValue = Linear(
            config.hiddenSize, config.qLoraRank, bias: config.attentionBias)
        self._qALayerNorm.wrappedValue = RMSNorm(dimensions: config.qLoraRank, eps: 1e-6)
        self._qBProj.wrappedValue = Linear(
            config.qLoraRank, config.numAttentionHeads * qHeadDim, bias: false)
        self._kvAProjWithMqa.wrappedValue = Linear(
            config.hiddenSize, config.kvLoraRank + config.qkRopeHeadDim,
            bias: config.attentionBias)
        self._kvALayerNorm.wrappedValue = RMSNorm(dimensions: config.kvLoraRank, eps: 1e-6)
        self._embedQ.wrappedValue = DeepseekMultiLinear(
            inputDims: config.qkNopeHeadDim, outputDims: config.kvLoraRank,
            numHeads: config.numAttentionHeads)
        self._unembedOut.wrappedValue = DeepseekMultiLinear(
            inputDims: config.kvLoraRank, outputDims: config.vHeadDim,
            numHeads: config.numAttentionHeads)
        self._oProj.wrappedValue = Linear(
            config.numAttentionHeads * config.vHeadDim, config.hiddenSize,
            bias: config.attentionBias)
        self._indexer.wrappedValue = DeepseekV32Indexer(config)
        // Main-attention RoPE stays traditional (interleaved) — upstream
        // keeps traditional=true here. This is DISTINCT from the indexer's
        // rope, which follows `indexer_rope_interleave` (default false).
        self.rope = initializeRope(
            dims: config.qkRopeHeadDim, base: config.ropeTheta, traditional: true,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXArray?, cache: CacheList?
    ) -> MLXArray {
        let b = x.dim(0)
        let l = x.dim(1)

        let qr = qALayerNorm(qAProj(x))
        var q = qBProj(qr)
        q = q.reshaped(b, l, numHeads, qHeadDim).swappedAxes(1, 2)  // [b,H,l,qHeadDim]
        let splitQ = split(q, indices: [qkNopeHeadDim], axis: -1)
        var qNope = splitQ[0]  // [b,H,l,qkNope]
        var qPe = splitQ[1]  // [b,H,l,qkRope]

        let compressed = kvAProjWithMqa(x)
        let splitKv = split(compressed, indices: [kvLoraRank], axis: -1)
        var kvLatent = kvALayerNorm(splitKv[0])  // [b,l,kvLora]
        var kPe = splitKv[1].reshaped(b, l, 1, qkRopeHeadDim).swappedAxes(1, 2)  // [b,1,l,qkRope]

        let offset = cache?[0].ropeOffset
        qPe = applyRotaryPosition(rope, to: qPe, offset: offset)
        kPe = applyRotaryPosition(rope, to: kPe, offset: offset)

        kvLatent = kvLatent.expandedDimensions(axis: 1)  // [b,1,l,kvLora]
        if let cache {
            (kvLatent, kPe) = cache[0].update(keys: kvLatent, values: kPe)
        }

        // Indexer top-k selection (threads cache[1] so the indexer's keys
        // accumulate across decode steps). Returns nil when the total key
        // count <= index_topk → dense attention.
        let topk = indexer(x, qr, mask, cache: cache?[1])
        var effMask = mask
        if let topk {
            if l == 1 {
                // Decode: gather the top-k latent rows along the key axis.
                let idx = topk[0..., 0..., 0, 0...].expandedDimensions(axis: -1)  // [b,1,topk,1]
                let kvIdx = broadcast(idx, to: Array(idx.shape.dropLast()) + [kvLatent.dim(-1)])
                kvLatent = takeAlong(kvLatent, kvIdx, axis: 2)
                let peIdx = broadcast(idx, to: Array(idx.shape.dropLast()) + [kPe.dim(-1)])
                kPe = takeAlong(kPe, peIdx, axis: 2)
                if let m = effMask { effMask = takeAlong(m, topk, axis: -1) }
            } else {
                // Prefill: scatter a sparse boolean mask over the key axis.
                var shape = topk.shape
                shape[shape.count - 1] = kvLatent.dim(2)
                var sparse = zeros(shape, type: Bool.self)
                sparse = putAlong(sparse, topk, values: MLXArray(true), axis: -1)
                if let m = effMask { sparse = sparse .&& m }
                effMask = sparse
            }
        }

        var peScores = matmul(qPe * scale, kPe.swappedAxes(-1, -2))  // [b,H,l,s]
        if let m = effMask {
            peScores = MLX.where(m, peScores, MLXArray(-Float.greatestFiniteMagnitude))
        }

        let k: MLXArray
        let v: MLXArray
        if l == 1 {
            qNope = embedQ(qNope)  // [b,H,l,kvLora]
            k = kvLatent
            v = kvLatent
        } else {
            k = embedQ(kvLatent, transpose: false)  // [b,H,s,qkNope]
            v = unembedOut(kvLatent)  // [b,H,s,vHead]
        }

        var output = scaledDotProductAttention(
            queries: qNope, keys: k, values: v, scale: scale, mask: peScores)
        if l == 1 {
            output = unembedOut(output)  // [b,H,l,vHead]
        }
        output = output.swappedAxes(1, 2).reshaped(b, l, -1)  // [b,l,H*vHead]
        return oProj(output)
    }
}
