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

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x.matmul(weight.swappedAxes(-1, -2))
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
        _ x: MLXArray, _ qr: MLXArray, _ mask: MLXArray?
    ) -> MLXArray? {
        let b = x.dim(0)
        let s = x.dim(1)

        var q = wqB(qr)
        q = q.reshaped(b, s, nHeads, headDim).swappedAxes(1, 2)  // [b, nHeads, s, headDim]
        var k = wk(x)
        k = kNorm(k)
        k = k.reshaped(b, 1, s, headDim)  // [b, 1, s, headDim]

        q = applyRotaryPosition(rope, to: q, offset: nil)
        k = applyRotaryPosition(rope, to: k, offset: nil)

        // Short-circuit: nothing to prune when every key fits in top-k.
        if s <= indexTopK { return nil }

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
