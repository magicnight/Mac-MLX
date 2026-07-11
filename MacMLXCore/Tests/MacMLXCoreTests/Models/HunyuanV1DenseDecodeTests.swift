import Testing
import Foundation
@testable import MacMLXCore

/// `HunyuanV1DenseConfiguration` decode contract: the switches and RoPE knobs the
/// port branches on must decode from `config.json` with the documented permissive
/// fallbacks. Pure JSON decode — no MLX, no Metal — so it runs ungated under both
/// bare `swift test` and xcodebuild. The end-to-end effect of each switch on the
/// computed logits is proven separately by the adversarial `HunyuanV1Dense…
/// ParityTests` fixtures (which invert every switch across two configs).
@Suite("HunyuanV1Dense decode")
struct HunyuanV1DenseDecodeTests {

    private func decode(_ json: String) throws -> HunyuanV1DenseConfiguration {
        try JSONDecoder().decode(HunyuanV1DenseConfiguration.self, from: Data(json.utf8))
    }

    /// Explicit `head_dim` is taken verbatim; when it is NOT `hidden / heads`,
    /// `resolvedHeadDim` returns the explicit value (the o_proj-shape path the
    /// realistic fixture pins).
    @Test("explicit head_dim wins, even when != hidden/heads")
    func explicitHeadDim() throws {
        let config = try decode("""
        {
          "model_type": "hunyuan_v1_dense",
          "hidden_size": 64,
          "num_attention_heads": 4,
          "head_dim": 24
        }
        """)
        #expect(config.headDim == 24)
        #expect(config.resolvedHeadDim == 24)  // not hidden/heads == 16
    }

    /// Missing `head_dim` falls back to `hidden_size / num_attention_heads`.
    @Test("missing head_dim falls back to hidden/heads")
    func headDimFallback() throws {
        let config = try decode("""
        {
          "model_type": "hunyuan_v1_dense",
          "hidden_size": 64,
          "num_attention_heads": 4
        }
        """)
        #expect(config.headDim == nil)
        #expect(config.resolvedHeadDim == 16)
    }

    /// `rope_scaling.alpha` is consumed as the DynamicNTK scaling factor.
    @Test("rope_scaling.alpha is read")
    func ropeScalingAlpha() throws {
        let config = try decode("""
        {
          "model_type": "hunyuan_v1_dense",
          "rope_scaling": {"type": "dynamic", "alpha": 1000.0, "factor": 1.0}
        }
        """)
        #expect(config.scalingAlpha == 1000.0)
    }

    /// Missing `rope_scaling` → alpha 1.0 (plain RoPE, base unchanged).
    @Test("missing rope_scaling falls back to alpha 1.0")
    func ropeScalingMissingFallback() throws {
        let config = try decode("""
        { "model_type": "hunyuan_v1_dense" }
        """)
        #expect(config.ropeScaling == nil)
        #expect(config.scalingAlpha == 1.0)
    }

    /// A present `rope_scaling` WITHOUT `alpha` → alpha 1.0 (permissive decode:
    /// unlike the Python `__post_init__`, we do NOT raise on missing keys).
    @Test("rope_scaling without alpha falls back to 1.0, no raise")
    func ropeScalingWithoutAlpha() throws {
        let config = try decode("""
        {
          "model_type": "hunyuan_v1_dense",
          "rope_scaling": {"type": "dynamic", "factor": 1.0}
        }
        """)
        #expect(config.scalingAlpha == 1.0)
    }

    /// `tie_word_embeddings` decodes both states; the default (Python `ModelArgs`)
    /// is `false`.
    @Test(arguments: [true, false])
    func tieWordEmbeddingsBothStates(_ tie: Bool) throws {
        let config = try decode("""
        { "model_type": "hunyuan_v1_dense", "tie_word_embeddings": \(tie) }
        """)
        #expect(config.tieWordEmbeddings == tie)
    }

    /// `use_qk_norm` decodes both states; the default (Python `ModelArgs`) is
    /// `true`.
    @Test(arguments: [true, false])
    func useQkNormBothStates(_ qk: Bool) throws {
        let config = try decode("""
        { "model_type": "hunyuan_v1_dense", "use_qk_norm": \(qk) }
        """)
        #expect(config.useQkNorm == qk)
    }

    /// `attention_bias` decodes both states; the default (Python `ModelArgs`) is
    /// `false`.
    @Test(arguments: [true, false])
    func attentionBiasBothStates(_ bias: Bool) throws {
        let config = try decode("""
        { "model_type": "hunyuan_v1_dense", "attention_bias": \(bias) }
        """)
        #expect(config.attentionBias == bias)
    }

    /// Defaults land where the Python `ModelArgs` dataclass puts them when the
    /// keys are absent (the permissive-decode contract).
    @Test("dataclass defaults when keys absent")
    func dataclassDefaults() throws {
        let config = try decode("""
        { "model_type": "hunyuan_v1_dense" }
        """)
        #expect(config.useQkNorm == true)
        #expect(config.attentionBias == false)
        #expect(config.tieWordEmbeddings == false)
        #expect(config.ropeTheta == 10000)
        #expect(config.maxPositionEmbeddings == 32768)
    }
}
