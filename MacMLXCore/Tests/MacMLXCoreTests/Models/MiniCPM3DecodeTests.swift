import Testing
import Foundation
@testable import MacMLXCore

/// `MiniCPM3Configuration` decode contract: the switches, muP scaling constants,
/// and longrope knobs the port branches on must decode from `config.json` with the
/// documented permissive fallbacks. Pure JSON decode — no MLX, no Metal — so it
/// runs ungated under both bare `swift test` and xcodebuild. The end-to-end effect
/// of each switch/scaling on the computed logits is proven separately by the
/// adversarial `MiniCPM3ModelParityTests` fixtures (which invert every one across
/// two configs).
@Suite("MiniCPM3 decode")
struct MiniCPM3DecodeTests {

    private func decode(_ json: String) throws -> MiniCPM3Configuration {
        try JSONDecoder().decode(MiniCPM3Configuration.self, from: Data(json.utf8))
    }

    /// Absolute-tolerance element compare for the longrope factor arrays (JSON
    /// number → Float can differ in the last bit from a Swift float literal).
    private func approxEqual(_ a: [Float], _ b: [Float], tol: Float = 1e-6) -> Bool {
        guard a.count == b.count else { return false }
        return zip(a, b).allSatisfy { abs($0 - $1) <= tol }
    }

    /// Defaults land where the Python `ModelArgs` dataclass puts them when the keys
    /// are absent (the permissive-decode contract). The structural dims fall back to
    /// the shipped MiniCPM3-4B values.
    @Test("dataclass defaults when keys absent")
    func dataclassDefaults() throws {
        let config = try decode("""
        { "model_type": "minicpm3" }
        """)
        #expect(config.hiddenSize == 2560)
        #expect(config.dimModelBase == 256)
        #expect(config.numHiddenLayers == 62)
        #expect(config.intermediateSize == 6400)
        #expect(config.numAttentionHeads == 40)
        #expect(config.numKeyValueHeads == 40)
        #expect(config.rmsNormEps == 1e-5)
        #expect(config.vocabSize == 73448)
        #expect(config.qLoraRank == 768)
        #expect(config.qkNopeHeadDim == 64)
        #expect(config.qkRopeHeadDim == 32)
        #expect(config.kvLoraRank == 256)
        #expect(config.scaleEmb == 12)
        #expect(config.scaleDepth == 1.4)
        #expect(config.maxPositionEmbeddings == 32768)
        #expect(config.attentionBias == false)
        #expect(config.ropeTraditional == false)
        #expect(config.tieWordEmbeddings == false)
        // Derived MLA dims.
        #expect(config.vHeadDim == 64)  // hidden / heads = 2560 / 40
        #expect(config.qHeadDim == 96)  // qk_nope + qk_rope = 64 + 32
    }

    /// `rope_theta` defaults to the Python dataclass `1_000_000.0`, NOT the
    /// Llama-lineage `10_000`. The shipped config OMITS the key, so this default is
    /// what runs — the single easiest value to get wrong.
    @Test("rope_theta defaults to 1_000_000 (NOT 10_000)")
    func ropeThetaDefault() throws {
        let config = try decode("""
        { "model_type": "minicpm3" }
        """)
        #expect(config.ropeTheta == 1_000_000.0)
    }

    /// An explicit `rope_theta` is read verbatim (the fixtures pin 10_000).
    @Test("rope_theta is read verbatim when present")
    func ropeThetaExplicit() throws {
        let config = try decode("""
        { "model_type": "minicpm3", "rope_theta": 10000.0 }
        """)
        #expect(config.ropeTheta == 10000.0)
    }

    /// `attention_bias` decodes both states; the default (Python `ModelArgs`) is
    /// `false`.
    @Test(arguments: [true, false])
    func attentionBiasBothStates(_ bias: Bool) throws {
        let config = try decode("""
        { "model_type": "minicpm3", "attention_bias": \(bias) }
        """)
        #expect(config.attentionBias == bias)
    }

    /// `tie_word_embeddings` decodes both states; the default is `false` (UNTIED —
    /// the shipped checkpoint carries an `lm_head`).
    @Test(arguments: [true, false])
    func tieWordEmbeddingsBothStates(_ tie: Bool) throws {
        let config = try decode("""
        { "model_type": "minicpm3", "tie_word_embeddings": \(tie) }
        """)
        #expect(config.tieWordEmbeddings == tie)
    }

    /// The muP scaling constants — `scale_emb`, `scale_depth`, `dim_model_base` —
    /// are read verbatim (each drives a distinct numerical scaling).
    @Test("muP scaling constants are read")
    func muPConstantsRead() throws {
        let config = try decode("""
        {
          "model_type": "minicpm3",
          "scale_emb": 3,
          "scale_depth": 0.7,
          "dim_model_base": 16
        }
        """)
        #expect(config.scaleEmb == 3)
        #expect(config.scaleDepth == 0.7)
        #expect(config.dimModelBase == 16)
    }

    /// longrope `long_factor` / `original_max_position_embeddings` are pulled from
    /// the `rope_scaling` dict verbatim when present.
    @Test("rope_scaling long_factor and original_max are read")
    func ropeScalingRead() throws {
        let config = try decode("""
        {
          "model_type": "minicpm3",
          "rope_scaling": {
            "type": "longrope",
            "long_factor": [1.2, 1.7],
            "short_factor": [9.9, 9.9],
            "original_max_position_embeddings": 8192
          }
        }
        """)
        #expect(approxEqual(config.ropeLongFactor, [1.2, 1.7]))
        #expect(config.ropeOriginalMaxPositionEmbeddings == 8192)
    }

    /// `original_max_position_embeddings` falls back to `4096` when the
    /// `rope_scaling` dict omits it (Python `.get(..., 4096)`).
    @Test("rope_scaling missing original_max falls back to 4096")
    func ropeScalingOriginalMaxFallback() throws {
        let config = try decode("""
        {
          "model_type": "minicpm3",
          "rope_scaling": { "type": "longrope", "long_factor": [1.5] }
        }
        """)
        #expect(config.ropeOriginalMaxPositionEmbeddings == 4096)
    }

    /// When `rope_scaling` is absent entirely, `long_factor` falls back to the
    /// scalar `[1.0]` and `original_max` to `4096` (Python `.get` defaults) — a
    /// plain, unscaled rotary frequency table.
    @Test("rope_scaling absent: long_factor [1.0], original_max 4096")
    func ropeScalingAbsentFallbacks() throws {
        let config = try decode("""
        { "model_type": "minicpm3" }
        """)
        #expect(config.ropeScaling == nil)
        #expect(approxEqual(config.ropeLongFactor, [1.0]))
        #expect(config.ropeOriginalMaxPositionEmbeddings == 4096)
    }
}
