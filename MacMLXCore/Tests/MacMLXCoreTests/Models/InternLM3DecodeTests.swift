import Testing
import Foundation
@testable import MacMLXCore

/// `InternLM3Configuration` decode contract: the switches and RoPE knobs the port
/// branches on must decode from `config.json` with the documented permissive
/// fallbacks, and the corrected DynamicNTK RoPE resolution (position scale, dynamic
/// factor, and the per-length base) must compute the reference values. Pure Swift —
/// no MLX, no Metal — so it runs ungated under both bare `swift test` and xcodebuild.
/// The end-to-end effect of each switch on the computed logits is proven separately
/// by the adversarial `InternLM3ModelParityTests` fixtures.
@Suite("InternLM3 decode")
struct InternLM3DecodeTests {

    private func decode(_ json: String) throws -> InternLM3Configuration {
        try JSONDecoder().decode(InternLM3Configuration.self, from: Data(json.utf8))
    }

    /// Defaults land where the Python `ModelArgs` dataclass puts them when the keys
    /// are absent; the structural dims fall back to the shipped InternLM3-8B values.
    @Test("dataclass defaults when keys absent")
    func dataclassDefaults() throws {
        let config = try decode("""
        { "model_type": "internlm3" }
        """)
        #expect(config.hiddenSize == 4096)
        #expect(config.numHiddenLayers == 48)
        #expect(config.intermediateSize == 10240)
        #expect(config.numAttentionHeads == 32)
        #expect(config.rmsNormEps == 1e-5)
        #expect(config.vocabSize == 128512)
        #expect(config.bias == false)
        #expect(config.qkvBias == false)
        #expect(config.maxPositionEmbeddings == 32768)
        #expect(config.ropeTheta == 10000)
        #expect(config.ropeTraditional == false)
        #expect(config.tieWordEmbeddings == false)
        // num_key_value_heads absent → defaults to num_attention_heads.
        #expect(config.numKeyValueHeads == 32)
        // head_dim is always hidden/heads = 4096/32 = 128.
        #expect(config.headDim == 128)
    }

    /// `num_key_value_heads`, when ABSENT, defaults to `num_attention_heads` (the
    /// Python `__post_init__` rule) — tracking a NON-default head count.
    @Test("num_key_value_heads absent defaults to num_attention_heads")
    func kvHeadsDefaultsToAttentionHeads() throws {
        let config = try decode("""
        { "model_type": "internlm3", "num_attention_heads": 20 }
        """)
        #expect(config.numKeyValueHeads == 20)
    }

    /// An explicit `num_key_value_heads` is read verbatim (the shipped 8B uses 2 —
    /// an aggressive 16:1 GQA).
    @Test("explicit num_key_value_heads is read verbatim")
    func kvHeadsExplicit() throws {
        let config = try decode("""
        { "model_type": "internlm3", "num_attention_heads": 32, "num_key_value_heads": 2 }
        """)
        #expect(config.numKeyValueHeads == 2)
    }

    /// The config's explicit `head_dim` field is NOT consumed — `headDim` is always
    /// `hidden_size / num_attention_heads`, even when the JSON carries a bogus value
    /// (the decoy the `dynamic_active` parity fixture relies on).
    @Test("explicit head_dim field is ignored (always hidden/heads)")
    func headDimNotConsumed() throws {
        let config = try decode("""
        {
          "model_type": "internlm3",
          "hidden_size": 32,
          "num_attention_heads": 4,
          "head_dim": 999
        }
        """)
        #expect(config.headDim == 8)  // 32 / 4, NOT 999
    }

    /// `rope_theta` is read verbatim when present (the shipped 8B overrides the
    /// dataclass default to 5e7).
    @Test("rope_theta is read verbatim when present")
    func ropeThetaExplicit() throws {
        let config = try decode("""
        { "model_type": "internlm3", "rope_theta": 50000000.0 }
        """)
        #expect(config.ropeTheta == 50_000_000.0)
    }

    /// `bias` (MLP gate/up/down) decodes both states; default false.
    @Test(arguments: [true, false])
    func biasBothStates(_ bias: Bool) throws {
        let config = try decode("""
        { "model_type": "internlm3", "bias": \(bias) }
        """)
        #expect(config.bias == bias)
    }

    /// `qkv_bias` (attention q/k/v AND o) decodes both states; default false.
    @Test(arguments: [true, false])
    func qkvBiasBothStates(_ bias: Bool) throws {
        let config = try decode("""
        { "model_type": "internlm3", "qkv_bias": \(bias) }
        """)
        #expect(config.qkvBias == bias)
    }

    /// `tie_word_embeddings` decodes both states; default false (the shipped 8B is
    /// UNTIED — it carries an `lm_head`).
    @Test(arguments: [true, false])
    func tieWordEmbeddingsBothStates(_ tie: Bool) throws {
        let config = try decode("""
        { "model_type": "internlm3", "tie_word_embeddings": \(tie) }
        """)
        #expect(config.tieWordEmbeddings == tie)
    }

    // MARK: - Corrected DynamicNTK RoPE resolution

    /// `dynamic` scaling: position scale is 1.0 (defect A — NOT the buggy 2.0), the
    /// config factor is the dynamic-base driver (defect B), and the base grows by the
    /// NTK formula once the sequence length exceeds `max_position_embeddings`.
    @Test("dynamic: scale 1.0, factor consumed, NTK base fires past max_position")
    func ropeDynamic() throws {
        let config = try decode("""
        {
          "model_type": "internlm3",
          "hidden_size": 32,
          "num_attention_heads": 4,
          "rope_theta": 10000.0,
          "max_position_embeddings": 6,
          "rope_scaling": { "rope_type": "dynamic", "factor": 4.0 }
        }
        """)
        #expect(config.ropeScalingType == "dynamic")
        #expect(config.ropeScalingFactor == 4.0)
        #expect(config.ropePositionScale == 1.0)  // defect A: no position doubling
        #expect(config.ropeDynamicFactor == 4.0)  // defect B: config factor consumed

        // seqLen ≤ max_position → static base (the common fast path).
        #expect(config.ropeBase(sequenceLength: 6) == 10000.0)
        #expect(config.ropeBase(sequenceLength: 5) == 10000.0)

        // seqLen > max_position → NTK base with the CONFIG factor (4), head_dim 8.
        //   base = theta * ((f*seqLen/maxPos) - (f-1)) ** (d/(d-2))
        let f = 4.0, seqLen = 8.0, maxPos = 6.0, d = 8.0
        let expected = Float(10000.0 * Foundation.pow((f * seqLen / maxPos) - (f - 1), d / (d - 2)))
        #expect(abs(config.ropeBase(sequenceLength: 8) - expected) < 0.5)

        // A port that reused the (buggy 2.0) position scale in place of the factor
        // would compute a materially smaller base; the config factor (4) must win.
        let buggyFactor2 = Float(
            10000.0 * Foundation.pow((2.0 * seqLen / maxPos) - 1.0, d / (d - 2)))
        #expect(config.ropeBase(sequenceLength: 8) > buggyFactor2 + 1000)
    }

    /// `linear` scaling: position scale is 1/factor (the one branch upstream got
    /// right), and the base is STATIC `rope_theta` even for very long sequences (no
    /// dynamic NTK growth on the linear path).
    @Test("linear: scale 1/factor, static base")
    func ropeLinear() throws {
        let config = try decode("""
        {
          "model_type": "internlm3",
          "rope_theta": 10000.0,
          "max_position_embeddings": 6,
          "rope_scaling": { "rope_type": "linear", "factor": 4.0 }
        }
        """)
        #expect(config.ropeScalingType == "linear")
        #expect(config.ropePositionScale == 0.25)  // 1 / 4
        #expect(config.ropeDynamicFactor == nil)  // linear keeps a static base
        #expect(config.ropeBase(sequenceLength: 8) == 10000.0)  // no NTK growth
        #expect(config.ropeBase(sequenceLength: 100000) == 10000.0)
    }

    /// No `rope_scaling`: plain RoPE — position scale 1.0 (defect A on the no-scaling
    /// path: the buggy upstream would use 2.0 here) and a static base.
    @Test("no rope_scaling: plain rope, scale 1.0, static base")
    func ropeAbsent() throws {
        let config = try decode("""
        { "model_type": "internlm3", "rope_theta": 10000.0 }
        """)
        #expect(config.ropeScaling == nil)
        #expect(config.ropeScalingType == nil)
        #expect(config.ropePositionScale == 1.0)
        #expect(config.ropeDynamicFactor == nil)
        #expect(config.ropeBase(sequenceLength: 999_999) == 10000.0)
    }

    /// An UNKNOWN `rope_type` is handled permissively (no raise — the intentional
    /// divergence from the Python `__post_init__`): treated as no scaling (plain
    /// RoPE, scale 1.0, static base).
    @Test("unknown rope_type is permissive: treated as no scaling")
    func ropeUnknownTypePermissive() throws {
        let config = try decode("""
        {
          "model_type": "internlm3",
          "rope_theta": 10000.0,
          "max_position_embeddings": 6,
          "rope_scaling": { "rope_type": "yarn", "factor": 4.0 }
        }
        """)
        #expect(config.ropeScalingType == "yarn")
        #expect(config.ropePositionScale == 1.0)
        #expect(config.ropeDynamicFactor == nil)
        #expect(config.ropeBase(sequenceLength: 8) == 10000.0)  // no NTK growth
    }
}
