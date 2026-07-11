import Testing
import Foundation
@testable import MacMLXCore

/// `Cohere2Configuration` decode contract: the switches and interleave knobs the
/// port branches on must decode from `config.json` with the documented permissive
/// fallbacks. Pure JSON decode — no MLX, no Metal — so it runs ungated under both
/// bare `swift test` and xcodebuild. The end-to-end effect of each switch on the
/// computed logits is proven separately by the adversarial
/// `Cohere2ModelParityTests` fixtures (which invert every switch across two
/// configs).
///
/// NOTE — the `head_dim * num_attention_heads == hidden_size` invariant is
/// enforced by a `precondition` in `Cohere2Attention.init` (a mismatch would
/// silently mis-shape the attention reshape). A `precondition` failure calls
/// `fatalError`, which neither XCTest nor swift-testing can trap, so it is NOT
/// unit-tested here; the parity fixtures pin the correct head_dim path instead.
@Suite("Cohere2 decode")
struct Cohere2DecodeTests {

    private func decode(_ json: String) throws -> Cohere2Configuration {
        try JSONDecoder().decode(Cohere2Configuration.self, from: Data(json.utf8))
    }

    /// Defaults land where the Python `ModelArgs` dataclass puts them when the
    /// keys are absent (the permissive-decode contract). Note `logit_scale`
    /// defaults to the dataclass `0.0625`, NOT the shipped checkpoint's `0.25`.
    @Test("dataclass defaults when keys absent")
    func dataclassDefaults() throws {
        let config = try decode("""
        { "model_type": "cohere2" }
        """)
        #expect(config.hiddenSize == 4096)
        #expect(config.headDim == 128)
        #expect(config.numHiddenLayers == 32)
        #expect(config.intermediateSize == 14336)
        #expect(config.numAttentionHeads == 32)
        #expect(config.numKeyValueHeads == 8)
        #expect(config.ropeTheta == 50000.0)
        #expect(config.vocabSize == 256000)
        #expect(config.layerNormEps == 1e-5)
        #expect(config.logitScale == 0.0625)
        #expect(config.attentionBias == false)
        #expect(config.layerNormBias == false)
        #expect(config.slidingWindow == 4096)
        #expect(config.slidingWindowPattern == 4)
    }

    /// `logit_scale` is read verbatim — the shipped checkpoint overrides the
    /// dataclass default (0.0625) with 0.25, so the value must come from config.
    @Test("logit_scale is read (shipped 0.25 overrides the 0.0625 default)")
    func logitScaleIsRead() throws {
        let config = try decode("""
        { "model_type": "cohere2", "logit_scale": 0.25 }
        """)
        #expect(config.logitScale == 0.25)
    }

    /// `attention_bias` decodes both states; the default (Python `ModelArgs`) is
    /// `false`.
    @Test(arguments: [true, false])
    func attentionBiasBothStates(_ bias: Bool) throws {
        let config = try decode("""
        { "model_type": "cohere2", "attention_bias": \(bias) }
        """)
        #expect(config.attentionBias == bias)
    }

    /// `layer_norm_bias` decodes both states; the default (Python `ModelArgs`) is
    /// `false`.
    @Test(arguments: [true, false])
    func layerNormBiasBothStates(_ bias: Bool) throws {
        let config = try decode("""
        { "model_type": "cohere2", "layer_norm_bias": \(bias) }
        """)
        #expect(config.layerNormBias == bias)
    }

    /// `sliding_window` and `sliding_window_pattern` — the interleave knobs the
    /// mask/cache/RoPE selection branches on — decode verbatim.
    @Test("sliding_window and sliding_window_pattern are read")
    func slidingKnobsAreRead() throws {
        let config = try decode("""
        {
          "model_type": "cohere2",
          "sliding_window": 4096,
          "sliding_window_pattern": 4
        }
        """)
        #expect(config.slidingWindow == 4096)
        #expect(config.slidingWindowPattern == 4)
    }

    /// An explicit `head_dim` is taken verbatim (the shipped checkpoint sets 128).
    @Test("head_dim is read verbatim")
    func headDimIsRead() throws {
        let config = try decode("""
        { "model_type": "cohere2", "head_dim": 8 }
        """)
        #expect(config.headDim == 8)
    }
}
