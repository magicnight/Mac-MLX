import Testing
import Foundation
@testable import MacMLXCore

/// Track E — API-compatibility extensions to `GenerationParameters`
/// (logit_bias, XTC, logprobs, KV-cache quantization). Pure decode/clamp/
/// default logic; no Metal required.
@Suite("GenerationParameters Track E fields")
struct GenerationParametersTrackETests {

    // MARK: Defaults

    @Test("all Track E fields default to nil/false")
    func defaults() {
        let p = GenerationParameters()
        #expect(p.logitBias == nil)
        #expect(p.xtcProbability == nil)
        #expect(p.xtcThreshold == nil)
        #expect(p.logprobs == false)
        #expect(p.topLogprobs == nil)
        #expect(p.kvBits == nil)
        #expect(p.kvGroupSize == nil)
        #expect(p.quantizedKVStart == nil)
    }

    // MARK: logit_bias

    @Test("empty logit_bias map normalizes to nil")
    func emptyLogitBiasIsNil() {
        #expect(GenerationParameters.normalizeLogitBias([:]) == nil)
        #expect(GenerationParameters(logitBias: [:]).logitBias == nil)
    }

    @Test("logit_bias values clamp to [-100, 100]")
    func logitBiasClamps() {
        let p = GenerationParameters(logitBias: [1: 250, 2: -250, 3: 5])
        #expect(p.logitBias?[1] == 100)
        #expect(p.logitBias?[2] == -100)
        #expect(p.logitBias?[3] == 5)
    }

    // MARK: XTC

    @Test("xtc_probability clamps to [0, 1]")
    func xtcProbabilityClamps() {
        #expect(GenerationParameters(xtcProbability: 2.0).xtcProbability == 1.0)
        #expect(GenerationParameters(xtcProbability: -1.0).xtcProbability == 0.0)
        #expect(GenerationParameters(xtcProbability: 0.5).xtcProbability == 0.5)
    }

    @Test("xtc_threshold clamps to [0, 0.5]")
    func xtcThresholdClamps() {
        #expect(GenerationParameters(xtcThreshold: 0.9).xtcThreshold == 0.5)
        #expect(GenerationParameters(xtcThreshold: -0.2).xtcThreshold == 0.0)
        #expect(GenerationParameters(xtcThreshold: 0.1).xtcThreshold == 0.1)
    }

    // MARK: logprobs

    @Test("top_logprobs clamps to 0...10")
    func topLogprobsClamps() {
        #expect(GenerationParameters(topLogprobs: 50).topLogprobs == 10)
        #expect(GenerationParameters(topLogprobs: -3).topLogprobs == 0)
        #expect(GenerationParameters(topLogprobs: 5).topLogprobs == 5)
    }

    // MARK: KV-cache quantization

    @Test("kv_bits snaps to the discrete supported set {2,3,4,6,8}")
    func kvBitsSnapsToSupportedSet() {
        #expect(GenerationParameters(kvBits: 1).kvBits == 2)
        #expect(GenerationParameters(kvBits: 16).kvBits == 8)
        #expect(GenerationParameters(kvBits: 4).kvBits == 4)
        // In-between widths snap to the NEAREST supported value (ties -> smaller):
        // 5 is equidistant from 4 and 6 -> 4; 7 is equidistant from 6 and 8 -> 6.
        #expect(GenerationParameters(kvBits: 5).kvBits == 4)
        #expect(GenerationParameters(kvBits: 7).kvBits == 6)
    }

    @Test("kv_group_size snaps to {32,64,128}; quantized_kv_start clamps to non-negative")
    func kvGroupAndStartClamp() {
        // Group size is a discrete set, not a range: out-of-set values snap
        // to the nearest supported size instead of failing mid-generation.
        #expect(GenerationParameters(kvGroupSize: 0).kvGroupSize == 32)
        #expect(GenerationParameters(kvGroupSize: 128).kvGroupSize == 128)
        #expect(GenerationParameters(kvGroupSize: 50).kvGroupSize == 64)
        #expect(GenerationParameters(kvGroupSize: 1000).kvGroupSize == 128)
        #expect(GenerationParameters(quantizedKVStart: -5).quantizedKVStart == 0)
        #expect(GenerationParameters(quantizedKVStart: 100).quantizedKVStart == 100)
    }

    // MARK: Codable

    @Test("legacy JSON without Track E keys decodes with defaults")
    func legacyDecode() throws {
        let json = """
        {"temperature":0.7,"topP":0.95,"maxTokens":2048,"stream":true}
        """
        let p = try JSONDecoder().decode(GenerationParameters.self, from: Data(json.utf8))
        #expect(p.logitBias == nil)
        #expect(p.xtcProbability == nil)
        #expect(p.logprobs == false)
        #expect(p.kvBits == nil)
    }

    @Test("Track E fields clamp on the raw-JSON decode path, not just the init")
    func rawDecodeClamps() throws {
        let json = """
        {"temperature":0.7,"topP":0.95,"maxTokens":2048,"stream":true,\
        "kvBits":99,"topLogprobs":99,"xtcProbability":9.0,"quantizedKVStart":-1}
        """
        let p = try JSONDecoder().decode(GenerationParameters.self, from: Data(json.utf8))
        #expect(p.kvBits == 8)
        #expect(p.topLogprobs == 10)
        #expect(p.xtcProbability == 1.0)
        #expect(p.quantizedKVStart == 0)
    }

    @Test("Track E fields round-trip through Codable")
    func roundTrip() throws {
        let p = GenerationParameters(
            temperature: 0.8, topP: 0.9, maxTokens: 100, stream: false,
            logitBias: [42: 3.5], xtcProbability: 0.3, xtcThreshold: 0.15,
            logprobs: true, topLogprobs: 4, kvBits: 4, kvGroupSize: 32, quantizedKVStart: 8)
        let data = try JSONEncoder().encode(p)
        let back = try JSONDecoder().decode(GenerationParameters.self, from: data)
        #expect(back == p)
    }

    // MARK: OpenAI logit_bias parsing (string keys)

    @Test("logit_bias parses OpenAI string keys, clamps, drops unparseable")
    func logitBiasFromOpenAI() {
        let parsed = GenerationParameters.logitBias(
            fromOpenAI: ["50256": -250, "1234": 250, "notAnInt": 5])
        #expect(parsed?[50256] == -100)  // clamped
        #expect(parsed?[1234] == 100)     // clamped
        #expect(parsed?.count == 2)       // "notAnInt" dropped
    }

    @Test("nil / empty / all-unparseable OpenAI logit_bias is nil")
    func logitBiasFromOpenAIEmpty() {
        #expect(GenerationParameters.logitBias(fromOpenAI: nil) == nil)
        #expect(GenerationParameters.logitBias(fromOpenAI: [:]) == nil)
        #expect(GenerationParameters.logitBias(fromOpenAI: ["nope": 5]) == nil)
    }
}
