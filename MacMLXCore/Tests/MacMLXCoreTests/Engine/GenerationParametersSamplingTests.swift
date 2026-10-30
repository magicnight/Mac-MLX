import MLXLMCommon
import Testing
@testable import MacMLXCore

/// `top_p: 0` keeps only the most probable token, so it samples greedily.
/// Apple's Foundation Models client sends exactly that for `.greedy`, without
/// a temperature; mlx-swift-lm treats a `topP` of 0 as "no nucleus filter" and
/// would sample the whole vocabulary at the default temperature instead.
@Suite("GenerationParameters sampling temperature")
struct GenerationParametersSamplingTests {

    @Test("top_p 0 samples greedily whatever the temperature")
    func zeroTopPIsGreedy() {
        #expect(GenerationParameters(temperature: 0.7, topP: 0).samplingTemperature == 0)
        #expect(GenerationParameters(temperature: 1.5, topP: -0.2).samplingTemperature == 0)
    }

    @Test("a positive top_p keeps the requested temperature")
    func positiveTopPKeepsTemperature() {
        #expect(GenerationParameters(temperature: 0.7, topP: 0.95).samplingTemperature == 0.7)
        #expect(GenerationParameters(temperature: 0.7, topP: 1).samplingTemperature == 0.7)
        #expect(GenerationParameters(temperature: 0.7, topP: 0.01).samplingTemperature == 0.7)
        #expect(GenerationParameters(temperature: 0, topP: 0.95).samplingTemperature == 0)
    }

    /// Both engine paths build mlx-swift-lm's parameters through
    /// `mlxGenerateParameters`, so this is the temperature they sample at.
    @Test("the mlx-swift-lm parameters carry the sampling temperature")
    func mlxParametersSampleGreedilyAtZeroTopP() {
        #expect(GenerationParameters(temperature: 0.7, topP: 0).mlxGenerateParameters.temperature == 0)
        let sampled = GenerationParameters(temperature: 0.7, topP: 0.9, maxTokens: 300).mlxGenerateParameters
        #expect(sampled.temperature == Float(0.7))
        #expect(sampled.topP == Float(0.9))
        #expect(sampled.maxTokens == 300)
    }
}
