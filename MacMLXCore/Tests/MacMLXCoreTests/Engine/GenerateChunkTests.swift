import Testing
@testable import MacMLXCore

@Test
func generateChunkInitialiserDefaultsAreEmpty() {
    let c = GenerateChunk(text: "Hello")
    #expect(c.text == "Hello")
    #expect(c.finishReason == nil)
    #expect(c.usage == nil)
}

@Test
func finishReasonRawValuesMatchOpenAI() {
    #expect(FinishReason.stop.rawValue == "stop")
    #expect(FinishReason.length.rawValue == "length")
    #expect(FinishReason.error.rawValue == "error")
}

@Test
func tokenUsageSumsCorrectly() {
    let u = TokenUsage(promptTokens: 12, completionTokens: 30)
    #expect(u.totalTokens == 42)
}
