import Testing
@testable import MacMLXCore

@Test
func engineErrorDescriptionsAreUserFriendly() {
    #expect(EngineError.modelNotLoaded.errorDescription == "No model is currently loaded.")
    #expect(EngineError.modelNotFound("Qwen3-8B-4bit").errorDescription?.contains("Qwen3-8B-4bit") == true)
    #expect(EngineError.engineNotReady.errorDescription?.contains("not ready") == true)
}

@Test
func engineErrorIsEquatable() {
    #expect(EngineError.modelNotLoaded == EngineError.modelNotLoaded)
    #expect(EngineError.modelNotFound("a") == EngineError.modelNotFound("a"))
    #expect(EngineError.modelNotFound("a") != EngineError.modelNotFound("b"))
}
