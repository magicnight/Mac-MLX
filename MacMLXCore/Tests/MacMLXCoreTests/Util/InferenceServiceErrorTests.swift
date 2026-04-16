import Testing
@testable import MacMLXCore

@Test
func inferenceServiceErrorDescriptionsMentionRelevantContext() {
    #expect(InferenceServiceError.pythonNotFound.errorDescription?.contains("Python") == true)
    #expect(InferenceServiceError.portAlreadyInUse(8000).errorDescription?.contains("8000") == true)
    #expect(InferenceServiceError.backendCrashed(exitCode: 137).errorDescription?.contains("137") == true)
}
