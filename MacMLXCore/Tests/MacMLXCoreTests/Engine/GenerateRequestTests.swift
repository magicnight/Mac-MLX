import Testing
import Foundation
@testable import MacMLXCore

@Test
func chatMessageRoundTripsThroughJSON() throws {
    let m = ChatMessage(role: .user, content: "hello")
    let data = try JSONEncoder().encode(m)
    let back = try JSONDecoder().decode(ChatMessage.self, from: data)
    #expect(back == m)
}

@Test
func generateRequestDefaultsAreReasonable() {
    let req = GenerateRequest(model: "Qwen3-8B-4bit", messages: [
        ChatMessage(role: .user, content: "hi")
    ])
    #expect(req.parameters.temperature == 0.7)
    #expect(req.parameters.maxTokens == 2048)
    #expect(req.parameters.stream == true)
    #expect(req.systemPrompt == nil)
}

@Test
func generateRequestPrependsSystemPromptToMessages() {
    let req = GenerateRequest(
        model: "x",
        messages: [ChatMessage(role: .user, content: "hi")],
        systemPrompt: "You are a helpful assistant."
    )
    let merged = req.allMessages
    #expect(merged.count == 2)
    #expect(merged[0].role == .system)
    #expect(merged[0].content == "You are a helpful assistant.")
    #expect(merged[1].role == .user)
}

@Test
func generateRequestWithoutSystemPromptReturnsMessagesUnchanged() {
    let req = GenerateRequest(
        model: "x",
        messages: [ChatMessage(role: .user, content: "hi")]
    )
    #expect(req.allMessages.count == 1)
}
