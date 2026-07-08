import Testing
import Foundation
@testable import MacMLXCore

@Suite("ChatMessage tool fields")
struct ChatMessageToolTests {

    @Test("MessageRole.tool raw value matches OpenAI")
    func toolRoleRawValue() {
        #expect(MessageRole.tool.rawValue == "tool")
    }

    /// Pre-v0.5 conversation JSON has no `toolCallID` / `toolCalls` keys. The
    /// decoder must default both to nil so existing on-disk messages load.
    @Test("legacy JSON without tool fields decodes with nil tool fields")
    func legacyDecodesWithNilToolFields() throws {
        let legacy = """
        {"id":"1FAA0000-0000-0000-0000-000000000001","role":"assistant","content":"hi"}
        """
        let decoded = try JSONDecoder().decode(ChatMessage.self, from: Data(legacy.utf8))
        #expect(decoded.toolCallID == nil)
        #expect(decoded.toolCalls == nil)
        #expect(decoded.images.isEmpty)
    }

    @Test(".tool role round-trips with its correlating tool-call id")
    func toolRoleRoundTrips() throws {
        let m = ChatMessage(role: .tool, content: "result text", toolCallID: "call_1")
        let data = try JSONEncoder().encode(m)
        let back = try JSONDecoder().decode(ChatMessage.self, from: data)
        #expect(back == m)
        #expect(back.role == .tool)
        #expect(back.toolCallID == "call_1")
    }

    @Test("assistant message carrying tool calls round-trips")
    func assistantWithToolCallsRoundTrips() throws {
        let call = ToolCallRequest(
            id: "call_1", name: "get_weather", arguments: ["city": .string("Paris")])
        let m = ChatMessage(role: .assistant, content: "", toolCalls: [call])
        let data = try JSONEncoder().encode(m)
        let back = try JSONDecoder().decode(ChatMessage.self, from: data)
        #expect(back == m)
        #expect(back.toolCalls?.first?.name == "get_weather")
        #expect(back.toolCalls?.first?.arguments["city"] == JSONValue.string("Paris"))
    }
}
