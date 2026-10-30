import Foundation
import Testing

@testable import MacMLXCore

@Suite("ChatMessage.reasoning")
struct ChatMessageReasoningTests {

    @Test
    func defaultsToNil() {
        #expect(ChatMessage(role: .assistant, content: "Hi").reasoning == nil)
    }

    /// Conversations saved before the field existed have no `reasoning` key.
    @Test
    func jsonWithoutTheKeyDecodesWithNil() throws {
        let legacy = """
        {"id": "1FAA0000-0000-0000-0000-000000000001", "role": "assistant", "content": "Hi"}
        """
        let decoded = try JSONDecoder().decode(ChatMessage.self, from: Data(legacy.utf8))
        #expect(decoded.content == "Hi")
        #expect(decoded.reasoning == nil)
    }

    @Test
    func reasoningRoundTrips() throws {
        let original = ChatMessage(role: .assistant, content: "Done.", reasoning: "First, list the files.")
        let back = try JSONDecoder().decode(ChatMessage.self, from: JSONEncoder().encode(original))
        #expect(back == original)
        #expect(back.reasoning == "First, list the files.")
    }

    /// A message without reasoning encodes as it did before the field existed.
    @Test
    func nilReasoningIsNotEncoded() throws {
        let data = try JSONEncoder().encode(ChatMessage(role: .assistant, content: "Done."))
        let object = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(object["reasoning"] == nil)
    }
}
