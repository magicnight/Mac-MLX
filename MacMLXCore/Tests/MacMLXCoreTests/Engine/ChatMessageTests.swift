import Testing
import Foundation
@testable import MacMLXCore

@Suite("ChatMessage.images")
struct ChatMessageImagesTests {

    @Test
    func defaultsToEmptyImagesWhenInitWithoutField() {
        let m = ChatMessage(role: .user, content: "Hi")
        #expect(m.images.isEmpty)
    }

    @Test
    func preservesAttachedImages() {
        let img = ImageAttachment(
            fileURL: URL(fileURLWithPath: "/tmp/x.jpg"),
            mimeType: "image/jpeg"
        )
        let m = ChatMessage(role: .user, content: "What is this?", images: [img])
        #expect(m.images == [img])
    }

    /// Pre-v0.4.1 conversation JSON has no `images` field. The decoder
    /// must default-construct an empty array so existing on-disk
    /// conversations load unchanged after the upgrade.
    @Test
    func legacyJSONWithoutImagesFieldDecodesWithEmptyArray() throws {
        let legacy = """
        {
            "id": "1FAA0000-0000-0000-0000-000000000001",
            "role": "user",
            "content": "Hi"
        }
        """
        let data = Data(legacy.utf8)
        let decoded = try JSONDecoder().decode(ChatMessage.self, from: data)
        #expect(decoded.role == .user)
        #expect(decoded.content == "Hi")
        #expect(decoded.images.isEmpty)
    }

    @Test
    func newJSONRoundTripsImages() throws {
        let img = ImageAttachment(
            fileURL: URL(fileURLWithPath: "/tmp/cat.jpg"),
            mimeType: "image/jpeg"
        )
        let original = ChatMessage(role: .user, content: "Describe.", images: [img])
        let data = try JSONEncoder().encode(original)
        let back = try JSONDecoder().decode(ChatMessage.self, from: data)
        #expect(back.id == original.id)
        #expect(back.role == original.role)
        #expect(back.content == original.content)
        #expect(back.images.count == 1)
        #expect(back.images.first?.fileURL == img.fileURL)
        #expect(back.images.first?.mimeType == img.mimeType)
    }

    @Test
    func emptyImagesArrayRoundTrips() throws {
        let original = ChatMessage(role: .assistant, content: "Sure.")
        let data = try JSONEncoder().encode(original)
        let back = try JSONDecoder().decode(ChatMessage.self, from: data)
        #expect(back.images.isEmpty)
    }
}
