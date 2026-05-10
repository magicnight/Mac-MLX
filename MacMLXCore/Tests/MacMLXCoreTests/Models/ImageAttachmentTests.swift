import Testing
import Foundation
@testable import MacMLXCore

@Suite("ImageAttachment")
struct ImageAttachmentTests {

    @Test
    func roundTripsThroughJSON() throws {
        let url = URL(fileURLWithPath: "/tmp/cat.jpg")
        let original = ImageAttachment(fileURL: url, mimeType: "image/jpeg")
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(ImageAttachment.self, from: data)
        #expect(decoded == original)
    }

    @Test
    func mimeTypeFromKnownExtensions() {
        #expect(ImageAttachment.mimeType(forPathExtension: "jpg") == "image/jpeg")
        #expect(ImageAttachment.mimeType(forPathExtension: "JPEG") == "image/jpeg")
        #expect(ImageAttachment.mimeType(forPathExtension: "png") == "image/png")
        #expect(ImageAttachment.mimeType(forPathExtension: "webp") == "image/webp")
        #expect(ImageAttachment.mimeType(forPathExtension: "gif") == "image/gif")
        #expect(ImageAttachment.mimeType(forPathExtension: "heic") == "image/heic")
        #expect(ImageAttachment.mimeType(forPathExtension: "bmp") == "image/bmp")
    }

    @Test
    func mimeTypeFromUnknownExtensionIsNil() {
        #expect(ImageAttachment.mimeType(forPathExtension: "txt") == nil)
        #expect(ImageAttachment.mimeType(forPathExtension: "") == nil)
        #expect(ImageAttachment.mimeType(forPathExtension: "exe") == nil)
    }
}
