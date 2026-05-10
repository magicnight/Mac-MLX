import Testing
import Foundation
@testable import MacMLXCore

/// Filesystem-backed: serialised so swift-testing's parallel executor
/// doesn't thrash on the temp directory.
@Suite("ConversationStore — images persistence", .serialized)
struct ConversationStoreImagesTests {

    @Test
    func saveCopiesExternalImageIntoConversationDir() async throws {
        let temp = try TempDir()
        let store = ConversationStore(directory: temp.url)

        // Lay down a "user-picked" image outside the store.
        let pickedDir = temp.url.appendingPathComponent("picks", isDirectory: true)
        try FileManager.default.createDirectory(at: pickedDir, withIntermediateDirectories: true)
        let pickedURL = pickedDir.appendingPathComponent("cat.jpg", isDirectory: false)
        try Data("fake-bytes".utf8).write(to: pickedURL)

        let conv = Conversation(
            messages: [
                StoredMessage(
                    role: .user,
                    content: "Look at this",
                    images: [ImageAttachment(fileURL: pickedURL, mimeType: "image/jpeg")]
                )
            ]
        )

        try await store.save(conv)

        // Reload; image URL must point inside the per-conversation dir.
        let listed = try await store.list()
        #expect(listed.count == 1)
        let reloaded = try #require(listed.first)
        #expect(reloaded.messages.first?.images.count == 1)
        let savedURL = try #require(reloaded.messages.first?.images.first?.fileURL)
        let convImagesPrefix = temp.url
            .appendingPathComponent(conv.id.uuidString, isDirectory: true)
            .appendingPathComponent("images", isDirectory: true)
            .path
        #expect(savedURL.path.hasPrefix(convImagesPrefix), "saved URL must be inside conversation images dir; got \(savedURL.path)")
        // Bytes survived the copy.
        let bytes = try Data(contentsOf: savedURL)
        #expect(bytes == Data("fake-bytes".utf8))
    }

    @Test
    func saveLeavesAlreadyInternalImageURLAlone() async throws {
        // If the URL already points into the conversation's images dir,
        // the second save must not re-copy + rename it.
        let temp = try TempDir()
        let store = ConversationStore(directory: temp.url)
        let convID = UUID()
        let imagesDir = temp.url
            .appendingPathComponent(convID.uuidString, isDirectory: true)
            .appendingPathComponent("images", isDirectory: true)
        try FileManager.default.createDirectory(at: imagesDir, withIntermediateDirectories: true)
        let internalURL = imagesDir.appendingPathComponent("already-here.jpg", isDirectory: false)
        try Data("internal".utf8).write(to: internalURL)

        let conv = Conversation(
            id: convID,
            messages: [
                StoredMessage(
                    role: .user,
                    content: "Look",
                    images: [ImageAttachment(fileURL: internalURL, mimeType: "image/jpeg")]
                )
            ]
        )
        try await store.save(conv)

        let listed = try await store.list()
        let reloaded = try #require(listed.first)
        let savedURL = try #require(reloaded.messages.first?.images.first?.fileURL)
        #expect(savedURL == internalURL, "internal URL should be preserved verbatim")
    }

    @Test
    func deleteRemovesConversationImagesDir() async throws {
        let temp = try TempDir()
        let store = ConversationStore(directory: temp.url)
        let pickedDir = temp.url.appendingPathComponent("picks", isDirectory: true)
        try FileManager.default.createDirectory(at: pickedDir, withIntermediateDirectories: true)
        let pickedURL = pickedDir.appendingPathComponent("cat.png", isDirectory: false)
        try Data("png-bytes".utf8).write(to: pickedURL)

        let conv = Conversation(
            messages: [
                StoredMessage(
                    role: .user,
                    content: "x",
                    images: [ImageAttachment(fileURL: pickedURL, mimeType: "image/png")]
                )
            ]
        )
        try await store.save(conv)

        // Confirm the conv directory exists.
        let convDir = temp.url.appendingPathComponent(conv.id.uuidString, isDirectory: true)
        #expect(FileManager.default.fileExists(atPath: convDir.path))

        try await store.delete(id: conv.id)

        #expect(!FileManager.default.fileExists(atPath: convDir.path), "conversation dir must be removed")
        // JSON sidecar gone too.
        let jsonURL = temp.url.appendingPathComponent("\(conv.id.uuidString).json", isDirectory: false)
        #expect(!FileManager.default.fileExists(atPath: jsonURL.path))
    }

    @Test
    func legacyStoredMessageJSONWithoutImagesDecodesWithEmptyArray() throws {
        let legacy = """
        {
            "id": "1FAA0000-0000-0000-0000-000000000001",
            "role": "user",
            "content": "Hello",
            "timestamp": 1700000000.0
        }
        """
        let data = Data(legacy.utf8)
        let decoder = JSONCoding.tolerantDecoder()
        let decoded = try decoder.decode(StoredMessage.self, from: data)
        #expect(decoded.images.isEmpty)
        #expect(decoded.role == .user)
    }
}

/// Auto-cleaning scratch dir used by the filesystem-backed tests.
private struct TempDir {
    let url: URL

    init() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("macmlx-conv-image-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        self.url = base
    }
}
