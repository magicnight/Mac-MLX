import Testing
import Foundation
@testable import MacMLXCore

// Wrapped in a @Suite struct so test-function names can mirror other
// stores' test names (e.g. BenchmarkStoreTests also has
// `listSortsNewestFirst`) without colliding at module scope.
@Suite
struct ConversationStoreTests {

private func makeTempStore() -> (ConversationStore, URL) {
    let dir = URL(filePath: NSTemporaryDirectory(), directoryHint: .isDirectory)
        .appending(path: "macmlx-convo-\(UUID().uuidString)", directoryHint: .isDirectory)
    return (ConversationStore(directory: dir), dir)
}

private func sample(
    id: UUID = UUID(),
    title: String = "New Chat",
    messages: [StoredMessage] = [],
    updatedAt: Date = Date()
) -> Conversation {
    Conversation(
        id: id,
        title: title,
        messages: messages,
        createdAt: Date(timeIntervalSince1970: 1_745_000_000),
        updatedAt: updatedAt,
        modelID: "Qwen3-8B-4bit",
        systemPrompt: "You are helpful."
    )
}

// MARK: - Round-trip

@Test
func saveAndLoadLatestRoundTrip() async throws {
    let (store, dir) = makeTempStore()
    defer { try? FileManager.default.removeItem(at: dir) }

    let convo = sample(
        messages: [
            StoredMessage(role: .user, content: "Hello"),
            StoredMessage(role: .assistant, content: "Hi"),
        ]
    )
    try await store.save(convo)
    let latest = try await store.loadLatest()
    #expect(latest?.id == convo.id)
    #expect(latest?.messages.count == 2)
    #expect(latest?.messages.last?.content == "Hi")
    #expect(latest?.systemPrompt == "You are helpful.")
}

@Test
func loadLatestReturnsNilOnEmptyStore() async throws {
    let (store, dir) = makeTempStore()
    defer { try? FileManager.default.removeItem(at: dir) }

    let latest = try await store.loadLatest()
    #expect(latest == nil)
}

// MARK: - Ordering

@Test
func listSortsNewestFirst() async throws {
    let (store, dir) = makeTempStore()
    defer { try? FileManager.default.removeItem(at: dir) }

    // Bypass `save()` — it overwrites `updatedAt` with `Date()` and then
    // encodes via `.iso8601`, which truncates to whole seconds. Rapid-fire
    // saves in-test would collapse into the same second and make sort
    // order undefined. We write three fixtures with explicit, well-
    // separated timestamps and verify `list()` sorts them correctly.
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    encoder.dateEncodingStrategy = .iso8601

    let older = sample(
        title: "Alpha",
        updatedAt: Date(timeIntervalSince1970: 1_745_000_000)
    )
    let middle = sample(
        title: "Beta",
        updatedAt: Date(timeIntervalSince1970: 1_745_000_100)
    )
    let newer = sample(
        title: "Gamma",
        updatedAt: Date(timeIntervalSince1970: 1_745_000_200)
    )
    for convo in [older, middle, newer] {
        let data = try encoder.encode(convo)
        let url = dir.appending(
            path: "\(convo.id.uuidString).json",
            directoryHint: .notDirectory
        )
        try data.write(to: url, options: .atomic)
    }

    let list = try await store.list()
    #expect(list.count == 3)
    #expect(list[0].title == "Gamma")  // newest
    #expect(list[2].title == "Alpha")  // oldest
}

// MARK: - Title derivation

@Test
func deriveTitleUsesFirstUserMessage() async throws {
    let (store, dir) = makeTempStore()
    defer { try? FileManager.default.removeItem(at: dir) }

    let convo = sample(
        messages: [
            StoredMessage(role: .user, content: "Explain transformer attention like I'm five"),
            StoredMessage(role: .assistant, content: "Imagine spotlights…"),
        ]
    )
    try await store.save(convo)
    let loaded = try await store.loadLatest()
    #expect(loaded?.title.hasPrefix("Explain transformer") == true)
    #expect(loaded?.title != "New Chat")
}

// MARK: - Deletion

@Test
func deleteRemovesConversation() async throws {
    let (store, dir) = makeTempStore()
    defer { try? FileManager.default.removeItem(at: dir) }

    let a = sample(title: "A")
    let b = sample(title: "B")
    try await store.save(a)
    try await store.save(b)

    try await store.delete(id: a.id)
    let remaining = try await store.list()
    #expect(remaining.count == 1)
    #expect(remaining.first?.id == b.id)
}

@Test
func deleteIsIdempotent() async throws {
    let (store, dir) = makeTempStore()
    defer { try? FileManager.default.removeItem(at: dir) }

    // Deleting a non-existent id must not throw.
    try await store.delete(id: UUID())
}

// MARK: - Corrupt-file tolerance

@Test
func corruptFilesDontBlockOtherLoads() async throws {
    let (store, dir) = makeTempStore()
    defer { try? FileManager.default.removeItem(at: dir) }

    let good = sample(title: "Survivor")
    try await store.save(good)

    // Drop a sibling corrupt file in the same directory.
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    try Data("not json".utf8).write(
        to: dir.appending(path: "\(UUID().uuidString).json", directoryHint: .notDirectory)
    )

    let list = try await store.list()
    #expect(list.count == 1)
    #expect(list.first?.id == good.id)
}

} // end @Suite ConversationStoreTests
