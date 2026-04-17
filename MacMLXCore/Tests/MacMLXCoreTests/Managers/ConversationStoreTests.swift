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

// MARK: - Date-precision regression (v0.3)

@Test
func rapidSavesPreserveOrderWithSubSecondPrecision() async throws {
    let (store, dir) = makeTempStore()
    defer { try? FileManager.default.removeItem(at: dir) }

    // Three saves ~50ms apart. Pre-v0.3 the encoder was `.iso8601` which
    // truncates to whole seconds, so all three collapsed to the same
    // timestamp and `list()` order was undefined. After the switch to
    // `.secondsSince1970` in `JSONCoding.precisionEncoder` each save
    // gets a distinct Double timestamp and order is stable.
    let a = sample(title: "Alpha")
    try await store.save(a)
    try await Task.sleep(nanoseconds: 50_000_000)
    let b = sample(title: "Beta")
    try await store.save(b)
    try await Task.sleep(nanoseconds: 50_000_000)
    let c = sample(title: "Gamma")
    try await store.save(c)

    let list = try await store.list()
    #expect(list.count == 3)
    #expect(list[0].title == "Gamma")
    #expect(list[2].title == "Alpha")
}

@Test
func decoderAcceptsLegacyISO8601Files() async throws {
    let (store, dir) = makeTempStore()
    defer { try? FileManager.default.removeItem(at: dir) }

    // Write a file in the pre-v0.3 format (ISO-8601 strings, no
    // fractional seconds) — simulates what v0.2 users already have on
    // disk. `list()` / `loadLatest()` must still round-trip it.
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let legacyEncoder = JSONEncoder()
    legacyEncoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    legacyEncoder.dateEncodingStrategy = .iso8601

    let legacy = sample(title: "Pre-v0.3 Chat")
    let data = try legacyEncoder.encode(legacy)
    let url = dir.appending(
        path: "\(legacy.id.uuidString).json",
        directoryHint: .notDirectory
    )
    try data.write(to: url, options: .atomic)

    let loaded = try await store.loadLatest()
    #expect(loaded?.id == legacy.id)
    #expect(loaded?.title == "Pre-v0.3 Chat")
}

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
