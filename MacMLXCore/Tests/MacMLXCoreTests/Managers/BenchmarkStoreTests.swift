import Testing
import Foundation
@testable import MacMLXCore

// MARK: - Helpers

/// Makes an isolated temp directory the store writes into, cleaned up by
/// the test host between runs. Mirrors ConversationStoreTests' shape.
private func makeTempStore() -> (BenchmarkStore, URL) {
    let dir = URL(filePath: NSTemporaryDirectory(), directoryHint: .isDirectory)
        .appending(path: "macmlx-bench-\(UUID().uuidString)", directoryHint: .isDirectory)
    return (BenchmarkStore(directory: dir), dir)
}

private func sample(
    id: UUID = UUID(),
    timestamp: Date = Date(),
    generationTPS: Double = 42.0
) -> BenchmarkResult {
    BenchmarkResult(
        id: id,
        modelID: "Qwen3-8B-4bit",
        engineID: .mlxSwift,
        promptTokens: 256,
        completionTokens: 200,
        runs: 3,
        promptTPS: 1200.0,
        generationTPS: generationTPS,
        ttftMs: 140.0,
        memoryUsedGB: 9.2,
        modelLoadTimeS: 4.1,
        timestamp: timestamp,
        system: SystemInfo(chip: "Apple M3 Pro", ramGB: 36, macOSVersion: "15.3.1"),
        macMLXVersion: "0.3.0",
        engineVersion: "mlx-swift-lm 3.31.3",
        notes: ""
    )
}

// MARK: - Tests

@Test
func saveAndLoadRoundTrip() async throws {
    let (store, dir) = makeTempStore()
    defer { try? FileManager.default.removeItem(at: dir) }

    let result = sample()
    try await store.save(result)
    let loaded = try await store.loadLatest()
    #expect(loaded?.id == result.id)
    #expect(loaded?.generationTPS == 42.0)
    #expect(loaded?.system.macOSVersion == "15.3.1")
}

@Test
func listSortsNewestFirst() async throws {
    let (store, dir) = makeTempStore()
    defer { try? FileManager.default.removeItem(at: dir) }

    let old = sample(timestamp: Date(timeIntervalSince1970: 1_745_000_000), generationTPS: 10)
    let mid = sample(timestamp: Date(timeIntervalSince1970: 1_746_000_000), generationTPS: 20)
    let new = sample(timestamp: Date(timeIntervalSince1970: 1_747_000_000), generationTPS: 30)
    try await store.save(old)
    try await store.save(new)
    try await store.save(mid)

    let list = try await store.list()
    #expect(list.count == 3)
    #expect(list[0].generationTPS == 30)  // newest
    #expect(list[1].generationTPS == 20)
    #expect(list[2].generationTPS == 10)
}

@Test
func deleteRemovesOneResult() async throws {
    let (store, dir) = makeTempStore()
    defer { try? FileManager.default.removeItem(at: dir) }

    let a = sample()
    let b = sample()
    try await store.save(a)
    try await store.save(b)

    try await store.delete(id: a.id)
    let remaining = try await store.list()
    #expect(remaining.count == 1)
    #expect(remaining.first?.id == b.id)
}

@Test
func deleteAllClearsStore() async throws {
    let (store, dir) = makeTempStore()
    defer { try? FileManager.default.removeItem(at: dir) }

    try await store.save(sample())
    try await store.save(sample())
    try await store.deleteAll()
    #expect(try await store.list().isEmpty)
}

@Test
func corruptFilesAreSkipped() async throws {
    let (store, dir) = makeTempStore()
    defer { try? FileManager.default.removeItem(at: dir) }

    // Save one valid result then drop a corrupt sibling — the valid one
    // should still round-trip.
    let good = sample()
    try await store.save(good)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    try Data("not valid json".utf8).write(
        to: dir.appending(path: "\(UUID().uuidString).json", directoryHint: .notDirectory)
    )
    let list = try await store.list()
    #expect(list.count == 1)
    #expect(list.first?.id == good.id)
}

@Test
func loadLatestReturnsNilOnEmptyStore() async throws {
    let (store, dir) = makeTempStore()
    defer { try? FileManager.default.removeItem(at: dir) }
    let latest = try await store.loadLatest()
    #expect(latest == nil)
}
