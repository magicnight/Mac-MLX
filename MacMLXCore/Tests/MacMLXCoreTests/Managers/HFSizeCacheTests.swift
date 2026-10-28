import Testing
import Foundation
@testable import MacMLXCore

/// Filesystem-backed: each test reopens the same file with a fresh
/// instance, which is how the app sees the cache on its next launch.
@Suite("HFSizeCache", .serialized)
struct HFSizeCacheTests {

    @Test
    func sizeSurvivesANewInstance() async throws {
        let file = try TempDir().url.appendingPathComponent("sizes.json")
        await HFSizeCache(fileURL: file).put("mlx-community/Qwen3-8B-4bit", size: 4_600_000_000)

        let reopened = HFSizeCache(fileURL: file)
        #expect(await reopened.get("mlx-community/Qwen3-8B-4bit") == 4_600_000_000)
    }

    @Test
    func putAfterReopenKeepsEarlierEntries() async throws {
        let file = try TempDir().url.appendingPathComponent("sizes.json")
        await HFSizeCache(fileURL: file).put("a/one", size: 1)
        await HFSizeCache(fileURL: file).put("b/two", size: 2)

        let reopened = HFSizeCache(fileURL: file)
        #expect(await reopened.get("a/one") == 1)
        #expect(await reopened.get("b/two") == 2)
    }

    @Test
    func entryOlderThanTTLIsMissing() async throws {
        let file = try TempDir().url.appendingPathComponent("sizes.json")
        // Written in the on-disk format `flush()` produces. The fresh
        // entry proves the file decoded, so the stale one's nil is the
        // TTL and not a failed read.
        let fresh = Date().addingTimeInterval(-60).ISO8601Format()
        let stale = Date().addingTimeInterval(-HFSizeCache.ttl - 60).ISO8601Format()
        let json = """
            {"entries":{
              "fresh/model":{"sizeBytes":10,"fetchedAt":"\(fresh)"},
              "stale/model":{"sizeBytes":20,"fetchedAt":"\(stale)"}
            }}
            """
        try Data(json.utf8).write(to: file)

        let cache = HFSizeCache(fileURL: file)
        #expect(await cache.get("fresh/model") == 10)
        #expect(await cache.get("stale/model") == nil)
    }
}

private struct TempDir {
    let url: URL
    init() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("macmlx-hf-size-cache-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        self.url = base
    }
}
