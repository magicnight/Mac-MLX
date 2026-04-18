import XCTest
@testable import MacMLXCore
import MLXLMCommon
import MLX

final class PromptCacheStoreTests: XCTestCase {

    /// mlx-swift's SwiftPM build does not always bundle `default.metallib`
    /// alongside the test binary — in that case any `MLXArray` op aborts
    /// the test process with a fatalError from the C++ side. Detect the
    /// bundle up front and skip MLX-dependent tests so we still exercise
    /// the pure LRU / miss paths in the store.
    private func requireMetalOrSkip() throws {
        let bundle = Bundle(identifier: "mlx-swift_Cmlx.resources")
            ?? Bundle.allBundles.first(where: { $0.bundlePath.contains("Cmlx") })
        let metallib = bundle?.url(forResource: "default", withExtension: "metallib")
        if metallib == nil {
            throw XCTSkip("Requires default.metallib (SPM test binaries often lack it — run under xcodebuild)")
        }
    }

    /// Build a minimal single-layer [KVCache] from known keys/values.
    /// Sufficient for roundtrip — shape is [1, n_heads, seq, head_dim].
    private func makeSyntheticSnapshot(seqLen: Int) -> PromptCacheSnapshot {
        let keys = MLXArray.zeros([1, 1, seqLen, 4])
        let values = MLXArray.ones([1, 1, seqLen, 4])
        let layer = KVCacheSimple()
        _ = layer.update(keys: keys, values: values)
        return PromptCacheSnapshot([layer])
    }

    private func tmpRoot() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "mlxkv-\(UUID().uuidString)", directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    func testPutThenGetHitsHotTier() async throws {
        try requireMetalOrSkip()
        let store = PromptCacheStore(root: tmpRoot(), hotCapacity: 4)
        let key = PromptCacheKey(modelID: "M", tokens: [1, 2, 3])

        await store.put(key: key, snapshot: makeSyntheticSnapshot(seqLen: 3))
        let got = await store.get(key)

        XCTAssertNotNil(got)
    }

    func testHotEvictionWritesToCold() async throws {
        try requireMetalOrSkip()
        let root = tmpRoot()
        let store = PromptCacheStore(root: root, hotCapacity: 1)

        let k1 = PromptCacheKey(modelID: "M", tokens: [1])
        let k2 = PromptCacheKey(modelID: "M", tokens: [2])

        await store.put(key: k1, snapshot: makeSyntheticSnapshot(seqLen: 1))
        await store.put(key: k2, snapshot: makeSyntheticSnapshot(seqLen: 1))

        // k1 should have been evicted from hot → written to cold.
        let coldFile = k1.shardedFileURL(under: root)
        XCTAssertTrue(FileManager.default.fileExists(atPath: coldFile.path))
    }

    func testColdLookupRestores() async throws {
        try requireMetalOrSkip()
        let root = tmpRoot()
        let store = PromptCacheStore(root: root, hotCapacity: 1)

        let k1 = PromptCacheKey(modelID: "M", tokens: [1])
        let k2 = PromptCacheKey(modelID: "M", tokens: [2])

        await store.put(key: k1, snapshot: makeSyntheticSnapshot(seqLen: 1))
        await store.put(key: k2, snapshot: makeSyntheticSnapshot(seqLen: 1))

        // k1 was evicted from hot, but cold should restore.
        let restored = await store.get(k1)
        XCTAssertNotNil(restored)
    }

    func testMissReturnsNil() async {
        let store = PromptCacheStore(root: tmpRoot(), hotCapacity: 4)
        let k = PromptCacheKey(modelID: "M", tokens: [99])
        let got = await store.get(k)
        XCTAssertNil(got)
    }
}
