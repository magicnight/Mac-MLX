import XCTest
@testable import MacMLXCore

final class PromptCacheKeyTests: XCTestCase {

    func testSameModelAndTokensProduceSameKey() {
        let a = PromptCacheKey(modelID: "Qwen3-8B-4bit", tokens: [1, 2, 3, 4])
        let b = PromptCacheKey(modelID: "Qwen3-8B-4bit", tokens: [1, 2, 3, 4])
        XCTAssertEqual(a.hashString, b.hashString)
    }

    func testDifferentTokensProduceDifferentKeys() {
        let a = PromptCacheKey(modelID: "Qwen3-8B-4bit", tokens: [1, 2, 3, 4])
        let b = PromptCacheKey(modelID: "Qwen3-8B-4bit", tokens: [1, 2, 3, 5])
        XCTAssertNotEqual(a.hashString, b.hashString)
    }

    func testDifferentModelsProduceDifferentKeys() {
        let a = PromptCacheKey(modelID: "Qwen3-8B-4bit", tokens: [1, 2, 3])
        let b = PromptCacheKey(modelID: "Llama-3-8B-4bit", tokens: [1, 2, 3])
        XCTAssertNotEqual(a.hashString, b.hashString)
    }

    func testHashStringIsHexLowercase() {
        let k = PromptCacheKey(modelID: "m", tokens: [1])
        XCTAssertTrue(k.hashString.allSatisfy { "0123456789abcdef".contains($0) })
        XCTAssertEqual(k.hashString.count, 64)  // sha256
    }

    func testShardedFilenameSplitsByFirstHexChar() {
        let k = PromptCacheKey(modelID: "m", tokens: [1])
        let url = k.shardedFileURL(under: URL(filePath: "/tmp/kv"))
        // /tmp/kv/<first-char>/<fullhash>.safetensors
        let comps = url.pathComponents.suffix(3)
        XCTAssertEqual(comps.count, 3)
        // Middle component is the 1-char shard dir.
        XCTAssertEqual(comps.dropFirst().first?.count, 1)
        XCTAssertTrue(url.pathExtension == "safetensors")
    }
}
