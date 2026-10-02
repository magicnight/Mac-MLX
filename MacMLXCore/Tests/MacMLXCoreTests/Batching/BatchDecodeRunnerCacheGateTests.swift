import Foundation
import MLX
import MLXLMCommon
import XCTest
@testable import MacMLXCore

/// The dense-cache gate the batched decode path runs before allocating any
/// streams. These assertions used to live with `BatchPositionedCacheWrapper`,
/// whose factory enforced the same predicate; the wrapper is gone (core
/// v0.32.0 fixed the RoPE defect it shimmed) but the gate is not, and this is
/// its direct, model-free coverage.
///
/// `requireMLXRuntimeOrSkip()` gates the cases, since the caches are
/// MLX-backed, so a bare `swift test` (no metallib) skips cleanly.
final class BatchDecodeRunnerCacheGateTests: XCTestCase {

    func testAcceptsPlainDenseCaches() throws {
        try requireMLXRuntimeOrSkip()
        XCTAssertTrue(
            BatchDecodeRunner.areDenseBatchableCaches([KVCacheSimple(), KVCacheSimple(), KVCacheSimple()]))
    }

    func testAcceptsRotatingKVCache() throws {
        try requireMLXRuntimeOrSkip()
        XCTAssertTrue(
            BatchDecodeRunner.areDenseBatchableCaches([RotatingKVCache(maxSize: 512, keep: 0), KVCacheSimple()]))
    }

    func testRefusesCacheListRatherThanCrashingLater() throws {
        try requireMLXRuntimeOrSkip()
        // Hybrid models reach a `CacheList`'s children through concrete-type
        // subscripts, and its plain `update` is a fatalError.
        XCTAssertFalse(
            BatchDecodeRunner.areDenseBatchableCaches([CacheList(MambaCache(), KVCacheSimple())]))
    }

    func testRefusesQuantizedKVCacheRatherThanCrashingLater() throws {
        try requireMLXRuntimeOrSkip()
        // Its real update is reached through a capability probe the batched
        // path does not perform; the plain `update` is a fatalError.
        XCTAssertFalse(BatchDecodeRunner.areDenseBatchableCaches([QuantizedKVCache()]))
    }

    func testRefusesWholeBatchIfAnyCacheIsUnsafe() throws {
        try requireMLXRuntimeOrSkip()
        XCTAssertFalse(
            BatchDecodeRunner.areDenseBatchableCaches([KVCacheSimple(), CacheList(MambaCache(), KVCacheSimple())]))
    }
}
