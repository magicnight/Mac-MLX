import XCTest
@testable import MacMLXCore

/// Stub engine for pool tests — no Metal/MLX required. Implements
/// the minimum InferenceEngine surface the pool touches: load,
/// unload, engineID. Generate throws since it shouldn't be called.
private actor StubEngine: InferenceEngine {
    nonisolated let engineID: EngineID = .mlxSwift
    var status: EngineStatus = .idle
    var loadedModel: LocalModel?
    var version: String = "stub"

    func load(_ model: LocalModel) async throws {
        loadedModel = model
        status = .ready(model: model.id)
    }

    func unload() async throws {
        loadedModel = nil
        status = .idle
    }

    nonisolated func generate(_ request: GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error> {
        AsyncThrowingStream { cont in
            cont.finish(throwing: EngineError.modelNotLoaded)
        }
    }

    func healthCheck() async -> Bool { true }
}

final class ModelPoolTests: XCTestCase {

    private func mkModel(_ id: String, size: Int64 = 1_000_000_000) -> LocalModel {
        LocalModel(
            id: id,
            displayName: id,
            directory: FileManager.default.temporaryDirectory,
            sizeBytes: size,
            format: .mlx,
            quantization: nil,
            parameterCount: nil,
            architecture: nil
        )
    }

    func testLoadAddsToPool() async throws {
        let pool = ModelPool(maxBytes: 4_000_000_000, engineFactory: { _ in StubEngine() })
        let m = mkModel("A", size: 1_000_000_000)
        _ = try await pool.load(m)
        let residents = await pool.residentModelIDs()
        XCTAssertEqual(residents, ["A"])
    }

    func testLoadReuseExistingInstance() async throws {
        let pool = ModelPool(maxBytes: 4_000_000_000, engineFactory: { _ in StubEngine() })
        let m = mkModel("A", size: 1_000_000_000)
        let e1 = try await pool.load(m) as AnyObject
        let e2 = try await pool.load(m) as AnyObject
        XCTAssertTrue(e1 === e2)
    }

    func testOverBudgetEvictsLRU() async throws {
        let pool = ModelPool(maxBytes: 2_500_000_000, engineFactory: { _ in StubEngine() })
        _ = try await pool.load(mkModel("A", size: 1_000_000_000))
        _ = try await pool.load(mkModel("B", size: 1_000_000_000))
        // Budget has 2.5 GB, A+B = 2 GB fits.
        _ = try await pool.load(mkModel("C", size: 1_000_000_000))
        // A+B+C = 3 GB — over. Oldest (A) evicted.
        let residents = await pool.residentModelIDs()
        XCTAssertFalse(residents.contains("A"))
        XCTAssertTrue(residents.contains("B"))
        XCTAssertTrue(residents.contains("C"))
    }

    func testPinnedNotEvicted() async throws {
        let pool = ModelPool(maxBytes: 2_500_000_000, engineFactory: { _ in StubEngine() })
        _ = try await pool.load(mkModel("A", size: 1_000_000_000))
        await pool.setPinned("A", true)
        _ = try await pool.load(mkModel("B", size: 1_000_000_000))
        _ = try await pool.load(mkModel("C", size: 1_000_000_000))
        // A is pinned → B (next-oldest) evicted instead.
        let residents = await pool.residentModelIDs()
        XCTAssertTrue(residents.contains("A"))
        XCTAssertFalse(residents.contains("B"))
        XCTAssertTrue(residents.contains("C"))
    }

    func testUnloadRemovesFromPool() async throws {
        let pool = ModelPool(maxBytes: 4_000_000_000, engineFactory: { _ in StubEngine() })
        _ = try await pool.load(mkModel("A"))
        await pool.unload("A")
        let residents = await pool.residentModelIDs()
        XCTAssertTrue(residents.isEmpty)
    }

    func testEngineForReturnsNilWhenNotLoaded() async {
        let pool = ModelPool(maxBytes: 4_000_000_000, engineFactory: { _ in StubEngine() })
        let e = await pool.engine(for: "A")
        XCTAssertNil(e)
    }

    // MARK: - Idle TTL (v0.5.1)

    func testSweepIdleUnloadsExpiredEntry() async throws {
        let pool = ModelPool(maxBytes: 4_000_000_000, engineFactory: { _ in StubEngine() })
        _ = try await pool.load(mkModel("A"), ttlSeconds: 1)
        // A far-future `now` → A has been idle far longer than its 1s TTL.
        await pool.sweepIdle(now: Date().addingTimeInterval(10_000))
        let residents = await pool.residentModelIDs()
        XCTAssertFalse(residents.contains("A"))
    }

    func testSweepIdleSkipsPinnedEntry() async throws {
        let pool = ModelPool(maxBytes: 4_000_000_000, engineFactory: { _ in StubEngine() })
        _ = try await pool.load(mkModel("A"), ttlSeconds: 1)
        await pool.setPinned("A", true)
        // Even far past the TTL, a pinned entry is exempt from the sweep.
        await pool.sweepIdle(now: Date().addingTimeInterval(10_000))
        let residents = await pool.residentModelIDs()
        XCTAssertTrue(residents.contains("A"))
    }

    func testSweepIdleSkipsWithinTTLEntry() async throws {
        let pool = ModelPool(maxBytes: 4_000_000_000, engineFactory: { _ in StubEngine() })
        _ = try await pool.load(mkModel("A"), ttlSeconds: 3600)
        // `now` ≈ load time → idle well under the 1h TTL, so A survives.
        await pool.sweepIdle(now: Date())
        let residents = await pool.residentModelIDs()
        XCTAssertTrue(residents.contains("A"))
    }

    func testSweepIdleNeverSweepsNilTTLEntry() async throws {
        let pool = ModelPool(maxBytes: 4_000_000_000, engineFactory: { _ in StubEngine() })
        _ = try await pool.load(mkModel("A"))  // no TTL configured
        await pool.sweepIdle(now: Date().addingTimeInterval(10_000))
        let residents = await pool.residentModelIDs()
        XCTAssertTrue(residents.contains("A"))
    }

    func testSweepIdleSkipsInFlightGeneratingEntry() async throws {
        let pool = ModelPool(maxBytes: 4_000_000_000, engineFactory: { _ in StubEngine() })
        _ = try await pool.load(mkModel("A"), ttlSeconds: 1)
        // Mark A as actively generating — even far past its 1s TTL it must
        // NOT be swept out from under an in-flight stream (A4 hazard fix).
        await pool.setGenerating("A", true)
        await pool.sweepIdle(now: Date().addingTimeInterval(10_000))
        var residents = await pool.residentModelIDs()
        XCTAssertTrue(residents.contains("A"), "generating entry must survive the sweep")

        // Once the generation finishes, the same expired entry is reclaimed —
        // proving the in-flight marker (not something else) is what spared it.
        await pool.setGenerating("A", false)
        await pool.sweepIdle(now: Date().addingTimeInterval(10_000))
        residents = await pool.residentModelIDs()
        XCTAssertFalse(residents.contains("A"), "cleared entry past TTL must sweep")
    }

    // MARK: - v0.5.3 stability wave (POOL-1 / POOL-3)

    /// POOL-1: mirrors the pin/unpin sequence `EngineCoordinator.load`
    /// now performs on every successful load — pin the newly-active
    /// model, then unpin whichever model was previously active. A former-
    /// active model that's been unpinned must become evictable again;
    /// the newly-pinned current model must survive budget pressure.
    /// (`EngineCoordinator` itself has no test target — this validates
    /// the `ModelPool`-side invariant its fix depends on.)
    func testPinThenUnpinPreviousAllowsEviction() async throws {
        let pool = ModelPool(maxBytes: 2_500_000_000, engineFactory: { _ in StubEngine() })
        _ = try await pool.load(mkModel("A", size: 1_000_000_000))
        await pool.setPinned("A", true)  // A becomes "current" — pinned

        _ = try await pool.load(mkModel("B", size: 1_000_000_000))
        await pool.setPinned("B", true)   // B becomes "current" — pinned
        await pool.setPinned("A", false)  // ...and A (no longer current) is unpinned

        // Budget 2.5 GB: A(1)+B(1) = 2 GB fits. Loading C(1) pushes to
        // 3 GB — over budget. A is unpinned + oldest → must be evicted;
        // B (pinned) must survive.
        _ = try await pool.load(mkModel("C", size: 1_000_000_000))
        let residents = await pool.residentModelIDs()
        XCTAssertFalse(residents.contains("A"), "unpinned former-active model must become evictable")
        XCTAssertTrue(residents.contains("B"), "pinned current model must survive budget pressure")
        XCTAssertTrue(residents.contains("C"))
    }

    /// POOL-3: `evict(toFit:)` must skip `isGenerating` entries, mirroring
    /// the guard `sweepIdle` already applies. Without this, a concurrent
    /// `load(_:)` could evict a model that's actively mid-generation.
    func testEvictSkipsGeneratingEntry() async throws {
        let pool = ModelPool(maxBytes: 2_500_000_000, engineFactory: { _ in StubEngine() })
        _ = try await pool.load(mkModel("A", size: 1_000_000_000))
        await pool.setGenerating("A", true)
        _ = try await pool.load(mkModel("B", size: 1_000_000_000))
        // A+B = 2 GB fits budget so far.
        _ = try await pool.load(mkModel("C", size: 1_000_000_000))
        // A+B+C = 3 GB — over. A is oldest but isGenerating → must be
        // SKIPPED; B (next-oldest, not generating) is evicted instead.
        var residents = await pool.residentModelIDs()
        XCTAssertTrue(residents.contains("A"), "generating entry must survive evict(toFit:)")
        XCTAssertFalse(residents.contains("B"), "non-generating next-oldest should be evicted instead")
        XCTAssertTrue(residents.contains("C"))

        // Once generation finishes, A becomes evictable again — proving
        // the in-flight marker (not something else) is what spared it.
        await pool.setGenerating("A", false)
        _ = try await pool.load(mkModel("D", size: 1_000_000_000))
        residents = await pool.residentModelIDs()
        XCTAssertFalse(residents.contains("A"), "cleared entry becomes evictable again once over budget")
    }
}
