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
}
