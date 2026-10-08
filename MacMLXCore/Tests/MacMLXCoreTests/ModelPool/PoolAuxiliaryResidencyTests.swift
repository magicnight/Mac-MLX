import Foundation
import Testing
import os

@testable import MacMLXCore

// MARK: - PoolAuxiliaryResidency (#130)
//
// The residency the app hands the server, over a real pool: what the server
// asks lands under the kind-scoped key, an admission is in use until the
// request ends it, a hit's use refreshes the clock and spares the entry in
// one step, and forget drops it without its release.

@Suite("PoolAuxiliaryResidency (#130)")
struct PoolAuxiliaryResidencyTests {

    private actor Stub: InferenceEngine {
        nonisolated let engineID: EngineID = .mlxSwift
        var status: EngineStatus = .idle
        var loadedModel: LocalModel?
        var version: String = "stub"
        func load(_ model: LocalModel) async throws { loadedModel = model; status = .ready(model: model.id) }
        func unload() async throws { loadedModel = nil; status = .idle }
        nonisolated func generate(_ request: GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error> {
            AsyncThrowingStream { $0.finish(throwing: EngineError.modelNotLoaded) }
        }
        func healthCheck() async -> Bool { true }
    }

    private func model(_ id: String, format: ModelFormat = .reranker) -> LocalModel {
        LocalModel(
            id: id, displayName: id, directory: FileManager.default.temporaryDirectory,
            sizeBytes: 1_000_000_000, format: format, quantization: nil, parameterCount: nil, architecture: nil)
    }

    /// A parameters store in a temporary directory, so the TTL a test saves
    /// is the one the residency reads.
    private func temporaryParameters() throws -> ModelParametersStore {
        let directory = FileManager.default.temporaryDirectory
            .appending(path: "aux-residency-params-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return ModelParametersStore(directory: directory)
    }

    @Test("admission carries the model's TTL from the parameters store")
    func admissionCarriesTheTTL() async throws {
        let pool = ModelPool(maxBytes: 10_000_000_000) { _ in Stub() }
        let parameters = try temporaryParameters()
        try await parameters.save(ModelParameters(ttlSeconds: 1), for: "qwen")
        let residency = PoolAuxiliaryResidency(pool: pool, parameters: parameters)
        let released = OSAllocatedUnfairLock<Int>(initialState: 0)
        try await residency.admit(.reranker, model("qwen"), load: {}, release: { released.withLock { $0 += 1 } })
        await residency.endUse(.reranker, "qwen")
        await pool.sweepIdle(now: Date().addingTimeInterval(5))
        let resident = await pool.residentModelIDs()
        #expect(resident == [], "past the TTL the store named, the sweep reclaims it")
        let deadline = Date().addingTimeInterval(60)
        while released.withLock({ $0 }) == 0, Date() < deadline {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        #expect(released.withLock { $0 } == 1)
    }

    @Test("an admission is in use until the request ends it")
    func admissionIsInUseUntilEnded() async throws {
        let pool = ModelPool(maxBytes: 1_500_000_000) { _ in Stub() }
        let residency = PoolAuxiliaryResidency(pool: pool, parameters: try temporaryParameters())
        let released = OSAllocatedUnfairLock<Int>(initialState: 0)
        try await residency.admit(.reranker, model("qwen"), load: {}, release: { released.withLock { $0 += 1 } })
        var resident = await pool.residentModelIDs()
        #expect(resident == ["reranker:qwen"])
        _ = try await pool.load(model("chat", format: .mlx))
        #expect(released.withLock { $0 } == 0, "in use from its admission, it is spared though the load is over budget")
        await residency.endUse(.reranker, "qwen")
        _ = try await pool.load(model("chat-2", format: .mlx))
        #expect(released.withLock { $0 } == 1, "ended, it is the LRU victim and the pool released it through the residency")
        resident = await pool.residentModelIDs()
        #expect(!resident.contains("reranker:qwen"))
    }

    @Test("a hit's use refreshes the clock, so a model used after a chat model outlives it")
    func beginUseRefreshesTheClock() async throws {
        let pool = ModelPool(maxBytes: 2_500_000_000) { _ in Stub() }
        let residency = PoolAuxiliaryResidency(pool: pool, parameters: try temporaryParameters())
        let released = OSAllocatedUnfairLock<Int>(initialState: 0)
        try await residency.admit(.reranker, model("qwen"), load: {}, release: { released.withLock { $0 += 1 } })
        await residency.endUse(.reranker, "qwen")
        _ = try await pool.load(model("a", format: .mlx))
        let held = await residency.beginUse(.reranker, "qwen")
        #expect(held)
        await residency.endUse(.reranker, "qwen")
        _ = try await pool.load(model("b", format: .mlx))
        #expect(released.withLock { $0 } == 0, "used after `a`, the reranker is no longer the LRU victim")
        var resident = await pool.residentModelIDs()
        #expect(resident == ["b", "reranker:qwen"])
        _ = try await pool.load(model("c", format: .mlx))
        #expect(released.withLock { $0 } == 1, "older than `b`, it goes next")
        resident = await pool.residentModelIDs()
        #expect(resident == ["b", "c"])
    }

    @Test("a hit's use spares the entry until it ends")
    func beginUseSparesUntilEnded() async throws {
        let pool = ModelPool(maxBytes: 1_500_000_000) { _ in Stub() }
        let residency = PoolAuxiliaryResidency(pool: pool, parameters: try temporaryParameters())
        let released = OSAllocatedUnfairLock<Int>(initialState: 0)
        try await residency.admit(.reranker, model("qwen"), load: {}, release: { released.withLock { $0 += 1 } })
        await residency.endUse(.reranker, "qwen")
        let held = await residency.beginUse(.reranker, "qwen")
        #expect(held)
        _ = try await pool.load(model("chat", format: .mlx))
        #expect(released.withLock { $0 } == 0, "in use, it is spared though the load is over budget")
        await residency.endUse(.reranker, "qwen")
        _ = try await pool.load(model("chat-2", format: .mlx))
        #expect(released.withLock { $0 } == 1, "ended, it goes")
    }

    @Test("a hit's use on a model the pool has let go is refused")
    func beginUseOnAModelThePoolLetGoIsRefused() async throws {
        let pool = ModelPool(maxBytes: 1_500_000_000) { _ in Stub() }
        let residency = PoolAuxiliaryResidency(pool: pool, parameters: try temporaryParameters())
        let released = OSAllocatedUnfairLock<Int>(initialState: 0)
        try await residency.admit(.reranker, model("qwen"), load: {}, release: { released.withLock { $0 += 1 } })
        await residency.endUse(.reranker, "qwen")
        _ = try await pool.load(model("chat", format: .mlx))
        #expect(released.withLock { $0 } == 1, "over budget, the pool let it go")
        let held = await residency.beginUse(.reranker, "qwen")
        #expect(!held, "the pool no longer holds it: the server must swap, not compute on it")
        await residency.endUse(.reranker, "qwen")
        let resident = await pool.residentModelIDs()
        #expect(resident == ["chat"], "an end for a model the pool let go changes nothing")
    }

    @Test("forget drops the entry without its release")
    func forgetDropsWithoutTheRelease() async throws {
        let pool = ModelPool(maxBytes: 1_500_000_000) { _ in Stub() }
        let residency = PoolAuxiliaryResidency(pool: pool, parameters: try temporaryParameters())
        let released = OSAllocatedUnfairLock<Int>(initialState: 0)
        try await residency.admit(.embedder, model("bge"), load: {}, release: { released.withLock { $0 += 1 } })
        await residency.endUse(.embedder, "bge")
        await residency.forget(.embedder, "bge")
        let resident = await pool.residentModelIDs()
        #expect(resident == [])
        _ = try await pool.load(model("chat", format: .mlx))
        #expect(released.withLock { $0 } == 0, "forget never runs the release, and there is nothing left to evict")
    }
}
