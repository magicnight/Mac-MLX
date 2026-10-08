import XCTest
import os
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
        await pool.beginGenerating("A")
        await pool.sweepIdle(now: Date().addingTimeInterval(10_000))
        var residents = await pool.residentModelIDs()
        XCTAssertTrue(residents.contains("A"), "generating entry must survive the sweep")

        // Once the generation finishes, the same expired entry is reclaimed —
        // proving the in-flight marker (not something else) is what spared it.
        await pool.endGenerating("A")
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
        await pool.beginGenerating("A")
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
        await pool.endGenerating("A")
        _ = try await pool.load(mkModel("D", size: 1_000_000_000))
        residents = await pool.residentModelIDs()
        XCTAssertFalse(residents.contains("A"), "cleared entry becomes evictable again once over budget")
    }

    // MARK: - v0.5.3 P2 hardening (POOL-4 / A3)

    /// Poll until `gate` reports a caller has parked, or fail after `timeout`.
    private func waitUntilParked(
        _ gate: LoadGate,
        timeout: TimeInterval = 5,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while !(await gate.isParked()) {
            if Date() > deadline {
                XCTFail("gate never parked within \(timeout)s", file: file, line: line)
                return
            }
            try await Task.sleep(nanoseconds: 5_000_000)  // 5ms
        }
    }

    /// POOL-4: two concurrent loads of DIFFERENT models must account for each
    /// other's in-flight bytes. The pool reserves a load's cost BEFORE it
    /// suspends on the load task, so the SECOND load's `evict(toFit:)` sees
    /// the FIRST's reservation and frees room the first load hasn't yet
    /// committed to `entries`. Observable: a pre-existing, evictable third
    /// model (C) is reclaimed by the second load's budget-aware sweep —
    /// which could only happen if that sweep counted the first load's pending
    /// bytes. Pre-fix (reservations ignored) C survives and A+B+C blow past
    /// the budget.
    func testConcurrentLoadsOfDifferentModelsRespectCombinedBudget() async throws {
        let gateA = LoadGate()
        let gateB = LoadGate()
        // Budget 25 bytes; each model costs 10. Loads for "A"/"B" park in
        // their engine's `load()` so both sit in-flight simultaneously; any
        // other model ("C") loads immediately.
        let pool = ModelPool(maxBytes: 25, engineFactory: { model in
            switch model.id {
            case "A": return GatedStubEngine(gate: gateA)
            case "B": return GatedStubEngine(gate: gateB)
            default:  return GatedStubEngine(gate: nil)
            }
        })

        // Pre-existing evictable resident. C(10) alone is well within budget.
        _ = try await pool.load(mkModel("C", size: 10))

        // Pre-build the models OUTSIDE the `Task` blocks below: `mkModel` is an
        // instance method, so calling it inside a `Task` would capture the
        // (non-Sendable) test case. `LocalModel` is Sendable, so handing the
        // prebuilt value to the actor is race-free.
        let modelA = mkModel("A", size: 10)
        let modelB = mkModel("B", size: 10)

        // Load A: its evict sees only C(10) ≤ target(15), evicts nothing, then
        // reserves A=10 and parks in engine.load.
        let loadA = Task { try await pool.load(modelA) }
        try await waitUntilParked(gateA)

        // Load B: its evict now sees resident C(10) + pending A(10) = 20 >
        // target(15), so it must evict the oldest evictable entry (C) before
        // reserving B=10 and parking. (Pre-fix: it would see only C(10) ≤ 15
        // and evict nothing.)
        let loadB = Task { try await pool.load(modelB) }
        try await waitUntilParked(gateB)

        // Mid-flight — before either load completes — C must already be gone,
        // proving B's evict (not the loads completing) reclaimed it.
        let midResidents = await pool.residentModelIDs()
        XCTAssertFalse(midResidents.contains("C"), "second load's budget-aware evict must reclaim C mid-flight")

        // Release both loads and let them finish.
        await gateA.open()
        await gateB.open()
        _ = try await loadA.value
        _ = try await loadB.value

        let residents = await pool.residentModelIDs()
        XCTAssertTrue(residents.contains("A"))
        XCTAssertTrue(residents.contains("B"))
        XCTAssertFalse(residents.contains("C"), "C stays evicted — combined A+B already fills the budget")
    }

    /// A3: the in-flight guard is a REFCOUNT, not a bool. Two overlapping
    /// generations on the same model (e.g. GUI chat + a server request) must
    /// keep the entry protected from BOTH sweep and evict until every one has
    /// ended — a bool let whichever finished first clear the guard while the
    /// other was still streaming.
    func testGeneratingRefcountProtectsUntilBalanced() async throws {
        let pool = ModelPool(maxBytes: 2_500_000_000, engineFactory: { _ in StubEngine() })
        _ = try await pool.load(mkModel("A", size: 1_000_000_000), ttlSeconds: 1)

        // Two concurrent generations begin.
        await pool.beginGenerating("A")
        await pool.beginGenerating("A")
        // One ends (refcount 2 → 1): still in flight, must survive a
        // far-future idle sweep.
        await pool.endGenerating("A")

        await pool.sweepIdle(now: Date().addingTimeInterval(10_000))
        var residents = await pool.residentModelIDs()
        XCTAssertTrue(residents.contains("A"), "one end of two must NOT expose A to the sweep")

        // Push over budget (A+B+C = 3 GB > 2.5 GB). A is still generating
        // (refcount 1) so evict must skip it; B (not generating) goes instead.
        _ = try await pool.load(mkModel("B", size: 1_000_000_000))
        _ = try await pool.load(mkModel("C", size: 1_000_000_000))
        residents = await pool.residentModelIDs()
        XCTAssertTrue(residents.contains("A"), "refcount > 0 must be skipped by evict(toFit:)")
        XCTAssertFalse(residents.contains("B"), "non-generating entry evicted instead of the still-generating A")

        // Second generation ends (refcount 1 → 0): A is finally reclaimable.
        await pool.endGenerating("A")
        await pool.sweepIdle(now: Date().addingTimeInterval(10_000))
        residents = await pool.residentModelIDs()
        XCTAssertFalse(residents.contains("A"), "balanced refcount past TTL must sweep")
    }

    /// A3: `endGenerating` clamps at 0 — an unbalanced end must never drive
    /// the refcount negative. If it underflowed to -1, a later single
    /// `beginGenerating` would only bring it back to 0 (still unprotected)
    /// instead of 1, and A would be wrongly swept.
    func testEndGeneratingClampsAtZeroNoUnderflow() async throws {
        let pool = ModelPool(maxBytes: 2_500_000_000, engineFactory: { _ in StubEngine() })
        _ = try await pool.load(mkModel("A", size: 1_000_000_000), ttlSeconds: 1)

        await pool.beginGenerating("A")   // 0 → 1
        await pool.endGenerating("A")     // 1 → 0
        await pool.endGenerating("A")     // clamp: stays 0, must NOT go to -1

        // A single begin must reach exactly 1 and re-protect A.
        await pool.beginGenerating("A")   // 0 → 1 (not -1 → 0)
        await pool.sweepIdle(now: Date().addingTimeInterval(10_000))
        let residents = await pool.residentModelIDs()
        XCTAssertTrue(residents.contains("A"), "one begin after an unbalanced end must re-protect A (clamp prevented underflow)")
    }

    // MARK: - v0.5.3 P3 cleanup wave (P3-5)

    /// P3-5: an `unload(id)` that races an in-flight `load(id)` must NOT be
    /// lost. While the load is parked in `engine.load()` the entry isn't
    /// resident yet, so a naive unload no-ops and the completing load
    /// resurrects it. The fix drops a tombstone the completing load honors —
    /// discarding its fresh engine AND releasing the byte reservation. Verified
    /// two ways: the model is not resident afterwards, and a later pair of
    /// loads that fit the budget ONLY if the reservation was released both
    /// survive (a leaked reservation would have forced one out).
    func testUnloadDuringInFlightLoadDoesNotResurrectOrLeakReservation() async throws {
        let gate = LoadGate()
        // Budget 20 bytes; each model costs 10. "A" parks in engine.load so we
        // can unload it mid-flight; everything else loads immediately.
        let pool = ModelPool(maxBytes: 20, engineFactory: { model in
            model.id == "A" ? GatedStubEngine(gate: gate) : GatedStubEngine(gate: nil)
        })

        let modelA = mkModel("A", size: 10)
        let loadA = Task { try await pool.load(modelA) }
        try await waitUntilParked(gate)

        // Unload A while its load is still parked — the entry isn't resident yet,
        // so this must leave a tombstone rather than silently no-op.
        await pool.unload("A")

        // Release the gate → A's load completes and must honor the tombstone.
        await gate.open()
        _ = try? await loadA.value

        var residents = await pool.residentModelIDs()
        XCTAssertFalse(residents.contains("A"), "unload during in-flight load must not resurrect the model")

        // Reservation-leak probe: B(10) + C(10) = 20 fits the budget EXACTLY,
        // but only if A's 10-byte in-flight reservation was released. A leaked
        // reservation would make loading C see B(10) + pendingA(10) = 20 >
        // target(10) and evict B.
        _ = try await pool.load(mkModel("B", size: 10))
        _ = try await pool.load(mkModel("C", size: 10))
        residents = await pool.residentModelIDs()
        XCTAssertTrue(residents.contains("B"), "B must survive — a leaked A reservation would have forced its eviction")
        XCTAssertTrue(residents.contains("C"))
        XCTAssertFalse(residents.contains("A"))
    }

    // MARK: - Auxiliary residents (#130)

    /// Records what the pool makes happen and when, from closures that may
    /// not touch an actor (the engine factory is synchronous).
    private typealias Events = OSAllocatedUnfairLock<[String]>

    /// A chat engine that records its unload, so a test can see a victim go
    /// before the next model is built.
    private actor RecordingEngine: InferenceEngine {
        nonisolated let engineID: EngineID = .mlxSwift
        var status: EngineStatus = .idle
        var loadedModel: LocalModel?
        var version: String = "recording"
        private let events: Events

        init(events: Events) { self.events = events }

        func load(_ model: LocalModel) async throws {
            loadedModel = model
            status = .ready(model: model.id)
            events.withLock { $0.append("loaded \(model.id)") }
        }

        func unload() async throws {
            let id = loadedModel?.id ?? "?"
            events.withLock { $0.append("unloaded \(id)") }
            loadedModel = nil
            status = .idle
        }

        nonisolated func generate(_ request: GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error> {
            AsyncThrowingStream { $0.finish(throwing: EngineError.modelNotLoaded) }
        }

        func healthCheck() async -> Bool { true }
    }

    /// Something a release closure may capture; its deinit tells when the
    /// pool let go of the closure.
    private final class Captured: @unchecked Sendable {
        private let events: Events
        init(events: Events) { self.events = events }
        deinit { events.withLock { $0.append("freed") } }
    }

    private func auxiliary(_ id: String, size: Int64 = 1_000_000_000) -> LocalModel {
        LocalModel(
            id: id,
            displayName: id,
            directory: FileManager.default.temporaryDirectory,
            sizeBytes: size,
            format: .reranker,
            quantization: nil,
            parameterCount: nil,
            architecture: nil
        )
    }

    /// A release that waits until the test lets it finish.
    private actor ReleaseGate {
        private var opened = false
        func open() { opened = true }
        func wait() async {
            while !opened {
                try? await Task.sleep(nanoseconds: 10_000_000)
            }
        }
    }

    /// Poll (never a fixed sleep) for a detached unload or a sweep.
    private func waitUntil(_ condition: @escaping () -> Bool) async {
        let deadline = Date().addingTimeInterval(60)
        while !condition(), Date() < deadline {
            try? await Task.sleep(nanoseconds: 20_000_000)
        }
    }

    func testAuxiliaryKeysAreScopedByKind() {
        XCTAssertEqual(ModelPool.auxiliaryKey(.reranker, "qwen"), "reranker:qwen")
        XCTAssertEqual(ModelPool.auxiliaryKey(.embedder, "qwen"), "embedder:qwen")
    }

    func testAdmitAuxiliaryUnloadsTheVictimBeforeItsLoadRuns() async throws {
        let events = Events(initialState: [])
        let pool = ModelPool(maxBytes: 2_500_000_000) { _ in RecordingEngine(events: events) }
        _ = try await pool.load(mkModel("a"))
        _ = try await pool.load(mkModel("b"))
        try await pool.admitAuxiliary(
            .reranker, auxiliary("r"),
            load: {
                let resident = await pool.residentModelIDs()
                events.withLock { $0.append("load resident=\(resident)") }
            },
            release: { events.withLock { $0.append("release") } }
        )
        let resident = await pool.residentModelIDs()
        XCTAssertEqual(resident, ["b", "reranker:r"], "the LRU chat model makes room for the reranker")
        XCTAssertEqual(
            events.withLock { $0 },
            ["loaded a", "loaded b", "unloaded a", "load resident=[\"b\"]"],
            "the victim is unloaded before the auxiliary load runs, and admission releases nothing")
    }

    func testAuxiliaryIsReleasedAndItsReleaseDroppedBeforeALaterLoadBuildsItsEngine() async throws {
        let events = Events(initialState: [])
        let pool = ModelPool(maxBytes: 1_500_000_000) { _ in
            events.withLock { $0.append("factory") }
            return StubEngine()
        }
        func admitWithCapture() async throws {
            let captured = Captured(events: events)
            try await pool.admitAuxiliary(
                .reranker, auxiliary("r"), load: {},
                release: {
                    _ = captured
                    events.withLock { $0.append("release") }
                })
        }
        try await admitWithCapture()
        _ = try await pool.load(mkModel("c"))
        XCTAssertEqual(
            events.withLock { $0 }, ["release", "freed", "factory"],
            "the reranker is released, and what its release captured is freed, before the new engine is even built")
        let resident = await pool.residentModelIDs()
        XCTAssertEqual(resident, ["c"])
    }

    func testAdmitReclaimsIdleEntriesFirst() async throws {
        let pool = ModelPool(maxBytes: 10_000_000_000) { _ in StubEngine() }
        let events = Events(initialState: [])
        try await pool.admitAuxiliary(
            .reranker, auxiliary("r"), ttlSeconds: 1, load: {},
            release: { events.withLock { $0.append("release r") } })
        // Nothing but the sweep that admission runs can release `r` (the
        // budget is never under pressure): keep admitting until it does.
        let deadline = Date().addingTimeInterval(60)
        while events.withLock({ $0.isEmpty }), Date() < deadline {
            try await pool.admitAuxiliary(.embedder, auxiliary("e"), load: {}, release: {})
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        XCTAssertEqual(events.withLock { $0 }, ["release r"], "admission sweeps idle entries like a load does")
    }

    func testSweepIdleReclaimsAnAuxiliaryPastItsTTL() async throws {
        let pool = ModelPool(maxBytes: 10_000_000_000) { _ in StubEngine() }
        let events = Events(initialState: [])
        try await pool.admitAuxiliary(
            .reranker, auxiliary("r"), ttlSeconds: 1, load: {},
            release: { events.withLock { $0.append("release") } })
        await pool.sweepIdle(now: Date().addingTimeInterval(0.5))
        let keptResident = await pool.residentModelIDs()
        XCTAssertEqual(keptResident, ["reranker:r"], "within its TTL it stays")
        await pool.sweepIdle(now: Date().addingTimeInterval(5))
        let resident = await pool.residentModelIDs()
        XCTAssertEqual(resident, [], "past its TTL the entry leaves the resident set at once")
        await waitUntil { events.withLock { $0 } == ["release"] }
        XCTAssertEqual(events.withLock { $0 }, ["release"], "and its owner is told to let go")
    }

    func testInUseAuxiliaryIsSparedUntilBalanced() async throws {
        let pool = ModelPool(maxBytes: 1_500_000_000) { _ in StubEngine() }
        let events = Events(initialState: [])
        try await pool.admitAuxiliary(
            .reranker, auxiliary("r"), load: {},
            release: { events.withLock { $0.append("release") } })
        _ = await pool.beginUsingAuxiliary(.reranker, "r")
        _ = try await pool.load(mkModel("c"))
        XCTAssertEqual(events.withLock { $0 }, [], "a reranker scoring a request is never evicted")
        let both = await pool.residentModelIDs()
        XCTAssertEqual(both, ["c", "reranker:r"])
        await pool.endUsingAuxiliary(.reranker, "r")
        _ = try await pool.load(mkModel("d"))
        XCTAssertEqual(events.withLock { $0 }, ["release"], "once balanced it is the LRU victim again")
    }

    func testBeginningUseOfAnEntryThePoolLetGoIsRefused() async throws {
        let pool = ModelPool(maxBytes: 1_500_000_000) { _ in StubEngine() }
        let events = Events(initialState: [])
        try await pool.admitAuxiliary(
            .reranker, auxiliary("r"), load: {},
            release: { events.withLock { $0.append("release") } })
        let held = await pool.beginUsingAuxiliary(.reranker, "r")
        XCTAssertTrue(held, "resident, so the use begins")
        await pool.endUsingAuxiliary(.reranker, "r")
        _ = try await pool.load(mkModel("c"))
        XCTAssertEqual(events.withLock { $0 }, ["release"], "over budget, the pool let the reranker go")
        let refused = await pool.beginUsingAuxiliary(.reranker, "r")
        XCTAssertFalse(refused, "the pool no longer holds it, so nothing is marked and the owner is told")
        let unknown = await pool.beginUsingAuxiliary(.embedder, "never")
        XCTAssertFalse(unknown)
    }

    func testAnAdmissionInUseIsSparedUntilEnded() async throws {
        let pool = ModelPool(maxBytes: 1_500_000_000) { _ in StubEngine() }
        let events = Events(initialState: [])
        try await pool.admitAuxiliary(
            .reranker, auxiliary("r"), inUse: true, load: {},
            release: { events.withLock { $0.append("release") } })
        _ = try await pool.load(mkModel("c"))
        XCTAssertEqual(events.withLock { $0 }, [], "in use from its admission, with no gap for an eviction")
        await pool.endUsingAuxiliary(.reranker, "r")
        _ = try await pool.load(mkModel("d"))
        XCTAssertEqual(events.withLock { $0 }, ["release"], "ended, it is the LRU victim again")
    }

    func testUnloadRunsTheAuxiliaryReleaseAndForgetDoesNot() async throws {
        let pool = ModelPool(maxBytes: 10_000_000_000) { _ in StubEngine() }
        let events = Events(initialState: [])
        try await pool.admitAuxiliary(
            .reranker, auxiliary("r"), load: {},
            release: { events.withLock { $0.append("release reranker") } })
        try await pool.admitAuxiliary(
            .embedder, auxiliary("e"), load: {},
            release: { events.withLock { $0.append("release embedder") } })
        await pool.unload(ModelPool.auxiliaryKey(.reranker, "r"))
        XCTAssertEqual(events.withLock { $0 }, ["release reranker"])
        await pool.forgetAuxiliary(.embedder, "e")
        XCTAssertEqual(events.withLock { $0 }, ["release reranker"], "forget drops the entry without the release")
        let resident = await pool.residentModelIDs()
        XCTAssertEqual(resident, [])
        await pool.forgetAuxiliary(.reranker, "never-there")
        await pool.unload("never-there")
    }

    func testUnloadLeavesTheResidentSetBeforeItsReleaseRuns() async throws {
        let pool = ModelPool(maxBytes: 10_000_000_000) { _ in StubEngine() }
        let gate = ReleaseGate()
        let events = Events(initialState: [])
        try await pool.admitAuxiliary(
            .reranker, auxiliary("r"), load: {},
            release: {
                events.withLock { $0.append("release started") }
                await gate.wait()
                events.withLock { $0.append("release finished") }
            })
        let unloading = Task { await pool.unload(ModelPool.auxiliaryKey(.reranker, "r")) }
        await waitUntil { events.withLock { $0 } == ["release started"] }
        XCTAssertEqual(events.withLock { $0 }, ["release started"], "the release is under way")
        // A new admission under the same key lands while the release waits.
        try await pool.admitAuxiliary(.reranker, auxiliary("r"), load: {}, release: {})
        await gate.open()
        await unloading.value
        let resident = await pool.residentModelIDs()
        XCTAssertEqual(resident, ["reranker:r"], "the entry admitted during the release is kept")
    }

    func testFailedAuxiliaryLoadRollsBackItsReservation() async throws {
        struct Boom: Error {}
        let pool = ModelPool(maxBytes: 2_500_000_000) { _ in StubEngine() }
        var threw = false
        do {
            try await pool.admitAuxiliary(.reranker, auxiliary("r"), load: { throw Boom() }, release: {})
        } catch {
            threw = true
        }
        XCTAssertTrue(threw, "the owner's error comes back")
        let after = await pool.residentModelIDs()
        XCTAssertEqual(after, [])
        // A leaked 1 GB reservation would make the second load evict the first.
        _ = try await pool.load(mkModel("a"))
        _ = try await pool.load(mkModel("b"))
        let resident = await pool.residentModelIDs()
        XCTAssertEqual(resident, ["a", "b"], "nothing stays reserved for a load that failed")
    }

    func testAdmitUnderAnExistingKeyReplacesTheEntryWithoutReleasingIt() async throws {
        // A budget the two admissions would not both fit: without the forget
        // at the top of admit, the first entry would be evicted (and its
        // release run) to make room for the second.
        let pool = ModelPool(maxBytes: 1_500_000_000) { _ in StubEngine() }
        let events = Events(initialState: [])
        try await pool.admitAuxiliary(
            .reranker, auxiliary("r"), load: {},
            release: { events.withLock { $0.append("release first") } })
        try await pool.admitAuxiliary(
            .reranker, auxiliary("r"), load: {},
            release: { events.withLock { $0.append("release second") } })
        XCTAssertEqual(events.withLock { $0 }, [], "re-admission is the owner replacing what it already let go of")
        await pool.unload(ModelPool.auxiliaryKey(.reranker, "r"))
        XCTAssertEqual(events.withLock { $0 }, ["release second"])
    }

    func testAChatModelAndARerankerWithTheSameIDAreSeparateEntries() async throws {
        let pool = ModelPool(maxBytes: 10_000_000_000) { _ in StubEngine() }
        let events = Events(initialState: [])
        _ = try await pool.load(mkModel("qwen"))
        await pool.setPinned("qwen", true)
        try await pool.admitAuxiliary(
            .reranker, auxiliary("qwen"), load: {},
            release: { events.withLock { $0.append("release") } })
        let resident = await pool.residentModelIDs()
        XCTAssertEqual(resident, ["qwen", "reranker:qwen"])
        let stillPinned = await pool.isPinned("qwen")
        XCTAssertTrue(stillPinned, "the reranker's admission leaves the chat entry alone")
        await pool.forgetAuxiliary(.reranker, "qwen")
        let chatStays = await pool.residentModelIDs()
        XCTAssertEqual(chatStays, ["qwen"])
        let engine = await pool.engine(for: "qwen")
        XCTAssertNotNil(engine)
    }

    func testMarkingAnAuxiliaryInUseRefreshesItsLRUClock() async throws {
        let pool = ModelPool(maxBytes: 2_500_000_000) { _ in StubEngine() }
        let events = Events(initialState: [])
        try await pool.admitAuxiliary(
            .reranker, auxiliary("r"), load: {},
            release: { events.withLock { $0.append("release") } })
        _ = try await pool.load(mkModel("a"))
        _ = await pool.beginUsingAuxiliary(.reranker, "r")
        await pool.endUsingAuxiliary(.reranker, "r")
        _ = try await pool.load(mkModel("b"))
        XCTAssertEqual(events.withLock { $0 }, [], "used after `a`, the reranker is no longer the LRU victim")
        let resident = await pool.residentModelIDs()
        XCTAssertEqual(resident, ["b", "reranker:r"])
    }
}

// MARK: - Test doubles for the concurrency tests

/// Actor gate an engine's `load()` can park on until the test opens it —
/// lets two loads sit in-flight simultaneously so the pool's pending-byte
/// accounting (POOL-4) is observable mid-flight.
private actor LoadGate {
    private var isOpen = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    private var parked = false

    /// Suspend until `open()` is called. Records that a caller has parked
    /// (before suspending) so the test can synchronise on it. Returns
    /// immediately if already open.
    func park() async {
        if isOpen { return }
        parked = true
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            if isOpen {
                cont.resume()
            } else {
                waiters.append(cont)
            }
        }
    }

    /// Release every parked caller and let future `park()` calls pass through.
    func open() {
        isOpen = true
        let pending = waiters
        waiters.removeAll()
        for cont in pending { cont.resume() }
    }

    /// Whether a caller has entered `park()` (set before it suspends).
    func isParked() -> Bool { parked }
}

/// Stub engine whose `load()` optionally parks on a `LoadGate`, so a test can
/// hold a load in-flight. Otherwise mirrors `StubEngine`.
private actor GatedStubEngine: InferenceEngine {
    nonisolated let engineID: EngineID = .mlxSwift
    var status: EngineStatus = .idle
    var loadedModel: LocalModel?
    var version: String = "gated-stub"
    private let gate: LoadGate?

    init(gate: LoadGate?) { self.gate = gate }

    func load(_ model: LocalModel) async throws {
        if let gate { await gate.park() }
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
