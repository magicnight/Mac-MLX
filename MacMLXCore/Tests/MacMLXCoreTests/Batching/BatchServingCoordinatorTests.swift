// Copyright © 2026 macMLX. English comments only.

import MLXLMCommon
import Testing

@testable import MacMLXCore

// MARK: - BatchServingCoordinator (A2d-2) MLX-free clause tests
//
// These lock the engine seam's five-clause concurrency contract at the layer that
// needs no model and no Metal — mirroring how `BatchSchedulerLogicTests` covered
// A2c's scheduler with a scripted `BatchInferenceCore`. The MLX drive loop is
// exercised by the gated real-model E2E; here every decision the contract turns on
// (drain epoch, admission refusal, per-row cancellation, batch-size headroom) is a
// plain `swift test`.

@Suite("BatchServingCoordinator")
struct BatchServingCoordinatorTests {

    /// Records whether an async task has completed, observable across isolation.
    private actor CompletionFlag {
        private(set) var done = false
        func set() { done = true }
        func value() -> Bool { done }
    }

    private func makePending(
        _ coordinator: BatchServingCoordinator, prompt: [Int] = [1, 2, 3]
    ) async -> (BatchServingCoordinator.Pending, AsyncThrowingStream<GenerateChunk, Error>) {
        let id = await coordinator.nextRequestID()
        let (stream, continuation) = AsyncThrowingStream<GenerateChunk, Error>.makeStream()
        let config = BatchSlotConfig(
            promptTokens: prompt, parameters: GenerateParameters(maxTokens: 8, temperature: 0))
        return (
            BatchServingCoordinator.Pending(id: id, config: config, continuation: continuation),
            stream
        )
    }

    // MARK: Clause 1 — submit/drain epoch

    /// The first row admitted finds the loop idle and must launch it; a second row
    /// admitted while it drives must NOT (one drive loop owns the cohort).
    @Test
    func firstAdmissionClaimsTheDriveLoop() async throws {
        let coordinator = BatchServingCoordinator()
        let (p0, _) = await makePending(coordinator)
        let (p1, _) = await makePending(coordinator)

        let a0 = await coordinator.enqueueAndClaim(p0)
        let a1 = await coordinator.enqueueAndClaim(p1)

        #expect(a0.accepted && a0.shouldStartDrive, "the first row must claim the drive loop")
        #expect(a1.accepted && !a1.shouldStartDrive, "a second row must not start a second loop")
    }

    /// Once a drain has begun, every admission is refused (the seam returns nil and
    /// the request takes the legacy path) — no row is admitted partway through a
    /// drain.
    @Test
    func admissionRefusedWhileDraining() async throws {
        let coordinator = BatchServingCoordinator()
        await coordinator.beginDrain()  // idle → returns immediately, epoch now closed

        let (pending, _) = await makePending(coordinator)
        let admission = await coordinator.enqueueAndClaim(pending)

        #expect(!admission.accepted, "a submit during the drain epoch must be refused")
        #expect(await coordinator.isDraining)
    }

    /// After the swap completes, `resumeAdmissions` re-opens the epoch so the same
    /// engine instance (CLI cold-swap) serves the new resident model.
    @Test
    func resumeAdmissionsReopensTheEpoch() async throws {
        let coordinator = BatchServingCoordinator()
        await coordinator.beginDrain()
        await coordinator.resumeAdmissions()

        let (pending, _) = await makePending(coordinator)
        let admission = await coordinator.enqueueAndClaim(pending)

        #expect(admission.accepted, "admissions must resume after the drain epoch ends")
        #expect(!(await coordinator.isDraining))
    }

    // MARK: Clause 4 — drain waits for the cohort, lock-free

    /// A drain issued while a cohort is decoding blocks until the loop reports idle
    /// (`finishDrive`) — the swap waits for rows.
    @Test
    func drainBlocksUntilCohortFinishes() async throws {
        let coordinator = BatchServingCoordinator()
        let (pending, _) = await makePending(coordinator)
        _ = await coordinator.enqueueAndClaim(pending)  // claims driving = true

        let flag = CompletionFlag()
        let drain = Task {
            await coordinator.beginDrain()
            await flag.set()
        }

        // Let beginDrain register its waiter and suspend.
        for _ in 0..<20 { await Task.yield() }
        #expect(await coordinator.isDraining, "drain must have started")
        #expect(!(await flag.value()), "drain must still be blocked while the cohort decodes")

        // The loop drains and exits.
        let redrive = await coordinator.finishDrive()
        await drain.value
        #expect(redrive == false, "a draining finishDrive must release, not re-drive")
        #expect(await flag.value(), "drain must complete once the cohort is idle")
    }

    /// A cohort that FAILS while a drain is open must fail every still-QUEUED row
    /// fast — not leave it hanging until the 120s stall watchdog. The drive loop
    /// calls `finishDrive` after `failAll` (which only fails the ACTIVE rows); the
    /// draining branch skips the re-drive, so `finishDrive` must itself flush the
    /// orphaned queue.
    @Test
    func queuedRowsFailFastWhenCohortFailsDuringDrain() async throws {
        // completionBatchSize 4, prefillBatchSize 2 → one tick admits exactly two
        // rows and leaves the rest queued.
        let coordinator = BatchServingCoordinator(completionBatchSize: 4, prefillBatchSize: 2)

        var streams: [AsyncThrowingStream<GenerateChunk, Error>] = []
        for _ in 0..<4 {
            let (pending, stream) = await makePending(coordinator)
            streams.append(stream)
            _ = await coordinator.enqueueAndClaim(pending)  // first claims driving = true
        }

        // Admit 2 rows (ids 0,1) → active; ids 2,3 stay queued.
        let tick = await coordinator.takeTick(active: [])
        #expect(tick.admits.count == 2, "prefill headroom admits exactly two rows")

        // Open a drain from a separate task; it parks on the drive claim.
        let drain = Task { await coordinator.beginDrain() }
        for _ in 0..<20 { await Task.yield() }
        #expect(await coordinator.isDraining, "drain epoch must be open")

        // Cohort failure while draining: the loop failed its ACTIVE rows and now
        // calls finishDrive. The still-queued rows must be flushed with an error.
        let redrive = await coordinator.finishDrive()
        #expect(redrive == false, "a draining finishDrive releases, never re-drives")

        // Both still-queued rows (2,3) terminate with an error promptly — no
        // reliance on the stall watchdog.
        for stream in streams[2...] {
            var failed = false
            do {
                for try await _ in stream {}
            } catch {
                failed = true
            }
            #expect(failed, "a queued row must error when the cohort dies during drain")
        }

        await drain.value  // finishDrive resumed the drain waiter
    }

    // MARK: Clause 2 — a dropped stream evicts its row

    /// A cancellation of a still-QUEUED row finishes its stream immediately and drops
    /// it, so it is never admitted to the cohort.
    @Test
    func cancellingQueuedRowFinishesItsStreamAndDropsIt() async throws {
        let coordinator = BatchServingCoordinator()
        let (pending, stream) = await makePending(coordinator)
        _ = await coordinator.enqueueAndClaim(pending)

        await coordinator.markCancelled(pending.id)

        // The stream finishes with no chunks (dropped before decode).
        var chunks = 0
        for try await _ in stream { chunks += 1 }
        #expect(chunks == 0, "a queued-then-cancelled row emits nothing")

        // It is gone from the queue: the next tick admits nothing.
        let tick = await coordinator.takeTick(active: [])
        #expect(tick.admits.isEmpty, "a cancelled queued row must not be admitted")
    }

    /// A cancellation of an ALREADY-DECODING row surfaces in the next tick so the
    /// loop evicts it from the running cohort.
    @Test
    func cancellingActiveRowSurfacesInNextTick() async throws {
        let coordinator = BatchServingCoordinator()
        let (pending, _) = await makePending(coordinator)
        _ = await coordinator.enqueueAndClaim(pending)

        // First tick admits the row; it is now "active".
        let first = await coordinator.takeTick(active: [])
        #expect(first.admits.map { $0.id } == [pending.id])

        // Consumer drops the stream mid-decode.
        await coordinator.markCancelled(pending.id)

        // Next tick, with the row reported active, hands the eviction to the loop.
        let second = await coordinator.takeTick(active: [pending.id])
        #expect(
            second.cancelledActive.contains(pending.id),
            "a cancelled decoding row must be handed to the loop for eviction")
    }

    /// A late cancel (row already finished and evicted) is dropped, not retained —
    /// the cancellation set cannot accumulate dead ids.
    @Test
    func lateCancelDoesNotAccumulate() async throws {
        let coordinator = BatchServingCoordinator()
        let (pending, _) = await makePending(coordinator)
        _ = await coordinator.enqueueAndClaim(pending)
        _ = await coordinator.takeTick(active: [])  // admit

        // Row finished and left the cohort: the loop now reports it gone.
        _ = await coordinator.takeTick(active: [])
        // A late cancel arrives for the departed id.
        await coordinator.markCancelled(pending.id)
        // Next tick (row not active) must NOT surface it — and must prune it.
        let tick = await coordinator.takeTick(active: [])
        #expect(tick.cancelledActive.isEmpty, "a late cancel for a departed row is dropped")
    }

    // MARK: Batch-size headroom

    /// `takeTick` admits at most `prefillBatchSize` per step and never exceeds
    /// `completionBatchSize` total.
    @Test
    func takeTickRespectsBatchSizeLimits() async throws {
        let coordinator = BatchServingCoordinator(completionBatchSize: 3, prefillBatchSize: 2)
        for _ in 0..<5 {
            let (pending, _) = await makePending(coordinator)
            _ = await coordinator.enqueueAndClaim(pending)
        }

        // Idle cohort: first step takes prefillBatchSize (2).
        let first = await coordinator.takeTick(active: [])
        #expect(first.admits.count == 2, "at most prefillBatchSize rows per step")

        // With 2 already active, only 1 more fits under completionBatchSize (3).
        let second = await coordinator.takeTick(active: [10, 11])
        #expect(second.admits.count == 1, "must not exceed completionBatchSize total")

        // Cohort full: no further admits.
        let third = await coordinator.takeTick(active: [10, 11, 12])
        #expect(third.admits.isEmpty, "a full cohort admits nothing")
    }

    /// When a race leaves rows queued as the loop tries to exit (and no drain is
    /// pending), `finishDrive` keeps the loop running for a fresh cohort.
    @Test
    func finishDriveRedrivesWhenRaceLeavesWorkQueued() async throws {
        let coordinator = BatchServingCoordinator()
        let (pending, _) = await makePending(coordinator)
        _ = await coordinator.enqueueAndClaim(pending)
        // Loop decided to exit but never took this row: finishDrive must re-drive.
        let redrive = await coordinator.finishDrive()
        #expect(redrive == true, "queued work at exit must keep the loop driving")
    }

    // MARK: H3 — teardown when the container vanished

    /// If the drive loop finds the container gone, `abortAllQueued` must finish EVERY
    /// queued row's stream with the error, reset `driving`, and leave the coordinator
    /// so a subsequent `beginDrain` does NOT hang (the deadlock this guards against).
    @Test
    func abortAllQueuedFinishesRowsAndUnblocksDrain() async throws {
        let coordinator = BatchServingCoordinator()
        let (p0, s0) = await makePending(coordinator)
        let (p1, s1) = await makePending(coordinator)
        _ = await coordinator.enqueueAndClaim(p0)  // claims driving = true
        _ = await coordinator.enqueueAndClaim(p1)

        await coordinator.abortAllQueued(BatchServingUnavailableError())

        // Every queued row is terminated with the abort error (not left dangling).
        await #expect(throws: BatchServingUnavailableError.self) {
            for try await _ in s0 { Issue.record("an aborted row must emit no chunks") }
        }
        await #expect(throws: BatchServingUnavailableError.self) {
            for try await _ in s1 { Issue.record("an aborted row must emit no chunks") }
        }

        // driving was reset + drain waiters resumed: a subsequent beginDrain returns
        // immediately instead of waiting forever on a loop that will never run. (If
        // abort left driving stuck true, this await would hang and time the test out.)
        await coordinator.beginDrain()
        #expect(await coordinator.isDraining, "drain must complete after an abort")

        // Nothing is left queued: the next tick admits nothing.
        let tick = await coordinator.takeTick(active: [])
        #expect(tick.admits.isEmpty, "aborted rows must be gone from the queue")
    }

    // MARK: M1 — batch only under concurrency

    /// An idle, uncontended request (its submit window is the only one in flight) goes
    /// solo — the seam returns nil and the request keeps the legacy single-stream
    /// path, preserving the engine prompt cache. No row is enqueued.
    @Test
    func idleUncontendedRequestGoesSolo() async throws {
        let coordinator = BatchServingCoordinator()
        await coordinator.enterSubmission()
        let (pending, _) = await makePending(coordinator)
        let decision = await coordinator.claimSoloOrEnqueue(pending)
        await coordinator.exitSubmission()

        guard case .solo = decision else {
            Issue.record("an idle, uncontended request must go solo, got \(decision)")
            return
        }
        let tick = await coordinator.takeTick(active: [])
        #expect(tick.admits.isEmpty, "a solo request must not enqueue a row")
    }

    /// Two overlapping submit windows (a concurrent burst) form a batch: the first
    /// claimant seeds the cohort (launches the drive loop), the second joins it.
    @Test
    func concurrentSubmissionsFormABatch() async throws {
        let coordinator = BatchServingCoordinator()
        await coordinator.enterSubmission()
        await coordinator.enterSubmission()

        let (p0, _) = await makePending(coordinator)
        let d0 = await coordinator.claimSoloOrEnqueue(p0)
        let (p1, _) = await makePending(coordinator)
        let d1 = await coordinator.claimSoloOrEnqueue(p1)
        await coordinator.exitSubmission()
        await coordinator.exitSubmission()

        guard case .admitted(let start0) = d0 else {
            Issue.record("a contended request must join the batch, got \(d0)")
            return
        }
        #expect(start0, "the first batched row must launch the drive loop")
        guard case .admitted(let start1) = d1 else {
            Issue.record("a contended request must join the batch, got \(d1)")
            return
        }
        #expect(!start1, "a second batched row must not start a second loop")
    }

    /// The drain epoch outranks the solo heuristic: while draining, even an idle,
    /// uncontended request is rejected (clause 1), never granted solo.
    @Test
    func drainRefusalWinsOverSolo() async throws {
        let coordinator = BatchServingCoordinator()
        await coordinator.beginDrain()  // idle → returns immediately, epoch closed
        await coordinator.enterSubmission()  // otherwise solo-eligible
        let (pending, _) = await makePending(coordinator)
        let decision = await coordinator.claimSoloOrEnqueue(pending)
        await coordinator.exitSubmission()

        guard case .rejected = decision else {
            Issue.record("a submit during the drain epoch must be rejected, got \(decision)")
            return
        }
    }

    /// A request that is ALONE in its window still JOINS an already-running cohort
    /// (rather than going solo) — once a batch exists, new work batches with it.
    @Test
    func loneRequestJoinsARunningCohort() async throws {
        let coordinator = BatchServingCoordinator()
        let (p0, _) = await makePending(coordinator)
        _ = await coordinator.enqueueAndClaim(p0)  // driving = true

        await coordinator.enterSubmission()
        let (p1, _) = await makePending(coordinator)
        let decision = await coordinator.claimSoloOrEnqueue(p1)
        await coordinator.exitSubmission()

        guard case .admitted(let start) = decision else {
            Issue.record("a request must join the running cohort, got \(decision)")
            return
        }
        #expect(!start, "the cohort is already driving, so no new loop starts")
    }
}
