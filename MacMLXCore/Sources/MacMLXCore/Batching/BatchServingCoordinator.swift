// Copyright © 2026 macMLX. English comments only.

import MLXLMCommon

/// MLX-free coordination core for the engine-side ``BatchGenerationServing`` seam
/// (Track A wave A2d-2). Owns the `Sendable` admission state — the pending queue,
/// the submit/drain epoch gate, per-row cancellation, and the single-drive-loop
/// claim — so the concurrency clauses of ``BatchGenerationServing`` can be unit
/// tested in CI with no model and no Metal.
///
/// ## Why this is split out of the engine
/// The MLX drive loop lives inside a long-lived `ModelContainer.perform` closure
/// (it owns the model + running `[BatchKVCache]` + per-row ``BatchDecodeSlot``s,
/// none of which are `Sendable`, and it holds the container's serial-access mutex
/// for the cohort's lifetime — see ``MLXSwiftEngine`` batch serving). That closure
/// cannot itself be exercised without Metal. This actor carries every decision the
/// five-clause contract turns on but that needs no model, so those decisions are
/// covered by a plain `swift test`, mirroring how A2c split ``BatchInferenceCore``
/// (one production driver + a scripted stub) from ``BatchScheduler``'s logic.
///
/// ## Clauses enforced here
///  - **submit/drain epoch (clause 1):** ``enqueueAndClaim(_:)`` refuses (the seam
///    returns `nil`) once ``beginDrain()`` has started, and keeps refusing until
///    ``resumeAdmissions()`` runs after the swap on the next load. Both live on this
///    one serial actor, so no row is admitted partway through a drain.
///  - **drain waits for the cohort (clause 4 liveness):** ``beginDrain()`` blocks
///    until the drive loop reports idle via ``finishDrive()``, awaiting a private
///    continuation — it never touches the server's FIFO generation lock.
///  - **dropped stream evicts its row (clause 2):** ``markCancelled(_:)`` finishes a
///    still-queued row's stream immediately (it never reached the cohort), or, for
///    a decoding row, surfaces the id in the next ``Tick`` so the loop evicts it.
///
/// Only `Sendable` values cross this boundary: ``BatchSlotConfig`` and the stream
/// ``AsyncThrowingStream/Continuation`` (which is `Sendable`) in, ``Tick`` out.
actor BatchServingCoordinator {

    /// One admitted-but-not-yet-decoding request: its stable id, its `Sendable`
    /// cohort config, and its `Sendable` output continuation. All three cross into
    /// the drive loop's isolation domain, so all three are `Sendable`.
    struct Pending: Sendable {
        let id: Int
        let config: BatchSlotConfig
        let continuation: AsyncThrowingStream<GenerateChunk, Error>.Continuation
    }

    /// The outcome of an ``enqueueAndClaim(_:)`` call: whether the row was admitted
    /// (a `false` routes the caller to the legacy path) and whether THIS caller must
    /// launch the drive loop (`true` only for the row that found the loop idle).
    struct Admission: Sendable {
        let accepted: Bool
        let shouldStartDrive: Bool
    }

    /// The outcome of a ``claimSoloOrEnqueue(_:)`` call — the "batch only under
    /// concurrency" admission heuristic (M1):
    ///  - ``rejected``: a drain epoch is open (clause 1) — the seam returns `nil` and
    ///    the request takes the legacy single-stream path.
    ///  - ``solo``: the seam was idle (no cohort driving, nothing queued, no other
    ///    solo admission in its window) — the seam returns `nil` so the request runs
    ///    on the legacy single-stream path, which keeps the engine's prompt cache
    ///    warm across a single client's successive turns (the flagship agent case).
    ///    No row, no promise, no stream (clauses 2/3 never fire).
    ///  - ``admitted``: a cohort is already forming (driving or queued, or a solo
    ///    admission is mid-window), so this request JOINS the batch; `startDrive` is
    ///    `true` only for the row that must launch the drive loop.
    enum SubmitDecision: Sendable {
        case rejected
        case solo
        case admitted(startDrive: Bool)
    }

    /// One scheduling step's worth of work for the drive loop: rows to admit
    /// (already dequeued, bounded by the batch-size limits) and the ids of
    /// currently-decoding rows whose consumer cancelled (evict them).
    struct Tick: Sendable {
        let admits: [Pending]
        let cancelledActive: Set<Int>
    }

    /// Max concurrent decoding rows (Python `completion_batch_size`).
    private let completionBatchSize: Int
    /// Max rows admitted per scheduling step (Python `prefill_batch_size`).
    private let prefillBatchSize: Int

    private var queue: [Pending] = []
    /// Ids of decoding rows whose consumer cancelled, not yet handed to the loop.
    private var cancelledActive: Set<Int> = []
    /// The loop's current row ids as of the last ``takeTick(active:)`` (plus that
    /// tick's fresh admits), so ``markCancelled(_:)`` can tell "still decoding" from
    /// "already gone" without asking the loop.
    private var activeIDs: Set<Int> = []
    private var draining = false
    private var driving = false
    private var drainWaiters: [CheckedContinuation<Void, Never>] = []
    private var nextID = 0
    /// Requests currently inside their `submit` window — bracketed by
    /// ``enterSubmission()`` (before admission tokenization) and ``exitSubmission()``
    /// (as `submit` returns). This is the concurrency signal for the "batch only under
    /// concurrency" heuristic (M1): a request goes solo only when it is the SOLE
    /// submission in flight. Incrementing BEFORE tokenization is what makes it
    /// reliable — several truly-concurrent requests all enter here before any of them
    /// reaches its claim, so the first claimant already sees the others and seeds a
    /// cohort they all join, even though the admission actor serializes their
    /// tokenization.
    private var activeSubmissions = 0

    init(completionBatchSize: Int = 32, prefillBatchSize: Int = 8) {
        self.completionBatchSize = max(1, completionBatchSize)
        self.prefillBatchSize = max(1, min(prefillBatchSize, completionBatchSize))
    }

    // MARK: - Submission

    /// Assign a fresh, never-reused stable id for a new request. The caller needs it
    /// BEFORE building the stream so the stream's `onTermination` can key its
    /// cancellation back to ``markCancelled(_:)``.
    func nextRequestID() -> Int {
        let id = nextID
        nextID += 1
        return id
    }

    /// Clause 1 + drive-claim, atomically. Refuses once a drain has begun (so the
    /// seam returns `nil`); otherwise enqueues and reports whether this caller must
    /// launch the loop. Claiming `driving` here — in the same actor step as the
    /// enqueue — closes the race where a row is committed but no loop is running yet
    /// (``beginDrain()`` would then wait forever): after this returns `accepted`,
    /// `driving` is already claimed, and the caller's synchronous `Task { … }` tail
    /// (no `await` before it) is guaranteed to start the loop.
    func enqueueAndClaim(_ pending: Pending) -> Admission {
        guard !draining else { return Admission(accepted: false, shouldStartDrive: false) }
        queue.append(pending)
        if driving {
            return Admission(accepted: true, shouldStartDrive: false)
        }
        driving = true
        return Admission(accepted: true, shouldStartDrive: true)
    }

    /// Open a `submit` window (M1). Called at the top of the seam's `submit`, BEFORE
    /// admission tokenization, so concurrent requests are all counted before any of
    /// them claims. Balanced by ``exitSubmission()``.
    func enterSubmission() {
        activeSubmissions += 1
    }

    /// Close a `submit` window (M1). Clamped at zero so an unbalanced call can never
    /// go negative and permanently starve the solo path.
    func exitSubmission() {
        if activeSubmissions > 0 { activeSubmissions -= 1 }
    }

    /// Clause 1 + the "batch only under concurrency" heuristic (M1), atomically (one
    /// actor step, no `await` gap — so the idle/solo verdict can't be split by an
    /// interleaving admission). Order matters: the drain epoch is checked FIRST so a
    /// refusal always wins over a solo grant; then a seam that is idle AND carrying no
    /// other in-flight submission grants `.solo` (legacy path, prompt-cache
    /// preserving); otherwise the request joins the batch via the same enqueue+claim
    /// as ``enqueueAndClaim(_:)``.
    ///
    /// The `activeSubmissions <= 1` guard is what turns the literal "idle → solo" rule
    /// into a WORKING heuristic. Without it, `!driving && queue.isEmpty` is chicken-
    /// and-egg: every cold-start request (including the first of a concurrent burst)
    /// finds the seam idle and goes solo, so a cohort never forms at all. Counting the
    /// in-flight submissions lets the first claimant of a concurrent burst see the
    /// others (they all entered their window before it claimed) and seed a cohort the
    /// rest join — while a lone sequential request, whose window never overlaps the
    /// next turn's, still goes solo and keeps the engine prompt cache warm.
    ///
    /// Concurrency note (accepted, per review): two requests whose submit windows do
    /// NOT overlap — the first returns before the second enters — each see
    /// `activeSubmissions == 1` and both go `.solo`, running as two serialized legacy
    /// requests on the container mutex. That misses batching ONCE; it is a throughput
    /// miss, never a correctness problem (each still produces its own correct output).
    func claimSoloOrEnqueue(_ pending: Pending) -> SubmitDecision {
        if draining { return .rejected }
        if !driving && queue.isEmpty && activeSubmissions <= 1 {
            return .solo
        }
        // Not idle / concurrent → join the forming or running cohort.
        // `enqueueAndClaim` re-checks `draining` (already false here) and never
        // refuses, so `accepted` holds.
        let admission = enqueueAndClaim(pending)
        return .admitted(startDrive: admission.shouldStartDrive)
    }

    // MARK: - Drive loop

    /// Pull the next scheduling step. `active` is the loop's current row ids, so the
    /// coordinator can intersect them with pending cancellations and prune ids whose
    /// rows already left the cohort. Admits up to the batch-size headroom.
    func takeTick(active: [Int]) -> Tick {
        let activeSet = Set(active)
        let cancels = cancelledActive.intersection(activeSet)

        let room = completionBatchSize - active.count
        let take = max(0, min(room, prefillBatchSize, queue.count))
        let admits = Array(queue.prefix(take))
        queue.removeFirst(take)
        let admitIDs = Set(admits.map { $0.id })

        // Retain only cancellations still relevant — active this tick or freshly
        // admitted (so a row cancelled in the one-tick window between admit and its
        // first appearance in `active` is still evicted next tick). Everything else
        // (handed off now, or a row that already finished and left) is dropped, so
        // this set can't accumulate over a long-lived engine.
        cancelledActive = cancelledActive.subtracting(cancels).intersection(activeSet.union(admitIDs))
        activeIDs = activeSet.union(admitIDs)
        return Tick(admits: admits, cancelledActive: cancels)
    }

    /// The drive loop drained its cohort and wants to exit. Returns `true` to keep
    /// the loop running for a fresh cohort when a race left new rows queued (and no
    /// drain is pending); returns `false` after releasing the drive claim and
    /// resuming any ``beginDrain()`` waiter. Either way `driving` stays consistent
    /// with the caller's next action.
    func finishDrive() -> Bool {
        if !draining && !queue.isEmpty {
            return true
        }
        driving = false
        activeIDs = []
        cancelledActive = []
        resumeDrainWaiters()
        return false
    }

    /// Tear down the whole admission state when the drive loop cannot run at all —
    /// the resident container vanished between admission and drive-loop start (H3).
    /// Finishes EVERY queued row's stream with `error` (admission was a promise; there
    /// is no legacy fall-back left for an already-admitted row), then releases the
    /// drive claim and resumes any ``beginDrain()`` waiter.
    ///
    /// Without this, the container-nil path would drop the `true`/`false` drive
    /// verdict on the floor: queued rows would hang forever, `driving` would stay
    /// stuck `true`, and the next ``beginDrain()`` — issued from under the server's
    /// FIFO generation lock — would wait on a loop that never runs, deadlocking the
    /// server. Resuming drain waiters here is what keeps that swap/unload live.
    func abortAllQueued(_ error: Error) {
        let doomed = queue
        queue = []
        for pending in doomed {
            pending.continuation.finish(throwing: error)
        }
        driving = false
        activeIDs = []
        cancelledActive = []
        resumeDrainWaiters()
    }

    // MARK: - Cancellation (clause 2)

    /// The consumer dropped this request's stream. If it is still queued (never
    /// decoded), finish its stream now and drop it. If it is decoding, remember the
    /// id so the next ``takeTick(active:)`` tells the loop to evict it.
    ///
    /// A late cancel (fired after the row already finished and left the cohort) is
    /// harmless: the id is not queued and not in `activeIDs`, so it is inserted but
    /// pruned on the very next ``takeTick(active:)``.
    func markCancelled(_ id: Int) {
        if let index = queue.firstIndex(where: { $0.id == id }) {
            let pending = queue.remove(at: index)
            pending.continuation.finish()
            return
        }
        cancelledActive.insert(id)
    }

    // MARK: - Drain epoch (clauses 1 + 4)

    /// Begin a drain: refuse further admissions and block until the drive loop has
    /// fully drained its cohort (``finishDrive()`` with nothing left). Returns
    /// immediately when no loop is running. Never acquires the server's generation
    /// lock — it only awaits a private continuation the loop resumes.
    ///
    /// Known deferred (M2, fast-follow): there is no grace-period cap or forced
    /// cancellation here — the drain waits for the cohort to reach zero naturally, so
    /// a model swap can stall for up to a full row's generation. This is a latency
    /// cliff, not a correctness bug (H1's step budget still keeps single-stream
    /// `container.prepare` responsive throughout the drain); bounding it with a
    /// grace-period-then-cancel is a follow-up.
    func beginDrain() async {
        draining = true
        guard driving else { return }
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            drainWaiters.append(continuation)
        }
    }

    /// Re-open admissions after a model swap/reload has completed (the same engine
    /// instance is reused across a CLI cold-swap, so its coordinator must be told the
    /// drain epoch is over). A no-op when not draining.
    func resumeAdmissions() {
        draining = false
    }

    /// Whether admissions are currently refused (drain in progress). Test/inspection
    /// hook — the seam decides `nil` via ``enqueueAndClaim(_:)``'s result, not this.
    var isDraining: Bool {
        draining
    }

    private func resumeDrainWaiters() {
        guard !drainWaiters.isEmpty else { return }
        let waiters = drainWaiters
        drainWaiters = []
        for waiter in waiters {
            waiter.resume()
        }
    }
}
