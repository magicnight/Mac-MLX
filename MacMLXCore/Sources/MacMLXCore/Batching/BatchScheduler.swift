// Copyright © 2026 macMLX. English comments only.

import MLXLMCommon

/// Continuous-batching scheduler (Track A wave A2c): serves many concurrent
/// generation requests over ONE resident model by batching them at the decode
/// step, admitting new requests and evicting finished ones mid-flight.
///
/// This is the Swift analogue of mlx-lm's `BatchGenerator` (`generate.py`),
/// specialised to macMLX's hard constraint (mlx-swift#441) that only the DECODE
/// phase may batch: prefill is per-row B=1. It wires the A2b ragged cache surface
/// (``BatchKVCache`` `extend`/`filter`) into a live loop through the
/// ``BatchInferenceCore`` seam.
///
/// ## Lifecycle
/// ``submit(_:)`` returns one `AsyncThrowingStream<GenerateChunk, Error>` per
/// request and enqueues it. A single background drive loop runs the Python
/// `_next` schedule while any work remains:
///  1. **decode** the running cohort one lockstep step (``BatchInferenceCore/decode(_:)``),
///  2. **evict** rows that just finished (EOS / stop string / max-tokens /
///     consumer cancellation) via ``BatchInferenceCore/evict(keeping:)``,
///  3. **admit** waiting requests up to `completionBatchSize`, prefilling each
///     B=1 and merging it onto the cohort (``BatchInferenceCore/admit(_:)``).
/// The loop exits when the queue and cohort are both empty; ``submit(_:)``
/// restarts it. Because rows are admitted whenever there is room, a long request
/// no longer blocks short ones behind it (unlike a static cohort).
///
/// ## Masking vs shrinking
/// Unlike A2a's equal-length ``BatchDecodeRunner`` (which MASKS finished rows and
/// keeps decoding full width), A2c SHRINKS: a finished row is evicted with
/// ``BatchKVCache/filter(batchIndices:)`` so the cohort stops spending compute on
/// it. Eviction carries a `.item()` host-sync, so the loop performs at most one
/// `evict` per decode step (all rows that finished on the same step are dropped
/// together).
///
/// ## Per-slot cancellation
/// Each stream's `onTermination` reports a consumer cancellation back to the
/// actor, which drops that row (from the queue if still waiting, or from the
/// cohort on the next step if decoding) WITHOUT disturbing the other rows —
/// closing the A2a debt where an abandoned stream could never stop its work.
///
/// ## Isolation / Swift 6
/// The `actor` is the single isolation domain that owns the non-`Sendable`
/// ``BatchInferenceCore`` (model + running `[BatchKVCache]` + per-row samplers)
/// and every live ``BatchDecodeSlot``. Only `Sendable` values cross the
/// boundary: `BatchSlotConfig` in, `GenerateChunk`s out through each stream's
/// continuation. `core`/`tokenizer` are handed in via `sending` so the actor
/// takes exclusive ownership. Heavy MLX work runs on the actor's executor and is
/// therefore serialised — exactly the single serial ModelContext access the
/// non-batched path uses, but batched at the step level.
///
/// Internal to `MacMLXCore` (like ``BatchSlotConfig`` / ``BatchStepEvaluator``);
/// the in-module server (A2d) is its consumer.
actor BatchScheduler {

    /// One request tracked by the scheduler: its stable id (for cancellation
    /// matching), its cohort config, and its live per-row decode/stream state.
    /// A private implementation detail of the actor's book-keeping.
    private struct ScheduledRow {
        let id: Int
        let config: BatchSlotConfig
        let slot: BatchDecodeSlot
    }

    // MARK: Injected model-level state

    private let core: any BatchInferenceCore
    private let tokenizer: any Tokenizer
    private let eosTokenIds: Set<Int>
    private let unknownTokenId: Int?

    // MARK: Configuration

    /// Max concurrent decoding rows (Python `completion_batch_size`, default 32).
    private let completionBatchSize: Int
    /// Max rows admitted per scheduling step (Python `prefill_batch_size`,
    /// default 8), bounding the per-step B=1 prefill burst.
    private let prefillBatchSize: Int
    /// Hard per-row ceiling on generated tokens (safety net above each row's own
    /// `maxTokens`).
    private let globalMaxTokens: Int

    // MARK: Mutable state (actor-isolated)

    private var queue: [ScheduledRow] = []
    private var active: [ScheduledRow] = []
    private var cancelledIDs: Set<Int> = []
    private var isDriving = false
    private var nextID = 0

    /// - Parameters:
    ///   - core: the MLX seam (real ``ModelBatchInferenceCore`` or a stub);
    ///     handed over with `sending` so the actor owns it exclusively.
    ///   - tokenizer: detokenizes each slot's stream (per-row incremental).
    ///   - eosTokenIds: the stop-token set (model config + tokenizer EOS).
    ///   - unknownTokenId: treated as a stop token, like the single-stream path.
    ///   - completionBatchSize: max concurrent decoding rows (default 32).
    ///   - prefillBatchSize: max rows admitted per step (default 8).
    ///   - globalMaxTokens: hard per-row token ceiling (default 4096).
    init(
        core: sending any BatchInferenceCore,
        tokenizer: sending any Tokenizer,
        eosTokenIds: Set<Int>,
        unknownTokenId: Int?,
        completionBatchSize: Int = 32,
        prefillBatchSize: Int = 8,
        globalMaxTokens: Int = 4096
    ) {
        self.core = core
        self.tokenizer = tokenizer
        self.eosTokenIds = eosTokenIds
        self.unknownTokenId = unknownTokenId
        self.completionBatchSize = max(1, completionBatchSize)
        self.prefillBatchSize = max(1, min(prefillBatchSize, completionBatchSize))
        self.globalMaxTokens = globalMaxTokens
    }

    // MARK: - Submission

    /// Enqueue one request and return its per-slot output stream. The stream
    /// yields `GenerateChunk`s as the row decodes and finishes with a terminal
    /// chunk (or an error). Dropping/cancelling the stream removes the row.
    ///
    /// Actor-isolated so the stable id is assigned and the row enqueued
    /// atomically; the stream is still available to the caller as soon as the
    /// `await` returns.
    func submit(_ config: BatchSlotConfig) -> AsyncThrowingStream<GenerateChunk, Error> {
        let id = nextID
        nextID += 1

        let (stream, continuation) = AsyncThrowingStream<GenerateChunk, Error>.makeStream()
        let slot = BatchDecodeSlot(
            row: id,
            promptTokens: config.promptTokens,
            parameters: config.parameters,
            maxTokens: config.parameters.maxTokens,
            eosTokenIds: eosTokenIds,
            unknownTokenId: unknownTokenId,
            stopStrings: config.stopStrings,
            textDecoder: NaiveIncrementalTextDecoder(tokenizer: tokenizer),
            continuation: continuation
        )
        queue.append(ScheduledRow(id: id, config: config, slot: slot))

        // Report a consumer cancellation (stream dropped) back to the actor.
        // `.finished` (our own natural finish) is ignored — only a consumer
        // abandon removes an otherwise-live row.
        continuation.onTermination = { @Sendable reason in
            if case .cancelled = reason {
                Task { await self.markCancelled(id) }
            }
        }

        driveIfNeeded()
        return stream
    }

    private func markCancelled(_ id: Int) {
        // A late cancellation can arrive after its row already finished and
        // left the books (`onTermination` also fires when the consumer's task
        // is cancelled post-completion, and the hop back to the actor is
        // async). Only track ids still queued or active, so `cancelledIDs`
        // cannot accumulate dead entries over a long-lived scheduler.
        guard queue.contains(where: { $0.id == id })
            || active.contains(where: { $0.id == id })
        else { return }
        cancelledIDs.insert(id)
        driveIfNeeded()
    }

    // MARK: - Drive loop

    private func driveIfNeeded() {
        guard !isDriving else { return }
        isDriving = true
        Task { await self.runLoop() }
    }

    /// Run the `_next` schedule until the queue and cohort are both empty.
    private func runLoop() async {
        defer { isDriving = false }
        while true {
            // Exit decision with NO `await` before the return: a `submit`
            // racing this either enqueued before it (loop continues and admits)
            // or runs entirely after `isDriving` is cleared (and restarts us).
            if active.isEmpty && queue.isEmpty {
                return
            }
            do {
                try stepDecode()
                try stepAdmit()
            } catch {
                failAll(error)
                return
            }
            // Yield so pending `submit`/`markCancelled` calls interleave between
            // decode steps rather than starving behind the synchronous MLX work.
            await Task.yield()
        }
    }

    /// One lockstep decode over the running cohort, then evict rows that just
    /// finished (natural stop or consumer cancellation).
    private func stepDecode() throws {
        guard !active.isEmpty else { return }

        // Consumer-cancelled rows: finish (a no-op yield to an already-dead
        // stream) so they are excluded below and evicted after this step.
        for row in active where cancelledIDs.contains(row.id) && !row.slot.isFinished {
            row.slot.fail(CancellationError())
        }

        let feedback = active.map { $0.slot.feedbackToken }
        let next = try core.decode(feedback)
        guard next.count == active.count else {
            throw BatchUnsupportedError.evaluatorContractViolation(
                expected: active.count, got: next.count)
        }

        for (index, row) in active.enumerated() where !row.slot.isFinished {
            row.slot.ingest(next[index])
            // Hard per-row ceiling above the row's own maxTokens.
            if !row.slot.isFinished, row.slot.generatedTokens.count >= globalMaxTokens {
                row.slot.finishAtCap()
            }
        }
        evictFinishedRows()
    }

    /// Admit waiting requests up to `completionBatchSize`, prefilling each B=1
    /// and merging it onto the cohort.
    private func stepAdmit() throws {
        guard active.count < completionBatchSize, !queue.isEmpty else { return }
        let take = min(completionBatchSize - active.count, prefillBatchSize, queue.count)
        guard take > 0 else { return }

        let batch = Array(queue.prefix(take))
        queue.removeFirst(take)

        var admits: [ScheduledRow] = []
        admits.reserveCapacity(batch.count)
        for row in batch {
            // A consumer that abandoned while queued: drop without admitting.
            if cancelledIDs.contains(row.id) {
                row.slot.fail(CancellationError())
                cancelledIDs.remove(row.id)
                continue
            }
            // Zero-cap rows emit nothing — finish now, never admit to the cohort
            // (matches the A2a runner's maxTokens==0 guard; avoids a prefill +
            // immediate evict for a row that would emit a single spurious token).
            if globalMaxTokens == 0 || row.slot.maxTokens == 0 {
                row.slot.finishAtCap()
                continue
            }
            admits.append(row)
        }
        guard !admits.isEmpty else { return }

        let firstTokens = try core.admit(admits.map { $0.config })
        guard firstTokens.count == admits.count else {
            throw BatchUnsupportedError.evaluatorContractViolation(
                expected: admits.count, got: firstTokens.count)
        }

        for (index, row) in admits.enumerated() {
            row.slot.ingest(firstTokens[index])
            if !row.slot.isFinished, row.slot.generatedTokens.count >= globalMaxTokens {
                row.slot.finishAtCap()
            }
            active.append(row)
        }
        // A row may stop on its very first token (EOS / stop string / maxTokens
        // == 1); it was merged into the cohort above, so shrink it back out now.
        evictFinishedRows()
    }

    /// Drop every finished row from the cohort in ONE `evict` (the host-sync in
    /// `filter` is paid once per step). No-op when no row finished.
    private func evictFinishedRows() {
        let survivors = active.enumerated().filter { !$0.element.slot.isFinished }
        guard survivors.count < active.count else { return }
        // Release cancellation bookkeeping for the rows leaving the cohort.
        for row in active where row.slot.isFinished {
            cancelledIDs.remove(row.id)
        }
        core.evict(keeping: survivors.map { $0.offset })
        active = survivors.map { $0.element }
    }

    /// Fail every open stream (queued and active) and reset — a core MLX error
    /// aborts the whole cohort (the batched forward is shared across rows).
    private func failAll(_ error: Error) {
        for row in active where !row.slot.isFinished {
            row.slot.fail(error)
        }
        for row in queue where !row.slot.isFinished {
            row.slot.fail(error)
        }
        active = []
        queue = []
        cancelledIDs = []
        core.evict(keeping: [])
    }
}
