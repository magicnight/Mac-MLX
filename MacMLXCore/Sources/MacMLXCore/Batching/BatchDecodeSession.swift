// Copyright © 2026 macMLX. English comments only.

import MLXLMCommon

/// The MLX-side per-cohort driver for the engine ``BatchGenerationServing`` seam
/// (Track A wave A2d-2): owns the running ``BatchInferenceCore`` plus one
/// ``BatchDecodeSlot`` per live row, and is pumped step-by-step by the engine's
/// drive loop (which runs inside a long-lived `ModelContainer.perform`, holding the
/// container's serial-access mutex for the cohort's lifetime).
///
/// It reuses A2c's admit / decode / evict primitives and per-row stop-string /
/// EOS / max-tokens / fan-out logic, but — unlike ``BatchScheduler`` — it does NOT
/// own the model on its own actor executor. §7d of the v0.6 master plan requires
/// the scheduler to run inside the engine's isolation domain, because a production
/// model is only ever resident inside a `ModelContainer` and `perform`'s
/// `sending R: Sendable` constraint will not surrender the non-`Sendable`
/// `model`/`tokenizer` to build a separate ``BatchScheduler`` actor. So the cohort
/// state lives HERE, inside the closure, and only `Sendable` values cross the
/// boundary: ``BatchServingCoordinator/Pending`` in, `GenerateChunk`s out through
/// each slot's continuation.
///
/// ## Continuous, not static
/// Admission and eviction are mid-flight: ``admit(_:)`` merges new rows onto the
/// running cohort (each B=1 prefilled — the proven scalar-offset path that
/// sidesteps mlx-swift#441), and ``decodeStep(cancelled:)`` shrinks the cohort as
/// rows finish (EOS / stop string / max-tokens / consumer cancellation) via
/// ``BatchInferenceCore/evict(keeping:)``. This is the same continuous schedule as
/// ``BatchScheduler``, so a long request no longer blocks short ones behind it.
///
/// ## Isolation
/// A `final class` holding non-`Sendable` MLX state. Not `Sendable`: construct and
/// pump it inside ONE `container.perform` closure. The `BatchInferenceCore` seam is
/// injected so this driver's orchestration is unit-tested MLX-free with a scripted
/// core (as A2c tests ``BatchScheduler``), while production uses
/// ``ModelBatchInferenceCore`` over the real model.
final class BatchDecodeSession {

    /// One live row: its stable id (for cancellation matching) and its per-row
    /// decode/stream state. Aligned in cohort order with the core's internal rows.
    private struct Row {
        let id: Int
        let slot: BatchDecodeSlot
    }

    private let core: any BatchInferenceCore
    private let tokenizer: any Tokenizer
    private let eosTokenIds: Set<Int>
    private let unknownTokenId: Int?
    private let globalMaxTokens: Int

    private var rows: [Row] = []

    init(
        core: any BatchInferenceCore,
        tokenizer: any Tokenizer,
        eosTokenIds: Set<Int>,
        unknownTokenId: Int?,
        globalMaxTokens: Int
    ) {
        self.core = core
        self.tokenizer = tokenizer
        self.eosTokenIds = eosTokenIds
        self.unknownTokenId = unknownTokenId
        self.globalMaxTokens = globalMaxTokens
    }

    /// Live row ids in cohort order — handed to the coordinator each tick so it can
    /// intersect pending cancellations and tell the loop which rows to evict.
    var activeIDs: [Int] { rows.map { $0.id } }

    /// Whether the cohort is empty (drive loop exits when this and the queue agree).
    var isEmpty: Bool { rows.isEmpty }

    /// One lockstep decode over the live cohort. First fails any rows whose consumer
    /// cancelled this tick (`cancelled`), then runs one batched forward over the
    /// rest, ingests per row, and evicts every row that just finished (natural stop
    /// or the cancellations above) in a single ``BatchInferenceCore/evict(keeping:)``.
    func decodeStep(cancelled: Set<Int>) throws {
        for row in rows where cancelled.contains(row.id) && !row.slot.isFinished {
            row.slot.fail(CancellationError())
        }
        guard !rows.isEmpty else { return }

        // Known deferred (L2, fast-follow): if EVERY row was cancelled just above, the
        // rows are marked finished but not yet evicted, so this still runs one batched
        // forward over an all-finished cohort whose output is discarded by `evictFinished`.
        // Skipping that single wasted step is a micro-optimization, not correctness.
        let feedback = rows.map { $0.slot.feedbackToken }
        let next = try core.decode(feedback)
        guard next.count == rows.count else {
            throw BatchUnsupportedError.evaluatorContractViolation(
                expected: rows.count, got: next.count)
        }
        for (index, row) in rows.enumerated() where !row.slot.isFinished {
            row.slot.ingest(next[index])
            // Hard per-row ceiling above the row's own maxTokens.
            if !row.slot.isFinished, row.slot.generatedTokens.count >= globalMaxTokens {
                row.slot.finishAtCap()
            }
        }
        evictFinished()
    }

    /// Prefill each new request B=1 and merge it onto the running cohort, returning
    /// nothing (each slot streams its own tokens). Zero-cap rows are finished
    /// without ever entering the cohort (mirrors ``BatchScheduler``). On a core
    /// failure, the slots built for THIS admit are failed before rethrowing, so a
    /// dropped admit never leaks a live-but-never-finished stream.
    func admit(_ pendings: [BatchServingCoordinator.Pending]) throws {
        guard !pendings.isEmpty else { return }

        var configs: [BatchSlotConfig] = []
        var admitted: [Row] = []
        configs.reserveCapacity(pendings.count)
        admitted.reserveCapacity(pendings.count)
        for pending in pendings {
            let slot = BatchDecodeSlot(
                row: pending.id,
                promptTokens: pending.config.promptTokens,
                parameters: pending.config.parameters,
                maxTokens: pending.config.parameters.maxTokens,
                eosTokenIds: eosTokenIds,
                unknownTokenId: unknownTokenId,
                stopStrings: pending.config.stopStrings,
                textDecoder: NaiveIncrementalTextDecoder(tokenizer: tokenizer),
                continuation: pending.continuation
            )
            // A zero-cap row emits nothing — finish now, never admit to the cohort
            // (matches A2c: avoids a prefill + immediate evict for a single spurious
            // token).
            if globalMaxTokens == 0 || slot.maxTokens == 0 {
                slot.finishAtCap()
                continue
            }
            configs.append(pending.config)
            admitted.append(Row(id: pending.id, slot: slot))
        }
        guard !configs.isEmpty else { return }

        let firstTokens: [Int]
        do {
            firstTokens = try core.admit(configs)
        } catch {
            for row in admitted where !row.slot.isFinished { row.slot.fail(error) }
            throw error
        }
        guard firstTokens.count == admitted.count else {
            let violation = BatchUnsupportedError.evaluatorContractViolation(
                expected: admitted.count, got: firstTokens.count)
            for row in admitted where !row.slot.isFinished { row.slot.fail(violation) }
            throw violation
        }

        for (index, row) in admitted.enumerated() {
            row.slot.ingest(firstTokens[index])
            if !row.slot.isFinished, row.slot.generatedTokens.count >= globalMaxTokens {
                row.slot.finishAtCap()
            }
            rows.append(row)
        }
        // A row may stop on its very first token (EOS / stop string / maxTokens==1);
        // it was merged above, so shrink it back out now.
        evictFinished()
    }

    /// Fail every open stream and drop the cohort — a shared-forward MLX error
    /// aborts all rows (the batched decode is shared across them).
    func failAll(_ error: Error) {
        for row in rows where !row.slot.isFinished { row.slot.fail(error) }
        rows = []
        core.evict(keeping: [])
    }

    /// Drop every finished row from the cohort in ONE `evict` (the host-sync in
    /// `filter` is paid once per step). No-op when no row finished.
    private func evictFinished() {
        let survivors = rows.enumerated().filter { !$0.element.slot.isFinished }
        guard survivors.count < rows.count else { return }
        core.evict(keeping: survivors.map { $0.offset })
        rows = survivors.map { $0.element }
    }
}
