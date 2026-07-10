// Copyright © 2026 macMLX. English comments only.

import MLXLMCommon

/// The batched-decode core (Track A wave A2a): drives an EQUAL-LENGTH cohort
/// through a single batched prefill and a lockstep decode loop, fanning out
/// per-slot `AsyncThrowingStream<GenerateChunk, Error>` results.
///
/// This is the v1-min increment of continuous batching — it proves per-row
/// sampling, per-row stop, finished-row masking, and per-slot fan-out on top of
/// A1's ``BatchPositionedCacheWrapper``. It deliberately does NOT do: ragged /
/// left-padded prompts (A2b), dynamic admission/eviction (A2c), or server
/// integration (A2d).
///
/// ## Two layers
///  1. **Scheduling core** — ``run()`` plus the injected ``BatchStepEvaluator``
///     and per-slot ``BatchDecodeSlot``s. All MLX-free apart from the evaluator,
///     so the stop/EOS/max-tokens/masking/fan-out logic is unit tested in CI with
///     a scripted stub.
///  2. **Production assembly** — ``make(model:tokenizer:eosTokenIds:cohort:globalMaxTokens:)``
///     runs the ``batchPositioned(_:batch:)`` coverage gate, builds the real
///     ``ModelBatchStepEvaluator`` + `NaiveStreamingDetokenizer`-backed slots, and
///     returns the runner plus one stream per row.
///
/// ## Masking, not shrinking
/// The cohort always decodes full width `B`. When a row finishes, its slot
/// freezes (emits nothing more) but the runner keeps feeding it a pad token so
/// the batch shape — and A1's single shared RoPE offset — advance in lockstep
/// with the still-live rows. The loop ends when ALL rows are finished or the
/// global step cap is hit.
///
/// ## Isolation
/// `run()` drives the whole cohort to completion synchronously and holds
/// non-`Sendable` MLX state (via the evaluator + slots). Construct and run it
/// within a single isolation domain — inside `ModelContainer.perform` today, or
/// the A2c `BatchScheduler` actor later. Only the returned streams (and their
/// `Sendable` `GenerateChunk`s) cross back out to each request's consumer;
/// because `run()` buffers into the streams and finishes them, consumers may
/// drain after it returns.
final class BatchDecodeRunner {
    private let evaluator: any BatchStepEvaluator
    private let slots: [BatchDecodeSlot]
    private let globalMaxTokens: Int

    /// Scheduling-core initializer. Prefer ``make(model:tokenizer:eosTokenIds:cohort:globalMaxTokens:)``
    /// for the production path; this initializer is the seam the CI logic tests
    /// use with a scripted evaluator and scripted-decoder slots.
    ///
    /// - Parameters:
    ///   - evaluator: turns tokens into per-row next tokens (real model or stub).
    ///   - slots: one per row, row-aligned with the evaluator's output (`slots[i].row == i`).
    ///   - globalMaxTokens: hard ceiling on tokens per row (prefill token + decode
    ///     steps) so an unbounded row cannot loop forever.
    init(evaluator: any BatchStepEvaluator, slots: [BatchDecodeSlot], globalMaxTokens: Int) {
        self.evaluator = evaluator
        self.slots = slots
        self.globalMaxTokens = globalMaxTokens
    }

    /// Per-slot generated token IDs, row-ordered. Read after ``run()`` for parity
    /// checks (the streams carry text; this carries the raw token trajectory).
    var slotTokens: [[Int]] {
        slots.map { $0.generatedTokens }
    }

    /// Prefill once, then decode in lockstep until every row is finished (EOS /
    /// stop string / max-tokens) or the global cap is reached. On an evaluator
    /// error or cancellation, every still-open slot stream is failed and (for an
    /// evaluator error) the error is rethrown.
    func run() throws {
        do {
            // These two guards used to sit outside this `do/catch`, so a throw
            // here would leave any already-constructed slots' continuations
            // live-but-never-finished (a stream leak) under the raw
            // `init(evaluator:slots:globalMaxTokens:)` seam. Keeping them inside
            // routes both through the shared `catch` below, which fails every
            // still-open slot before rethrowing (a no-op when `slots` is empty).
            guard !slots.isEmpty else { throw BatchUnsupportedError.emptyCohort }
            let lengths = slots.map { $0.promptTokens.count }
            if Set(lengths).count != 1 {
                throw BatchUnsupportedError.unequalPromptLengths(lengths)
            }

            // Single batched prefill → the first token per row.
            let firstTokens = try evaluator.prefill(slots.map { $0.promptTokens })
            guard firstTokens.count == slots.count else {
                throw BatchUnsupportedError.evaluatorContractViolation(
                    expected: slots.count, got: firstTokens.count)
            }
            for (index, slot) in slots.enumerated() {
                // cap == 0 (this row's own `maxTokens`, or the whole cohort's
                // `globalMaxTokens`) must emit nothing. Ingesting the prefill
                // token unconditionally would still emit 1 token at a 0 cap, so
                // finish the row immediately instead — `feedbackToken` still
                // falls back cleanly (its existing empty-history pad logic),
                // preserving MASK-not-shrink for the rest of the cohort.
                if globalMaxTokens == 0 || slot.maxTokens == 0 {
                    slot.finishAtCap()
                } else {
                    slot.ingest(firstTokens[index])
                }
            }

            // Lockstep decode. `stepIndex` counts the prefill token as step 0, so
            // the ceiling bounds total tokens per row at `globalMaxTokens`.
            var stepIndex = 1
            while !slots.allSatisfy({ $0.isFinished }) && stepIndex < globalMaxTokens {
                if Task.isCancelled {
                    failAll(CancellationError())
                    return
                }
                let fed = slots.map { $0.feedbackToken }
                let nextTokens = try evaluator.step(fed)
                guard nextTokens.count == slots.count else {
                    throw BatchUnsupportedError.evaluatorContractViolation(
                        expected: slots.count, got: nextTokens.count)
                }
                for (index, slot) in slots.enumerated() where !slot.isFinished {
                    slot.ingest(nextTokens[index])
                }
                stepIndex += 1
            }

            // Global cap hit with rows still live → close them as length-limited.
            for slot in slots where !slot.isFinished {
                slot.finishAtCap()
            }
        } catch {
            failAll(error)
            throw error
        }
    }

    private func failAll(_ error: Error) {
        for slot in slots where !slot.isFinished {
            slot.fail(error)
        }
    }

    // MARK: - Production assembly

    /// Build a runner + per-row streams for a real model, applying the coverage
    /// gate. Call this — and ``run()`` — inside the model's isolation domain
    /// (`ModelContainer.perform` or the A2c actor).
    ///
    /// - Throws: ``BatchUnsupportedError/emptyCohort`` for an empty cohort,
    ///   ``BatchUnsupportedError/unequalPromptLengths(_:)`` for a ragged cohort
    ///   (A2b territory), or ``BatchUnsupportedError/cacheNotBatchable`` when
    ///   ``batchPositioned(_:batch:)`` refuses the model's caches. On any throw,
    ///   no streams are created and the caller must route the request(s) through
    ///   the sequential path.
    ///
    /// - Parameters:
    ///   - model: the resident language model (must read `cache.ropeOffset`; the
    ///     model-architecture allowlist is the A2c scheduler's responsibility).
    ///   - tokenizer: used only to detokenize each slot's stream.
    ///   - eosTokenIds: the stop-token set (e.g. from the model config + tokenizer
    ///     EOS); the caller owns this policy.
    ///   - cohort: one ``BatchSlotConfig`` per row, all equal prompt length.
    ///   - globalMaxTokens: hard per-row token ceiling (default 4096).
    static func make(
        model: any LanguageModel,
        tokenizer: any Tokenizer,
        eosTokenIds: Set<Int>,
        cohort: [BatchSlotConfig],
        globalMaxTokens: Int = 4096
    ) throws -> (runner: BatchDecodeRunner, streams: [AsyncThrowingStream<GenerateChunk, Error>]) {
        guard !cohort.isEmpty else { throw BatchUnsupportedError.emptyCohort }
        let lengths = cohort.map { $0.promptTokens.count }
        if Set(lengths).count != 1 {
            throw BatchUnsupportedError.unequalPromptLengths(lengths)
        }

        let batch = cohort.count
        // `parameters: nil` means v1 intentionally discards any per-request KV
        // cache options (`maxKVSize`, `kvBits`, …) from `cohort`'s parameters —
        // the batched path has no consumer for per-row cache configuration
        // before A2d, and all rows must share one cache shape/kind to be
        // batch-positioned below.
        //
        // Coverage gate: refuse before allocating any streams if the model's
        // caches cannot be safely batch-positioned.
        guard let caches = batchPositioned(model.newCache(parameters: nil), batch: batch) else {
            throw BatchUnsupportedError.cacheNotBatchable
        }

        let evaluator = ModelBatchStepEvaluator(
            model: model,
            caches: caches,
            cohortParameters: cohort.map { $0.parameters }
        )

        var slots: [BatchDecodeSlot] = []
        var streams: [AsyncThrowingStream<GenerateChunk, Error>] = []
        slots.reserveCapacity(batch)
        streams.reserveCapacity(batch)
        for (index, config) in cohort.enumerated() {
            let (stream, continuation) = AsyncThrowingStream<GenerateChunk, Error>.makeStream()
            let slot = BatchDecodeSlot(
                row: index,
                promptTokens: config.promptTokens,
                parameters: config.parameters,
                maxTokens: config.parameters.maxTokens,
                eosTokenIds: eosTokenIds,
                unknownTokenId: tokenizer.unknownTokenId,
                stopStrings: config.stopStrings,
                textDecoder: NaiveIncrementalTextDecoder(tokenizer: tokenizer),
                continuation: continuation
            )
            slots.append(slot)
            streams.append(stream)
        }

        let runner = BatchDecodeRunner(
            evaluator: evaluator, slots: slots, globalMaxTokens: globalMaxTokens)
        return (runner, streams)
    }
}
