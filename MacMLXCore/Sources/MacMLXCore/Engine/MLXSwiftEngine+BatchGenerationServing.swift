// Copyright © 2026 macMLX. English comments only.

import MLXLLM
import MLXLMCommon

/// A2d-2: `MLXSwiftEngine` IS the continuous-batching seam. §7d of the v0.6 master
/// plan (option A, ratified) puts the scheduler inside the engine's isolation
/// domain — a production model is only ever resident inside a `ModelContainer`,
/// whose `perform`'s `sending R: Sendable` constraint will not surrender the
/// non-`Sendable` `model`/`tokenizer` to build a separate ``BatchScheduler`` actor.
/// So the engine owns the batch drive loop, running it inside a long-lived
/// `container.perform` closure, and coordinates the loop against its own other
/// `container.perform`/`prepare` calls through ONE model-mutex primitive.
///
/// ## The model-mutex primitive (converged, no new lock)
/// `ModelContainer` wraps a `SerialAccessContainer`, whose `AsyncMutex` holds
/// exclusive access for the ENTIRE duration of a `perform` closure — including
/// across `await` suspensions inside it (unlike an actor, which is re-entrant and
/// is therefore *not* a lock; this is the D1 review lesson made load-bearing). The
/// batch drive loop runs as one such closure for its cohort's lifetime, so it is
/// already mutually exclusive with every other `container.perform` on the same
/// model — no additional gate is introduced. Admission tokenization is the one
/// piece that must NOT wait behind the loop (or the four-client cohort would
/// serialize), so it runs off-lock through a processor captured once at load, which
/// is exactly what upstream `ModelContainer.prepare` itself does (fetch processor,
/// run it outside the read lock).
///
/// ## Coverage is decided once, at load
/// The dense-cache + verified-`ropeOffset` architecture gates (and the stop-token
/// set + processor) are captured by a cheap MLX-free probe at load, while the mutex
/// is free — never inside ``submit(_:)`` (which must not wait behind the loop).
/// ``submit(_:)`` then decides `nil`-vs-stream synchronously from that cached
/// verdict (clause 3: admission is a promise), and a resident model that is a VLM
/// tower, off the allowlist, or non-dense simply reports `false` and every request
/// falls to the legacy single-stream path.
extension MLXSwiftEngine: BatchGenerationServing {

    // MARK: - Coverage capture (called from load / unload)

    /// Probe whether the just-loaded LLM can be served on the batched path and, if
    /// so, capture its stop-token set + input processor for off-lock admission.
    /// Runs while the container mutex is free (at load), so the batched decode loop
    /// can later hold it for whole cohorts without this probe ever contending. Also
    /// re-opens the coordinator's drain epoch — the same engine instance is reused
    /// across a CLI cold-swap, so a prior drain must be cleared for the new model.
    func refreshBatchServingCoverage(container: ModelContainer) async {
        clearBatchServingState()
        await batchCoordinator.resumeAdmissions()

        let probe:
            (
                covered: Bool, eosTokenIds: Set<Int>, unknownTokenId: Int?,
                messageGenerator: (any MessageGenerator)?
            ) =
                await container.perform { context in
                    // Both gates: cache SHAPE (dense KVCacheSimple only) AND model
                    // ARCHITECTURE (verified `ropeOffset`). The probe is cheap — type
                    // checks plus an empty `newCache` array, no MLX compute — matching
                    // `ModelBatchInferenceCore.ensureCoverage`.
                    let covered =
                        BatchModelAllowlist.contains(context.model)
                        && BatchCacheConverter.makeBatchCaches(
                            from: context.model.newCache(parameters: nil), leftPadding: [0]) != nil
                    // Complete EOS set, exactly as upstream `buildStopTokenIds`: config
                    // ids + tokenizer EOS + encoded `extraEOSTokens`.
                    var eos = context.configuration.eosTokenIds
                    if let tokenizerEOS = context.tokenizer.eosTokenId { eos.insert(tokenizerEOS) }
                    for token in context.configuration.extraEOSTokens {
                        if let id = context.tokenizer.convertTokenToId(token) { eos.insert(id) }
                    }
                    // Capture the resident model's own `MessageGenerator` (its Default
                    // vs NoSystem choice is decided HERE, from the tokenizer's chat
                    // template). Both concrete generators are stateless `Sendable`
                    // structs that retain neither the model nor the tokenizer, so the
                    // value is safe to carry out of the mutex and reuse on the
                    // admission actor's SEPARATE tokenizer (H2).
                    let generator =
                        (context.model as? LLMModel)?.messageGenerator(tokenizer: context.tokenizer)
                    return (covered, eos, context.tokenizer.unknownTokenId, generator)
                }

        // Coverage needs the cache/arch gates AND the pieces admission tokenization
        // needs: the model's message generator and its on-disk directory. Any missing
        // → decline batching (every request falls to the legacy single-stream path).
        guard probe.covered, let messageGenerator = probe.messageGenerator,
            let directory = loadedModel?.directory
        else { return }

        // Load the admission processor's DEDICATED tokenizer — a DISTINCT instance
        // from `context.tokenizer`, loaded once here off the mutex at model load — so
        // admission tokenization never races the drive loop's `context.tokenizer`
        // detokenizer (H2). It loads from the SAME directory, so its template/vocab
        // match the legacy path exactly. A failure just declines batching.
        let admissionTokenizer: any Tokenizer
        do {
            admissionTokenizer = try await HuggingFaceTokenizerLoader().load(from: directory)
        } catch {
            return
        }

        batchServingCoverage = true
        batchServingEOSTokenIds = probe.eosTokenIds
        batchServingUnknownTokenId = probe.unknownTokenId
        batchServingContainer = container
        batchAdmissionProcessor = BatchAdmissionProcessor(
            tokenizer: admissionTokenizer, messageGenerator: messageGenerator)
    }

    /// Reset all batched-path state (a VLM load, an unload, or a failed probe): every
    /// request then takes the legacy single-stream path.
    func clearBatchServingState() {
        batchServingCoverage = false
        batchServingEOSTokenIds = []
        batchServingUnknownTokenId = nil
        batchAdmissionProcessor = nil
        batchServingContainer = nil
    }

    // MARK: - BatchGenerationServing

    public func submit(
        _ request: GenerateRequest
    ) async -> AsyncThrowingStream<GenerateChunk, Error>? {
        // Clause 3 (admission is a promise): every coverage check that decides
        // batchability runs synchronously, from the cached load-time verdict, before
        // any stream is handed back. A VLM/off-allowlist/non-dense resident model, a
        // request naming a DIFFERENT model (routes to the legacy swap path), or a
        // speculative request all decline here with zero work performed.
        guard batchServingCoverage,
            let admissionProcessor = batchAdmissionProcessor,
            let loaded = loadedModel,
            request.draftModelID == nil,
            request.model == loaded.id || request.model == loaded.displayName
        else { return nil }

        // Open the submit window BEFORE tokenizing so a concurrent burst is all
        // counted before any of us claims (M1). Balanced on every exit path below.
        await batchCoordinator.enterSubmission()
        defer { Task { await self.batchCoordinator.exitSubmission() } }

        // Tokenize OFF the container mutex, on the serial admission actor's DEDICATED
        // tokenizer (H2) — never the drive loop's `context.tokenizer`, so admission
        // races neither a concurrent admission nor the loop's detokenizer. A failure
        // here just declines to batch; the legacy path re-tokenizes and surfaces any
        // real error cleanly.
        let promptTokens: [Int]
        do {
            let userInput = batchServingUserInput(from: request)
            promptTokens = try await admissionProcessor.prepare(userInput)
        } catch {
            return nil
        }
        guard !promptTokens.isEmpty else { return nil }

        let params = request.parameters
        let config = BatchSlotConfig(
            promptTokens: promptTokens,
            parameters: GenerateParameters(
                maxTokens: params.maxTokens,
                temperature: Float(params.temperature),
                topP: Float(params.topP)),
            // GenerateRequest carries no stop-strings field, so the batched path has
            // none to honour — parity with what the single-stream path receives.
            stopStrings: [])

        let id = await batchCoordinator.nextRequestID()
        let (stream, continuation) = AsyncThrowingStream<GenerateChunk, Error>.makeStream()
        // Clause 2 (a dropped stream evicts its row): a cancelled/abandoned iterator
        // — the COMMON case, fired by `ChunkIteratorBox.deinit` when a body never
        // runs — removes the row. `.finished`/`.failure` are our own terminal cases
        // and need no eviction. Harmless on the `.solo`/`.rejected` paths below, which
        // `finish()` (→ `.finished`, not `.cancelled`) an id that was never enqueued.
        continuation.onTermination = { @Sendable reason in
            if case .cancelled = reason {
                Task { await self.batchCoordinator.markCancelled(id) }
            }
        }

        // M1 "batch only under concurrency": an idle, uncontended seam returns `.solo`
        // → the request takes the legacy single-stream path, which keeps the engine
        // prompt cache warm across one client's successive turns (the flagship agent
        // case). A drain epoch returns `.rejected` (clause 1). Otherwise the request
        // joins the forming/running cohort.
        let decision = await batchCoordinator.claimSoloOrEnqueue(
            BatchServingCoordinator.Pending(id: id, config: config, continuation: continuation))
        switch decision {
        case .rejected, .solo:
            // Not batched: finish the never-returned stream so it is not left
            // dangling, and route the request to the legacy single-stream path.
            continuation.finish()
            return nil
        case .admitted(let startDrive):
            // The synchronous `Task { … }` launch (no `await` before it) is what makes
            // the claim safe: `driving` is already set, so a concurrent `beginDrain`
            // will wait for THIS loop.
            if startDrive {
                startBatchDriveLoop()
            }
            return stream
        }
    }

    public func drainForModelChange() async {
        // Clause 4 (never re-enter the caller's lock): the server calls this from
        // UNDER its FIFO generation lock; we only await the coordinator, which awaits
        // the drive loop — no `container.perform`, no generation lock, so a swap
        // waits for rows while rows never wait for the swap.
        await batchCoordinator.beginDrain()
    }

    // MARK: - Drive loop (inside container.perform)

    /// Launch the single background drive loop. Called synchronously by the row that
    /// claimed `driving` via ``BatchServingCoordinator/claimSoloOrEnqueue(_:)`` (which
    /// enqueues + claims in one actor step).
    func startBatchDriveLoop() {
        Task { await self.runBatchDriveLoop() }
    }

    /// Run continuous batched decode for the resident model, pulling admissions /
    /// cancellations from the coordinator each step. The cohort's session lives across
    /// bounded `container.perform` SEGMENTS (H1): each segment runs at most
    /// ``batchServingStepBudget`` decode steps, then releases the model mutex and
    /// re-acquires it, so single-stream `container.prepare` never waits behind the
    /// cohort's whole lifetime. Also loops across COHORTS while a race leaves work
    /// queued (`finishDrive` re-drive).
    private func runBatchDriveLoop() async {
        guard let container = batchServingContainer else {
            // H3: the resident model went away between admission and drive-loop start
            // (an unload or failed swap raced admission). Admission was a binding
            // promise — there is no legacy fall-back left for an already-enqueued row
            // — so finish every queued row with a clear error and release the drive /
            // drain state. Using `finishDrive` here instead would drop the queued rows
            // on the floor: they would hang forever, `driving` would stay `true`, and
            // the next `beginDrain` (under the server's FIFO lock) would deadlock.
            await batchCoordinator.abortAllQueued(BatchServingUnavailableError())
            return
        }
        let coordinator = batchCoordinator
        let eosTokenIds = batchServingEOSTokenIds
        let unknownTokenId = batchServingUnknownTokenId
        let globalMaxTokens = Self.batchServingGlobalMaxTokens
        let stepBudget = Self.batchServingStepBudget

        // H1: the cohort's `BatchDecodeSession` (non-`Sendable` MLX state — the running
        // KV cache + per-row slots) is carried ACROSS `perform` segments in a
        // `NonSendableBox`, the same @unchecked-Sendable handoff this file uses for
        // other non-`Sendable` mlx values.
        //
        // Why carrying it is safe: the container wraps ONE `ModelContext`, so
        // `context.model`/`context.tokenizer` are the SAME instances each segment (the
        // session built against them in segment 1 stays valid in segment 2). There is
        // exactly ONE drive loop (the coordinator's `driving` claim guarantees it), and
        // the session is ONLY ever touched inside this loop's own serial `perform`
        // segments — never concurrently. Between segments the mutex is free for other
        // `perform`s (legacy single-stream / speculative), but they build their own
        // state and never see this session; the model weights they read are the
        // already-evaluated constants the session also only reads. Actor isolation
        // (this loop advances on the engine actor between segments) plus the single-
        // drive invariant means the cross-segment handoff is race-free.
        var carried: NonSendableBox<BatchDecodeSession>?
        var keepDriving = true
        while keepDriving {
            let inbound = carried
            let outcome: NonSendableBox<BatchDecodeSession>? = await container.perform { context in
                let session =
                    inbound?.value
                    ?? BatchDecodeSession(
                        core: ModelBatchInferenceCore(model: context.model),
                        tokenizer: context.tokenizer,
                        eosTokenIds: eosTokenIds,
                        unknownTokenId: unknownTokenId,
                        globalMaxTokens: globalMaxTokens)
                do {
                    var steps = 0
                    while true {
                        let tick = await coordinator.takeTick(active: session.activeIDs)
                        if !session.isEmpty || !tick.cancelledActive.isEmpty {
                            try session.decodeStep(cancelled: tick.cancelledActive)
                            steps += 1
                        }
                        if !tick.admits.isEmpty {
                            try session.admit(tick.admits)
                        }
                        // Cohort drained and nothing newly admitted → the queue was
                        // empty; end the cohort (discard the session) and let
                        // `finishDrive` catch any late race.
                        if session.isEmpty && tick.admits.isEmpty { return nil }
                        // H1: step budget spent while the cohort is still live → yield
                        // the mutex. Carry the SAME session out so the next segment
                        // resumes it (no re-prefill, no lost rows). `driving` stays set
                        // and `finishDrive` is deliberately NOT called on this path.
                        if steps >= stepBudget && !session.isEmpty {
                            return NonSendableBox(session)
                        }
                        // Interleave pending submit/cancel hops between MLX steps.
                        await Task.yield()
                    }
                } catch {
                    // A shared-forward MLX error (or a defensive contract violation)
                    // aborts the whole cohort — every row shares the batched decode.
                    session.failAll(error)
                    return nil
                }
            }
            if let outcome {
                // Budget yield: the cohort is still live. Keep `driving` set — calling
                // `finishDrive` here would release a `beginDrain` waiter while rows are
                // still decoding — and re-enter `perform` with the SAME session. A
                // concurrent `beginDrain` stays blocked until the cohort truly empties.
                carried = outcome
            } else {
                // The cohort drained (or aborted). Only now ask the coordinator whether
                // a race left fresh work queued (re-drive) or the loop should release
                // the drive claim and resume any drain waiter.
                carried = nil
                keepDriving = await coordinator.finishDrive()
            }
        }
    }

    // MARK: - Prompt assembly

    /// Build the `UserInput` for a batched request, mirroring `runGeneration`'s
    /// text-only mapping (chat messages, tool specs, template kwargs). The batched
    /// path is dense-LLM-only by the coverage gate, so image attachments are dropped
    /// exactly as the single-stream LLM path drops them.
    ///
    /// `nonisolated`: it reads only the `Sendable` `request` and static helpers, so
    /// its result is a disconnected value — safe to hand to the off-actor
    /// `processor.prepare(input:)` without a data-race diagnostic.
    private nonisolated func batchServingUserInput(from request: GenerateRequest) -> UserInput {
        let chatMessages: [Chat.Message] = request.allMessages.map { message in
            Self.upstreamChatMessage(from: message, images: [])
        }
        let toolSpecs = request.tools.map { Self.toolSpecs(from: $0) }
        return UserInput(
            chat: chatMessages,
            tools: toolSpecs,
            additionalContext: request.templateKwargs?.mapValues { $0.toSendable() })
    }
}
