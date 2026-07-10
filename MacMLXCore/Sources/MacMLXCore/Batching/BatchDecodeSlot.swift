// Copyright © 2026 macMLX. English comments only.

import MLXLMCommon

/// One row of a batched decode: its immutable config, its live decode state, and
/// its output stream. Owns everything needed to turn a stream of sampled token
/// IDs (from ``BatchStepEvaluator``) into that request's per-slot
/// `AsyncThrowingStream<GenerateChunk, Error>`.
///
/// All of this is pure Swift — no MLX, no tokenizer beyond the injected
/// ``IncrementalTextDecoder`` — so the stop / EOS / max-tokens / masking / fan-out
/// logic is unit testable in CI with a scripted decoder.
///
/// ## Finished-row masking (v1 = MASK, not shrink)
/// Once a slot finishes, ``ingest(_:)`` is a no-op and the slot is FROZEN: the
/// runner keeps feeding the cohort full width `B` (the row's pad token via
/// ``feedbackToken``) so live rows keep decoding, but this slot emits nothing
/// further. This wastes compute on the finished row (documented v1 trade-off;
/// shrink-on-finish needs the A2b `BatchKVCache.filter`).
///
/// ## Isolation
/// A `final class` (reference per-slot state, cleanly ownable by the driving
/// loop). Not `Sendable`: it is created and mutated on a single isolation domain
/// (the same one that drives ``BatchDecodeRunner``). Only its
/// `AsyncThrowingStream.Continuation` (which IS `Sendable`) crosses out to the
/// request's consumer.
final class BatchDecodeSlot {
    /// This slot's row index in the batch, `0..<B`.
    let row: Int

    /// The (already equal-length) prompt token IDs for this row.
    let promptTokens: [Int]

    /// Per-row sampling configuration. Read by ``ModelBatchStepEvaluator`` to
    /// build this row's sampler/processor; the scheduling logic here does not use
    /// it. Kept on the slot so a slot fully describes one request.
    let parameters: GenerateParameters

    /// Upper bound on emitted (non-stop) tokens for this row, or `nil` for none.
    let maxTokens: Int?

    /// End-of-sequence / stop token IDs (EOS set + any extras the caller adds).
    let eosTokenIds: Set<Int>

    /// The tokenizer's unknown-token ID, treated as a stop token like upstream.
    let unknownTokenId: Int?

    private let continuation: AsyncThrowingStream<GenerateChunk, Error>.Continuation
    private var stopFilter: SlotStopStringFilter
    private var textDecoder: any IncrementalTextDecoder

    /// Generated token IDs so far (excludes the prompt and any swallowed stop
    /// token). Primarily for tests / parity checks.
    private(set) var generatedTokens: [Int] = []

    /// `true` once this row has stopped (EOS, stop string, or max-tokens). Frozen
    /// thereafter.
    private(set) var isFinished = false

    /// Why the row stopped, once finished.
    private(set) var finishReason: FinishReason?

    init(
        row: Int,
        promptTokens: [Int],
        parameters: GenerateParameters,
        maxTokens: Int?,
        eosTokenIds: Set<Int>,
        unknownTokenId: Int?,
        stopStrings: Set<String>,
        textDecoder: any IncrementalTextDecoder,
        continuation: AsyncThrowingStream<GenerateChunk, Error>.Continuation
    ) {
        self.row = row
        self.promptTokens = promptTokens
        self.parameters = parameters
        self.maxTokens = maxTokens
        self.eosTokenIds = eosTokenIds
        self.unknownTokenId = unknownTokenId
        self.stopFilter = SlotStopStringFilter(stopStrings: stopStrings)
        self.textDecoder = textDecoder
        self.continuation = continuation
    }

    /// The token to feed this row back on the NEXT batched step. Live rows feed
    /// their most recent token; a finished row keeps feeding its last token as a
    /// pad so the batch stays width `B` and the shared RoPE offset advances in
    /// lockstep with the live rows. The fallback (empty history) only applies if
    /// the very first token was itself a stop token.
    var feedbackToken: Int {
        generatedTokens.last ?? eosTokenIds.first ?? unknownTokenId ?? 0
    }

    /// Feed one freshly sampled token ID for this row. Emits any decodable text
    /// to the slot's stream and decides whether the row stops (EOS/unknown, a
    /// completed stop string, or reaching max-tokens). Returns `true` iff the row
    /// just finished. A no-op returning `false` once already finished (masking).
    ///
    /// Ordering matches upstream `generateLoopTask`: the EOS/unknown check comes
    /// first and such a token is swallowed (not emitted, not counted), exactly as
    /// upstream's default `includeStopToken == false`.
    @discardableResult
    func ingest(_ token: Int) -> Bool {
        guard !isFinished else { return false }

        // EOS / unknown: stop without emitting or counting the token. Flush any
        // text the stop-string filter was holding back (upstream onGenerationEnd).
        if token == unknownTokenId || eosTokenIds.contains(token) {
            flushHeldText()
            finish(reason: .stop)
            return true
        }

        generatedTokens.append(token)

        let piece = textDecoder.decode(token)
        let result = stopFilter.process(piece)
        if let text = result.text, !text.isEmpty {
            continuation.yield(GenerateChunk(text: text))
        }
        if result.stopped {
            // A stop string completed: the pre-stop text was already emitted and
            // the rest is truncated (the filter's buffer is cleared).
            finish(reason: .stop)
            return true
        }

        if let maxTokens, generatedTokens.count >= maxTokens {
            flushHeldText()
            finish(reason: .length)
            return true
        }
        return false
    }

    /// Terminate a still-running slot at the cohort's global step cap, flushing
    /// any held-back text. Used by ``BatchDecodeRunner`` for rows that never hit a
    /// natural stop.
    func finishAtCap() {
        guard !isFinished else { return }
        flushHeldText()
        finish(reason: .length)
    }

    /// Fail this slot's stream (e.g. an MLX error or cancellation aborting the
    /// whole cohort). No-op if already finished.
    func fail(_ error: Error) {
        guard !isFinished else { return }
        isFinished = true
        continuation.finish(throwing: error)
    }

    private func flushHeldText() {
        if let residual = stopFilter.finish(), !residual.isEmpty {
            continuation.yield(GenerateChunk(text: residual))
        }
    }

    private func finish(reason: FinishReason) {
        isFinished = true
        finishReason = reason
        continuation.yield(
            GenerateChunk(
                text: "",
                finishReason: reason,
                usage: TokenUsage(
                    promptTokens: promptTokens.count,
                    completionTokens: generatedTokens.count
                )
            )
        )
        continuation.finish()
    }
}
