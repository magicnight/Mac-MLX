// Copyright © 2026 macMLX. English comments only.

import XCTest

@testable import MacMLXCore
import MLXLMCommon

/// Primary correctness proof for the A2a batched-decode SCHEDULING / masking
/// logic — per-row stop, finished-row freezing, fan-out routing, clean cohort
/// completion, and refusal on bad cohorts.
///
/// These are MLX-free: a ``ScriptedStepEvaluator`` supplies per-row tokens and a
/// ``ScriptedTextDecoder`` supplies per-token text, so the full
/// ``BatchDecodeRunner`` / ``BatchDecodeSlot`` machinery runs under a plain
/// `swift test` in CI with no model and no Metal runtime. (End-to-end per-row
/// SAMPLING correctness — that greedy argmax picks the right token from real
/// `[B, vocab]` logits — is proven by the model-gated
/// `BatchDecodeRunnerModelTests`.)
final class BatchDecodeCoreLogicTests: XCTestCase {

    // MARK: - Scripted seams

    /// Feeds scripted per-row tokens: `script[0]` is the prefill result, and
    /// `script[k]` (k ≥ 1) is decode step `k`. Records the fed-back tokens so
    /// tests can assert finished rows are masked with a pad.
    private final class ScriptedStepEvaluator: BatchStepEvaluator {
        let script: [[Int]]
        private(set) var fedHistory: [[Int]] = []
        private var stepCall = 0

        init(_ script: [[Int]]) { self.script = script }

        func prefill(_ promptRows: [[Int]]) throws -> [Int] { script[0] }

        func step(_ fed: [Int]) throws -> [Int] {
            fedHistory.append(fed)
            stepCall += 1
            return script[min(stepCall, script.count - 1)]
        }
    }

    /// Deterministic token→text mapping (no tokenizer). Unmapped tokens decode to
    /// the empty string, which is the natural "no visible text yet" case.
    private struct ScriptedTextDecoder: IncrementalTextDecoder {
        let map: [Int: String]
        func decode(_ token: Int) -> String { map[token] ?? "" }
    }

    // MARK: - Builders

    private func makeSlot(
        row: Int,
        prompt: [Int] = [1, 2, 3],
        maxTokens: Int? = nil,
        eos: Set<Int> = [],
        unknown: Int? = nil,
        stops: Set<String> = [],
        decoderMap: [Int: String] = [:]
    ) -> (BatchDecodeSlot, AsyncThrowingStream<GenerateChunk, Error>) {
        let (stream, continuation) = AsyncThrowingStream<GenerateChunk, Error>.makeStream()
        let slot = BatchDecodeSlot(
            row: row,
            promptTokens: prompt,
            parameters: GenerateParameters(),
            maxTokens: maxTokens,
            eosTokenIds: eos,
            unknownTokenId: unknown,
            stopStrings: stops,
            textDecoder: ScriptedTextDecoder(map: decoderMap),
            continuation: continuation
        )
        return (slot, stream)
    }

    private struct SlotOutcome {
        var texts: [String] = []
        var finishReason: FinishReason?
        var usage: TokenUsage?
        var failure: Error?
        var fullText: String { texts.joined() }
    }

    private func drain(
        _ stream: AsyncThrowingStream<GenerateChunk, Error>
    ) async -> SlotOutcome {
        var outcome = SlotOutcome()
        do {
            for try await chunk in stream {
                if let reason = chunk.finishReason {
                    outcome.finishReason = reason
                    outcome.usage = chunk.usage
                } else {
                    outcome.texts.append(chunk.text)
                }
            }
        } catch {
            outcome.failure = error
        }
        return outcome
    }

    // MARK: - Fan-out routing

    func testFanOutRoutesEachRowsTokensToItsOwnSlot() async throws {
        // Three rows, distinct per-row trajectories, no early stop. Proves the
        // runner routes evaluator column `r` to slot `r` with no cross-talk, and
        // that a whole cohort finishes cleanly at the global cap.
        let script: [[Int]] = [
            [10, 20, 30],  // prefill
            [11, 21, 31],  // decode 1
            [12, 22, 32],  // decode 2
            [13, 23, 33],  // decode 3
        ]
        let evaluator = ScriptedStepEvaluator(script)
        var slots: [BatchDecodeSlot] = []
        var streams: [AsyncThrowingStream<GenerateChunk, Error>] = []
        for row in 0..<3 {
            let (slot, stream) = makeSlot(row: row)
            slots.append(slot)
            streams.append(stream)
        }
        let runner = BatchDecodeRunner(
            evaluator: evaluator, slots: slots, globalMaxTokens: script.count)

        try runner.run()

        XCTAssertEqual(runner.slotTokens[0], [10, 11, 12, 13])
        XCTAssertEqual(runner.slotTokens[1], [20, 21, 22, 23])
        XCTAssertEqual(runner.slotTokens[2], [30, 31, 32, 33])
        for slot in slots {
            XCTAssertTrue(slot.isFinished)
            XCTAssertEqual(slot.finishReason, .length)
        }

        // Every stream closes with a terminal finish-reason chunk carrying usage.
        for (row, stream) in streams.enumerated() {
            let outcome = await drain(stream)
            XCTAssertEqual(outcome.finishReason, .length, "row \(row) terminal reason")
            XCTAssertEqual(outcome.usage?.completionTokens, 4, "row \(row) completion count")
            XCTAssertEqual(outcome.usage?.promptTokens, 3, "row \(row) prompt count")
        }
    }

    // MARK: - Per-row EOS

    func testEOSStopsOnlyTheHittingRowAndIsNotEmitted() async throws {
        // Row 0 hits EOS (99) at decode step 2; row 1 keeps going to the cap.
        let script: [[Int]] = [
            [10, 20],  // prefill
            [11, 21],  // decode 1
            [99, 22],  // decode 2 — row 0 EOS
            [12, 23],  // decode 3 — row 0 masked, row 1 live
            [13, 24],  // decode 4
        ]
        let evaluator = ScriptedStepEvaluator(script)
        let (slot0, stream0) = makeSlot(row: 0, eos: [99])
        let (slot1, stream1) = makeSlot(row: 1, eos: [99])
        let runner = BatchDecodeRunner(
            evaluator: evaluator, slots: [slot0, slot1], globalMaxTokens: script.count)

        try runner.run()

        // EOS is swallowed: not appended, not counted.
        XCTAssertEqual(slot0.generatedTokens, [10, 11])
        XCTAssertEqual(slot0.finishReason, .stop)
        // Row 1 was unaffected and ran to the cap.
        XCTAssertEqual(slot1.generatedTokens, [20, 21, 22, 23, 24])
        XCTAssertEqual(slot1.finishReason, .length)

        let out0 = await drain(stream0)
        let out1 = await drain(stream1)
        XCTAssertEqual(out0.finishReason, .stop)
        XCTAssertEqual(out0.usage?.completionTokens, 2)
        XCTAssertEqual(out1.finishReason, .length)
        XCTAssertEqual(out1.usage?.completionTokens, 5)
    }

    func testUnknownTokenStopsRowLikeEOS() async throws {
        let script: [[Int]] = [[10], [11], [7], [12]]  // unknown = 7 at step 2
        let evaluator = ScriptedStepEvaluator(script)
        let (slot, _) = makeSlot(row: 0, unknown: 7)
        let runner = BatchDecodeRunner(
            evaluator: evaluator, slots: [slot], globalMaxTokens: script.count)

        try runner.run()

        XCTAssertEqual(slot.generatedTokens, [10, 11])
        XCTAssertEqual(slot.finishReason, .stop)
    }

    // MARK: - Finished-row masking

    func testFinishedRowFreezesWhileOthersContinueAndIsFedAPad() async throws {
        // Row 0 stops at max-tokens = 1 (after the prefill token). Row 1 stops on
        // EOS at step 2. Row 2 runs to the cap. Assert each row's trajectory is
        // frozen at its own stop and that finished rows are fed their pad token.
        let script: [[Int]] = [
            [10, 20, 30],  // prefill
            [11, 21, 31],  // decode 1 — row 0 already frozen (max=1)
            [12, 99, 32],  // decode 2 — row 1 EOS
            [13, 23, 33],  // decode 3
            [14, 24, 34],  // decode 4
        ]
        let evaluator = ScriptedStepEvaluator(script)
        let (slot0, _) = makeSlot(row: 0, maxTokens: 1)
        let (slot1, _) = makeSlot(row: 1, eos: [99])
        let (slot2, _) = makeSlot(row: 2)
        let runner = BatchDecodeRunner(
            evaluator: evaluator, slots: [slot0, slot1, slot2], globalMaxTokens: script.count)

        try runner.run()

        XCTAssertEqual(slot0.generatedTokens, [10], "row 0 frozen after 1 token")
        XCTAssertEqual(slot0.finishReason, .length)
        XCTAssertEqual(slot1.generatedTokens, [20, 21], "row 1 frozen at EOS (EOS swallowed)")
        XCTAssertEqual(slot1.finishReason, .stop)
        XCTAssertEqual(slot2.generatedTokens, [30, 31, 32, 33, 34], "row 2 ran to cap")
        XCTAssertEqual(slot2.finishReason, .length)

        // Masking evidence: once frozen, a row is fed back its last real token
        // (its pad) on every subsequent step, keeping the batch width B.
        // fedHistory[k] is the feed for decode step k+1.
        // Row 0 froze after the prefill token 10 → always fed 10.
        for fed in evaluator.fedHistory {
            XCTAssertEqual(fed[0], 10, "frozen row 0 must be fed its pad (last token)")
        }
        // Row 1 froze after token 21 (EOS not appended) at decode step 2, so from
        // the decode-step-3 feed onward it is padded with 21.
        XCTAssertEqual(evaluator.fedHistory[2][1], 21, "frozen row 1 padded with last token")
        XCTAssertEqual(evaluator.fedHistory[3][1], 21, "frozen row 1 stays padded")
    }

    // MARK: - Max tokens

    func testMaxTokensStopsRowWithLengthReason() async throws {
        let script: [[Int]] = [[10], [11], [12], [13], [14]]
        let evaluator = ScriptedStepEvaluator(script)
        let (slot, stream) = makeSlot(row: 0, maxTokens: 3)
        let runner = BatchDecodeRunner(
            evaluator: evaluator, slots: [slot], globalMaxTokens: script.count)

        try runner.run()

        XCTAssertEqual(slot.generatedTokens, [10, 11, 12], "exactly maxTokens emitted")
        XCTAssertEqual(slot.finishReason, .length)
        let outcome = await drain(stream)
        XCTAssertEqual(outcome.usage?.completionTokens, 3)
    }

    func testMaxTokensZeroEmitsNothingAndOtherRowsAreUnaffected() async throws {
        // F5: cap == 0 must emit ZERO tokens. Before the fix, the prefill token
        // was ingested unconditionally, so a maxTokens=0 row still emitted 1
        // token — an off-by-one against the "cap == 0 emits nothing" contract.
        let script: [[Int]] = [
            [10, 20],  // prefill
            [11, 21],  // decode 1
            [12, 22],  // decode 2
        ]
        let evaluator = ScriptedStepEvaluator(script)
        let (slot0, stream0) = makeSlot(row: 0, maxTokens: 0)
        let (slot1, stream1) = makeSlot(row: 1, maxTokens: 3)
        let runner = BatchDecodeRunner(
            evaluator: evaluator, slots: [slot0, slot1], globalMaxTokens: script.count)

        try runner.run()

        XCTAssertEqual(slot0.generatedTokens, [], "maxTokens=0 row must emit nothing")
        XCTAssertEqual(slot0.finishReason, .length)
        XCTAssertEqual(slot1.generatedTokens, [20, 21, 22], "batchmate unaffected by the 0-cap row")
        XCTAssertEqual(slot1.finishReason, .length)

        let out0 = await drain(stream0)
        XCTAssertEqual(out0.finishReason, .length)
        XCTAssertEqual(out0.usage?.completionTokens, 0)
        XCTAssertEqual(out0.texts, [], "no text chunks for the 0-cap row")

        let out1 = await drain(stream1)
        XCTAssertEqual(out1.finishReason, .length)
        XCTAssertEqual(out1.usage?.completionTokens, 3)
    }

    // MARK: - Stop strings

    func testStopStringCompletedAcrossTokensTruncatesAndStops() async throws {
        // Tokens detokenize to "hello " then "STOP" then "world". The stop string
        // "STOP" completes on the second token: pre-stop text is emitted, the
        // rest is truncated, and the row stops. Token 3 is never reached.
        let script: [[Int]] = [[1], [2], [3]]
        let evaluator = ScriptedStepEvaluator(script)
        let (slot, stream) = makeSlot(
            row: 0, stops: ["STOP"], decoderMap: [1: "hello ", 2: "STOP", 3: "world"])
        let runner = BatchDecodeRunner(
            evaluator: evaluator, slots: [slot], globalMaxTokens: script.count)

        try runner.run()

        XCTAssertEqual(slot.finishReason, .stop)
        // The stop token (2) is still counted as generated, but "world" (token 3)
        // is never produced.
        XCTAssertEqual(slot.generatedTokens, [1, 2])
        let outcome = await drain(stream)
        XCTAssertEqual(outcome.fullText, "hello ", "text after the stop string is truncated")
        XCTAssertEqual(outcome.finishReason, .stop)
    }

    func testPartialStopPrefixIsHeldBackThenFlushedAtEOS() async throws {
        // "foo ST" holds back "ST" (a prefix of "STOP"); the next token is EOS,
        // which is not a stop-string completion, so the held-back "ST" is flushed.
        let script: [[Int]] = [[1], [99]]  // 99 = EOS
        let evaluator = ScriptedStepEvaluator(script)
        let (slot, stream) = makeSlot(
            row: 0, eos: [99], stops: ["STOP"], decoderMap: [1: "foo ST"])
        let runner = BatchDecodeRunner(
            evaluator: evaluator, slots: [slot], globalMaxTokens: script.count)

        try runner.run()

        XCTAssertEqual(slot.finishReason, .stop)
        let outcome = await drain(stream)
        XCTAssertEqual(outcome.fullText, "foo ST", "held-back prefix flushed when EOS ends the row")
    }

    func testStopStringSpanningMultipleTokenChunksTruncatesCleanly() async throws {
        // Unlike testStopStringCompletedAcrossTokensTruncatesAndStops (where a
        // SINGLE token's text completes the stop string), here "STOP" is split
        // across TWO decode tokens' text ("ST" then "OP"), proving the filter's
        // buffer correctly accumulates a match across a token boundary and never
        // leaks the held-back "ST" fragment before the match completes.
        let script: [[Int]] = [[1], [2], [3]]
        let evaluator = ScriptedStepEvaluator(script)
        let (slot, stream) = makeSlot(
            row: 0, stops: ["STOP"], decoderMap: [1: "hello ", 2: "ST", 3: "OP"])
        let runner = BatchDecodeRunner(
            evaluator: evaluator, slots: [slot], globalMaxTokens: script.count)

        try runner.run()

        XCTAssertEqual(slot.finishReason, .stop)
        // All three tokens (including both halves of the split stop string) are
        // counted as generated, matching the single-chunk stop-string test.
        XCTAssertEqual(slot.generatedTokens, [1, 2, 3])
        let outcome = await drain(stream)
        XCTAssertEqual(
            outcome.fullText, "hello ", "no fragment of the split stop string ever leaks out")
        XCTAssertEqual(outcome.finishReason, .stop)
    }

    // MARK: - Refusal

    func testEmptyCohortThrows() {
        let evaluator = ScriptedStepEvaluator([[0]])
        let runner = BatchDecodeRunner(evaluator: evaluator, slots: [], globalMaxTokens: 4)
        XCTAssertThrowsError(try runner.run()) { error in
            XCTAssertEqual(error as? BatchUnsupportedError, .emptyCohort)
        }
    }

    func testUnequalPromptLengthsThrows() {
        let evaluator = ScriptedStepEvaluator([[10, 20]])
        let (slot0, _) = makeSlot(row: 0, prompt: [1, 2, 3])
        let (slot1, _) = makeSlot(row: 1, prompt: [1, 2])  // different length
        let runner = BatchDecodeRunner(
            evaluator: evaluator, slots: [slot0, slot1], globalMaxTokens: 4)
        XCTAssertThrowsError(try runner.run()) { error in
            XCTAssertEqual(error as? BatchUnsupportedError, .unequalPromptLengths([3, 2]))
        }
    }

    func testUnequalPromptLengthsFailsEverySlotStream() async throws {
        // F1 regression: the emptyCohort/unequalPromptLengths guards used to
        // throw OUTSIDE the `do/catch`, so under the raw
        // `init(evaluator:slots:globalMaxTokens:)` seam (live continuations
        // already attached to `slots`) the streams were never terminated — a
        // leak. Assert every slot's stream now receives the same error instead
        // of hanging forever.
        let evaluator = ScriptedStepEvaluator([[10, 20]])
        let (slot0, stream0) = makeSlot(row: 0, prompt: [1, 2, 3])
        let (slot1, stream1) = makeSlot(row: 1, prompt: [1, 2])  // different length
        let runner = BatchDecodeRunner(
            evaluator: evaluator, slots: [slot0, slot1], globalMaxTokens: 4)

        XCTAssertThrowsError(try runner.run()) { error in
            XCTAssertEqual(error as? BatchUnsupportedError, .unequalPromptLengths([3, 2]))
        }

        let out0 = await drain(stream0)
        let out1 = await drain(stream1)
        XCTAssertEqual(
            out0.failure as? BatchUnsupportedError, .unequalPromptLengths([3, 2]),
            "row 0 stream must be terminated with the error, not leaked")
        XCTAssertEqual(
            out1.failure as? BatchUnsupportedError, .unequalPromptLengths([3, 2]),
            "row 1 stream must be terminated with the error, not leaked")
    }

    // MARK: - Evaluator contract

    func testEvaluatorReturningWrongTokenCountFailsAllStreamsWithContractViolation() async throws {
        // F2: the evaluator returns only 1 token for a 2-row cohort, violating
        // the `BatchStepEvaluator` contract (exactly `slots.count` tokens per
        // call). Before the fix this indexed out of bounds (a trap); now it
        // must fail every stream with `evaluatorContractViolation` instead.
        let evaluator = ScriptedStepEvaluator([[10]])
        let (slot0, stream0) = makeSlot(row: 0)
        let (slot1, stream1) = makeSlot(row: 1)
        let runner = BatchDecodeRunner(
            evaluator: evaluator, slots: [slot0, slot1], globalMaxTokens: 4)

        XCTAssertThrowsError(try runner.run()) { error in
            XCTAssertEqual(
                error as? BatchUnsupportedError,
                .evaluatorContractViolation(expected: 2, got: 1))
        }

        let out0 = await drain(stream0)
        let out1 = await drain(stream1)
        XCTAssertEqual(
            out0.failure as? BatchUnsupportedError,
            .evaluatorContractViolation(expected: 2, got: 1))
        XCTAssertEqual(
            out1.failure as? BatchUnsupportedError,
            .evaluatorContractViolation(expected: 2, got: 1))
    }

    // MARK: - Clean completion

    func testCohortWhereEveryRowStopsNaturallyFinishesCleanly() async throws {
        // Row 0 EOS at step 1, row 1 max-tokens 2 → both stop before the cap.
        let script: [[Int]] = [[10, 20], [99, 21], [11, 22], [12, 23]]
        let evaluator = ScriptedStepEvaluator(script)
        let (slot0, stream0) = makeSlot(row: 0, eos: [99])
        let (slot1, stream1) = makeSlot(row: 1, maxTokens: 2, eos: [99])
        let runner = BatchDecodeRunner(
            evaluator: evaluator, slots: [slot0, slot1], globalMaxTokens: script.count)

        try runner.run()

        XCTAssertTrue(slot0.isFinished && slot1.isFinished)
        XCTAssertEqual(slot0.finishReason, .stop)
        XCTAssertEqual(slot1.finishReason, .length)
        let out0 = await drain(stream0)
        let out1 = await drain(stream1)
        XCTAssertEqual(out0.finishReason, .stop)
        XCTAssertNil(out0.failure)
        XCTAssertEqual(out1.finishReason, .length)
        XCTAssertNil(out1.failure)
    }

    func testEvaluatorErrorFailsAllOpenStreamsAndRethrows() async throws {
        struct BoomError: Error {}
        final class ThrowingEvaluator: BatchStepEvaluator {
            func prefill(_ promptRows: [[Int]]) throws -> [Int] { [10, 20] }
            func step(_ fed: [Int]) throws -> [Int] { throw BoomError() }
        }
        let (slot0, stream0) = makeSlot(row: 0)
        let (slot1, stream1) = makeSlot(row: 1)
        let runner = BatchDecodeRunner(
            evaluator: ThrowingEvaluator(), slots: [slot0, slot1], globalMaxTokens: 8)

        XCTAssertThrowsError(try runner.run()) { XCTAssertTrue($0 is BoomError) }

        let out0 = await drain(stream0)
        let out1 = await drain(stream1)
        XCTAssertTrue(out0.failure is BoomError, "row 0 stream should surface the failure")
        XCTAssertTrue(out1.failure is BoomError, "row 1 stream should surface the failure")
    }

    // MARK: - Cancellation

    func testCancelledTaskFailsEverySlotStreamWithCancellationError() async throws {
        // `run()`'s `Task.isCancelled` check sits inside the decode loop (not
        // before the prefill ingest), so the script needs enough steps that the
        // loop actually reaches it before every row would finish naturally.
        let script: [[Int]] = Array(repeating: [10, 20], count: 10)
        let evaluator = ScriptedStepEvaluator(script)
        let (slot0, stream0) = makeSlot(row: 0)
        let (slot1, stream1) = makeSlot(row: 1)
        let runner = BatchDecodeRunner(
            evaluator: evaluator, slots: [slot0, slot1], globalMaxTokens: script.count)

        let task = Task {
            // Yield at least once so `task.cancel()` — issued synchronously
            // right after this task is created, before any suspension on the
            // caller's side — is guaranteed to already be visible once `run()`
            // reaches its per-step `Task.isCancelled` check.
            await Task.yield()
            try runner.run()
        }
        task.cancel()
        _ = try? await task.value

        let out0 = await drain(stream0)
        let out1 = await drain(stream1)
        XCTAssertTrue(out0.failure is CancellationError, "row 0 stream should surface cancellation")
        XCTAssertTrue(out1.failure is CancellationError, "row 1 stream should surface cancellation")
    }
}
