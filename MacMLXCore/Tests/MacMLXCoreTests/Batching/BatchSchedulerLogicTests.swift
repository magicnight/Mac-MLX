// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLXLMCommon
import XCTest

@testable import MacMLXCore

/// MLX-free proof of the ``BatchScheduler`` continuous-batching SCHEDULING logic:
/// admission timing, batch-full queueing, prefill→decode migration, row-exit
/// shrink, cancellation, and error fan-out. The model forward + cache surgery is
/// stubbed behind a scripted ``BatchInferenceCore``, so these run under a plain
/// `swift test` in CI with no Metal (the numeric ragged-decode parity is the
/// Metal-gated `BatchSchedulerModelTests`).
final class BatchSchedulerLogicTests: XCTestCase {

    // MARK: - Doubles

    /// Minimal `Tokenizer` conformance — the scheduler only uses it to build each
    /// slot's detokenizer, and these tests assert on token COUNTS / finish
    /// reasons, never on decoded text, so `decode` returning `""` is sufficient.
    private struct MockTokenizer: Tokenizer {
        let bosToken: String? = nil
        let eosToken: String? = nil
        let unknownToken: String? = nil
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    /// Scripted ``BatchInferenceCore``: each admitted row consumes the next
    /// trajectory from `scripts` (in admission order); `script[0]` is the token
    /// ``admit(_:)`` returns, `script[k]` (k ≥ 1) is the k-th ``decode(_:)``
    /// token. It records every admit/decode/evict call so the scheduling logic
    /// is directly observable. `@unchecked Sendable` via an `NSLock` because the
    /// scheduler actor drives it while the test inspects its records.
    private final class ScriptedInferenceCore: BatchInferenceCore, @unchecked Sendable {
        private let lock = NSLock()
        private var pending: [[Int]]
        private var rows: [(script: [Int], pos: Int)] = []
        private var recordedAdmits: [[[Int]]] = []
        private var recordedDecodes: [[Int]] = []
        private var recordedEvicts: [[Int]] = []

        init(scripts: [[Int]]) { self.pending = scripts }

        var admitCalls: [[[Int]]] { lock.withLock { recordedAdmits } }
        var decodeFeedback: [[Int]] { lock.withLock { recordedDecodes } }
        var evictKeeps: [[Int]] { lock.withLock { recordedEvicts } }

        var rowCount: Int { lock.withLock { rows.count } }

        func admit(_ configs: [BatchSlotConfig]) throws -> [Int] {
            lock.withLock {
                recordedAdmits.append(configs.map { $0.promptTokens })
                var firstTokens: [Int] = []
                for _ in configs {
                    let script = pending.isEmpty ? [0] : pending.removeFirst()
                    rows.append((script: script, pos: 1))  // pos 1: admit returned script[0]
                    firstTokens.append(script.first ?? 0)
                }
                return firstTokens
            }
        }

        func decode(_ feedback: [Int]) throws -> [Int] {
            lock.withLock {
                recordedDecodes.append(feedback)
                var next: [Int] = []
                for index in rows.indices {
                    let (script, pos) = rows[index]
                    next.append(pos < script.count ? script[pos] : (script.last ?? 0))
                    rows[index].pos = pos + 1
                }
                return next
            }
        }

        func evict(keeping keepRows: [Int]) {
            lock.withLock {
                recordedEvicts.append(keepRows)
                if keepRows.isEmpty {
                    rows = []
                } else if keepRows.count != rows.count {
                    rows = keepRows.map { rows[$0] }
                }
            }
        }
    }

    /// Throws on the first ``decode(_:)`` — the shared batched forward failing.
    private final class ThrowingInferenceCore: BatchInferenceCore, @unchecked Sendable {
        struct Boom: Error, Equatable {}
        private let lock = NSLock()
        private var count = 0

        var rowCount: Int { lock.withLock { count } }
        func admit(_ configs: [BatchSlotConfig]) throws -> [Int] {
            lock.withLock { count += configs.count }
            return configs.map { _ in 1 }
        }
        func decode(_ feedback: [Int]) throws -> [Int] { throw Boom() }
        func evict(keeping keepRows: [Int]) { lock.withLock { count = keepRows.count } }
    }

    // MARK: - Helpers

    private struct SlotOutcome {
        var texts: [String] = []
        var finishReason: FinishReason?
        var usage: TokenUsage?
        var failure: Error?
    }

    private func drain(_ stream: AsyncThrowingStream<GenerateChunk, Error>) async -> SlotOutcome {
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

    private func config(_ prompt: [Int], maxTokens: Int? = nil) -> BatchSlotConfig {
        BatchSlotConfig(
            promptTokens: prompt,
            parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0))
    }

    private func makeScheduler(
        _ core: sending any BatchInferenceCore,
        cap: Int, prefill: Int, eos: Set<Int> = [], global: Int = 4096
    ) -> BatchScheduler {
        BatchScheduler(
            core: core, tokenizer: MockTokenizer(), eosTokenIds: eos, unknownTokenId: nil,
            completionBatchSize: cap, prefillBatchSize: prefill, globalMaxTokens: global)
    }

    // MARK: - Migration (prefill first token → decode)

    func testSingleRequestMigratesPrefillTokenIntoDecode() async throws {
        let core = ScriptedInferenceCore(scripts: [[10, 11, 12]])
        let scheduler = makeScheduler(core, cap: 4, prefill: 4)

        let outcome = await drain(await scheduler.submit(config([1, 2, 3], maxTokens: 3)))

        XCTAssertEqual(outcome.finishReason, .length)
        XCTAssertEqual(outcome.usage?.completionTokens, 3)
        XCTAssertEqual(outcome.usage?.promptTokens, 3)
        // One admit with the row's prompt; the admit's first token (10) is fed
        // to decode 1 (→11), which is fed to decode 2 (→12). Two decode steps.
        XCTAssertEqual(core.admitCalls, [[[1, 2, 3]]])
        XCTAssertEqual(core.decodeFeedback, [[10], [11]])
    }

    func testEOSStopsRowAndIsSwallowed() async throws {
        let core = ScriptedInferenceCore(scripts: [[10, 11, 99]])
        let scheduler = makeScheduler(core, cap: 4, prefill: 4, eos: [99])

        let outcome = await drain(await scheduler.submit(config([1, 2], maxTokens: 10)))

        // 99 is the EOS: swallowed, not counted. Two emitted tokens (10, 11).
        XCTAssertEqual(outcome.finishReason, .stop)
        XCTAssertEqual(outcome.usage?.completionTokens, 2)
    }

    // MARK: - Row-exit shrink

    func testRowExitShrinksCohortAndSurvivorContinues() async throws {
        // row0 (index 0) runs long; row1 (index 1) is short and — admitted within
        // a step or two of row0 — always finishes first, so it is the one evicted
        // while row0 keeps decoding. Draining is sequential because the drive
        // loop is a separate Task that buffers both streams concurrently. The big
        // 30-vs-3 length gap makes the eviction ORDER independent of the exact
        // admission interleave (row1 always finishes long before row0).
        let core = ScriptedInferenceCore(scripts: [
            Array(repeating: 5, count: 30),
            [20, 21, 22],
        ])
        let scheduler = makeScheduler(core, cap: 4, prefill: 4, global: 30)

        let stream0 = await scheduler.submit(config([1], maxTokens: 30))
        let stream1 = await scheduler.submit(config([2], maxTokens: 3))
        let outcome0 = await drain(stream0)
        let outcome1 = await drain(stream1)

        XCTAssertEqual(outcome0.finishReason, .length)
        XCTAssertEqual(outcome0.usage?.completionTokens, 30)
        XCTAssertEqual(outcome1.finishReason, .length)
        XCTAssertEqual(outcome1.usage?.completionTokens, 3)

        // The two rows overlapped in the cohort (a decode step carried both);
        // then row1 (index 1) was evicted first, leaving row0 (index 0), and the
        // cohort finally drained to empty.
        XCTAssertTrue(
            core.decodeFeedback.contains { $0.count == 2 }, "the rows must batch together")
        XCTAssertEqual(core.evictKeeps, [[0], []], "row1 evicted first, then the cohort drains")
    }

    // MARK: - Admission timing / batch-full queueing

    func testBatchFullQueuesExcessAndNeverExceedsCap() async throws {
        // 5 identical-length rows, cap 2: they must decode in waves of ≤2.
        let scripts = (0..<5).map { _ in [30, 31, 32] }
        let core = ScriptedInferenceCore(scripts: scripts)
        let scheduler = makeScheduler(core, cap: 2, prefill: 2)

        // Submit all five up front; the drive loop (a separate Task) decodes them
        // in waves while their streams buffer, so sequential draining is safe.
        var streams: [AsyncThrowingStream<GenerateChunk, Error>] = []
        for row in 0..<5 {
            streams.append(await scheduler.submit(config([row], maxTokens: 3)))
        }
        var outcomes: [SlotOutcome] = []
        for stream in streams { outcomes.append(await drain(stream)) }

        // Every row completed its full 3-token trajectory.
        XCTAssertEqual(outcomes.count, 5)
        for outcome in outcomes {
            XCTAssertEqual(outcome.finishReason, .length)
            XCTAssertEqual(outcome.usage?.completionTokens, 3)
        }
        // The cohort never exceeded the cap: no admit wave and no decode step
        // ever carried more than 2 rows, and all 5 rows were admitted in total.
        XCTAssertTrue(core.admitCalls.allSatisfy { $0.count <= 2 }, "admit wave exceeded cap")
        XCTAssertEqual(core.admitCalls.reduce(0) { $0 + $1.count }, 5, "all rows must be admitted")
        XCTAssertTrue(core.decodeFeedback.allSatisfy { $0.count <= 2 }, "decode width exceeded cap")
        XCTAssertGreaterThanOrEqual(core.admitCalls.count, 3, "5 rows at cap 2 need ≥3 waves")
    }

    // MARK: - Zero-cap short-circuit (F5)

    func testZeroMaxTokensRowEmitsNothingAndIsNotAdmitted() async throws {
        let core = ScriptedInferenceCore(scripts: [[10]])
        let scheduler = makeScheduler(core, cap: 4, prefill: 4)

        let outcome = await drain(await scheduler.submit(config([1, 2], maxTokens: 0)))

        XCTAssertEqual(outcome.finishReason, .length)
        XCTAssertEqual(outcome.usage?.completionTokens, 0)
        XCTAssertTrue(outcome.texts.isEmpty)
        // A zero-cap row emits nothing, so it is finished at submission-drain and
        // never prefilled into the cohort.
        XCTAssertTrue(core.admitCalls.isEmpty, "a zero-cap row must never be admitted")
    }

    // MARK: - Cancellation

    func testCancellingQueuedRequestDropsItWithoutAdmitting() async throws {
        // cap 1: row0 occupies the only slot for its whole (long) run; row1
        // waits in the queue the entire time. Cancelling row1's consumer must
        // drop it — it is never prefilled — without disturbing row0.
        let core = ScriptedInferenceCore(scripts: [
            Array(repeating: 5, count: 40),
            [7, 7, 7],
        ])
        let scheduler = makeScheduler(core, cap: 1, prefill: 1, global: 40)

        let stream0 = await scheduler.submit(config([100], maxTokens: 40))
        let stream1 = await scheduler.submit(config([200], maxTokens: 3))

        // Cancel row1's consumer while it is still queued behind row0. The
        // inline consumer captures only the (Sendable) stream, so cancelling its
        // task tears down the stream and fires `onTermination(.cancelled)`.
        let consumer1 = Task { for try await _ in stream1 {} }
        consumer1.cancel()
        _ = try? await consumer1.value

        // row0 decodes to its 40-token cap, unaffected.
        let outcome0 = await drain(stream0)
        XCTAssertEqual(outcome0.finishReason, .length)
        XCTAssertEqual(outcome0.usage?.completionTokens, 40)

        // row1 (prompt [200]) was dropped from the queue: never admitted.
        let admittedPrompts = core.admitCalls.flatMap { $0 }
        XCTAssertTrue(admittedPrompts.contains([100]), "row0 must have been admitted")
        XCTAssertFalse(
            admittedPrompts.contains([200]), "a cancelled-while-queued row must never be admitted")
    }

    // MARK: - Error fan-out

    func testCoreErrorFailsAllOpenStreams() async throws {
        let core = ThrowingInferenceCore()
        let scheduler = makeScheduler(core, cap: 4, prefill: 4)

        let stream0 = await scheduler.submit(config([1], maxTokens: 10))
        let stream1 = await scheduler.submit(config([2], maxTokens: 10))
        let outcome0 = await drain(stream0)
        let outcome1 = await drain(stream1)

        // The shared batched forward threw on the first decode step: every open
        // stream fails, none produces a natural finish.
        XCTAssertTrue(outcome0.failure is ThrowingInferenceCore.Boom)
        XCTAssertTrue(outcome1.failure is ThrowingInferenceCore.Boom)
        XCTAssertNil(outcome0.finishReason)
        XCTAssertNil(outcome1.finishReason)
    }
}
