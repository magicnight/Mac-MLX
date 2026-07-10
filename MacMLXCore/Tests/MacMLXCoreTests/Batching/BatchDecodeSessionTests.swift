// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLXLMCommon
import XCTest

@testable import MacMLXCore

/// MLX-free proof of the A2d-2 ``BatchDecodeSession`` per-cohort driver: admit /
/// lockstep decode / finished-row shrink / consumer-cancel eviction, with the model
/// forward stubbed behind a scripted ``BatchInferenceCore`` (the same doubles style
/// as `BatchSchedulerLogicTests`). The real ragged-decode numerics are the
/// Metal-gated `BatchSchedulerModelTests`; this isolates the orchestration the seam
/// runs INSIDE `container.perform`.
final class BatchDecodeSessionTests: XCTestCase {

    // MARK: Doubles

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

    /// Scripted core: `script[0]` is the token `admit` returns for a row, `script[k]`
    /// (k ≥ 1) is the k-th `decode` token. Tracks its own row set so `evict` shrinks
    /// in lockstep with the session.
    private final class ScriptedCore: BatchInferenceCore, @unchecked Sendable {
        private var pending: [[Int]]
        private var rows: [(script: [Int], pos: Int)] = []
        private(set) var decodeCalls = 0

        init(scripts: [[Int]]) { self.pending = scripts }

        var rowCount: Int { rows.count }

        func admit(_ configs: [BatchSlotConfig]) throws -> [Int] {
            var firstTokens: [Int] = []
            for _ in configs {
                let script = pending.isEmpty ? [0] : pending.removeFirst()
                rows.append((script: script, pos: 1))
                firstTokens.append(script.first ?? 0)
            }
            return firstTokens
        }

        func decode(_ feedback: [Int]) throws -> [Int] {
            decodeCalls += 1
            var next: [Int] = []
            for index in rows.indices {
                let (script, pos) = rows[index]
                next.append(pos < script.count ? script[pos] : (script.last ?? 0))
                rows[index].pos = pos + 1
            }
            return next
        }

        func evict(keeping keepRows: [Int]) {
            if keepRows.isEmpty {
                rows = []
            } else if keepRows.count != rows.count {
                rows = keepRows.map { rows[$0] }
            }
        }
    }

    private struct SlotOutcome {
        var texts: [String] = []
        var finishReason: FinishReason?
        var completionTokens: Int?
        var failure: Error?
    }

    private func makePending(
        _ id: Int, maxTokens: Int = 4096
    ) -> (BatchServingCoordinator.Pending, AsyncThrowingStream<GenerateChunk, Error>) {
        let (stream, continuation) = AsyncThrowingStream<GenerateChunk, Error>.makeStream()
        let config = BatchSlotConfig(
            promptTokens: [1, 2, 3],
            parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0))
        return (
            BatchServingCoordinator.Pending(id: id, config: config, continuation: continuation),
            stream
        )
    }

    private func drain(_ stream: AsyncThrowingStream<GenerateChunk, Error>) async -> SlotOutcome {
        var outcome = SlotOutcome()
        do {
            for try await chunk in stream {
                if !chunk.text.isEmpty { outcome.texts.append(chunk.text) }
                if let reason = chunk.finishReason { outcome.finishReason = reason }
                if let usage = chunk.usage { outcome.completionTokens = usage.completionTokens }
            }
        } catch {
            outcome.failure = error
        }
        return outcome
    }

    // MARK: Tests

    /// Two rows admitted together each decode their OWN scripted trajectory to EOS,
    /// finishing with the right completion-token count — no cross-row bleed.
    func testTwoRowsEachDecodeTheirOwnTrajectoryToEOS() async throws {
        // Row A: 1,2,<eos>  → 2 completion tokens. Row B: 3,4,5,<eos> → 3 tokens.
        let core = ScriptedCore(scripts: [[1, 2, 99], [3, 4, 5, 99]])
        let session = BatchDecodeSession(
            core: core, tokenizer: MockTokenizer(),
            eosTokenIds: [99], unknownTokenId: nil, globalMaxTokens: 4096)

        let (p0, s0) = makePending(0)
        let (p1, s1) = makePending(1)
        try session.admit([p0, p1])
        while !session.isEmpty { try session.decodeStep(cancelled: []) }

        let o0 = await drain(s0)
        let o1 = await drain(s1)
        XCTAssertEqual(o0.finishReason, .stop)
        XCTAssertEqual(o0.completionTokens, 2, "row A stops at EOS after 2 tokens")
        XCTAssertEqual(o1.finishReason, .stop)
        XCTAssertEqual(o1.completionTokens, 3, "row B stops at EOS after 3 tokens")
    }

    /// A shorter row finishing first is EVICTED (cohort shrinks) while the longer row
    /// keeps decoding — the continuous-batching payoff over static masking.
    func testShorterRowIsEvictedWhileLongerContinues() async throws {
        let core = ScriptedCore(scripts: [[1, 99], [2, 3, 4, 5, 99]])
        let session = BatchDecodeSession(
            core: core, tokenizer: MockTokenizer(),
            eosTokenIds: [99], unknownTokenId: nil, globalMaxTokens: 4096)

        let (p0, s0) = makePending(0)
        let (p1, s1) = makePending(1)
        try session.admit([p0, p1])
        XCTAssertEqual(session.activeIDs, [0, 1], "both rows admitted (row 0 emitted its first token)")
        try session.decodeStep(cancelled: [])  // row 0 hits EOS on its first decode → evicted
        XCTAssertEqual(session.activeIDs, [1], "the finished short row is shrunk out of the cohort")

        while !session.isEmpty { try session.decodeStep(cancelled: []) }

        let o0 = await drain(s0)
        let o1 = await drain(s1)
        XCTAssertEqual(o0.completionTokens, 1, "row 0 emits its single pre-EOS token")
        XCTAssertEqual(o1.completionTokens, 4, "row 1 keeps decoding to its own EOS")
    }

    /// Cancelling a decoding row fails its stream and evicts it, leaving the sibling
    /// row's decode undisturbed (clause 2, active half).
    func testCancellingActiveRowEvictsItAndSpareSibling() async throws {
        let core = ScriptedCore(scripts: [[1, 2, 3, 4, 5, 6], [7, 8, 9, 99]])
        let session = BatchDecodeSession(
            core: core, tokenizer: MockTokenizer(),
            eosTokenIds: [99], unknownTokenId: nil, globalMaxTokens: 4096)

        let (p0, s0) = makePending(0)
        let (p1, s1) = makePending(1)
        try session.admit([p0, p1])
        try session.decodeStep(cancelled: [])          // both advance
        try session.decodeStep(cancelled: [p0.id])     // row 0 cancelled + evicted
        XCTAssertEqual(session.activeIDs, [1], "only the sibling row remains")
        while !session.isEmpty { try session.decodeStep(cancelled: []) }

        let o0 = await drain(s0)
        let o1 = await drain(s1)
        XCTAssertTrue(o0.failure is CancellationError, "the cancelled row's stream fails")
        XCTAssertEqual(o1.finishReason, .stop, "the sibling finishes normally")
        XCTAssertEqual(o1.completionTokens, 3)
    }

    /// A shared-forward error aborts every open row (the batched decode is shared).
    func testCoreErrorFailsWholeCohort() async throws {
        struct Boom: Error {}
        final class ThrowingCore: BatchInferenceCore, @unchecked Sendable {
            private var count = 0
            var rowCount: Int { count }
            func admit(_ configs: [BatchSlotConfig]) throws -> [Int] {
                count += configs.count
                return configs.map { _ in 1 }
            }
            func decode(_ feedback: [Int]) throws -> [Int] { throw Boom() }
            func evict(keeping keepRows: [Int]) { count = keepRows.count }
        }
        let session = BatchDecodeSession(
            core: ThrowingCore(), tokenizer: MockTokenizer(),
            eosTokenIds: [99], unknownTokenId: nil, globalMaxTokens: 4096)

        let (p0, s0) = makePending(0)
        let (p1, s1) = makePending(1)
        try session.admit([p0, p1])
        XCTAssertThrowsError(try session.decodeStep(cancelled: [])) { error in
            session.failAll(error)
        }

        let o0 = await drain(s0)
        let o1 = await drain(s1)
        XCTAssertTrue(o0.failure is Boom, "row 0 stream fails with the shared error")
        XCTAssertTrue(o1.failure is Boom, "row 1 stream fails with the shared error")
    }

    /// A zero-cap request never enters the cohort — it finishes immediately without a
    /// prefill (mirrors A2c's `stepAdmit` zero-cap guard).
    func testZeroMaxTokensNeverAdmitted() async throws {
        let core = ScriptedCore(scripts: [[1, 2, 99]])
        let session = BatchDecodeSession(
            core: core, tokenizer: MockTokenizer(),
            eosTokenIds: [99], unknownTokenId: nil, globalMaxTokens: 4096)

        let (p0, s0) = makePending(0, maxTokens: 0)
        try session.admit([p0])
        XCTAssertTrue(session.isEmpty, "a zero-cap row is never admitted to the cohort")

        let o0 = await drain(s0)
        XCTAssertEqual(o0.completionTokens, 0)
        XCTAssertEqual(o0.finishReason, .length)
    }
}
