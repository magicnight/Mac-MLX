// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLX
import MLXLMCommon
import Testing
import XCTest

@testable import MacMLXCore

// MARK: - JSONConstraintProcessor decision core (Track C — H1 / M3)
//
// The processor's *masking policy* is factored into pure, MLX-free statics
// (`selectLegalToken` / `isLegal` / `lowestStopToken`) so it can be pinned down
// with a scripted `TokenVocabularyTable` and plain logits — no Metal, so these
// run in CI. What is NOT covered here is the `MLXArray` masking *mechanics*
// (building the one-hot / full mask, argmax over it); that needs a live Metal
// backend and is exercised by `JSONConstraintProcessorMaskTests` below (gated,
// local-only) and by the real-model E2E in `StructuredOutputModelTests`.
//
// Tradeoff: `bestLegalIndex` still owns the MLX ranking (argmax/argsort); the
// decision it feeds — "first legal token in descending-logit order", and "no
// legal token ⇒ force the lowest stop id" — lives in the statics tested here, so
// production and tests share exactly one policy.

@Suite("JSONConstraintProcessor decision core")
struct JSONConstraintProcessorDecisionTests {

    /// Token ids ranked highest-logit first — mirrors what `bestLegalIndex`
    /// feeds the pure selector after its MLX argsort.
    private func descending(_ logits: [Float]) -> [Int] {
        logits.enumerated().sorted { $0.element > $1.element }.map(\.offset)
    }

    private func table(_ vocab: [String], stop: Set<Int> = []) -> TokenVocabularyTable {
        TokenVocabularyTable(vocabularySize: vocab.count, stopTokenIDs: stop, decode: { vocab[$0] })
    }

    // MARK: 1 — greedy top token legal ⇒ chosen directly (== full-mask argmax)

    @Test
    func greedyTopLegalTokenIsSelectedDirectly() {
        // 0 "{" legal JSON start, 1 "abc" illegal, 2 "\"" legal.
        let table = table(["{", "abc", "\""])
        let state = ConstraintState.initial(for: .jsonObject)
        // Highest logit is id 0 "{", itself legal — so selection == argmax.
        let order = descending([3.0, 1.0, 2.0])   // [0, 2, 1]
        #expect(JSONConstraintProcessor.selectLegalToken(
            state: state, table: table, descendingLogitOrder: order) == 0)
    }

    // MARK: 2 — greedy top token illegal ⇒ highest-logit *legal* token

    @Test
    func greedyTopIllegalSelectsHighestLegal() {
        // 0 "abc" illegal, 1 "{" legal, 2 "\"" legal.
        let table = table(["abc", "{", "\""])
        let state = ConstraintState.initial(for: .jsonObject)
        // Order [0, 2, 1]: id 0 (top) is illegal, id 2 is the next and legal.
        let order = descending([9.0, 1.0, 5.0])
        #expect(JSONConstraintProcessor.selectLegalToken(
            state: state, table: table, descendingLogitOrder: order) == 2)
    }

    // MARK: 3 — all illegal ⇒ nil (⇒ processor forces the lowest stop id)

    @Test
    func allIllegalReturnsNil() {
        // 0 "}" 1 ":" 2 "abc" are all illegal at a fresh JSON start; id 3 is EOS,
        // illegal too because the document is not complete.
        let table = table(["}", ":", "abc", "</s>"], stop: [3])
        let state = ConstraintState.initial(for: .jsonObject)
        let order = descending([4.0, 3.0, 2.0, 1.0])
        #expect(JSONConstraintProcessor.selectLegalToken(
            state: state, table: table, descendingLogitOrder: order) == nil)
        // The forced-termination token is the lowest in-range stop id (H1).
        #expect(JSONConstraintProcessor.lowestStopToken(in: [3], vocab: 4) == 3)
    }

    // MARK: 4 — EOS legal only in an accepting (complete) state

    @Test
    func stopTokenIsLegalOnlyWhenComplete() {
        let table = table(["{", "</s>"], stop: [1])
        let incomplete = ConstraintState.initial(for: .jsonObject)
        #expect(JSONConstraintProcessor.isLegal(1, state: incomplete, table: table) == false)

        // Walk a full document; EOS is now legal and normally selected.
        guard let complete = incomplete.walk(Array("{}".utf8)) else {
            Issue.record("'{}' should walk to a complete state")
            return
        }
        #expect(complete.isComplete)
        #expect(JSONConstraintProcessor.isLegal(1, state: complete, table: table) == true)
        // "{" is illegal after a complete top-level value, so EOS (id 1) wins.
        #expect(JSONConstraintProcessor.selectLegalToken(
            state: complete, table: table, descendingLogitOrder: [0, 1]) == 1)
    }

    // MARK: lowestStopToken — determinism + range filtering

    @Test
    func lowestStopTokenPicksMinimumInRange() {
        #expect(JSONConstraintProcessor.lowestStopToken(in: [7, 2, 5], vocab: 10) == 2)
        #expect(JSONConstraintProcessor.lowestStopToken(in: [9, 12], vocab: 10) == 9)   // 12 out of range
        #expect(JSONConstraintProcessor.lowestStopToken(in: [12, 20], vocab: 10) == nil) // none in range
        #expect(JSONConstraintProcessor.lowestStopToken(in: [-1, 3], vocab: 10) == 3)    // negative excluded
        #expect(JSONConstraintProcessor.lowestStopToken(in: [], vocab: 10) == nil)
    }

    // MARK: schema (C2) decision — mid-value legality

    @Test
    func schemaValueTypeConstrainsSelection() {
        // Schema {"age": integer}. After `{"age":`, only a sign/digit may start
        // the value — a quote or letter token is illegal.
        let object = JSONSchemaObject(properties: [.init(name: "age", type: .integer)], required: [])
        let table = table(["\"", "7", "abc"])   // 0 quote, 1 digit, 2 letters
        guard let atValue = ConstraintState.schema(SchemaConstraintState(schema: object))
            .walk(Array("{\"age\":".utf8)) else {
            Issue.record("prefix should walk")
            return
        }
        // Order [0, 2, 1]: quote (illegal for integer) then letters (illegal)
        // then the digit (legal) — the digit must be chosen.
        #expect(JSONConstraintProcessor.selectLegalToken(
            state: atValue, table: table, descendingLogitOrder: [0, 2, 1]) == 1)
    }
}

// MARK: - JSONConstraintProcessor MLXArray masking (gated, local-only)
//
// GATED — needs a live Metal backend (`MLXArray` construction `fatalError`s under
// bare `swift test`), so `requireMLXRuntimeOrSkip()` skips it in the SPM/CI job
// and it runs under xcodebuild. It locks the *mechanics* the CI decision tests
// cannot touch: the actual `-inf` mask the processor returns and that a forced
// EOS does not advance the constraint state.
final class JSONConstraintProcessorMaskTests: XCTestCase {

    /// A tokenizer over a fixed vocabulary: `decode([id])` is the token string,
    /// so the processor builds a correct `TokenVocabularyTable` from it. Only the
    /// members the table needs are non-trivial.
    private struct ScriptedTokenizer: Tokenizer {
        let vocab: [String]
        let bosToken: String? = nil
        let eosToken: String? = nil
        let unknownToken: String? = nil
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.compactMap { $0 >= 0 && $0 < vocab.count ? vocab[$0] : nil }.joined()
        }
        func convertTokenToId(_ token: String) -> Int? { vocab.firstIndex(of: token) }
        func convertIdToToken(_ id: Int) -> String? {
            id >= 0 && id < vocab.count ? vocab[id] : nil
        }
        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    private func processor(vocab: [String], stop: Set<Int>, greedy: Bool) -> JSONConstraintProcessor {
        JSONConstraintProcessor(
            format: .jsonObject,
            inner: nil,
            cache: TokenVocabularyCache(),
            modelID: "test-scripted",
            tokenizer: ScriptedTokenizer(vocab: vocab),
            stopTokenIDs: stop,
            greedy: greedy
        )
    }

    /// H1: no legal token ⇒ the mask keeps ONLY the lowest stop id (clean EOS),
    /// and `didSample(EOS)` leaves the constraint state unadvanced.
    func testForcedEOSWhenNoLegalToken() throws {
        try requireMLXRuntimeOrSkip()
        // 0 "}" 1 ":" 2 "abc" all illegal at a fresh JSON start; 3 is EOS.
        let vocab = ["}", ":", "abc", "</s>"]
        let processor = processor(vocab: vocab, stop: [3], greedy: true)

        // Logits favor an illegal token; EOS has the lowest logit.
        let logits = MLXArray([5.0, 4.0, 9.0, 1.0] as [Float]).reshaped([1, 4])
        let masked = processor.process(logits: logits).reshaped([4])
        masked.eval()
        let values = masked.asArray(Float.self)

        // Only EOS (id 3) survives; everything else is -inf, so argmax → 3.
        XCTAssertEqual(values[0], -Float.infinity)
        XCTAssertEqual(values[1], -Float.infinity)
        XCTAssertEqual(values[2], -Float.infinity)
        XCTAssertEqual(values[3], 1.0, accuracy: 1e-4)
        XCTAssertEqual(argMax(masked, axis: -1).item(Int.self), 3)

        // didSample(EOS) must not advance the state (no freeze residue): EOS just
        // terminates the stream.
        var advanced = processor
        let before = advanced.state.diagnosticDescription
        advanced.didSample(token: MLXArray(Int32(3)))
        XCTAssertEqual(advanced.state.diagnosticDescription, before)
    }

    /// Greedy path, top token illegal: the highest-logit *legal* token is kept
    /// (one-hot) and all others masked — exercises the argsort fallback +
    /// `selectLegalToken` + `applyMask(keepingOnly:)`.
    func testGreedyKeepsHighestLegalToken() throws {
        try requireMLXRuntimeOrSkip()
        // Top logit is id 1 "abc" (illegal); id 0 "{" is the legal fallback.
        let vocab = ["{", "abc", "}"]
        let processor = processor(vocab: vocab, stop: [], greedy: true)

        let logits = MLXArray([3.0, 9.0, 2.0] as [Float]).reshaped([1, 3])
        let masked = processor.process(logits: logits).reshaped([3])
        masked.eval()
        let values = masked.asArray(Float.self)

        // Only the legal "{" (id 0) keeps a finite logit.
        XCTAssertGreaterThan(values[0], -Float.infinity)
        XCTAssertEqual(values[1], -Float.infinity)
        XCTAssertEqual(values[2], -Float.infinity)
        XCTAssertEqual(argMax(masked, axis: -1).item(Int.self), 0)
    }

    /// Sampling (full-mask) path: every illegal token is masked, the legal one
    /// keeps its (penalty-adjusted) logit.
    func testFullMaskLeavesOnlyLegalTokens() throws {
        try requireMLXRuntimeOrSkip()
        let vocab = ["{", "abc", "}"]
        let processor = processor(vocab: vocab, stop: [], greedy: false)

        let logits = MLXArray([3.0, 9.0, 2.0] as [Float]).reshaped([1, 3])
        let masked = processor.process(logits: logits).reshaped([3])
        masked.eval()
        let values = masked.asArray(Float.self)

        XCTAssertGreaterThan(values[0], -Float.infinity)
        XCTAssertEqual(values[1], -Float.infinity)
        XCTAssertEqual(values[2], -Float.infinity)
        XCTAssertEqual(argMax(masked, axis: -1).item(Int.self), 0)
    }
}
