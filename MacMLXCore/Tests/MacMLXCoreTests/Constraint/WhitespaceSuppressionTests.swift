// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLX
import MLXLMCommon
import Testing
import XCTest

@testable import MacMLXCore

// MARK: - The latch and the pure selection policy (MLX-free)

@Suite("Whitespace runaway suppression")
struct WhitespaceSuppressionTests {

    private func table(_ vocab: [String], stop: Set<Int> = []) -> TokenVocabularyTable {
        TokenVocabularyTable(vocabularySize: vocab.count, stopTokenIDs: stop, decode: { vocab[$0] })
    }

    private func schemaState(_ object: JSONSchemaObject, after prefix: String) throws -> ConstraintState {
        try #require(ConstraintState.schema(SchemaConstraintState(schema: object)).walk(Array(prefix.utf8)))
    }

    private func jsonState(after prefix: String) throws -> ConstraintState {
        try #require(ConstraintState.json(JSONGrammarState()).walk(Array(prefix.utf8)))
    }

    private var flat: JSONSchemaObject {
        JSONSchemaObject(properties: [.init(name: "a", type: .string)], required: ["a"])
    }

    @Test
    func latchTripsAtTheThresholdAndStaysOn() {
        var latch = WhitespaceRunLatch(threshold: 3)
        latch.record(whitespaceOnly: true)
        latch.record(whitespaceOnly: true)
        #expect(!latch.isActive, "two in a row is still ordinary formatting")
        latch.record(whitespaceOnly: false)
        latch.record(whitespaceOnly: true)
        latch.record(whitespaceOnly: true)
        #expect(!latch.isActive, "a non-whitespace token resets the run")
        latch.record(whitespaceOnly: true)
        #expect(latch.isActive)
        latch.record(whitespaceOnly: false)
        #expect(latch.isActive, "once tripped it never releases")
    }

    @Test
    func whitespaceOnlyTokensAreRecognisedByBytes() {
        let table = table([" ", "\n", " \t\r\n", " x", "", "\"", "</s>"], stop: [6])
        #expect(table.isWhitespaceOnly(0))
        #expect(table.isWhitespaceOnly(1))
        #expect(table.isWhitespaceOnly(2))
        #expect(!table.isWhitespaceOnly(3))
        #expect(!table.isWhitespaceOnly(4), "an unusable token is not whitespace")
        #expect(!table.isWhitespaceOnly(5))
        #expect(!table.isWhitespaceOnly(6), "EOS is not whitespace")
    }

    @Test
    func insideStringIsReportedByBothAutomata() throws {
        #expect(try jsonState(after: "{\"k").isInsideString, "an object key is a string literal")
        #expect(try jsonState(after: "{\"k\":\"v").isInsideString)
        #expect(try jsonState(after: "{\"k\":\"v\\").isInsideString, "mid-escape")
        #expect(!(try jsonState(after: "{").isInsideString))
        #expect(!(try jsonState(after: "{\"k\":1").isInsideString))
        #expect(!(try jsonState(after: "{\"k\":\"v\"").isInsideString))

        #expect(try schemaState(flat, after: "{\"a").isInsideString, "a schema key is matched byte by byte like a literal")
        #expect(try schemaState(flat, after: "{\"a\":\"x").isInsideString)
        #expect(!(try schemaState(flat, after: "{").isInsideString))
        #expect(!(try schemaState(flat, after: "{\"a\":").isInsideString))
        #expect(!(try schemaState(flat, after: "{\"a\":\"x\"").isInsideString))
        let enumSchema = JSONSchemaObject(
            properties: [.init(name: "a", type: .stringEnum(["New York", "Paris"]))], required: ["a"])
        #expect(try schemaState(enumSchema, after: "{\"a\":\"New").isInsideString, "an enum literal may itself contain a space")
    }

    /// At a structural position with the latch on, the whitespace tokens the
    /// model prefers are withheld and the next legal token wins; with the latch
    /// off the policy is unchanged.
    @Test
    func suppressionSkipsWhitespaceAtStructuralPositions() throws {
        let table = table(["\n", "  ", "\"", "}"])
        let state = try schemaState(flat, after: "{")
        let order = [0, 1, 2, 3]
        #expect(JSONConstraintProcessor.selectLegalToken(
            state: state, table: table, descendingLogitOrder: order) == 0)
        #expect(JSONConstraintProcessor.selectLegalToken(
            state: state, table: table, descendingLogitOrder: order, suppressingWhitespace: true) == 2)
        #expect(!JSONConstraintProcessor.isLegal(0, state: state, table: table, suppressingWhitespace: true))
        #expect(JSONConstraintProcessor.isLegal(0, state: state, table: table))
    }

    /// Inside a string value a space is data, so suppression never applies
    /// there — the processor consults `isInsideString` before suppressing.
    @Test
    func suppressionLeavesStringContentAlone() throws {
        let table = table([" ", "lo\"", "}"])
        let state = try schemaState(flat, after: "{\"a\":\"hel")
        // The processor decides suppression from the state; the pure selector is
        // asked with suppression OFF for an in-string state, which is what
        // `JSONConstraintProcessor` does.
        #expect(state.isInsideString)
        #expect(JSONConstraintProcessor.selectLegalToken(
            state: state, table: table, descendingLogitOrder: [0, 1, 2]) == 0)
    }

    /// Suppression can never create a dead end: when whitespace is the only
    /// legal continuation the vocabulary offers, it is let through.
    @Test
    func suppressionFallsBackWhenOnlyWhitespaceIsLegal() throws {
        // At `{` only a key quote or `}` could follow, and this vocabulary has
        // neither — only whitespace is legal.
        let table = table(["\n", "x", ":"])
        let state = try schemaState(flat, after: "{")
        #expect(JSONConstraintProcessor.selectLegalToken(
            state: state, table: table, descendingLogitOrder: [1, 2, 0], suppressingWhitespace: true) == nil,
            "the pure selector reports that nothing non-whitespace is legal")
        // The processor's fallback is exercised through the Metal-gated test
        // below; here the latch-free selection shows whitespace is legal at all.
        #expect(JSONConstraintProcessor.selectLegalToken(
            state: state, table: table, descendingLogitOrder: [1, 2, 0]) == 0)
    }
}

// MARK: - The processor end to end (needs Metal for the MLX mask)

final class WhitespaceSuppressionMaskTests: XCTestCase {

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
        func convertIdToToken(_ id: Int) -> String? { id >= 0 && id < vocab.count ? vocab[id] : nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    /// `topKCap: 1` so the greedy path's first pass covers one token and every
    /// test also exercises the scan of the remaining vocabulary (and, under
    /// suppression, its unsuppressed rescan) — with a three-token vocabulary
    /// the default 256-token first pass would never reach those loops.
    private func processor(vocab: [String], greedy: Bool) -> JSONConstraintProcessor {
        JSONConstraintProcessor(
            format: .jsonSchema(JSONSchemaObject(properties: [.init(name: "a", type: .string)], required: ["a"])),
            inner: nil,
            cache: TokenVocabularyCache(),
            modelID: "test-whitespace-\(greedy)",
            tokenizer: ScriptedTokenizer(vocab: vocab),
            stopTokenIDs: [],
            greedy: greedy,
            topKCap: 1)
    }

    /// Spaces sampled INSIDE a string value do not count toward the latch: the
    /// state is consulted before the token advances it. The vocabulary offers
    /// `}` after the value, so a tripped latch would have something to prefer
    /// over the space (without that alternative the dead-end fallback would
    /// let the space through and hide a wrongly tripped latch).
    func testWhitespaceInsideAStringDoesNotCountTowardTheLatch() throws {
        try requireMLXRuntimeOrSkip()
        let vocab = [" ", "\"", "lo", "}"]
        let logits: [Float] = [9.0, 1.0, 2.0, 3.0]
        var processor = processor(vocab: vocab, greedy: true)
        processor.state = try XCTUnwrap(processor.state.walk(Array("{\"a\":\"hel".utf8)))
        _ = masked(processor, logits)   // resolves the table
        for _ in 0..<3 { processor.didSample(token: MLXArray(Int32(0))) }   // three spaces, all data
        processor.state = try XCTUnwrap(processor.state.walk(Array("lo\"".utf8)))
        // Back at a structural position (after the value) whitespace is still
        // the model's choice: the latch never tripped.
        let values = masked(processor, logits)
        XCTAssertGreaterThan(values[0], -Float.infinity, "the latch tripped on in-string spaces")
        XCTAssertEqual(argMax(MLXArray(values), axis: -1).item(Int.self), 0)
    }

    private func masked(_ processor: JSONConstraintProcessor, _ logits: [Float]) -> [Float] {
        let out = processor.process(logits: MLXArray(logits).reshaped([1, logits.count])).reshaped([logits.count])
        out.eval()
        return out.asArray(Float.self)
    }

    /// After `{` the model keeps preferring newlines. Three sampled whitespace
    /// tokens trip the latch; from then on the whitespace token is masked at
    /// this structural position on both the greedy and the sampling path, and
    /// the key quote wins.
    func testThreeWhitespaceTokensTripTheLatchAndMaskWhitespace() throws {
        try requireMLXRuntimeOrSkip()
        let vocab = ["\n", "\"", "a\":\"x\"}"]
        for greedy in [true, false] {
            var processor = processor(vocab: vocab, greedy: greedy)
            let logits: [Float] = [9.0, 2.0, 1.0]   // the model wants "\n"
            // Open the object (the automaton is advanced directly; the table is
            // resolved by the first `process` below, which `didSample` needs).
            processor.state = try XCTUnwrap(processor.state.walk(Array("{".utf8)))
            // Two whitespace tokens: still allowed.
            for _ in 0..<2 {
                let values = masked(processor, logits)
                XCTAssertGreaterThan(values[0], -Float.infinity, "greedy=\(greedy): whitespace still legal before the latch")
                processor.didSample(token: MLXArray(Int32(0)))
            }
            // Third: the latch trips on this sample.
            let before = masked(processor, logits)
            XCTAssertGreaterThan(before[0], -Float.infinity)
            processor.didSample(token: MLXArray(Int32(0)))
            // Now whitespace is withheld and the key quote is the choice.
            let after = masked(processor, logits)
            XCTAssertEqual(after[0], -Float.infinity, "greedy=\(greedy): whitespace masked once the latch is on")
            XCTAssertGreaterThan(after[1], -Float.infinity)
            XCTAssertEqual(argMax(MLXArray(after), axis: -1).item(Int.self), 1)
        }
    }

    /// The model's top token is illegal and whitespace is its second choice:
    /// with the latch on, the scan of the vocabulary beyond the top-K must skip
    /// the whitespace token and land on the key quote (third choice), on both
    /// paths. (With `topKCap: 1` the quote is only reachable through that scan.)
    func testRemainderScanSkipsWhitespaceOnceTheLatchIsOn() throws {
        try requireMLXRuntimeOrSkip()
        let vocab = ["\n", "\"", "a\":\"x\"}"]
        for greedy in [true, false] {
            var processor = processor(vocab: vocab, greedy: greedy)
            let logits: [Float] = [5.0, 1.0, 9.0]   // illegal top token, then "\n", then the quote
            processor.state = try XCTUnwrap(processor.state.walk(Array("{".utf8)))
            _ = masked(processor, logits)   // resolves the table
            for _ in 0..<3 { processor.didSample(token: MLXArray(Int32(0))) }
            let values = masked(processor, logits)
            XCTAssertEqual(values[0], -Float.infinity, "greedy=\(greedy): whitespace skipped in the remainder scan")
            XCTAssertEqual(values[2], -Float.infinity, "greedy=\(greedy): the illegal top token stays masked")
            XCTAssertEqual(argMax(MLXArray(values), axis: -1).item(Int.self), 1, "greedy=\(greedy): the quote wins")
        }
    }

    /// Inside the string value the same latch leaves the space token alone.
    func testLatchDoesNotMaskWhitespaceInsideAString() throws {
        try requireMLXRuntimeOrSkip()
        let vocab = [" ", "\"", "lo"]
        var processor = processor(vocab: vocab, greedy: true)
        // The table is resolved on the first `process`; `didSample` records
        // nothing before that, so prime it, then trip the latch at `{`.
        processor.state = try XCTUnwrap(processor.state.walk(Array("{".utf8)))
        _ = masked(processor, [9.0, 1.0, 2.0])
        for _ in 0..<3 { processor.didSample(token: MLXArray(Int32(0))) }
        XCTAssertEqual(masked(processor, [9.0, 1.0, 2.0])[0], -Float.infinity, "the latch is on at a structural position")
        processor.state = try XCTUnwrap(processor.state.walk(Array("\"a\":\"hel".utf8)))
        let values = masked(processor, [9.0, 1.0, 2.0])
        XCTAssertGreaterThan(values[0], -Float.infinity, "a space inside a string is data")
        XCTAssertEqual(argMax(MLXArray(values), axis: -1).item(Int.self), 0)
    }

    /// With the latch on but nothing non-whitespace legal in the vocabulary,
    /// whitespace is let through rather than forcing EOS on a dead end.
    func testLatchFallsBackWhenOnlyWhitespaceIsLegal() throws {
        try requireMLXRuntimeOrSkip()
        let vocab = ["\n", "x", ":"]
        for greedy in [true, false] {
            var processor = processor(vocab: vocab, greedy: greedy)
            processor.state = try XCTUnwrap(processor.state.walk(Array("{".utf8)))
            _ = masked(processor, [1.0, 9.0, 5.0])   // resolves the table
            for _ in 0..<3 { processor.didSample(token: MLXArray(Int32(0))) }
            let values = masked(processor, [1.0, 9.0, 5.0])
            XCTAssertGreaterThan(values[0], -Float.infinity, "greedy=\(greedy): the only legal token survives")
            XCTAssertEqual(argMax(MLXArray(values), axis: -1).item(Int.self), 0)
        }
    }
}
