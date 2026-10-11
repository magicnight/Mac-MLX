// Copyright © 2026 macMLX. English comments only.

import MLXLMCommon
import XCTest

@testable import MacMLXCore

/// The key a text generation records its prompt cache under (#142): the
/// prompt, the tokens that became text, and the stop token the loop
/// delivered last, which the iterator had stepped before the loop saw it.
/// MLX-free, so it runs under bare `swift test`.
final class MLXSwiftEnginePromptCacheKeyTests: XCTestCase {

    private let prompt = [10, 11, 12]
    private let generated = [20, 21]

    func testAStopTokenJoinsTheKey() {
        XCTAssertEqual(
            MLXSwiftEngine.promptCacheKey(
                inputTokens: prompt, generatedTokenIDs: generated, terminalStopToken: 2, stopReason: .stop),
            [10, 11, 12, 20, 21, 2])
    }

    func testABudgetStopRecordsWhatWasYielded() {
        XCTAssertEqual(
            MLXSwiftEngine.promptCacheKey(
                inputTokens: prompt, generatedTokenIDs: generated, terminalStopToken: nil, stopReason: .length),
            [10, 11, 12, 20, 21])
    }

    func testABudgetStopThatDeliveredAStopTokenKeepsIt() {
        XCTAssertEqual(
            MLXSwiftEngine.promptCacheKey(
                inputTokens: prompt, generatedTokenIDs: generated, terminalStopToken: 2, stopReason: .length),
            [10, 11, 12, 20, 21, 2])
    }

    func testAStopWithoutTheTokenRecordsNothing() {
        XCTAssertNil(
            MLXSwiftEngine.promptCacheKey(
                inputTokens: prompt, generatedTokenIDs: generated, terminalStopToken: nil, stopReason: .stop),
            "a loop that reports a stop token it did not deliver leaves a position no key names")
    }

    func testATokenAfterTheStopTokenRecordsNothing() {
        XCTAssertNil(
            MLXSwiftEngine.promptCacheKey(
                inputTokens: prompt, generatedTokenIDs: generated, terminalStopToken: 2,
                tokenAfterStopToken: true, stopReason: .stop),
            "a loop that delivered a token after the stop token holds positions in an order no key here names")
    }

    func testThePhaseSummaryLeavesTheStopTokenOutOfItsCount() {
        let config = EngineGenerationConfig(kvBits: nil, kvGroupSize: nil, quantizedKVStart: nil, batchSize: 1)
        let info = GenerateCompletionInfo(
            promptTokenCount: 24, generationTokenCount: 6, promptTime: 1, generationTime: 1, stopReason: .stop)
        XCTAssertEqual(
            MLXSwiftEngine.phaseSummary(config: config, completionInfo: info, stopTokenCounted: true).generationTokenCount,
            5)
        XCTAssertEqual(
            MLXSwiftEngine.phaseSummary(config: config, completionInfo: info).generationTokenCount, 6)
        XCTAssertEqual(
            MLXSwiftEngine.phaseSummary(config: config, completionInfo: nil, stopTokenCounted: true).generationTokenCount,
            0, "never below zero")
    }

    func testTheTrackerKeepsTheFirstStopTokenAndNotesWhatFollows() {
        var alone = MLXSwiftEngine.StopTokenTracker()
        XCTAssertFalse(alone.note(20, isStop: false))
        XCTAssertTrue(alone.note(2, isStop: true), "a stop token stays out of the text")
        XCTAssertEqual(alone.terminal, 2)
        XCTAssertFalse(alone.tokenAfterTerminal)

        var thenText = alone
        XCTAssertFalse(thenText.note(21, isStop: false), "text after the stop token still flows")
        XCTAssertTrue(thenText.tokenAfterTerminal, "and is noted: the loop no longer breaks on the stop token")

        var thenStop = alone
        XCTAssertTrue(thenStop.note(3, isStop: true), "a second stop token stays out of the text too")
        XCTAssertEqual(thenStop.terminal, 2, "the first stop token remains the terminal one")
        XCTAssertTrue(thenStop.tokenAfterTerminal, "and the second is noted as a token after it")
    }

    func testACancelOrAMissingCompletionRecordsNothing() {
        XCTAssertNil(
            MLXSwiftEngine.promptCacheKey(
                inputTokens: prompt, generatedTokenIDs: generated, terminalStopToken: nil, stopReason: .cancelled))
        XCTAssertNil(
            MLXSwiftEngine.promptCacheKey(
                inputTokens: prompt, generatedTokenIDs: generated, terminalStopToken: nil, stopReason: nil))
    }
}
