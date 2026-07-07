import XCTest
@testable import MacMLXCore

final class MessageSegmentTests: XCTestCase {

    func testPlainText() {
        XCTAssertEqual(
            MessageSegmenter.parse("Hello world"),
            [.text("Hello world")]
        )
    }

    func testEmptyString() {
        XCTAssertEqual(MessageSegmenter.parse(""), [])
    }

    func testBalancedTagsWithTrailingAnswer() {
        let got = MessageSegmenter.parse("<think>reason</think>Answer")
        XCTAssertEqual(got, [
            .think("reason", isClosed: true),
            .text("Answer")
        ])
    }

    func testStreamingOpenOnly() {
        let got = MessageSegmenter.parse("<think>partial")
        XCTAssertEqual(got, [.think("partial", isClosed: false)])
    }

    func testImplicitOpenerQwen3Style() {
        // qwen3 template injects `<think>` in the prompt; the stream
        // only contains the close tag and whatever comes before it.
        let got = MessageSegmenter.parse("reason here</think>Answer")
        XCTAssertEqual(got, [
            .think("reason here", isClosed: true),
            .text("Answer")
        ])
    }

    func testTextBeforeOpenTag() {
        let got = MessageSegmenter.parse("Intro<think>r</think>A")
        XCTAssertEqual(got, [
            .text("Intro"),
            .think("r", isClosed: true),
            .text("A")
        ])
    }

    func testMultipleThinkBlocks() {
        let got = MessageSegmenter.parse("<think>a</think>X<think>b</think>Y")
        XCTAssertEqual(got, [
            .think("a", isClosed: true),
            .text("X"),
            .think("b", isClosed: true),
            .text("Y")
        ])
    }

    func testImplicitOpenerPlusLaterPair() {
        // close-then-open: implicit think, then text, then normal think.
        let got = MessageSegmenter.parse("implicit</think>Answer <think>more</think>End")
        XCTAssertEqual(got, [
            .think("implicit", isClosed: true),
            .text("Answer "),
            .think("more", isClosed: true),
            .text("End")
        ])
    }

    func testCloseWithoutAnyThinkContent() {
        // Edge case: just a close tag at the start.
        let got = MessageSegmenter.parse("</think>Answer")
        XCTAssertEqual(got, [.text("Answer")])
    }

    // MARK: - splitReasoning (reasoning_content API convention)

    func testSplitReasoningBalanced() {
        let (reasoning, answer) = MessageSegmenter.splitReasoning("<think>reason</think>Answer")
        XCTAssertEqual(reasoning, "reason")
        XCTAssertEqual(answer, "Answer")
    }

    func testSplitReasoningQwen3ImplicitOpener() {
        // The #30 case: model emits reasoning + close tag but no opener
        // (qwen3's chat template injects `<think>` into the prompt).
        let (reasoning, answer) = MessageSegmenter.splitReasoning("reason here</think>Answer")
        XCTAssertEqual(reasoning, "reason here")
        XCTAssertEqual(answer, "Answer")
    }

    func testSplitReasoningNoTagsIsUntouched() {
        // Non-reasoning models: no reasoning, content passes through.
        let (reasoning, answer) = MessageSegmenter.splitReasoning("Just an answer")
        XCTAssertNil(reasoning)
        XCTAssertEqual(answer, "Just an answer")
    }

    func testSplitReasoningStreamingOpenOnly() {
        let (reasoning, answer) = MessageSegmenter.splitReasoning("<think>still thinking")
        XCTAssertEqual(reasoning, "still thinking")
        XCTAssertEqual(answer, "")
    }

    func testSplitReasoningEmpty() {
        let (reasoning, answer) = MessageSegmenter.splitReasoning("")
        XCTAssertNil(reasoning)
        XCTAssertEqual(answer, "")
    }

    func testSplitReasoningMultipleBlocksJoin() {
        let (reasoning, answer) = MessageSegmenter.splitReasoning(
            "<think>a</think>X<think>b</think>Y")
        XCTAssertEqual(reasoning, "ab")
        XCTAssertEqual(answer, "XY")
    }
}
