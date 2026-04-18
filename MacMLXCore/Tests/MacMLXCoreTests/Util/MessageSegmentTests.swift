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
}
