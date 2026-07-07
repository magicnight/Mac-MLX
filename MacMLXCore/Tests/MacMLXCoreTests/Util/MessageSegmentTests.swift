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

    // MARK: - ReasoningStreamSplitter (streaming, incremental)

    func testStreamSplitterImplicitReasoningWholeChunk() {
        var s = ReasoningStreamSplitter(startInReasoning: true)
        let (r, a) = s.push("reasoning here</think>the answer")
        XCTAssertEqual(r, "reasoning here")
        XCTAssertEqual(a, "the answer")
    }

    func testStreamSplitterImplicitReasoningAcrossChunks() {
        // qwen3 case: prompt opened <think>, model streams reasoning then
        // </think> then the answer, chunked arbitrarily.
        var s = ReasoningStreamSplitter(startInReasoning: true)
        var reasoning = "", answer = ""
        for chunk in ["rea", "soning", "</think>", "ans", "wer"] {
            let (r, a) = s.push(chunk)
            reasoning += r
            answer += a
        }
        let (r, a) = s.finish()
        reasoning += r
        answer += a
        XCTAssertEqual(reasoning, "reasoning")
        XCTAssertEqual(answer, "answer")
    }

    func testStreamSplitterTagSplitAcrossChunks() {
        // </think> arrives split as "</thi" + "nk>" — must not leak.
        var s = ReasoningStreamSplitter(startInReasoning: true)
        var reasoning = "", answer = ""
        for chunk in ["think</thi", "nk>done"] {
            let (r, a) = s.push(chunk)
            reasoning += r
            answer += a
        }
        XCTAssertEqual(reasoning, "think")
        XCTAssertEqual(answer, "done")
    }

    func testStreamSplitterNonReasoningModel() {
        // startInReasoning=false, no tags → everything is answer.
        var s = ReasoningStreamSplitter(startInReasoning: false)
        var answer = ""
        for chunk in ["plain ", "answer ", "no tags"] {
            let (r, a) = s.push(chunk)
            XCTAssertEqual(r, "")
            answer += a
        }
        XCTAssertEqual(answer, "plain answer no tags")
    }

    func testStreamSplitterExplicitThinkTag() {
        var s = ReasoningStreamSplitter(startInReasoning: false)
        let (r, a) = s.push("<think>reason</think>answer")
        XCTAssertEqual(r, "reason")
        XCTAssertEqual(a, "answer")
    }

    func testStreamSplitterFlushesFalseTagPrefixOnFinish() {
        // A trailing "<" that never becomes a tag must flush at finish.
        var s = ReasoningStreamSplitter(startInReasoning: false)
        let (r1, a1) = s.push("answer<")
        XCTAssertEqual(r1, "")
        XCTAssertEqual(a1, "answer")  // "<" held back as a possible tag
        let (r2, a2) = s.finish()
        XCTAssertEqual(r2, "")
        XCTAssertEqual(a2, "<")
    }

    // MARK: - promptOpensThink (streaming seed signal)

    func testPromptOpensThinkTrailingOpen() {
        // qwen3: prompt ends with the injected opener → open.
        XCTAssertTrue(MessageSegmenter.promptOpensThink("system\nuser: hi\nassistant\n<think>\n"))
    }

    func testPromptOpensThinkClosedBlock() {
        // A balanced block from a prior turn → not open.
        XCTAssertFalse(MessageSegmenter.promptOpensThink("<think>x</think>ok, ready"))
    }

    func testPromptOpensThinkNoTags() {
        XCTAssertFalse(MessageSegmenter.promptOpensThink("plain prompt, no tags"))
    }

    func testPromptOpensThinkReopenedAfterClose() {
        XCTAssertTrue(MessageSegmenter.promptOpensThink("<think>a</think>b<think>c"))
    }
}
