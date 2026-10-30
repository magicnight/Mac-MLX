import Foundation

/// Starts a streamed answer at its first non-whitespace text, as `/v1/messages`
/// and `/v1/responses` do. A thinking model emits blank lines after `</think>`,
/// which are not an answer; on a tool-only turn they made Apple's Foundation
/// Models client store a blank response and replay the reasoning on it, apart
/// from the tool calls. Feed it each answer piece in order.
struct AnswerStartTrimmer {
    private var started = false

    mutating func push(_ text: String) -> String {
        guard !started else { return text }
        let trimmed = String(text.drop(while: \.isWhitespace))
        started = !trimmed.isEmpty
        return trimmed
    }
}
