// Copyright © 2026 macMLX. English comments only.

/// How a `/v1/messages` reply returns the model's reasoning, from the
/// request's `thinking`.
///
/// Either way a returned block's `signature` carries the reasoning
/// (`ReasoningCarrier`), so a client that replays the block gives it back on
/// the next turn even when it never saw the text.
enum AnthropicThinkingReply: Equatable, Sendable {
    /// No `thinking`, or `disabled`: the reasoning is dropped.
    case none
    /// `enabled` / `adaptive`: a `thinking` block with the reasoning text.
    case shown
    /// `enabled` / `adaptive` with `display: "omitted"` (Claude Code's
    /// setting): a `thinking` block with empty text, as Anthropic returns one.
    case omitted

    init(type: String?, display: String?) {
        switch type {
        case "enabled", "adaptive": self = display == "omitted" ? .omitted : .shown
        default: self = .none
        }
    }
}
