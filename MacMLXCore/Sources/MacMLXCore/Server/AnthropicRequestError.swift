// Copyright © 2026 macMLX. English comments only.

/// An Anthropic Messages body macMLX cannot serve as sent (bad JSON, a
/// missing `max_tokens`, an unknown `tool_choice` type, a broken tool
/// history). `message` is the text of the 400 `invalid_request_error` that
/// `/v1/messages` and `/v1/messages/count_tokens` return.
struct AnthropicRequestError: Error, Equatable, Sendable {
    let message: String
}
