// Copyright © 2026 macMLX. English comments only.

/// An OpenAI Responses API request body (`POST /v1/responses`).
///
/// Only the fields macMLX acts on or echoes are declared; everything else a
/// client sends (`prompt_cache_key`, `user`, `service_tier`, `truncation`,
/// `safety_identifier`, `client_metadata`, `include`, …) is accepted and
/// ignored by omission. The polymorphic fields — `input`, `tools`,
/// `tool_choice`, `text`, `reasoning`, `metadata` — stay `JSONValue` and are
/// interpreted by `ResponsesInputTranslator`, which owns every 400 they can
/// produce. Every field is optional so a missing `model` or `input` is
/// reported by name rather than as a generic decode failure.
struct ResponsesRequest: Decodable, Sendable {
    let model: String?
    /// A string (one user message) or an array of input items.
    let input: JSONValue?
    /// Top-level instructions; joined ahead of any leading system/developer
    /// items into the system prompt.
    let instructions: String?
    let tools: [JSONValue]?
    let tool_choice: JSONValue?
    /// Echoed in the response object; not enforced (the engine reports every
    /// call the model makes).
    let parallel_tool_calls: Bool?
    /// `{"format":{…}}` — structured output.
    let text: JSONValue?
    /// `{"effort":…,"summary":…}`.
    let reasoning: JSONValue?
    let metadata: JSONValue?
    let stream: Bool?
    /// Accepted and ignored: macMLX stores no responses.
    let store: Bool?
    /// Rejected when non-null: macMLX stores no responses to continue from.
    let previous_response_id: String?
    /// Rejected when true: there is no stored response to poll.
    let background: Bool?
    let max_output_tokens: Int?
    let temperature: Double?
    let top_p: Double?
}
