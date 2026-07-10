/// One token (or token group) yielded from a streaming generation.
public struct GenerateChunk: Codable, Hashable, Sendable {
    public let text: String
    public let finishReason: FinishReason?
    public let usage: TokenUsage?
    /// Tool calls the model emitted this turn. Non-nil only on the terminal
    /// chunk of a generation that ended in tool calls (paired with
    /// `finishReason == .toolCalls`); nil for ordinary text chunks. Default
    /// nil keeps existing call sites and the synthesised `Codable` (which
    /// omits the key when absent) wire-compatible.
    public let toolCalls: [ToolCallRequest]?
    /// Speculative decoding acceptance counters for this generation (D1 —
    /// classic draft-model path only). Non-nil ONLY on the terminal chunk of
    /// a generation that actually ran the speculative path AND mlx-swift-lm
    /// returned telemetry for it; nil whenever speculative decoding wasn't
    /// requested/used — never fabricated. Default nil keeps existing call
    /// sites and the synthesised `Codable` wire-compatible.
    public let speculativeDecoding: SpeculativeDecodingUsage?
    /// Per-token logprobs for the token(s) this chunk carries (Track E, OpenAI
    /// `logprobs`). Non-nil ONLY when the request set `logprobs: true` and the
    /// engine ran the capture path; the array holds one entry per generated
    /// token in emission order. Nil for ordinary generations and on the terminal
    /// usage-only chunk. Default nil keeps existing call sites and the
    /// synthesised `Codable` wire-compatible.
    public let logprobs: [TokenLogprob]?

    public init(
        text: String,
        finishReason: FinishReason? = nil,
        usage: TokenUsage? = nil,
        toolCalls: [ToolCallRequest]? = nil,
        speculativeDecoding: SpeculativeDecodingUsage? = nil,
        logprobs: [TokenLogprob]? = nil
    ) {
        self.text = text
        self.finishReason = finishReason
        self.usage = usage
        self.toolCalls = toolCalls
        self.speculativeDecoding = speculativeDecoding
        self.logprobs = logprobs
    }
}

/// OpenAI-compatible finish reasons.
public enum FinishReason: String, Codable, Hashable, Sendable, CaseIterable {
    case stop
    case length
    case error
    /// The model stopped to call one or more tools. Raw value matches the
    /// OpenAI `finish_reason` string.
    case toolCalls = "tool_calls"
}

/// Token usage accounting at the end of a generation.
public struct TokenUsage: Codable, Hashable, Sendable {
    public let promptTokens: Int
    public let completionTokens: Int

    public init(promptTokens: Int, completionTokens: Int) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
    }

    public var totalTokens: Int { promptTokens + completionTokens }
}
