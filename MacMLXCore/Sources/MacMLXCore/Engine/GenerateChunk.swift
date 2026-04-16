/// One token (or token group) yielded from a streaming generation.
public struct GenerateChunk: Codable, Hashable, Sendable {
    public let text: String
    public let finishReason: FinishReason?
    public let usage: TokenUsage?

    public init(text: String, finishReason: FinishReason? = nil, usage: TokenUsage? = nil) {
        self.text = text
        self.finishReason = finishReason
        self.usage = usage
    }
}

/// OpenAI-compatible finish reasons.
public enum FinishReason: String, Codable, Hashable, Sendable, CaseIterable {
    case stop
    case length
    case error
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
