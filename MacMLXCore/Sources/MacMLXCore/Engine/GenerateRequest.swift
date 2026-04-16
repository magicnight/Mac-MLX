import Foundation

/// A turn in a chat conversation.
public struct ChatMessage: Codable, Hashable, Identifiable, Sendable {
    public let id: UUID
    public let role: MessageRole
    public let content: String

    public init(id: UUID = UUID(), role: MessageRole, content: String) {
        self.id = id
        self.role = role
        self.content = content
    }
}

/// OpenAI-compatible message roles.
public enum MessageRole: String, Codable, Hashable, Sendable, CaseIterable {
    case system
    case user
    case assistant
}

/// Sampling and length parameters.
public struct GenerationParameters: Codable, Hashable, Sendable {
    public var temperature: Double
    public var topP: Double
    public var maxTokens: Int
    public var stream: Bool

    public init(
        temperature: Double = 0.7,
        topP: Double = 0.95,
        maxTokens: Int = 2048,
        stream: Bool = true
    ) {
        self.temperature = temperature
        self.topP = topP
        self.maxTokens = maxTokens
        self.stream = stream
    }
}

/// Everything an inference engine needs to start a generation.
public struct GenerateRequest: Codable, Hashable, Sendable {
    public let model: String
    public let messages: [ChatMessage]
    public let systemPrompt: String?
    public var parameters: GenerationParameters

    public init(
        model: String,
        messages: [ChatMessage],
        systemPrompt: String? = nil,
        parameters: GenerationParameters = .init()
    ) {
        self.model = model
        self.messages = messages
        self.systemPrompt = systemPrompt
        self.parameters = parameters
    }

    /// Messages with the system prompt (if any) prepended.
    public var allMessages: [ChatMessage] {
        guard let systemPrompt, !systemPrompt.isEmpty else { return messages }
        return [ChatMessage(role: .system, content: systemPrompt)] + messages
    }
}
