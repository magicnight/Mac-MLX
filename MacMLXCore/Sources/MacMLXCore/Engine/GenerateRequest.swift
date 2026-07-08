import Foundation

/// A turn in a chat conversation.
public struct ChatMessage: Codable, Hashable, Identifiable, Sendable {
    public let id: UUID
    public let role: MessageRole
    public let content: String
    /// Image attachments. Empty for text-only messages — the common
    /// case. Backwards compatible: pre-v0.4.1 conversation JSON (which
    /// has no `images` key) decodes with an empty array, so existing
    /// user chats survive the upgrade unchanged.
    public let images: [ImageAttachment]

    public init(
        id: UUID = UUID(),
        role: MessageRole,
        content: String,
        images: [ImageAttachment] = []
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.images = images
    }

    private enum CodingKeys: String, CodingKey {
        case id, role, content, images
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try c.decode(UUID.self, forKey: .id)
        self.role = try c.decode(MessageRole.self, forKey: .role)
        self.content = try c.decode(String.self, forKey: .content)
        // Default to empty when the key is absent (legacy conversations).
        self.images = try c.decodeIfPresent([ImageAttachment].self, forKey: .images) ?? []
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
    /// Optional per-model chat-template kwargs (v0.5.1) forwarded to the
    /// Jinja chat template as `additionalContext` — e.g.
    /// `{"enable_thinking": true}` for Qwen3. Stored as `JSONValue` (not
    /// `[String: any Sendable]`) so `GenerateRequest` keeps its
    /// synthesised `Codable` / `Hashable`; the engine unwraps to the
    /// tokenizer's `Sendable` shape at the `UserInput` boundary.
    public var templateKwargs: [String: JSONValue]?

    public init(
        model: String,
        messages: [ChatMessage],
        systemPrompt: String? = nil,
        parameters: GenerationParameters = .init(),
        templateKwargs: [String: JSONValue]? = nil
    ) {
        self.model = model
        self.messages = messages
        self.systemPrompt = systemPrompt
        self.parameters = parameters
        self.templateKwargs = templateKwargs
    }

    /// Messages with the system prompt (if any) prepended.
    public var allMessages: [ChatMessage] {
        guard let systemPrompt, !systemPrompt.isEmpty else { return messages }
        return [ChatMessage(role: .system, content: systemPrompt)] + messages
    }
}
