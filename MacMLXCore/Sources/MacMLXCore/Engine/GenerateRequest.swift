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
    /// For a `.tool` message: the id of the assistant tool call this result
    /// answers. Nil for every non-tool turn. `decodeIfPresent` + default nil so
    /// pre-v0.5 conversation JSON (no such key) still decodes.
    public let toolCallID: String?
    /// For an `.assistant` message that itself issued tool calls: the calls it
    /// made, so a re-render reproduces the assistant's tool-call block. Nil
    /// otherwise. Back-compatible for the same reason as `toolCallID`.
    public let toolCalls: [ToolCallRequest]?

    public init(
        id: UUID = UUID(),
        role: MessageRole,
        content: String,
        images: [ImageAttachment] = [],
        toolCallID: String? = nil,
        toolCalls: [ToolCallRequest]? = nil
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.images = images
        self.toolCallID = toolCallID
        self.toolCalls = toolCalls
    }

    private enum CodingKeys: String, CodingKey {
        case id, role, content, images, toolCallID, toolCalls
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try c.decode(UUID.self, forKey: .id)
        self.role = try c.decode(MessageRole.self, forKey: .role)
        self.content = try c.decode(String.self, forKey: .content)
        // Default to empty/nil when the key is absent (legacy conversations).
        self.images = try c.decodeIfPresent([ImageAttachment].self, forKey: .images) ?? []
        self.toolCallID = try c.decodeIfPresent(String.self, forKey: .toolCallID)
        self.toolCalls = try c.decodeIfPresent([ToolCallRequest].self, forKey: .toolCalls)
    }
}

/// OpenAI-compatible message roles.
public enum MessageRole: String, Codable, Hashable, Sendable, CaseIterable {
    case system
    case user
    case assistant
    /// A tool result turn (chat-side MCP routing, v0.5). Raw value matches the
    /// OpenAI `tool` role. Additive: legacy conversation JSON never carries it,
    /// so existing on-disk data keeps decoding.
    case tool
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
    /// Optional tool specifications (v0.5) forwarded to the chat template as
    /// `UserInput.tools`. Each element is one OpenAI function spec
    /// (`{"type":"function","function":{name,description,parameters}}`), built
    /// from an MCP tool via `ToolValueBridge.openAIToolSpec(from:)`. Stored as
    /// `[JSONValue]` — not `[ToolSpec]` — so `GenerateRequest` keeps its
    /// synthesised `Codable` / `Hashable`; the engine unwraps to the
    /// tokenizer's `Sendable` shape at the `UserInput` boundary. Default nil
    /// and (being optional) decoded via the synthesised `decodeIfPresent`, so
    /// existing serialized requests keep decoding.
    public var tools: [JSONValue]?

    public init(
        model: String,
        messages: [ChatMessage],
        systemPrompt: String? = nil,
        parameters: GenerationParameters = .init(),
        templateKwargs: [String: JSONValue]? = nil,
        tools: [JSONValue]? = nil
    ) {
        self.model = model
        self.messages = messages
        self.systemPrompt = systemPrompt
        self.parameters = parameters
        self.templateKwargs = templateKwargs
        self.tools = tools
    }

    /// Messages with the system prompt (if any) prepended.
    public var allMessages: [ChatMessage] {
        guard let systemPrompt, !systemPrompt.isEmpty else { return messages }
        return [ChatMessage(role: .system, content: systemPrompt)] + messages
    }
}
