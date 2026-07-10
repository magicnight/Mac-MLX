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
    /// Model id of a draft model to speculate with (D1 — classic per-request
    /// draft-model speculative decoding, mirrors mlx-lm Python server's
    /// `draft_model` request field). Resolved by the engine the same way it
    /// resolves `model` — the id of a directory under the models root. `nil`
    /// (the default, and any value explicitly set back to `nil`) disables
    /// speculative decoding for this request and unloads any draft model the
    /// engine currently has resident — there is no separate "unload draft"
    /// verb. Ignored on VLM requests (text-only, D1).
    ///
    /// - Note: Silently falls back to plain (non-speculative) decoding when
    ///   either the target or draft model's KV cache isn't trimmable — e.g. a
    ///   hybrid/linear-attention architecture such as Qwen3.5, whose
    ///   GatedDeltaNet layers use a non-trimmable `MambaCache`. No error is
    ///   raised in that case; the response simply carries no
    ///   `speculativeDecoding` telemetry. See
    ///   `MLXSwiftEngine.canUseSpeculativeDecoding`.
    ///
    /// - Important: Mutually exclusive with continuous batching, mirroring
    ///   mlx-lm's `is_batchable` semantics — a batched decode request must
    ///   never also carry a draft model. Enforced at the HTTP gate by
    ///   `BatchRoutingPolicy.shouldAttemptBatch(hasDraftModel:)`: a non-nil
    ///   `draftModelID` always routes to the legacy single-stream path, so
    ///   the batched path never sees a request carrying one.
    public var draftModelID: String?
    /// Number of tokens the draft model proposes per speculative round.
    /// Clamped to `1...8` on every construction path (this initialiser AND
    /// JSON decode — see `init(from:)`) so a malformed or hostile request
    /// can't ask mlx-swift-lm for a pathological round size. `nil` (the
    /// default) defers to mlx-swift-lm's own default (2 as of 3.31.3).
    /// Meaningless when `draftModelID` is nil.
    public var numDraftTokens: Int?
    /// Structured-output constraint (Track C). When non-nil the engine
    /// constrains generation so the output is guaranteed well-formed JSON
    /// (`.jsonObject`, C1) or conforms to the compiled schema subset
    /// (`.jsonSchema`, C2), via a decode-time logit mask. Decoded by the server
    /// from OpenAI's `response_format` (unsupported schema features are rejected
    /// with a 400 before a request is ever built — see `ResponseFormatDecoder`).
    ///
    /// - Important: Mutually exclusive with both continuous batching and
    ///   speculative decoding in v1. The batch gate
    ///   (`BatchRoutingPolicy.shouldAttemptBatch(hasResponseFormat:)`) routes a
    ///   constrained request to the single-stream path, and the engine disables
    ///   any resident draft model for it — a draft proposal cannot be guaranteed
    ///   to honor the constraint, so the two never combine. Default nil and
    ///   decoded via `decodeIfPresent`, so pre-Track-C serialized requests keep
    ///   decoding.
    public var responseFormat: ResponseFormat?

    public init(
        model: String,
        messages: [ChatMessage],
        systemPrompt: String? = nil,
        parameters: GenerationParameters = .init(),
        templateKwargs: [String: JSONValue]? = nil,
        tools: [JSONValue]? = nil,
        draftModelID: String? = nil,
        numDraftTokens: Int? = nil,
        responseFormat: ResponseFormat? = nil
    ) {
        self.model = model
        self.messages = messages
        self.systemPrompt = systemPrompt
        self.parameters = parameters
        self.templateKwargs = templateKwargs
        self.tools = tools
        self.draftModelID = draftModelID
        self.numDraftTokens = Self.clampNumDraftTokens(numDraftTokens)
        self.responseFormat = responseFormat
    }

    private enum CodingKeys: String, CodingKey {
        case model, messages, systemPrompt, parameters, templateKwargs, tools
        case draftModelID, numDraftTokens, responseFormat
    }

    /// Custom decoder (mirrors `ChatMessage.init(from:)`) so the `1...8`
    /// `numDraftTokens` clamp is enforced on EVERY decode path, not just the
    /// memberwise initialiser above — a raw `JSONDecoder().decode(GenerateRequest.self,…)`
    /// (e.g. a persisted/replayed request) must not bypass it. All fields
    /// besides `model`/`messages`/`parameters` are optional and decoded via
    /// `decodeIfPresent`, so pre-D1 serialized requests keep decoding.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.model = try c.decode(String.self, forKey: .model)
        self.messages = try c.decode([ChatMessage].self, forKey: .messages)
        self.systemPrompt = try c.decodeIfPresent(String.self, forKey: .systemPrompt)
        self.parameters = try c.decode(GenerationParameters.self, forKey: .parameters)
        self.templateKwargs = try c.decodeIfPresent([String: JSONValue].self, forKey: .templateKwargs)
        self.tools = try c.decodeIfPresent([JSONValue].self, forKey: .tools)
        self.draftModelID = try c.decodeIfPresent(String.self, forKey: .draftModelID)
        self.numDraftTokens = Self.clampNumDraftTokens(
            try c.decodeIfPresent(Int.self, forKey: .numDraftTokens)
        )
        self.responseFormat = try c.decodeIfPresent(ResponseFormat.self, forKey: .responseFormat)
    }

    /// Clamp to mlx-swift-lm's sane speculative round-size range. `nil`
    /// passes through unchanged (defers to the upstream default).
    private static func clampNumDraftTokens(_ value: Int?) -> Int? {
        guard let value else { return nil }
        return Swift.min(8, Swift.max(1, value))
    }

    /// Messages with the system prompt (if any) prepended.
    public var allMessages: [ChatMessage] {
        guard let systemPrompt, !systemPrompt.isEmpty else { return messages }
        return [ChatMessage(role: .system, content: systemPrompt)] + messages
    }
}
