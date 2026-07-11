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

    // MARK: Track E — API-compatibility extensions
    //
    // All optional and decoded via `decodeIfPresent` (see `init(from:)`), so
    // every pre-Track-E serialized `GenerationParameters` keeps decoding with
    // these defaulting to nil / disabled.

    /// Additive per-token logit bias (OpenAI `logit_bias`). Maps a token id to a
    /// bias in `[-100, 100]` added to that token's logit before sampling, matching
    /// mlx-lm's `logit_bias` processor. `nil` (and an empty map) ⇒ no bias.
    /// Composed BEFORE any structured-output constraint mask (bias first,
    /// constraint last — the constraint's last-applied invariant).
    public var logitBias: [Int: Float]?

    /// XTC (Exclude Top Choices) sampling probability (mlx-lm `xtc_probability`).
    /// With probability `p`, every token whose probability exceeds `xtcThreshold`
    /// is removed EXCEPT the least-probable such token, before the base sampler
    /// runs. `nil` or `0` ⇒ XTC disabled. Meaningful only at `temperature > 0`.
    public var xtcProbability: Float?

    /// XTC probability threshold (mlx-lm `xtc_threshold`). Tokens with probability
    /// strictly greater than this are candidates for exclusion. `nil` defers to
    /// mlx-lm's default (`0.1`) when `xtcProbability` is set; ignored otherwise.
    public var xtcThreshold: Float?

    /// Whether to emit per-token logprobs (OpenAI `logprobs`). When true the
    /// engine attaches each generated token's logprob (and up to `topLogprobs`
    /// alternatives) to the stream. Default false.
    public var logprobs: Bool

    /// Number of top-alternative logprobs per token (OpenAI `top_logprobs`,
    /// `0...10`). Only meaningful when `logprobs` is true. `nil` ⇒ 0 (the sampled
    /// token's logprob only).
    public var topLogprobs: Int?

    /// KV-cache quantization bit width (mlx-lm `kv_bits`). `nil` ⇒ no KV-cache
    /// quantization. Passed through to mlx-swift-lm's `GenerateParameters.kvBits`.
    public var kvBits: Int?

    /// KV-cache quantization group size (mlx-lm `kv_group_size`). `nil` defers to
    /// mlx-swift-lm's default (`64`). Meaningful only with `kvBits`.
    public var kvGroupSize: Int?

    /// Token offset at which to begin quantizing the KV cache (mlx-lm
    /// `quantized_kv_start`). `nil` defers to mlx-swift-lm's default (`0`).
    /// Meaningful only with `kvBits`.
    public var quantizedKVStart: Int?

    public init(
        temperature: Double = 0.7,
        topP: Double = 0.95,
        maxTokens: Int = 2048,
        stream: Bool = true,
        logitBias: [Int: Float]? = nil,
        xtcProbability: Float? = nil,
        xtcThreshold: Float? = nil,
        logprobs: Bool = false,
        topLogprobs: Int? = nil,
        kvBits: Int? = nil,
        kvGroupSize: Int? = nil,
        quantizedKVStart: Int? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.maxTokens = maxTokens
        self.stream = stream
        self.logitBias = Self.normalizeLogitBias(logitBias)
        self.xtcProbability = Self.clampXTCProbability(xtcProbability)
        self.xtcThreshold = Self.clampXTCThreshold(xtcThreshold)
        self.logprobs = logprobs
        self.topLogprobs = Self.clampTopLogprobs(topLogprobs)
        self.kvBits = Self.clampKVBits(kvBits)
        self.kvGroupSize = Self.clampKVGroupSize(kvGroupSize)
        self.quantizedKVStart = Self.clampQuantizedKVStart(quantizedKVStart)
    }

    private enum CodingKeys: String, CodingKey {
        case temperature, topP, maxTokens, stream
        case logitBias, xtcProbability, xtcThreshold, logprobs, topLogprobs
        case kvBits, kvGroupSize, quantizedKVStart
    }

    /// Custom decoder so every clamp/normalization runs on the raw-JSON decode
    /// path too (a persisted/replayed request must not bypass it), mirroring
    /// `GenerateRequest.init(from:)`. The four base fields decode as before; the
    /// Track-E additions are all optional so legacy JSON keeps decoding.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.temperature = try c.decode(Double.self, forKey: .temperature)
        self.topP = try c.decode(Double.self, forKey: .topP)
        self.maxTokens = try c.decode(Int.self, forKey: .maxTokens)
        self.stream = try c.decode(Bool.self, forKey: .stream)
        self.logitBias = Self.normalizeLogitBias(
            try c.decodeIfPresent([Int: Float].self, forKey: .logitBias))
        self.xtcProbability = Self.clampXTCProbability(
            try c.decodeIfPresent(Float.self, forKey: .xtcProbability))
        self.xtcThreshold = Self.clampXTCThreshold(
            try c.decodeIfPresent(Float.self, forKey: .xtcThreshold))
        self.logprobs = try c.decodeIfPresent(Bool.self, forKey: .logprobs) ?? false
        self.topLogprobs = Self.clampTopLogprobs(
            try c.decodeIfPresent(Int.self, forKey: .topLogprobs))
        self.kvBits = Self.clampKVBits(try c.decodeIfPresent(Int.self, forKey: .kvBits))
        self.kvGroupSize = Self.clampKVGroupSize(try c.decodeIfPresent(Int.self, forKey: .kvGroupSize))
        self.quantizedKVStart = Self.clampQuantizedKVStart(
            try c.decodeIfPresent(Int.self, forKey: .quantizedKVStart))
    }

    // MARK: Clamps (pure, unit-testable without Metal)

    /// Drop an empty map to nil and clamp each bias to OpenAI's `[-100, 100]`.
    static func normalizeLogitBias(_ value: [Int: Float]?) -> [Int: Float]? {
        guard let value, !value.isEmpty else { return nil }
        return value.mapValues { Swift.min(100, Swift.max(-100, $0)) }
    }

    /// Parse OpenAI's wire `logit_bias` — a map of STRING token ids to bias
    /// numbers (`{"50256": -100}`) — into the engine's `[Int: Float]`, then
    /// normalize (clamp + drop-empty). Unparseable keys are dropped. Public so
    /// the server can convert at the request boundary and tests can exercise the
    /// string-key parsing directly.
    public static func logitBias(fromOpenAI raw: [String: Double]?) -> [Int: Float]? {
        guard let raw, !raw.isEmpty else { return nil }
        var parsed: [Int: Float] = [:]
        for (key, value) in raw {
            if let id = Int(key) { parsed[id] = Float(value) }
        }
        return normalizeLogitBias(parsed)
    }

    /// Clamp XTC probability to `[0, 1]`; `0` (or nil) leaves XTC disabled.
    static func clampXTCProbability(_ value: Float?) -> Float? {
        guard let value else { return nil }
        return Swift.min(1, Swift.max(0, value))
    }

    /// Clamp XTC threshold to `[0, 0.5]` (mlx-lm's meaningful range — above 0.5 no
    /// two tokens can both exceed it, so XTC can never fire).
    static func clampXTCThreshold(_ value: Float?) -> Float? {
        guard let value else { return nil }
        return Swift.min(0.5, Swift.max(0, value))
    }

    /// Clamp `top_logprobs` to OpenAI's `0...10`.
    static func clampTopLogprobs(_ value: Int?) -> Int? {
        guard let value else { return nil }
        return Swift.min(10, Swift.max(0, value))
    }

    /// Snap `kv_bits` to the DISCRETE set MLX's affine quantizer supports
    /// ({2, 3, 4, 6, 8}) — it is a set, not a range. An in-between value
    /// (5, 7) would otherwise pass validation and fail mid-generation as a
    /// Metal/runtime error instead of being corrected at the boundary.
    ///
    /// `nil`, `0`, or a negative value all mean "no KV-cache quantization"
    /// (same semantics as nil). Snapping a non-positive value UP to the lowest
    /// supported width (2) would silently ENABLE the lossiest quantization the
    /// caller never asked for — and flip `bypassPromptCache` / the unbatchable
    /// routing that key off `kvBits != nil` — so it is dropped to nil instead.
    /// Only a strictly-positive value is snapped to the supported set.
    static func clampKVBits(_ value: Int?) -> Int? {
        guard let value, value > 0 else { return nil }
        let supported = [2, 3, 4, 6, 8]
        // Nearest supported width; ties resolve to the smaller (safer) width.
        return supported.min {
            (abs($0 - value), $0) < (abs($1 - value), $1)
        }
    }

    /// Snap `kv_group_size` to the DISCRETE set MLX quantization supports
    /// ({32, 64, 128}); nil defers to the upstream default of 64.
    static func clampKVGroupSize(_ value: Int?) -> Int? {
        guard let value else { return nil }
        let supported = [32, 64, 128]
        return supported.min {
            (abs($0 - value), $0) < (abs($1 - value), $1)
        }
    }

    /// Clamp `quantized_kv_start` to a non-negative offset.
    static func clampQuantizedKVStart(_ value: Int?) -> Int? {
        guard let value else { return nil }
        return Swift.max(0, value)
    }
}

/// Everything an inference engine needs to start a generation.
public struct GenerateRequest: Codable, Hashable, Sendable {
    public let model: String
    /// `var` (not `let`) so a caller that must vary only the conversation across
    /// otherwise-identical turns — the MCP tool loop (`ToolCallingSession`) — can
    /// mutate a COPY of the seed request instead of re-initialising it. Re-init
    /// silently drops every field not re-passed (draft model, response_format,
    /// adapters, …); mutating preserves them all by construction.
    public var messages: [ChatMessage]
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
    /// Per-request LoRA adapter to apply for this generation (Track E, mlx-lm
    /// `adapters`). A bare name resolved under `~/.mac-mlx/adapters/<name>` or an
    /// absolute path to an adapter directory. The engine reloads the model with
    /// the new (model, adapter) pairing only when it differs from what is resident
    /// — an unchanged pairing is a no-op, and `nil` unloads any resident adapter
    /// (reloads the base model), mirroring mlx-lm's server semantics. Default nil
    /// and decoded via `decodeIfPresent`, so pre-Track-E serialized requests keep
    /// decoding.
    ///
    /// - Important: Mutually exclusive with continuous batching in v1 (the batched
    ///   step evaluator runs the resident model without per-request adapter
    ///   swaps), so a request carrying `adapters` routes to the single-stream path.
    public var adapters: String?

    public init(
        model: String,
        messages: [ChatMessage],
        systemPrompt: String? = nil,
        parameters: GenerationParameters = .init(),
        templateKwargs: [String: JSONValue]? = nil,
        tools: [JSONValue]? = nil,
        draftModelID: String? = nil,
        numDraftTokens: Int? = nil,
        responseFormat: ResponseFormat? = nil,
        adapters: String? = nil
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
        self.adapters = Self.normalizeAdapters(adapters)
    }

    private enum CodingKeys: String, CodingKey {
        case model, messages, systemPrompt, parameters, templateKwargs, tools
        case draftModelID, numDraftTokens, responseFormat, adapters
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
        self.adapters = Self.normalizeAdapters(
            try c.decodeIfPresent(String.self, forKey: .adapters))
    }

    /// Trim an adapter identifier and collapse the empty string to nil (an empty
    /// `adapters` string means "no adapter", same as absent), so downstream
    /// combo/equality checks treat `""` and `nil` identically.
    static func normalizeAdapters(_ value: String?) -> String? {
        guard let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines),
              !trimmed.isEmpty
        else { return nil }
        return trimmed
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
