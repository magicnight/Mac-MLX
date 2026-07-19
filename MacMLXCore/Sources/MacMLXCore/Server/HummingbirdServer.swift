import Foundation
import HTTPTypes
import Hummingbird
import NIOCore
import NIOPosix
import ServiceLifecycle

// MARK: - OpenAI-compatible request/response types

/// OpenAI-compatible chat completion request body.
///
/// `Message.content` accepts either a plain string (text-only chat) or
/// an OpenAI multimodal content array of `{type, text|image_url}` parts
/// (v0.4.1+ — VLM models can read images this way). The decoder tries
/// string first, falls back to `[Part]`. See `MultimodalContent` below.
private struct ChatCompletionRequest: Decodable, Sendable {
    struct Message: Decodable, Sendable {
        let role: String
        /// Optional so an assistant tool-call turn (`{"content":null,
        /// "tool_calls":[…]}`) and any content-less turn still decode — the
        /// synthesised `decodeIfPresent` maps both an absent key AND JSON `null`
        /// to nil. A plain text/tool turn keeps its string as before.
        let content: MultimodalContent?
        /// Assistant `tool_calls` history: the calls a prior assistant turn
        /// issued, replayed by an agent client across rounds. Decoded into
        /// `ChatMessage.toolCalls` so the chat template reproduces the tool-call
        /// block and its pairing with the following `tool` results is satisfied.
        /// Optional + `decodeIfPresent` ⇒ every pre-tools client is unchanged.
        let tool_calls: [OpenAIToolCall]?
        /// For a `role:"tool"` turn: the id of the assistant call this result
        /// answers (OpenAI `tool_call_id`). Carried onto the `.tool`
        /// `ChatMessage` so the chat template pairs result ↔ call.
        let tool_call_id: String?
    }

    let model: String
    let messages: [Message]
    let stream: Bool?
    let temperature: Double?
    let top_p: Double?
    let max_tokens: Int?
    /// OpenAI `tools` — an array of `{"type":"function","function":{name,
    /// description,parameters}}` specs. Forwarded VERBATIM into
    /// `GenerateRequest.tools` (the chat template consumes exactly this shape,
    /// so no conversion happens here). Optional + synthesised `decodeIfPresent`
    /// ⇒ absent key decodes as nil, so every pre-tools client is unchanged.
    let tools: [JSONValue]?
    /// OpenAI `tool_choice`. Only the string `"none"` is honored here — it
    /// suppresses the tools so the model can't call any. Every other value
    /// (the strings `"auto"`/`"required"`, or a `{"type":"function",...}`
    /// object naming a specific function) currently means "offer the tools and
    /// let the model decide"; fine-grained forcing is a documented follow-up.
    /// Decoded as a free-form `JSONValue` so both the string and object forms
    /// round-trip without a bespoke enum.
    let tool_choice: JSONValue?
    /// Draft model id for classic per-request speculative decoding (D1).
    /// NOT an OpenAI-standard field — mirrors mlx-lm's Python server
    /// `draft_model` request field. Absent/`null` disables speculative
    /// decoding and unloads any draft model the engine currently has
    /// resident (see `GenerateRequest.draftModelID`).
    let draft_model: String?
    /// Draft tokens proposed per speculative round — NOT an OpenAI-standard
    /// field, mirrors mlx-lm's `num_draft_tokens`. Clamped to 1...8 by
    /// `GenerateRequest`; meaningless without `draft_model`.
    let num_draft_tokens: Int?
    /// OpenAI `response_format` (Track C structured output). Decoded as a
    /// free-form `JSONValue` and validated by `ResponseFormatDecoder`, which
    /// maps `{"type":"json_object"}` / `{"type":"json_schema",…}` to a
    /// `ResponseFormat` or rejects unsupported schema features with a 400.
    /// Absent/`null` (and `{"type":"text"}`) mean no constraint. Optional +
    /// synthesised `decodeIfPresent` ⇒ every pre-Track-C client is unchanged.
    let response_format: JSONValue?
    // MARK: Track E — API-compatibility extensions (all optional / back-compat)
    /// OpenAI `logit_bias`: STRING token id → additive bias in `[-100, 100]`.
    let logit_bias: [String: Double]?
    /// OpenAI `logprobs`: emit per-token logprobs when true.
    let logprobs: Bool?
    /// OpenAI `top_logprobs`: alternatives per token, `0...10`.
    let top_logprobs: Int?
    /// mlx-lm `xtc_probability` — per-step chance of applying XTC.
    let xtc_probability: Double?
    /// mlx-lm `xtc_threshold` — probability above which a token is an XTC candidate.
    let xtc_threshold: Double?
    /// mlx-lm `kv_bits` — KV-cache quantization bit width (nil ⇒ off).
    let kv_bits: Int?
    /// mlx-lm `kv_group_size` — KV-cache quantization group size.
    let kv_group_size: Int?
    /// mlx-lm `quantized_kv_start` — token offset to begin quantizing.
    let quantized_kv_start: Int?
    /// mlx-lm `adapters` — LoRA adapter name/path to hot-swap for this request
    /// (nil/absent ⇒ base model; an explicit change reloads the model with the
    /// new adapter). See `GenerateRequest.adapters`.
    let adapters: String?
}

/// One OpenAI `tool_calls[]` entry on a replayed assistant turn:
/// `{"id","type":"function","function":{"name","arguments":<JSON string>}}`.
/// `function.arguments` is a JSON-ENCODED STRING per the OpenAI wire format
/// (the same shape the server EMITS via `openAIToolCallObject`) — it is parsed
/// back into an object when building the internal `ToolCallRequest`. `id` is
/// optional for leniency (synthesised if a client omits it); `type` is decoded
/// but unused (only `"function"` tools exist today).
private struct OpenAIToolCall: Decodable, Sendable {
    let id: String?
    let type: String?
    let function: Function

    struct Function: Decodable, Sendable {
        let name: String
        /// Optional so a no-argument call (`arguments` absent) decodes; an empty
        /// or absent string means "no arguments" (`[:]`).
        let arguments: String?
    }
}

/// Legacy OpenAI text-completions body (`/v1/completions`, and the
/// `POST /` / `POST /v1` aliases): `{"model","prompt",...}` with a bare
/// `prompt` string and NO `messages` array. Decoded as a fallback when a
/// body fails to parse as `ChatCompletionRequest`, then wrapped into a
/// single user turn so these routes serve the same chat-format response
/// (the full text-completions response shape is out of scope). Honors
/// `max_tokens` / `temperature` / `top_p` / `stream` when present.
private struct LegacyCompletionRequest: Decodable, Sendable {
    let model: String
    let prompt: String
    let stream: Bool?
    let temperature: Double?
    let top_p: Double?
    let max_tokens: Int?
    /// See `ChatCompletionRequest.draft_model` — same non-standard D1 field,
    /// honored on this legacy route too since it shares `handleChatCompletions`.
    let draft_model: String?
    /// See `ChatCompletionRequest.num_draft_tokens`.
    let num_draft_tokens: Int?
    // Track E — honored on the legacy route too (shares `handleChatCompletions`).
    let logit_bias: [String: Double]?
    let logprobs: Bool?
    let top_logprobs: Int?
    let xtc_probability: Double?
    let xtc_threshold: Double?
    let kv_bits: Int?
    let kv_group_size: Int?
    let quantized_kv_start: Int?
    let adapters: String?
}

/// OpenAI multimodal content payload. Either a plain string (text-only
/// — backwards compat with every existing client) or an array of typed
/// parts (`text` and `image_url`). The decoder tries the string form
/// first; on failure it falls through to an array of parts so we don't
/// reject older clients that send a bare string.
private enum MultimodalContent: Decodable, Sendable {
    case string(String)
    case parts([Part])

    struct Part: Decodable, Sendable {
        let type: String                  // "text" or "image_url"
        let text: String?
        let image_url: ImageURL?
    }
    struct ImageURL: Decodable, Sendable {
        /// Either a `data:image/...;base64,XXXX` URL (only form we
        /// currently decode — see `extractImages()`) or `http(s)://`.
        /// `file://` is rejected by `extractImages()` for defence-in-
        /// depth even though the server is localhost-bound.
        let url: String
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let s = try? container.decode(String.self) {
            self = .string(s)
            return
        }
        let parts = try container.decode([Part].self)
        self = .parts(parts)
    }

    /// Concatenated text view — what the model sees as the prompt
    /// content for this turn. Image parts are ignored (their bytes
    /// flow into the engine separately via `extractImages()`).
    var text: String {
        switch self {
        case .string(let s):
            return s
        case .parts(let parts):
            return parts.compactMap { $0.type == "text" ? $0.text : nil }
                .joined(separator: "\n")
        }
    }

    /// Decode any base64 data URLs into `ImageAttachment` values backed
    /// by tmpfile copies. Caps:
    /// - 4 images per call (further parts silently dropped)
    /// - 10 MB per image (oversized parts silently dropped)
    /// - data URL only — `http(s)://` and `file://` are not fetched
    func extractImages() -> [ImageAttachment] {
        guard case .parts(let parts) = self else { return [] }
        var out: [ImageAttachment] = []
        for part in parts {
            guard part.type == "image_url",
                  let urlStr = part.image_url?.url,
                  let attachment = MultimodalContent.decodeDataURL(urlStr)
            else { continue }
            out.append(attachment)
            if out.count >= 4 { break }
        }
        return out
    }

    /// Best-effort base64 data-URL → on-disk image. Returns nil on
    /// any malformed input or unsupported MIME so callers can simply
    /// drop the part.
    private static func decodeDataURL(_ urlStr: String) -> ImageAttachment? {
        guard urlStr.hasPrefix("data:") else { return nil }
        let body = urlStr.dropFirst("data:".count)
        let split = body.split(separator: ",", maxSplits: 1, omittingEmptySubsequences: false)
        guard split.count == 2 else { return nil }
        let header = String(split[0])  // e.g. "image/png;base64"
        let payload = String(split[1])
        guard header.hasSuffix(";base64") else { return nil }
        let mime = String(header.dropLast(";base64".count))

        let ext: String
        switch mime.lowercased() {
        case "image/jpeg", "image/jpg": ext = "jpg"
        case "image/png":               ext = "png"
        case "image/webp":              ext = "webp"
        case "image/gif":               ext = "gif"
        case "image/heic":              ext = "heic"
        case "image/bmp":               ext = "bmp"
        default:                        return nil
        }

        guard let bytes = Data(base64Encoded: payload, options: .ignoreUnknownCharacters) else {
            return nil
        }
        // 10 MB per image cap.
        if bytes.count > 10 * 1024 * 1024 { return nil }

        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("macmlx-http-img-\(UUID().uuidString).\(ext)")
        do {
            try bytes.write(to: tmp)
            return ImageAttachment(fileURL: tmp, mimeType: mime)
        } catch {
            return nil
        }
    }
}

/// Request body for `/x/models/load`.
private struct LoadModelRequest: Decodable, Sendable {
    let model_path: String
}

// MARK: - Embeddings / rerank request types (v0.5.2)

/// OpenAI `/v1/embeddings` `input` field. Either a single string or an
/// array of strings. The decoder tries string first, then falls through
/// to `[String]` — mirroring `MultimodalContent`'s string-or-array shape.
private enum EmbeddingsInput: Decodable, Sendable {
    case string(String)
    case array([String])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let s = try? container.decode(String.self) {
            self = .string(s)
            return
        }
        self = .array(try container.decode([String].self))
    }

    /// Flattened list of texts to embed, in request order.
    var values: [String] {
        switch self {
        case .string(let s): return [s]
        case .array(let a): return a
        }
    }
}

/// OpenAI-compatible `/v1/embeddings` request body.
private struct EmbeddingsRequest: Decodable, Sendable {
    let model: String
    let input: EmbeddingsInput
    let encoding_format: String?
}

/// `/v1/rerank` request body (Cohere/Jina-style shape). Scored by a TRUE
/// cross-encoder when `model` is a `.reranker`, or the bi-encoder cosine
/// fallback when it's an `.embedder` — see `handleRerank`.
private struct RerankRequest: Decodable, Sendable {
    let model: String
    let query: String
    let documents: [String]
    let top_n: Int?
    /// When `true`, each result echoes the original `document` text alongside
    /// its `index` (Cohere/Jina parity). Optional; defaults to omitting it.
    let return_documents: Bool?
}

/// Ollama-compatible chat request body. Simpler than OpenAI's —
/// no `stream_options`, system message expressed the same way.
private struct OllamaChatRequest: Decodable, Sendable {
    struct Message: Decodable, Sendable {
        let role: String
        let content: String
    }
    struct Options: Decodable, Sendable {
        let temperature: Double?
        let top_p: Double?
        let num_predict: Int?  // Ollama's name for max_tokens
    }
    let model: String
    let messages: [Message]
    let stream: Bool?
    let options: Options?
}

/// Ollama-compatible /api/generate body (single-prompt completion).
private struct OllamaGenerateRequest: Decodable, Sendable {
    struct Options: Decodable, Sendable {
        let temperature: Double?
        let top_p: Double?
        let num_predict: Int?
    }
    let model: String
    let prompt: String
    let system: String?
    let stream: Bool?
    let options: Options?
}

// MARK: - Anthropic-compatible request/response types

/// Anthropic Messages API request body (`POST /v1/messages`).
///
/// Mirrors the subset of the Anthropic wire format macMLX supports:
/// a top-level `system` prompt (string or `[{type:"text", text}]`),
/// a `messages` array whose `content` is either a bare string or an
/// array of typed blocks (`text` / `image`), and the usual sampling
/// knobs. Unlike OpenAI, `max_tokens` is required by Anthropic's spec.
private struct AnthropicMessagesRequest: Decodable, Sendable {
    let model: String
    let max_tokens: Int
    let messages: [AnthropicMessage]
    let system: AnthropicSystem?
    let temperature: Double?
    let top_p: Double?
    let stream: Bool?
    /// Anthropic `tools` — `[{name, description?, input_schema}]`. Converted to
    /// the same internal OpenAI-function-spec `[JSONValue]` the OpenAI path
    /// forwards (`input_schema` ≙ OpenAI `function.parameters`) so both protocols
    /// feed the chat template identically. Optional + `decodeIfPresent` ⇒ every
    /// pre-tools client is unchanged.
    let tools: [AnthropicTool]?
    /// Anthropic `tool_choice`. Only `{type:"auto"}` (or an absent field) is
    /// honoured; `any`/`tool`/`none` are rejected with a 400 rather than
    /// silently altered (project rule: no silent degradation).
    let tool_choice: AnthropicToolChoice?
}

/// One Anthropic `tools[]` entry: `{name, description?, input_schema}`.
/// `input_schema` is the JSON-Schema object describing the tool's arguments —
/// Anthropic's analogue of OpenAI's `function.parameters`.
private struct AnthropicTool: Decodable, Sendable {
    let name: String
    let description: String?
    let input_schema: JSONValue?
}

/// Anthropic `tool_choice` object: `{type:"auto"|"any"|"tool"|"none", name?}`.
/// Decoded so the handler can reject the unsupported modes explicitly.
private struct AnthropicToolChoice: Decodable, Sendable {
    let type: String
    let name: String?
}

/// Anthropic top-level `system` field. Either a plain string or an
/// array of `{type:"text", text}` blocks (the SDK's structured form).
/// `text` flattens both into the single system-prompt string macMLX's
/// engine expects.
private enum AnthropicSystem: Decodable, Sendable {
    case string(String)
    case blocks([AnthropicBlock])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let s = try? container.decode(String.self) {
            self = .string(s)
            return
        }
        self = .blocks(try container.decode([AnthropicBlock].self))
    }

    /// Concatenated text of the system blocks (or the bare string).
    var text: String {
        switch self {
        case .string(let s):
            return s
        case .blocks(let blocks):
            return blocks.compactMap { $0.type == "text" ? $0.text : nil }
                .joined(separator: "\n")
        }
    }
}

/// One Anthropic message turn. `role` is "user" or "assistant";
/// `content` is a string or an array of typed content blocks.
private struct AnthropicMessage: Decodable, Sendable {
    let role: String
    let content: AnthropicContent
}

/// Anthropic message `content`. Either a bare string (text-only turn)
/// or an array of typed blocks (`text` and `image`). The decoder tries
/// the string form first, falling through to `[AnthropicBlock]`.
private enum AnthropicContent: Decodable, Sendable {
    case string(String)
    case blocks([AnthropicBlock])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let s = try? container.decode(String.self) {
            self = .string(s)
            return
        }
        self = .blocks(try container.decode([AnthropicBlock].self))
    }

    /// Concatenated text view — what the model sees as the prompt
    /// content for this turn. Image blocks are ignored (their bytes
    /// flow into the engine separately via `extractImages()`).
    var text: String {
        switch self {
        case .string(let s):
            return s
        case .blocks(let blocks):
            return blocks.compactMap { $0.type == "text" ? $0.text : nil }
                .joined(separator: "\n")
        }
    }

    /// The typed content blocks, or nil for the bare-string form — so the
    /// tool-turn decoder can inspect `tool_use` / `tool_result` blocks that the
    /// text-flattening `text` view discards.
    var blocks: [AnthropicBlock]? {
        if case .blocks(let b) = self { return b }
        return nil
    }

    /// Decode any base64 image blocks into `ImageAttachment` values
    /// backed by tmpfile copies. Caps mirror the OpenAI path:
    /// - 4 images per call (further blocks silently dropped)
    /// - 10 MB per image (oversized blocks silently dropped)
    ///
    /// Unlike OpenAI's data-URL form, Anthropic supplies `media_type` +
    /// raw base64 `data` directly, so we decode the bytes straight from
    /// `source.data` without any data-URL parsing.
    func extractImages() -> [ImageAttachment] {
        guard case .blocks(let blocks) = self else { return [] }
        var out: [ImageAttachment] = []
        for block in blocks {
            guard block.type == "image",
                  let source = block.source,
                  let attachment = AnthropicContent.decodeImageSource(source)
            else { continue }
            out.append(attachment)
            if out.count >= 4 { break }
        }
        return out
    }

    /// Best-effort base64 image block → on-disk image. Returns nil on
    /// malformed input or unsupported media type so callers can simply
    /// drop the block.
    private static func decodeImageSource(_ source: AnthropicImageSource) -> ImageAttachment? {
        let mime = source.media_type
        let ext: String
        switch mime.lowercased() {
        case "image/jpeg", "image/jpg": ext = "jpg"
        case "image/png":               ext = "png"
        case "image/webp":              ext = "webp"
        case "image/gif":               ext = "gif"
        case "image/heic":              ext = "heic"
        case "image/bmp":               ext = "bmp"
        default:                        return nil
        }

        guard let bytes = Data(base64Encoded: source.data, options: .ignoreUnknownCharacters) else {
            return nil
        }
        // 10 MB per image cap.
        if bytes.count > 10 * 1024 * 1024 { return nil }

        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("macmlx-http-img-\(UUID().uuidString).\(ext)")
        do {
            try bytes.write(to: tmp)
            return ImageAttachment(fileURL: tmp, mimeType: mime)
        } catch {
            return nil
        }
    }
}

/// One Anthropic content block. `text` is set for `type:"text"`; `source` for
/// `type:"image"`; the `tool_use` fields (`id`/`name`/`input`) for an assistant
/// `type:"tool_use"` block; the `tool_result` fields (`tool_use_id`/`content`/
/// `is_error`) for a user `type:"tool_result"` block. Every tool field is
/// optional so the existing text/image decode paths are unaffected.
private struct AnthropicBlock: Decodable, Sendable {
    let type: String            // "text" | "image" | "tool_use" | "tool_result"
    let text: String?
    let source: AnthropicImageSource?
    // tool_use (assistant turn): the call the model issued.
    let id: String?
    let name: String?
    let input: JSONValue?
    // tool_result (user turn): the answer to a prior tool_use.
    let tool_use_id: String?
    let content: AnthropicToolResultContent?
    let is_error: Bool?
}

/// Anthropic `tool_result.content`: either a bare string or an array of content
/// blocks (only `text` blocks are meaningful to a text model). `text` flattens
/// both into the single string fed back to the model as the tool result. The
/// decoder tries the string form first, falling through to `[AnthropicBlock]`.
private enum AnthropicToolResultContent: Decodable, Sendable {
    case string(String)
    case blocks([AnthropicBlock])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let s = try? container.decode(String.self) {
            self = .string(s)
            return
        }
        self = .blocks(try container.decode([AnthropicBlock].self))
    }

    /// Concatenated text of the result's text blocks (or the bare string).
    var text: String {
        switch self {
        case .string(let s):
            return s
        case .blocks(let blocks):
            return blocks.compactMap { $0.type == "text" ? $0.text : nil }
                .joined(separator: "\n")
        }
    }
}

/// Anthropic image block `source`. Only the base64 form is decoded
/// (`type:"base64"`); `media_type` is the IANA MIME and `data` is the
/// raw base64 payload (no data-URL prefix).
private struct AnthropicImageSource: Decodable, Sendable {
    let type: String            // "base64"
    let media_type: String
    let data: String
}

// MARK: - Tool-history request decoding (agent-tools wave)
//
// Both protocol surfaces replay tool history across rounds: OpenAI as
// `role:"tool"` turns + assistant `tool_calls`, Anthropic as `tool_result` /
// `tool_use` content blocks. These free functions decode either wire shape into
// the SAME internal `[ChatMessage]` the engine already knows how to render
// (`MLXSwiftEngine.upstreamChatMessage` maps `.tool` → an upstream tool turn and
// an assistant's `toolCalls` → its tool-call block), mirroring the history that
// `MCP/ToolCallingSession` builds for the GUI loop. A malformed tool block is
// surfaced as a `ToolHistoryDecodeError` (→ HTTP 400) — never silently dropped.

/// A tool-history block that could not be decoded into a well-formed internal
/// message. The handler maps it to an HTTP 400 (project rule: no silent
/// degradation — a half-formed tool turn would otherwise trip a chat-template
/// pairing assertion downstream).
private struct ToolHistoryDecodeError: Error {
    let message: String
}

/// Parse an OpenAI `function.arguments` JSON string into the internal arguments
/// object. An empty or absent string means "no arguments" (`[:]`); a non-empty
/// string that is not a JSON object is malformed (throws → 400).
private func decodeToolArgumentsJSON(_ raw: String?, tool: String) throws -> [String: JSONValue] {
    guard let trimmed = raw?.trimmingCharacters(in: .whitespacesAndNewlines), !trimmed.isEmpty else {
        return [:]
    }
    guard let data = trimmed.data(using: .utf8),
          let value = try? JSONDecoder().decode(JSONValue.self, from: data),
          case .object(let object) = value
    else {
        throw ToolHistoryDecodeError(
            message: "Malformed `arguments` for tool '\(tool)': expected a JSON object string."
        )
    }
    return object
}

/// Convert one decoded OpenAI `tool_calls[]` entry into a `ToolCallRequest`.
/// A missing `id` is synthesised (mirrors the engine's own `call_<uuid>`
/// fallback) so downstream pairing still has a stable correlator.
private func toolCallRequest(fromOpenAI call: OpenAIToolCall) throws -> ToolCallRequest {
    let name = call.function.name
    guard !name.isEmpty else {
        throw ToolHistoryDecodeError(message: "A `tool_calls` entry is missing its function name.")
    }
    let arguments = try decodeToolArgumentsJSON(call.function.arguments, tool: name)
    let id = call.id ?? "call_\(UUID().uuidString)"
    return ToolCallRequest(id: id, name: name, arguments: arguments)
}

/// Validate call ↔ result PAIRING across an already-decoded conversation, on
/// BOTH protocol surfaces (called from `decodeOpenAIMessages` and
/// `decodeAnthropicMessages`). Field-presence alone (every `.tool` message
/// carries a non-empty `toolCallID`) is not sufficient — this additionally
/// rejects, in one linear pass over the decoded `[ChatMessage]`:
///   - a `.tool` message whose `toolCallID` was never issued by a preceding
///     assistant `toolCalls` entry (an orphan result), including one whose
///     id has already been answered by an earlier `.tool` message (a second
///     result for the same call);
///   - an assistant `toolCalls` entry whose `id` was already issued earlier
///     in the SAME history (a duplicate call id, anywhere — not just within
///     one turn).
/// Every violation throws `ToolHistoryDecodeError` (→ HTTP 400) rather than
/// reaching the chat template half-formed.
private func validateToolCallPairing(_ messages: [ChatMessage]) throws {
    var everIssuedCallIDs: Set<String> = []
    var awaitingResultCallIDs: Set<String> = []
    for message in messages {
        if let toolCalls = message.toolCalls {
            for call in toolCalls {
                guard !everIssuedCallIDs.contains(call.id) else {
                    throw ToolHistoryDecodeError(
                        message: "Duplicate tool call id '\(call.id)': an assistant tool_calls "
                            + "entry must not reuse an id already issued earlier in the conversation."
                    )
                }
                everIssuedCallIDs.insert(call.id)
                awaitingResultCallIDs.insert(call.id)
            }
        }
        guard message.role == .tool else { continue }
        // The per-surface decoders (`decodeOpenAIMessages` / `toolResultMessage`)
        // already reject a missing/empty id before a `.tool` message is ever
        // constructed, so `toolCallID` is non-nil/non-empty here — but guard
        // defensively rather than assume.
        let id = message.toolCallID ?? ""
        guard awaitingResultCallIDs.contains(id) else {
            if everIssuedCallIDs.contains(id) {
                throw ToolHistoryDecodeError(
                    message: "Tool result for id '\(id)' answers a call that already has a result."
                )
            }
            throw ToolHistoryDecodeError(
                message: "Tool result references unknown tool_call id '\(id)'."
            )
        }
        awaitingResultCallIDs.remove(id)
    }
}

/// Decode OpenAI chat messages into the internal `[ChatMessage]`, fully wiring
/// and validating tool history. System turns are dropped here (re-prepended
/// via `GenerateRequest.systemPrompt`); unknown roles are dropped. A
/// `role:"tool"` turn requires a non-empty `tool_call_id` — a missing/empty
/// one throws rather than being admitted half-formed (`toolCallID: nil`),
/// symmetric with the Anthropic `tool_result` guard. An assistant turn's
/// `tool_calls` → `ChatMessage.toolCalls`. After per-message decode,
/// `validateToolCallPairing` rejects an orphaned/duplicate-answered tool
/// result or a duplicate assistant call id anywhere in the history.
private func decodeOpenAIMessages(_ messages: [ChatCompletionRequest.Message]) throws -> [ChatMessage] {
    let decoded = try messages.compactMap { msg -> ChatMessage? in
        guard let role = MessageRole(rawValue: msg.role), role != .system else {
            return nil
        }
        // Tool-result turn: the correlating id is required. Admitting one
        // without it (`toolCallID: nil`) is exactly the half-formed shape the
        // original pre-wave-2 guard existed to avoid, so this throws instead.
        if role == .tool {
            guard let toolCallID = msg.tool_call_id, !toolCallID.isEmpty else {
                throw ToolHistoryDecodeError(
                    message: "A `tool` message is missing its `tool_call_id`."
                )
            }
            return ChatMessage(
                role: .tool,
                content: msg.content?.text ?? "",
                toolCallID: toolCallID
            )
        }
        // Assistant turn that issued tool calls: decode them so the template
        // reproduces the tool-call block and the pairing assertion is satisfied.
        // Only the ASSISTANT role carries tool calls (OpenAI semantics). A user
        // message's `tool_calls` is ignored — decoding it would let a user turn
        // satisfy the role-blind `validateToolCallPairing` for a following tool
        // result, while the engine forwards only assistant tool calls to the
        // template (an opaque 500 / wrong prompt). Ignoring it lets the orphan-
        // result validation below 400 a mispaired history cleanly.
        let toolCalls = role == .assistant
            ? try msg.tool_calls?.map { try toolCallRequest(fromOpenAI: $0) }
            : nil
        return ChatMessage(
            role: role,
            content: msg.content?.text ?? "",
            images: msg.content?.extractImages() ?? [],
            toolCalls: (toolCalls?.isEmpty == false) ? toolCalls : nil
        )
    }
    try validateToolCallPairing(decoded)
    return decoded
}

/// The OpenAI function-spec `JSONValue` the chat template consumes
/// (`GenerateRequest.tools`), built from an Anthropic tool. `input_schema` maps
/// to OpenAI `function.parameters`. Mirrors `ToolValueBridge.openAIToolSpec`.
private func openAIToolSpec(fromAnthropic tool: AnthropicTool) -> JSONValue {
    .object([
        "type": .string("function"),
        "function": .object([
            "name": .string(tool.name),
            "description": .string(tool.description ?? ""),
            "parameters": tool.input_schema ?? .object([:]),
        ]),
    ])
}

/// Convert one Anthropic `tool_use` block into a `ToolCallRequest`. Both `id`
/// and `name` are required (the following `tool_result` pairs on `id`, so a
/// synthesised id would mispair); `input` is an arbitrary JSON object, absent ⇒
/// no-arg call, a present non-object value is malformed.
private func toolCallRequest(fromAnthropicUse block: AnthropicBlock) throws -> ToolCallRequest {
    guard let name = block.name, !name.isEmpty else {
        throw ToolHistoryDecodeError(message: "A `tool_use` block is missing its `name`.")
    }
    guard let id = block.id, !id.isEmpty else {
        throw ToolHistoryDecodeError(message: "The `tool_use` block for '\(name)' is missing its `id`.")
    }
    let arguments: [String: JSONValue]
    switch block.input {
    case .none, .some(.null):
        arguments = [:]
    case .some(.object(let object)):
        arguments = object
    case .some:
        throw ToolHistoryDecodeError(
            message: "The `input` of `tool_use` '\(name)' must be a JSON object."
        )
    }
    return ToolCallRequest(id: id, name: name, arguments: arguments)
}

/// Assistant turn (block form) → one `.assistant` message whose text is the
/// joined text blocks and whose `toolCalls` are the decoded `tool_use` blocks.
private func assistantMessage(
    fromAnthropic blocks: [AnthropicBlock],
    images: [ImageAttachment]
) throws -> ChatMessage {
    let text = blocks.compactMap { $0.type == "text" ? $0.text : nil }.joined(separator: "\n")
    let toolCalls = try blocks.compactMap { block -> ToolCallRequest? in
        guard block.type == "tool_use" else { return nil }
        return try toolCallRequest(fromAnthropicUse: block)
    }
    return ChatMessage(
        role: .assistant,
        content: text,
        images: images,
        toolCalls: toolCalls.isEmpty ? nil : toolCalls
    )
}

/// One Anthropic `tool_result` block → a `.tool` message. `tool_use_id` is
/// required (it is the correlator the template pairs on). `is_error` is decoded
/// so the block parses, but a text model has no dedicated error slot — the
/// result text (which the client already frames as an error when it sets the
/// flag) is fed back verbatim.
private func toolResultMessage(fromAnthropic block: AnthropicBlock) throws -> ChatMessage {
    guard let id = block.tool_use_id, !id.isEmpty else {
        throw ToolHistoryDecodeError(message: "A `tool_result` block is missing its `tool_use_id`.")
    }
    return ChatMessage(role: .tool, content: block.content?.text ?? "", toolCallID: id)
}

/// User turn (block form) → tool-result messages FIRST (so the result ↔ call
/// pairing the chat template asserts holds regardless of block order in the
/// turn), then one `.user` message for any remaining text/image content. A turn
/// mixing `tool_result` and text loses neither.
private func userMessages(
    fromAnthropic blocks: [AnthropicBlock],
    images: [ImageAttachment]
) throws -> [ChatMessage] {
    var out: [ChatMessage] = []
    for block in blocks where block.type == "tool_result" {
        out.append(try toolResultMessage(fromAnthropic: block))
    }
    let text = blocks.compactMap { $0.type == "text" ? $0.text : nil }.joined(separator: "\n")
    if !text.isEmpty || !images.isEmpty {
        out.append(ChatMessage(role: .user, content: text, images: images))
    }
    return out
}

/// Decode Anthropic message turns into internal `[ChatMessage]`, fully wiring
/// and validating tool blocks. Because one Anthropic turn can carry several
/// blocks, a turn may expand to multiple internal messages (see
/// `assistantMessage` / `userMessages`). Bare-string content is a plain text
/// turn. Unknown roles are dropped; a malformed tool block throws (→ 400).
/// After per-message decode, `validateToolCallPairing` rejects an
/// orphaned/duplicate-answered tool result or a duplicate `tool_use` id
/// anywhere in the history.
private func decodeAnthropicMessages(_ messages: [AnthropicMessage]) throws -> [ChatMessage] {
    var out: [ChatMessage] = []
    for msg in messages {
        guard let role = MessageRole(rawValue: msg.role), role != .system else {
            continue
        }
        guard let blocks = msg.content.blocks else {
            // Bare-string content — a plain text turn (never a tool turn).
            out.append(ChatMessage(role: role, content: msg.content.text))
            continue
        }
        if role == .assistant {
            out.append(try assistantMessage(fromAnthropic: blocks, images: msg.content.extractImages()))
        } else {
            out.append(
                contentsOf: try userMessages(fromAnthropic: blocks, images: msg.content.extractImages()))
        }
    }
    try validateToolCallPairing(out)
    return out
}

// MARK: - HummingbirdServer

/// OpenAI-compatible HTTP server backed by an `InferenceEngine`.
///
/// Routes:
///   GET  /health
///   GET  /v1/models
///   GET  /x/status
///   POST /v1/chat/completions  (non-streaming + SSE streaming)
///   POST /x/models/load
///   POST /x/models/unload
///
/// Cold-swap (v0.3.3): when a `/v1/chat/completions` request names a
/// model that isn't the currently-loaded one, the server resolves it
/// via a caller-supplied `ModelResolver` closure, unloads the current
/// model, loads the requested one, and proceeds. Concurrent requests
/// for the same model share the load; concurrent requests for
/// different models serialise on an actor-local in-flight-load Task.
public actor HummingbirdServer {
    // MARK: Types

    /// Async lookup from an OpenAI-style model ID (the exact string the
    /// caller put in the request's `model` field) to a `LocalModel` on
    /// disk that the server can load. Returns `nil` if the ID isn't in
    /// the user's model directory. The server maps that to an HTTP 404
    /// with OpenAI's `model_not_found` code.
    public typealias ModelResolver = @Sendable (String) async -> LocalModel?

    /// Optional hook the caller can install to perform the actual
    /// load through a layer above the raw engine (e.g. GUI's
    /// `EngineCoordinator`, which also maintains observable state
    /// for the menu bar / toolbar / parameters inspector). When nil,
    /// cold-swap goes straight to `engine.unload()` + `engine.load()`.
    public typealias LoadHook = @Sendable (LocalModel) async throws -> Void

    // MARK: State

    /// Re-resolved per request (SRV-1). MacMLXCore never imports app-target
    /// types, so the *current* engine is fetched through this closure rather
    /// than captured once: the CLI passes `{ engine }` (a single engine it
    /// mutates in place), while the GUI passes `{ await coordinator.activeEngine }`
    /// — its `ModelPool` mints a NEW `MLXSwiftEngine` per model, so a captured
    /// reference would answer from a stale (or evicted) model after a cold-swap.
    private let engineProvider: @Sendable () async -> (any InferenceEngine)?

    /// Caller-supplied resolver for cold-swap. Defaults to returning nil
    /// (i.e. cold-swap disabled — server behaves as pre-v0.3.3: only
    /// the explicitly-loaded model can answer). The CLI's
    /// `ServeCommand` wires this up against `ModelLibraryManager.scan`.
    private let modelResolver: ModelResolver

    /// Optional hook used by cold-swap instead of raw engine calls.
    /// GUI sets this to route through `EngineCoordinator.load(_:)` so
    /// `currentModel`, `status`, and `onModelLoaded` stay in sync.
    private let loadHook: LoadHook?

    /// Bearer token gate for the HTTP surface. `nil` (default) leaves the
    /// localhost server open — the dev default. When set, every `/v1/*`
    /// and `/api/*` request must carry `Authorization: Bearer <key>`;
    /// only the `/health` + `/v1/health` probes stay open.
    private let apiKey: String?

    /// Optional hook the server calls around each generation to mark the
    /// active model in-flight one layer up (the GUI's `ModelPool`), so a
    /// concurrent load can't LRU-evict a model mid-stream (POOL-3). Injected
    /// as a closure — like `loadHook` — so MacMLXCore stays free of app types.
    /// CLI leaves it nil (it has no pool).
    public typealias InFlightHook = @Sendable (_ modelID: String, _ active: Bool) async -> Void
    private let inFlightHook: InFlightHook?

    /// A2d continuous-batching seam (default `nil` = disabled → every request
    /// takes the legacy single-stream path, byte-for-byte unchanged). When
    /// installed, an OpenAI chat/completions request the ``BatchRoutingPolicy``
    /// deems eligible AND the seam accepts (``BatchGenerationServing/submit(_:)``
    /// returns a stream) is served batched, BYPASSING the FIFO generation lock —
    /// the seam's own admission provides concurrency control. A cold-swap first
    /// drains the seam (SRV-2 generalized). Injected like the other hooks so
    /// MacMLXCore stays free of scheduler construction; the CLI/GUI wire the real
    /// scheduler, tests wire a stub.
    private let batchServing: (any BatchGenerationServing)?

    /// Inter-chunk stall timeout in seconds (SRV-4 / issue #29). If a live
    /// generation emits no new chunk for this long, the watchdog cancels it,
    /// releases the generation lock, and fails loudly (HTTP 504 for
    /// non-streaming; an in-band error frame then close for streaming).
    /// `<= 0` disables the watchdog. Stall-based, NOT total-duration: a long
    /// generation that keeps producing tokens is never killed.
    private let stallTimeoutSeconds: TimeInterval

    /// In-flight cold-swap Task, if any. Guards against thrashing when
    /// two requests for different models arrive at once: the second one
    /// awaits the first's completion before checking whether it can
    /// reuse the newly-loaded model or needs its own swap.
    private var loadInFlight: Task<Void, Error>?

    /// Lazily-created embedding engine for `/v1/embeddings` + `/v1/rerank`
    /// (v0.5.2). A sibling to `engine` — embedders don't conform to
    /// `InferenceEngine`. Cold-swapped by `ensureEmbedderLoaded` when a
    /// request names a different embedder than is currently resident. MVP:
    /// a single engine, no pool (see `EmbeddingEngine`).
    private var embeddingEngine: EmbeddingEngine?

    /// Lazily-created cross-encoder reranker for `/v1/rerank` (v0.7). A
    /// sibling to `embeddingEngine`: when a rerank request names a
    /// `.reranker` model, `handleRerank` routes to this TRUE cross-encoder
    /// instead of the bi-encoder cosine fallback. Cold-swapped by
    /// `ensureRerankerLoaded` when a different reranker is requested. Single
    /// engine, no pool — matching the embedder MVP (see `RerankEngine`).
    private var rerankEngine: RerankEngine?

    /// Binary generation lock. MLX model state (tokenizer, KV cache,
    /// MLX allocator) is not safe to share across concurrent
    /// `generate()` calls — the actor serialises method entry, but
    /// `generate` returns an AsyncStream whose iteration happens
    /// outside the actor. Parallel clients (Zed + Immersive Translate
    /// + curl, say) therefore stomp on each other and either crash
    /// or hang. `generationLocked` + `generationWaiters` implement a
    /// FIFO semaphore across response-body iteration, so at most one
    /// generation streams at a time.
    private var generationLocked: Bool = false

    /// A parked acquirer. Keyed by `id` so a cancelled waiter can be removed
    /// from the queue before a release ever hands it ownership — the old
    /// non-cancellation-aware `withCheckedContinuation` let a dead waiter
    /// receive the lock and never release it (SRV-3b permanent deadlock).
    private struct GenerationWaiter {
        let id: UInt64
        let continuation: CheckedContinuation<Void, Error>
    }
    private var generationWaiters: [GenerationWaiter] = []
    private var nextGenerationWaiterID: UInt64 = 0

    // MARK: Generation lock + lifecycle
    //
    // Lock-scope invariant (SRV-2 + SRV-3): the generation lock is acquired
    // and released within the SAME lexical scope that runs the generation —
    // the actor method for non-streaming, the `ResponseBody` writer closure
    // for streaming. It is NEVER acquired before entering that scope, so a
    // scope Hummingbird never runs never acquires (no leak), and a scope that
    // runs always releases via `defer`. The cold-swap (`ensureModelLoaded`)
    // runs AFTER the acquire, under the lock, so a swap can never race an
    // in-flight generation (SRV-2). `ensureModelLoaded` never waits on this
    // lock, so holding it across the load cannot deadlock.

    /// Await the generation lock. Cancellation-aware: a parked acquirer whose
    /// task is cancelled is removed from the queue and throws `CancellationError`
    /// instead of silently receiving ownership later (SRV-3b). Callers MUST
    /// pair a successful (non-throwing) return with `releaseGenerationLock()`.
    func acquireGenerationLock() async throws {
        if !generationLocked {
            generationLocked = true
            return
        }
        let id = nextGenerationWaiterID
        nextGenerationWaiterID &+= 1
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
                generationWaiters.append(GenerationWaiter(id: id, continuation: cont))
            }
        } onCancel: {
            Task { await self.cancelGenerationWaiter(id) }
        }
        // A non-throwing return means a release handed us ownership.
    }

    /// Remove a still-parked waiter and fail its acquire. No-op if it was
    /// already handed ownership by a release (that owner still releases).
    private func cancelGenerationWaiter(_ id: UInt64) {
        guard let idx = generationWaiters.firstIndex(where: { $0.id == id }) else { return }
        let waiter = generationWaiters.remove(at: idx)
        waiter.continuation.resume(throwing: CancellationError())
    }

    /// Release the generation lock and hand off to the next live waiter.
    func releaseGenerationLock() {
        if !generationWaiters.isEmpty {
            let next = generationWaiters.removeFirst()
            next.continuation.resume(returning: ())
            // Ownership passes to `next`; `generationLocked` stays true.
            return
        }
        generationLocked = false
    }

    /// Acquire the lock, cold-swap to `requestedID` under it (SRV-2), and
    /// return the freshly-resolved active engine (SRV-1). The lock is HELD on
    /// a non-throwing return — the caller MUST `releaseGenerationLock()` on
    /// every exit path (via `defer`). Any thrown error releases the lock first,
    /// so a failure never leaks it.
    private func beginGeneration(_ requestedID: String) async throws -> any InferenceEngine {
        try await acquireGenerationLock()
        // Race guard: if a release handed us ownership but our task was
        // concurrently cancelled, drop the lock instead of generating under a
        // cancelled task.
        if Task.isCancelled {
            releaseGenerationLock()
            throw CancellationError()
        }
        do {
            try await ensureModelLoaded(requestedID)  // SRV-2: swap under the lock
        } catch {
            releaseGenerationLock()
            throw error
        }
        guard let engine = await engineProvider() else {  // SRV-1: re-resolve
            releaseGenerationLock()
            throw ModelSwapError.loadFailed(id: requestedID, reason: "No active engine after load")
        }
        return engine
    }

    /// Mark/unmark the active model in-flight one layer up (POOL-3). Best
    /// effort — a nil hook (CLI/tests) is a no-op.
    private func markInFlight(_ modelID: String, _ active: Bool) async {
        await inFlightHook?(modelID, active)
    }

    /// Map a cold-swap failure onto an OpenAI/Anthropic-style error
    /// `Response` (`model_not_found` / `load_failed`). Shared by every
    /// OpenAI + Anthropic response path — each now calls `beginGeneration`
    /// directly (SRV-2 moved the swap under the generation lock).
    private func openAIStyleSwapErrorResponse(_ err: ModelSwapError) -> Response {
        switch err {
        case .modelNotFound(let id):
            return errorResponse(
                status: .notFound,
                message: "Model not found: \(id). Download it via `macmlx pull \(id)` or check `macmlx list`.",
                code: "model_not_found"
            )
        case .loadFailed(let id, let reason):
            return errorResponse(
                status: .internalServerError,
                message: "Failed to load \(id): \(reason)",
                code: "load_failed"
            )
        }
    }

    /// Ollama-style variant — same shape, different wording/codes
    /// (`model_load_failed` vs `load_failed`) matching the pre-existing
    /// Ollama error responses.
    private func ollamaStyleSwapErrorResponse(_ err: ModelSwapError) -> Response {
        switch err {
        case .modelNotFound(let id):
            return errorResponse(status: .notFound, message: "Model not found: \(id)", code: "model_not_found")
        case .loadFailed(let id, let reason):
            return errorResponse(status: .internalServerError, message: "Failed to load \(id): \(reason)", code: "model_load_failed")
        }
    }

    /// The running ServiceGroup — held so `stop()` can trigger graceful shutdown.
    private var serviceGroup: ServiceGroup?
    private var serverTask: Task<Void, Error>?
    private var _listeningPort: Int?

    // Telemetry counters
    private var requestCount: Int = 0
    private var tokenCount: Int = 0
    private var startedAt: Date?

    // MARK: Public interface

    /// The port the server is actually listening on, or `nil` if not running.
    public var listeningPort: Int? { _listeningPort }

    /// `true` once the server has bound a port and is accepting connections.
    public var isRunning: Bool { _listeningPort != nil }

    // MARK: Init

    /// Default stall-watchdog timeout (SRV-4) when a caller doesn't
    /// specify one — mirrors `SettingsManager`'s persisted default.
    public static let defaultStallTimeoutSeconds: TimeInterval = 120

    /// Create a server without cold-swap support — a chat completion
    /// request whose `model` field doesn't match the engine's loaded
    /// model will fail at the engine layer. Back-compat convenience:
    /// wraps `engine` in a fixed provider (SRV-1's re-resolution is a
    /// no-op when there's only ever one engine instance, as with the CLI).
    public init(engine: any InferenceEngine, apiKey: String? = nil) {
        self.engineProvider = { engine }
        self.modelResolver = { _ in nil }
        self.loadHook = nil
        self.inFlightHook = nil
        self.apiKey = apiKey
        self.stallTimeoutSeconds = Self.defaultStallTimeoutSeconds
        self.batchServing = nil
    }

    /// Create a server with cold-swap support — chat completion
    /// requests naming a different model than currently loaded will
    /// trigger an unload + load before the request proceeds. Back-compat
    /// convenience — fixed provider, see `init(engine:apiKey:)`.
    public init(
        engine: any InferenceEngine,
        modelResolver: @escaping ModelResolver,
        apiKey: String? = nil
    ) {
        self.engineProvider = { engine }
        self.modelResolver = modelResolver
        self.loadHook = nil
        self.inFlightHook = nil
        self.apiKey = apiKey
        self.stallTimeoutSeconds = Self.defaultStallTimeoutSeconds
        self.batchServing = nil
    }

    /// Create a server with cold-swap + a custom load hook. Back-compat
    /// convenience — fixed provider, see `init(engine:apiKey:)`. Prefer
    /// `init(engineProvider:modelResolver:loadHook:inFlightHook:apiKey:stallTimeoutSeconds:)`
    /// for callers (like the GUI) whose active engine can change out from
    /// under a fixed reference.
    public init(
        engine: any InferenceEngine,
        modelResolver: @escaping ModelResolver,
        loadHook: @escaping LoadHook,
        apiKey: String? = nil
    ) {
        self.engineProvider = { engine }
        self.modelResolver = modelResolver
        self.loadHook = loadHook
        self.inFlightHook = nil
        self.apiKey = apiKey
        self.stallTimeoutSeconds = Self.defaultStallTimeoutSeconds
        self.batchServing = nil
    }

    /// Primary initialiser (SRV-1). `engineProvider` is invoked fresh on
    /// every request after `ensureModelLoaded` succeeds, so a caller whose
    /// active engine can change out from under it (the GUI's `ModelPool`
    /// mints a new `MLXSwiftEngine` per model) always generates against the
    /// model that's actually loaded. `inFlightHook` (POOL-3) lets the caller
    /// mark the active model busy in a pool one layer up; `stallTimeoutSeconds`
    /// (SRV-4) bounds inter-chunk silence during generation — `<= 0` disables
    /// the watchdog.
    public init(
        engineProvider: @escaping @Sendable () async -> (any InferenceEngine)?,
        modelResolver: @escaping ModelResolver = { _ in nil },
        loadHook: LoadHook? = nil,
        inFlightHook: InFlightHook? = nil,
        apiKey: String? = nil,
        stallTimeoutSeconds: TimeInterval = HummingbirdServer.defaultStallTimeoutSeconds,
        batchServing: (any BatchGenerationServing)? = nil
    ) {
        self.engineProvider = engineProvider
        self.modelResolver = modelResolver
        self.loadHook = loadHook
        self.inFlightHook = inFlightHook
        self.apiKey = apiKey
        self.stallTimeoutSeconds = stallTimeoutSeconds
        self.batchServing = batchServing
    }

    // MARK: Lifecycle

    /// Start the server on `preferredPort`, retrying up to +20 adjacent ports on bind failure.
    /// Returns the actual bound port.
    @discardableResult
    public func start(preferredPort: Int) async throws -> Int {
        guard !isRunning else { return _listeningPort! }

        startedAt = Date()

        let boundPort = try await attemptBind(startingAt: preferredPort)
        _listeningPort = boundPort
        return boundPort
    }

    /// Stop the server and release the port.
    ///
    /// Triggers a graceful shutdown of the underlying ServiceGroup and awaits
    /// full teardown so the port is released before this method returns.
    public func stop() async {
        guard let group = serviceGroup else { return }
        let task = serverTask
        serviceGroup = nil
        serverTask = nil
        _listeningPort = nil
        // Trigger graceful shutdown — this signals the ServiceGroup to stop
        // all its services (including the HTTP server) cleanly.
        await group.triggerGracefulShutdown()
        // Await the task so we know the port has been released.
        _ = try? await task?.value
    }

    // MARK: Private helpers

    private func attemptBind(startingAt preferredPort: Int) async throws -> Int {
        for offset in 0...20 {
            let port = preferredPort + offset
            if let actualPort = try? await bindServer(on: port) {
                return actualPort
            }
        }
        throw InferenceServiceError.noAvailablePort
    }

    /// Try to bind on the given port. Returns the actual bound port, or throws on failure.
    private func bindServer(on port: Int) async throws -> Int {
        // Use an AsyncStream to signal the bound port from `onServerRunning`.
        let (boundPortStream, continuation) = AsyncStream<Int>.makeStream()

        let router = buildRouter()
        let config = ApplicationConfiguration(
            address: .hostname("127.0.0.1", port: port),
            serverName: "macMLX"
        )

        let app = Application(
            router: router,
            configuration: config,
            onServerRunning: { channel in
                let actualPort = channel.localAddress?.port ?? port
                continuation.yield(actualPort)
                continuation.finish()
            }
        )

        // Build the ServiceGroup so we can trigger graceful shutdown later.
        let group = ServiceGroup(
            configuration: .init(
                services: [app],
                logger: app.logger
            )
        )

        let task = Task {
            do {
                try await group.run()
            } catch {
                // If the server fails to start (e.g. port in use), finish the
                // continuation so that `iterator.next()` below unblocks.
                continuation.finish()
                throw error
            }
        }
        self.serverTask = task
        self.serviceGroup = group

        // Race: wait for the port signal OR the task to fail (which finishes the stream).
        var iterator = boundPortStream.makeAsyncIterator()
        if let actual = await iterator.next() {
            return actual
        }

        // Stream finished without yielding a port — the task failed to bind.
        // Clean up: await task to propagate the error.
        self.serverTask = nil
        self.serviceGroup = nil
        do {
            try await task.value
        } catch {
            throw InferenceServiceError.portAlreadyInUse(port)
        }
        throw InferenceServiceError.portAlreadyInUse(port)
    }

    // MARK: Counters

    private func incrementRequest() { requestCount += 1 }
    private func incrementTokens(_ n: Int) { tokenCount += n }

    // MARK: Router

    private func buildRouter() -> Router<BasicRequestContext> {
        let router = Router<BasicRequestContext>()

        // CORS — browser-based clients (Open WebUI, ChatGPT Next Web,
        // Cursor's embedded webview, etc.) enforce CORS on fetch and
        // fail with "fetch error" if Access-Control-Allow-Origin is
        // missing or not `*`. curl ignores CORS, which is why the
        // terminal worked while the UI didn't. `.all` is the right
        // setting for a localhost-only API bound to 127.0.0.1 — the
        // inbound reachability boundary is already the loopback
        // interface, not the origin check.
        router.add(middleware: CORSMiddleware(
            allowOrigin: .all,
            allowHeaders: [
                .accept, .authorization, .contentType, .origin, .userAgent
            ],
            allowMethods: [.get, .post, .head, .options, .put, .delete],
            allowCredentials: false,
            maxAge: .seconds(3600)
        ))

        // Request logging — see RequestLoggingMiddleware at the bottom
        // of this file. Must come before routes so every inbound request
        // (including ones that will 404 at route-matching time) is
        // observed.
        router.add(middleware: RequestLoggingMiddleware())

        // Bearer-token auth — when the server is configured with an API
        // key, gate every route except the health probes. Added after
        // logging so rejected requests still show up in the Logs tab.
        // An empty string counts as "no auth" (matches the nil default):
        // installing the middleware with "" would expect the literal
        // header "Bearer " and 401 every normal request — a bricked
        // server that provides no security either.
        if let apiKey, !apiKey.isEmpty {
            router.add(middleware: BearerAuthMiddleware(apiKey: apiKey))
        }

        // Capture self for use in route closures.
        // The actor reference is Sendable, so this is safe under Swift 6.
        let server = self

        // GET /health
        router.get("/health") { _, _ -> Response in
            await server.incrementRequest()
            return try await server.handleHealth()
        }

        // GET /v1/models
        router.get("/v1/models") { _, _ -> Response in
            await server.incrementRequest()
            return try await server.handleModels()
        }

        // GET /x/status
        router.get("/x/status") { _, _ -> Response in
            await server.incrementRequest()
            return try await server.handleStatus()
        }

        // POST /v1/chat/completions
        router.post("/v1/chat/completions") { request, context -> Response in
            await server.incrementRequest()
            return try await server.handleChatCompletions(request: request, context: context)
        }

        // POST /v1 alias — some OpenAI-compat clients ("Custom API"
        // configs in older front-ends, LM-Studio-style probes, etc.)
        // POST their full chat payload straight to the base URL root
        // rather than /v1/chat/completions. Route them through the
        // same handler. Covers `POST /v1`, `POST /`, and the legacy
        // `/v1/completions` text-completion path (we serve the same
        // chat-format response — sufficient for "did this model work?"
        // connectivity tests).
        router.post("/v1") { request, context -> Response in
            await server.incrementRequest()
            return try await server.handleChatCompletions(request: request, context: context)
        }
        router.post("/") { request, context -> Response in
            await server.incrementRequest()
            return try await server.handleChatCompletions(request: request, context: context)
        }
        router.post("/v1/completions") { request, context -> Response in
            await server.incrementRequest()
            return try await server.handleChatCompletions(request: request, context: context)
        }

        // POST /v1/messages — Anthropic Messages API compatibility.
        // Claude Code, the Anthropic SDKs, and other Anthropic-speaking
        // clients POST here. We translate to and from the same
        // GenerateRequest the OpenAI path uses so one engine serves both.
        router.post("/v1/messages") { request, context -> Response in
            await server.incrementRequest()
            return try await server.handleAnthropicMessages(request: request, context: context)
        }

        // POST /v1/embeddings — OpenAI-compatible text embeddings (v0.5.2).
        // Served by the dedicated `EmbeddingEngine`, cold-swapped per model.
        router.post("/v1/embeddings") { request, context -> Response in
            await server.incrementRequest()
            return try await server.handleEmbeddings(request: request, context: context)
        }

        // POST /v1/rerank — bi-encoder rerank MVP (v0.5.2). Embeds the query
        // and documents with the same embedder and ranks by cosine similarity.
        router.post("/v1/rerank") { request, context -> Response in
            await server.incrementRequest()
            return try await server.handleRerank(request: request, context: context)
        }

        // POST /x/models/load
        router.post("/x/models/load") { request, context -> Response in
            await server.incrementRequest()
            return try await server.handleLoadModel(request: request, context: context)
        }

        // POST /x/models/unload
        router.post("/x/models/unload") { request, context -> Response in
            await server.incrementRequest()
            return try await server.handleUnloadModel(request: request, context: context)
        }

        // Root + /v1 "discovery" routes. OpenAI-compat clients (Open WebUI,
        // Cursor custom model, etc.) often probe these before committing
        // to the real endpoints; returning a tiny JSON body is friendlier
        // than letting them 404 and show "fetch error".
        router.get("/") { _, _ -> Response in
            await server.incrementRequest()
            return try await server.handleRootInfo()
        }
        // Hummingbird normalises trailing slashes, so `/v1` and `/v1/`
        // resolve to the same route — registering both throws at
        // runtime ("GET already has a handler"). A single registration
        // covers both shapes.
        router.get("/v1") { _, _ -> Response in
            await server.incrementRequest()
            return try await server.handleRootInfo()
        }

        // Health-check alias. Many clients probe /v1/health rather than
        // the bare /health we already expose.
        router.get("/v1/health") { _, _ -> Response in
            await server.incrementRequest()
            return try await server.handleHealth()
        }

        // Status alias under /v1 for clients that don't know about /x/.
        router.get("/v1/status") { _, _ -> Response in
            await server.incrementRequest()
            return try await server.handleStatus()
        }

        // Ollama-compatible API surface. Zed's "Ollama" provider, some
        // translation extensions, and misc CLIs probe these paths. We
        // translate to and from the equivalent OpenAI shape internally
        // so the same engine handles both protocols.
        router.get("/api/version") { _, _ -> Response in
            await server.incrementRequest()
            return try await server.handleOllamaVersion()
        }
        router.get("/api/tags") { _, _ -> Response in
            await server.incrementRequest()
            return try await server.handleOllamaTags()
        }
        router.post("/api/chat") { request, context -> Response in
            await server.incrementRequest()
            return try await server.handleOllamaChat(request: request, context: context)
        }
        router.post("/api/generate") { request, context -> Response in
            await server.incrementRequest()
            return try await server.handleOllamaGenerate(request: request, context: context)
        }
        router.post("/api/show") { _, _ -> Response in
            await server.incrementRequest()
            return try await server.handleOllamaShow()
        }

        // Defensive: some users misconfigure client base URL as
        // `http://localhost:8000/v1/chat/completions` (full endpoint path)
        // instead of `http://localhost:8000/v1` (base). The client then
        // appends `/chat/completions` on top, producing the doubled path.
        // Route the doubled form to the same handler.
        router.post("/v1/chat/completions/chat/completions") { request, context -> Response in
            await server.incrementRequest()
            return try await server.handleChatCompletions(request: request, context: context)
        }

        return router
    }

    // MARK: Route handlers

    private func handleHealth() throws -> Response {
        return try jsonResponse(["status": "ok"])
    }

    /// Minimal discovery payload. Returned at `/`, `/v1`, `/v1/` so
    /// OpenAI-compat clients that probe the root don't see a bare 404.
    private func handleRootInfo() throws -> Response {
        return try jsonResponseAny([
            "object": "api",
            "name": "macMLX",
            "version": "0.3.6",
            "openai_compatible": true,
            "endpoints": [
                "/v1/models",
                "/v1/chat/completions",
                "/v1/health",
                "/v1/status"
            ]
        ])
    }

    private func handleModels() async throws -> Response {
        // Pre-v0.3.3 this only listed the currently loaded model, because
        // that was the only one that could answer a chat completion. With
        // cold-swap (v0.3.3), any model the resolver can find is a valid
        // target, so we want `GET /v1/models` to reflect that. But we
        // don't have a "list all" primitive on the resolver — it's a
        // point-lookup. We keep listing the loaded model (compatibility)
        // and document that external clients wanting a full list should
        // use `macmlx list` or the GUI Models tab.
        let loaded = await engineProvider()?.loadedModel
        var data: [[String: String]] = []
        if let model = loaded {
            // Report the user-facing alias (v0.5.1) when one is set for
            // this model; otherwise fall back to its directory id. Empty
            // alias is treated as "no alias".
            let params = await ModelParametersStore().load(for: model.id)
            let reportedID: String
            if let alias = params.alias, !alias.isEmpty {
                reportedID = alias
            } else {
                reportedID = model.id
            }
            data = [["id": reportedID, "object": "model", "owned_by": "local"]]
        }
        let body: [String: Any] = ["object": "list", "data": data]
        return try jsonResponseAny(body)
    }

    private func handleStatus() async throws -> Response {
        let loaded = await engineProvider()?.loadedModel
        let uptime = startedAt.map { Date().timeIntervalSince($0) } ?? 0
        let totalMemory = MemoryProbe.totalMemoryGB()

        var body: [String: Any] = [
            "status": loaded != nil ? "loaded" : "idle",
            "memory_used_gb": MemoryProbe.residentMemoryGB(),
            "memory_total_gb": totalMemory,
            "uptime_seconds": uptime,
            "requests_total": requestCount,
            "tokens_generated_total": tokenCount,
        ]
        if let m = loaded {
            body["loaded_model"] = m.id
        } else {
            body["loaded_model"] = NSNull()
        }
        return try jsonResponseAny(body)
    }

    private func handleChatCompletions(
        request: Request,
        context: BasicRequestContext
    ) async throws -> Response {
        let buffer = try await request.body.collect(upTo: context.maxUploadSize)
        guard let data = buffer.getData(at: buffer.readerIndex, length: buffer.readableBytes) else {
            return errorResponse(
                status: .badRequest,
                message: "Empty request body",
                code: "invalid_request_error"
            )
        }

        let chatReq: ChatCompletionRequest
        do {
            chatReq = try JSONDecoder().decode(ChatCompletionRequest.self, from: data)
        } catch {
            // Legacy text-completions fallback (P3-1): `/v1/completions` (and
            // the `POST /` / `POST /v1` aliases) historically carry a bare
            // `prompt` with NO `messages`, so the chat decode above fails. Wrap
            // `prompt` into a single user turn and continue — these routes serve
            // the same chat-format response the route comment promises. A body
            // that is neither shape falls through to the original 400.
            guard let legacy = try? JSONDecoder().decode(LegacyCompletionRequest.self, from: data) else {
                return errorResponse(
                    status: .badRequest,
                    message: "Invalid JSON: \(error.localizedDescription)",
                    code: "invalid_request_error"
                )
            }
            chatReq = ChatCompletionRequest(
                model: legacy.model,
                messages: [ChatCompletionRequest.Message(
                    role: "user",
                    content: .string(legacy.prompt),
                    tool_calls: nil,
                    tool_call_id: nil
                )],
                stream: legacy.stream,
                temperature: legacy.temperature,
                top_p: legacy.top_p,
                max_tokens: legacy.max_tokens,
                // Legacy text-completions bodies carry no tools.
                tools: nil,
                tool_choice: nil,
                draft_model: legacy.draft_model,
                num_draft_tokens: legacy.num_draft_tokens,
                // Legacy text-completions bodies carry no response_format.
                response_format: nil,
                // Track E extensions are honored on the legacy route too.
                logit_bias: legacy.logit_bias,
                logprobs: legacy.logprobs,
                top_logprobs: legacy.top_logprobs,
                xtc_probability: legacy.xtc_probability,
                xtc_threshold: legacy.xtc_threshold,
                kv_bits: legacy.kv_bits,
                kv_group_size: legacy.kv_group_size,
                quantized_kv_start: legacy.quantized_kv_start,
                adapters: legacy.adapters
            )
        }

        // Extract system prompt from the first system message. This also
        // removes it from the downstream messages array — GenerateRequest
        // re-prepends the systemPrompt via its allMessages computed
        // property, so leaving it in both places produces a duplicate
        // system turn, which Qwen3 / Gemma / other strict Jinja chat
        // templates reject with a TemplateException.
        let systemPrompt = chatReq.messages.first(where: { $0.role == "system" })?.content?.text

        // Map the rest (user / assistant / tool), dropping unknown roles and the
        // now-separated system turns. Multimodal `content` arrays are split here:
        // text parts → `content`, image_url data URLs → `images` via
        // `extractImages()`. Tool history is fully wired (agent-tools wave):
        // assistant `tool_calls` → `ChatMessage.toolCalls`, and `role:"tool"`
        // turns → `.tool` messages carrying `tool_call_id`, so a chat template
        // that asserts call ↔ result pairing is satisfied. A malformed tool
        // block is a 400 (never a silent drop).
        let messages: [ChatMessage]
        do {
            messages = try decodeOpenAIMessages(chatReq.messages)
        } catch let error as ToolHistoryDecodeError {
            return errorResponse(
                status: .badRequest,
                message: error.message,
                code: "invalid_request_error"
            )
        }

        let params = GenerationParameters(
            temperature: chatReq.temperature ?? 0.7,
            topP: chatReq.top_p ?? 0.95,
            maxTokens: chatReq.max_tokens ?? 2048,
            stream: chatReq.stream ?? false,
            // Track E — the `GenerationParameters` initializer clamps every one of
            // these (see its clamp helpers), so a hostile/malformed value can't
            // reach the engine unbounded.
            logitBias: GenerationParameters.logitBias(fromOpenAI: chatReq.logit_bias),
            xtcProbability: chatReq.xtc_probability.map { Float($0) },
            xtcThreshold: chatReq.xtc_threshold.map { Float($0) },
            logprobs: chatReq.logprobs ?? false,
            topLogprobs: chatReq.top_logprobs,
            kvBits: chatReq.kv_bits,
            kvGroupSize: chatReq.kv_group_size,
            quantizedKVStart: chatReq.quantized_kv_start
        )

        // Tool pass-through (v0.5, OpenAI `/v1/chat/completions` only): forward
        // the request's `tools` into the chat template unless `tool_choice` is
        // the string "none", which suppresses tool use entirely. Passing an
        // empty `tools` array through is harmless — the engine gates its
        // tool-call detector on `tools?.isEmpty == false`, so `[]` behaves like
        // "no tools". Fine-grained forcing (a `tool_choice` object naming one
        // function) is a documented follow-up; any non-"none" value here means
        // "offer the tools".
        let toolChoiceIsNone: Bool
        if case .string("none") = chatReq.tool_choice {
            toolChoiceIsNone = true
        } else {
            toolChoiceIsNone = false
        }
        let toolsToSend = toolChoiceIsNone ? nil : chatReq.tools

        // Track C: decode + validate `response_format`. Unsupported schema
        // features or malformed bodies are rejected here with a 400 (aligned
        // with every other request-validation 400 on this path) rather than
        // silently downgraded.
        let responseFormat: ResponseFormat?
        do {
            responseFormat = try ResponseFormatDecoder.decode(chatReq.response_format)
        } catch let error as ResponseFormatError {
            return errorResponse(
                status: .badRequest,
                message: error.description,
                code: "invalid_request_error"
            )
        }

        // A vision request (image attachments) combined with `response_format`
        // (structured output) is rejected here, before generation starts, rather
        // than silently degraded: the VLM decode path carries no constraint mask,
        // so it would ignore the schema entirely (project rule: no silent
        // degradation). Checked at the request boundary — the same 400 the other
        // unsupported combos on this path return — because an engine-side throw
        // would surface mid-stream (headers already sent) rather than as a clean 400.
        if responseFormat != nil, messages.contains(where: { !$0.images.isEmpty }) {
            return errorResponse(
                status: .badRequest,
                message: "`response_format` (structured output) is not supported together "
                    + "with image inputs (vision) in this version.",
                code: "invalid_request_error"
            )
        }

        // `tools` + `response_format` together is rejected here too, symmetric with
        // the image/logprobs combo guards on this path. `toolsToSend` (post
        // `tool_choice:"none"` filtering) is the same signal the engine gates
        // `hasTools` on, so a request whose tools were suppressed via
        // `tool_choice:"none"` is unaffected. Without this, the request was
        // silently ACCEPTED and only degraded deep in the engine — `makeToolProcessor`
        // (Track E) disables just the tool-call PROCESSOR for this combination
        // (grammar-constrained output can't contain tool-call syntax anyway), which
        // is correct as defense-in-depth for the GUI's direct (non-HTTP)
        // `engine.generate` call, but left the HTTP path as the one unsupported
        // combo that didn't 400 like every sibling guard here.
        if responseFormat != nil, toolsToSend?.isEmpty == false {
            return errorResponse(
                status: .badRequest,
                message: "`tools` is not supported together with `response_format` "
                    + "(structured output) in this version.",
                code: "invalid_request_error"
            )
        }

        // Track E combination guards: reject combinations the engine cannot
        // honor, rather than silently dropping a requested feature. `logprobs`
        // capture is not composed with the constraint mask (structured output)
        // and cannot ride the speculative-decoding iterator, so both are 400s.
        // OpenAI semantics: `top_logprobs` requires `logprobs: true`.
        if !params.logprobs, let top = chatReq.top_logprobs, top > 0 {
            return errorResponse(
                status: .badRequest,
                message: "`top_logprobs` requires `logprobs` to be true.",
                code: "invalid_request_error"
            )
        }
        if params.logprobs {
            if responseFormat != nil {
                return errorResponse(
                    status: .badRequest,
                    message: "`logprobs` is not supported together with `response_format` "
                        + "(structured output) in this version.",
                    code: "invalid_request_error"
                )
            }
            if chatReq.draft_model != nil {
                return errorResponse(
                    status: .badRequest,
                    message: "`logprobs` is not supported together with `draft_model` "
                        + "(speculative decoding) in this version.",
                    code: "invalid_request_error"
                )
            }
        }

        let genRequest = GenerateRequest(
            model: chatReq.model,
            messages: messages,
            systemPrompt: systemPrompt,
            parameters: params,
            templateKwargs: await templateKwargs(for: chatReq.model),
            tools: toolsToSend,
            draftModelID: chatReq.draft_model,
            numDraftTokens: chatReq.num_draft_tokens,
            responseFormat: responseFormat,
            adapters: chatReq.adapters
        )

        // Cold-swap (v0.3.3) is no longer resolved here. SRV-2: the swap
        // must happen atomically with the generation lock, so each
        // responder below calls `beginGeneration(genRequest.model)` itself
        // (acquire → swap → re-resolve engine, all under the lock) instead
        // of us doing it here, unlocked, ahead of time.
        let wantsStream = chatReq.stream ?? false

        // A2d: route an eligible request to the continuous-batching path when a
        // seam is installed and accepts it. `submit` returning nil — no seam, a
        // VLM/uncoverable resident model, or a non-resident model — falls through
        // to the legacy single-stream responders below, byte-for-byte unchanged
        // and with no batched work performed (so no double-billing on fallback).
        // Only this OpenAI chat path (and its legacy `/v1/completions` alias,
        // which decodes into the same `chatReq`) is ever batched; every other
        // endpoint keeps the existing single-stream path untouched.
        // Track E: any per-request sampling/adapter feature the batched step
        // evaluator can't honor forces the single-stream path (see
        // `BatchRoutingPolicy`).
        let hasUnbatchableSamplingFeature =
            params.logitBias != nil
            || params.logprobs
            || (params.xtcProbability ?? 0) > 0
            || params.kvBits != nil
            || genRequest.adapters != nil
        if BatchRoutingPolicy.shouldAttemptBatch(
            batchingEnabled: batchServing != nil,
            hasDraftModel: genRequest.draftModelID != nil,
            hasResponseFormat: genRequest.responseFormat != nil,
            hasUnbatchableSamplingFeature: hasUnbatchableSamplingFeature
        ) {
            // MEDIUM#3: resolve the in-flight key the same alias-aware way the
            // single-stream path does (mirrors the resolve at ~line 1897) — the
            // resident model's own id when one is loaded, falling back to the
            // request's raw `model` field otherwise.
            let batchModelID = await engineProvider()?.loadedModel?.id ?? genRequest.model
            // MEDIUM#2 / POOL-3: mark in-flight BEFORE calling `submit`, not
            // after. Per the seam's contract (`submit` is a billing/timing
            // anchor — see `BatchGenerationServing`), admission IS the start of
            // decode, so by the time `submit` returns the row may already be
            // running; marking afterward would leave a window where a
            // concurrent load could LRU-evict the model out from under an
            // already-decoding row.
            await markInFlight(batchModelID, true)
            if let batchStream = await batchServing?.submit(genRequest) {
                // Submit accepted the row: ownership of the "false" mark passes
                // to the batch responder's own `defer`. Neither responder marks
                // `true` again — that would double-count.
                if wantsStream {
                    return batchStreamingChatResponse(genRequest: genRequest, modelID: batchModelID, stream: batchStream)
                } else {
                    return try await batchNonStreamingChatResponse(genRequest: genRequest, modelID: batchModelID, stream: batchStream)
                }
            }
            // Submit declined (nil): undo the speculative mark before falling
            // through to the legacy single-stream path below, which marks
            // in-flight itself — leaving this mark set would double-count.
            await markInFlight(batchModelID, false)
        }

        if wantsStream {
            return try await streamingChatResponse(genRequest: genRequest)
        } else {
            return try await nonStreamingChatResponse(genRequest: genRequest)
        }
    }

    // MARK: - Ollama compatibility

    /// Ollama `/api/version` — tiny JSON describing our API level.
    /// Clients use this as a reachability probe before issuing real
    /// requests; returning a value that looks like Ollama is enough.
    private func handleOllamaVersion() throws -> Response {
        return try jsonResponse([
            "version": "0.3.6-macmlx"
        ])
    }

    /// Ollama `/api/tags` — list of locally-available models. We
    /// translate from our existing `/v1/models` shape into Ollama's
    /// `{"models":[{name, size, modified_at, digest, details}]}` shape.
    private func handleOllamaTags() async throws -> Response {
        // Reuse the same "what's loaded + what the resolver can see"
        // logic as handleModels. For Ollama the list should be of
        // locally-available models regardless of load state.
        let currentID = await engineProvider()?.loadedModel?.id
        var entries: [[String: Any]] = []
        if let currentID {
            // Report the user-facing alias (v0.5.1) when one is set for this
            // model, mirroring `handleModels` / `GET /v1/models` (P3-2). Without
            // this, Ollama-compat clients saw the raw directory id while the
            // OpenAI list showed the alias (A3 inconsistency). Empty alias is
            // treated as "no alias".
            let params = await ModelParametersStore().load(for: currentID)
            let reportedID: String
            if let alias = params.alias, !alias.isEmpty {
                reportedID = alias
            } else {
                reportedID = currentID
            }
            entries.append([
                "name": reportedID,
                "model": reportedID,
                "modified_at": ISO8601DateFormatter().string(from: Date()),
                "size": 0,
                "digest": "",
                "details": [
                    "format": "mlx",
                    "family": "",
                    "parameter_size": "",
                    "quantization_level": ""
                ] as [String: Any]
            ])
        }
        return try jsonResponseAny([
            "models": entries
        ])
    }

    /// Ollama `/api/show` — minimal metadata response. Returns an empty-ish
    /// envelope so probing clients don't 404.
    private func handleOllamaShow() throws -> Response {
        return try jsonResponseAny([
            "modelfile": "",
            "parameters": "",
            "template": "",
            "details": [
                "format": "mlx",
                "family": "",
                "parameter_size": "",
                "quantization_level": ""
            ] as [String: Any]
        ])
    }

    /// Ollama `/api/chat` — translates to OpenAI-shape internally then
    /// serialises the response back into Ollama's envelope.
    private func handleOllamaChat(
        request: Request,
        context: BasicRequestContext
    ) async throws -> Response {
        let buffer = try await request.body.collect(upTo: context.maxUploadSize)
        guard let data = buffer.getData(at: buffer.readerIndex, length: buffer.readableBytes) else {
            return errorResponse(status: .badRequest, message: "Empty request body", code: "invalid_request_error")
        }
        let req: OllamaChatRequest
        do {
            req = try JSONDecoder().decode(OllamaChatRequest.self, from: data)
        } catch {
            return errorResponse(status: .badRequest, message: "Invalid JSON: \(error.localizedDescription)", code: "invalid_request_error")
        }

        let systemPrompt = req.messages.first(where: { $0.role == "system" })?.content
        let messages: [ChatMessage] = req.messages.compactMap { msg in
            // The agent-tools wave wired full tool history on the OpenAI and
            // Anthropic surfaces only. Ollama's `/api/chat` tool wire-format
            // differs (assistant `tool_calls[].function.arguments` is an OBJECT,
            // not a JSON string; results route by `tool_name`; calls carry no
            // `id`) AND this endpoint's response side emits no `tool_calls`, so a
            // working loop would need its own encode half too. Out of scope for
            // this wave: `.tool` turns are dropped rather than decoded
            // half-formed (which could trip a chat-template pairing assertion).
            guard let role = MessageRole(rawValue: msg.role),
                  role != .system, role != .tool
            else { return nil }
            return ChatMessage(role: role, content: msg.content)
        }

        let params = GenerationParameters(
            temperature: req.options?.temperature ?? 0.7,
            topP: req.options?.top_p ?? 0.95,
            maxTokens: req.options?.num_predict ?? 2048,
            stream: req.stream ?? false
        )

        let genRequest = GenerateRequest(
            model: req.model,
            messages: messages,
            systemPrompt: systemPrompt,
            parameters: params,
            templateKwargs: await templateKwargs(for: req.model)
        )

        // Cold-swap (SRV-2): resolved inside each responder, atomically with
        // the generation lock — see `beginGeneration`.
        //
        // Ollama defaults stream=true when omitted (opposite of OpenAI).
        // Zed, Immersive Translate, and most Ollama CLIs expect NDJSON
        // streaming unless they explicitly opt out.
        let wantsStream = req.stream ?? true
        if wantsStream {
            return try await streamingOllamaChatResponse(model: req.model, genRequest: genRequest)
        }
        return try await nonStreamingOllamaChatResponse(model: req.model, genRequest: genRequest)
    }

    /// Non-streaming Ollama /api/chat response. Shape:
    /// `{"model":"…","created_at":"…","message":{"role":"assistant","content":"…"},"done":true}`.
    private func nonStreamingOllamaChatResponse(model: String, genRequest: GenerateRequest) async throws -> Response {
        let engine: any InferenceEngine
        do {
            engine = try await beginGeneration(genRequest.model)
        } catch let err as ModelSwapError {
            return ollamaStyleSwapErrorResponse(err)
        }
        defer { Task { [weak self] in await self?.releaseGenerationLock() } }

        let modelID = await engine.loadedModel?.id ?? genRequest.model
        await markInFlight(modelID, true)
        defer { Task { [weak self] in await self?.markInFlight(modelID, false) } }

        let stream = await engine.generate(genRequest)
        let box = ChunkIteratorBox(stream)
        var fullText = ""
        loop: while true {
            switch try await nextGenerationStep(box, stallTimeout: stallTimeoutSeconds) {
            case .finished:
                break loop
            case .stalled:
                return errorResponse(
                    status: .gatewayTimeout,
                    message: "Generation stalled: no output for over \(Int(stallTimeoutSeconds))s.",
                    code: "generation_stalled"
                )
            case .chunk(let chunk):
                fullText += chunk.text
            }
        }
        return try jsonResponseAny([
            "model": model,
            "created_at": ISO8601DateFormatter().string(from: Date()),
            "message": [
                "role": "assistant",
                "content": fullText
            ] as [String: Any],
            "done": true
        ])
    }

    /// Streaming Ollama /api/chat — NDJSON (one JSON object per line,
    /// newline-delimited). Each line contains a partial assistant
    /// message; final line has `done:true`. This is the default when
    /// an Ollama-compat client doesn't explicitly set `stream:false`,
    /// so covers Zed, Immersive Translate, Ollama CLI, and most other
    /// Ollama-speaking tools.
    private func streamingOllamaChatResponse(model: String, genRequest: GenerateRequest) async throws -> Response {
        // A1: cheap pre-flight resolve check (no load, no lock) — reject
        // an unknown model with a real 404 before any streaming headers
        // are sent. The real (locked) resolve+load still happens inside
        // the ResponseBody closure below; SRV-2 atomicity is unchanged.
        let modelIsResolvable = await canResolveModel(model)
        if !modelIsResolvable {
            return ollamaStyleSwapErrorResponse(.modelNotFound(id: model))
        }

        // Lock-scope invariant (SRV-2/SRV-3): acquire, cold-swap, and
        // generate ALL happen inside the writer closure — see the
        // invariant comment above `acquireGenerationLock()`.
        let startedAt = Date()
        let server = self

        let responseBody = ResponseBody { writer in
            var completionTokens = 0

            let engine: any InferenceEngine
            do {
                engine = try await server.beginGeneration(genRequest.model)
            } catch {
                let msg = error.localizedDescription.replacingOccurrences(of: "\"", with: "\\\"")
                var buf = ByteBuffer()
                buf.writeString("{\"error\":\"\(msg)\",\"done\":true}\n")
                try? await writer.write(buf)
                try? await writer.finish(nil)
                return
            }
            defer { Task { await server.releaseGenerationLock() } }

            let modelID = await engine.loadedModel?.id ?? model
            await server.markInFlight(modelID, true)
            defer { Task { await server.markInFlight(modelID, false) } }

            let stream = await engine.generate(genRequest)
            let stallTimeout = await server.stallTimeoutSeconds
            let box = ChunkIteratorBox(stream)
            do {
                loop: while true {
                    switch try await nextGenerationStep(box, stallTimeout: stallTimeout) {
                    case .finished:
                        break loop
                    case .stalled:
                        var buf = ByteBuffer()
                        buf.writeString("{\"error\":\"Generation stalled: no output for over \(Int(stallTimeout))s.\",\"done\":true}\n")
                        try? await writer.write(buf)
                        try? await writer.finish(nil)
                        await server.incrementTokens(completionTokens)
                        return
                    case .chunk(let chunk):
                        // P3-4: accumulate the engine's real completion-token
                        // count (delivered on the terminal chunk's usage), not
                        // the number of SSE/NDJSON frames.
                        if let usage = chunk.usage { completionTokens = usage.completionTokens }
                        let payload: [String: Any] = [
                            "model": model,
                            "created_at": ISO8601DateFormatter().string(from: Date()),
                            "message": [
                                "role": "assistant",
                                "content": chunk.text
                            ] as [String: Any],
                            "done": false
                        ]
                        let jsonData = try JSONSerialization.data(withJSONObject: payload)
                        var buf = ByteBuffer()
                        buf.writeBytes(jsonData)
                        buf.writeString("\n")
                        try await writer.write(buf)
                    }
                }
            } catch {
                let msg = error.localizedDescription
                    .replacingOccurrences(of: "\"", with: "\\\"")
                var buf = ByteBuffer()
                buf.writeString("{\"error\":\"\(msg)\",\"done\":true}\n")
                try? await writer.write(buf)
                try? await writer.finish(nil)
                await server.incrementTokens(completionTokens)
                return
            }

            // Final frame: empty content + done:true with timing stats.
            let totalNanos = Int(Date().timeIntervalSince(startedAt) * 1_000_000_000)
            let donePayload: [String: Any] = [
                "model": model,
                "created_at": ISO8601DateFormatter().string(from: Date()),
                "message": [
                    "role": "assistant",
                    "content": ""
                ] as [String: Any],
                "done": true,
                "done_reason": "stop",
                "total_duration": totalNanos
            ]
            let doneData = try JSONSerialization.data(withJSONObject: donePayload)
            var doneBuf = ByteBuffer()
            doneBuf.writeBytes(doneData)
            doneBuf.writeString("\n")
            try await writer.write(doneBuf)
            try await writer.finish(nil)
            await server.incrementTokens(completionTokens)
        }

        var headers = HTTPFields()
        headers[.contentType] = "application/x-ndjson"
        headers[.cacheControl] = "no-cache"
        headers[.connection] = "keep-alive"
        return Response(status: .ok, headers: headers, body: responseBody)
    }

    /// Ollama `/api/generate` — text completion. Translate to our
    /// chat shape by wrapping the prompt in a single user message.
    private func handleOllamaGenerate(
        request: Request,
        context: BasicRequestContext
    ) async throws -> Response {
        let buffer = try await request.body.collect(upTo: context.maxUploadSize)
        guard let data = buffer.getData(at: buffer.readerIndex, length: buffer.readableBytes) else {
            return errorResponse(status: .badRequest, message: "Empty request body", code: "invalid_request_error")
        }
        let req: OllamaGenerateRequest
        do {
            req = try JSONDecoder().decode(OllamaGenerateRequest.self, from: data)
        } catch {
            return errorResponse(status: .badRequest, message: "Invalid JSON: \(error.localizedDescription)", code: "invalid_request_error")
        }

        let params = GenerationParameters(
            temperature: req.options?.temperature ?? 0.7,
            topP: req.options?.top_p ?? 0.95,
            maxTokens: req.options?.num_predict ?? 2048,
            stream: req.stream ?? false
        )

        let genRequest = GenerateRequest(
            model: req.model,
            messages: [ChatMessage(role: .user, content: req.prompt)],
            systemPrompt: req.system,
            parameters: params,
            templateKwargs: await templateKwargs(for: req.model)
        )

        // Cold-swap (SRV-2): resolved inside each responder, atomically with
        // the generation lock — see `beginGeneration`.
        let wantsStream = req.stream ?? true
        if wantsStream {
            return try await streamingOllamaGenerateResponse(model: req.model, genRequest: genRequest)
        }

        let engine: any InferenceEngine
        do {
            engine = try await beginGeneration(genRequest.model)
        } catch let err as ModelSwapError {
            return ollamaStyleSwapErrorResponse(err)
        }
        defer { Task { [weak self] in await self?.releaseGenerationLock() } }

        let modelID = await engine.loadedModel?.id ?? genRequest.model
        await markInFlight(modelID, true)
        defer { Task { [weak self] in await self?.markInFlight(modelID, false) } }

        let stream = await engine.generate(genRequest)
        let box = ChunkIteratorBox(stream)
        var fullText = ""
        loop: while true {
            switch try await nextGenerationStep(box, stallTimeout: stallTimeoutSeconds) {
            case .finished:
                break loop
            case .stalled:
                return errorResponse(
                    status: .gatewayTimeout,
                    message: "Generation stalled: no output for over \(Int(stallTimeoutSeconds))s.",
                    code: "generation_stalled"
                )
            case .chunk(let chunk):
                fullText += chunk.text
            }
        }
        return try jsonResponseAny([
            "model": req.model,
            "created_at": ISO8601DateFormatter().string(from: Date()),
            "response": fullText,
            "done": true
        ])
    }

    /// Streaming Ollama /api/generate — NDJSON, each line
    /// `{"model":...,"response":"partial","done":false}`.
    private func streamingOllamaGenerateResponse(model: String, genRequest: GenerateRequest) async throws -> Response {
        // A1: cheap pre-flight resolve check (no load, no lock) — reject
        // an unknown model with a real 404 before any streaming headers
        // are sent. The real (locked) resolve+load still happens inside
        // the ResponseBody closure below; SRV-2 atomicity is unchanged.
        // (Same gap as the /api/chat streaming path — /api/generate hits
        // the identical `beginGeneration`-inside-the-closure shape.)
        let modelIsResolvable = await canResolveModel(model)
        if !modelIsResolvable {
            return ollamaStyleSwapErrorResponse(.modelNotFound(id: model))
        }

        // Lock-scope invariant (SRV-2/SRV-3): acquire, cold-swap, and
        // generate ALL happen inside the writer closure.
        let startedAt = Date()
        let server = self

        let responseBody = ResponseBody { writer in
            var completionTokens = 0

            let engine: any InferenceEngine
            do {
                engine = try await server.beginGeneration(genRequest.model)
            } catch {
                let msg = error.localizedDescription.replacingOccurrences(of: "\"", with: "\\\"")
                var buf = ByteBuffer()
                buf.writeString("{\"error\":\"\(msg)\",\"done\":true}\n")
                try? await writer.write(buf)
                try? await writer.finish(nil)
                return
            }
            defer { Task { await server.releaseGenerationLock() } }

            let modelID = await engine.loadedModel?.id ?? model
            await server.markInFlight(modelID, true)
            defer { Task { await server.markInFlight(modelID, false) } }

            let stream = await engine.generate(genRequest)
            let stallTimeout = await server.stallTimeoutSeconds
            let box = ChunkIteratorBox(stream)
            do {
                loop: while true {
                    switch try await nextGenerationStep(box, stallTimeout: stallTimeout) {
                    case .finished:
                        break loop
                    case .stalled:
                        var buf = ByteBuffer()
                        buf.writeString("{\"error\":\"Generation stalled: no output for over \(Int(stallTimeout))s.\",\"done\":true}\n")
                        try? await writer.write(buf)
                        try? await writer.finish(nil)
                        await server.incrementTokens(completionTokens)
                        return
                    case .chunk(let chunk):
                        // P3-4: accumulate the engine's real completion-token
                        // count (delivered on the terminal chunk's usage), not
                        // the number of SSE/NDJSON frames.
                        if let usage = chunk.usage { completionTokens = usage.completionTokens }
                        let payload: [String: Any] = [
                            "model": model,
                            "created_at": ISO8601DateFormatter().string(from: Date()),
                            "response": chunk.text,
                            "done": false
                        ]
                        let jsonData = try JSONSerialization.data(withJSONObject: payload)
                        var buf = ByteBuffer()
                        buf.writeBytes(jsonData)
                        buf.writeString("\n")
                        try await writer.write(buf)
                    }
                }
            } catch {
                let msg = error.localizedDescription
                    .replacingOccurrences(of: "\"", with: "\\\"")
                var buf = ByteBuffer()
                buf.writeString("{\"error\":\"\(msg)\",\"done\":true}\n")
                try? await writer.write(buf)
                try? await writer.finish(nil)
                await server.incrementTokens(completionTokens)
                return
            }
            let totalNanos = Int(Date().timeIntervalSince(startedAt) * 1_000_000_000)
            let donePayload: [String: Any] = [
                "model": model,
                "created_at": ISO8601DateFormatter().string(from: Date()),
                "response": "",
                "done": true,
                "done_reason": "stop",
                "total_duration": totalNanos
            ]
            let doneData = try JSONSerialization.data(withJSONObject: donePayload)
            var doneBuf = ByteBuffer()
            doneBuf.writeBytes(doneData)
            doneBuf.writeString("\n")
            try await writer.write(doneBuf)
            try await writer.finish(nil)
            await server.incrementTokens(completionTokens)
        }

        var headers = HTTPFields()
        headers[.contentType] = "application/x-ndjson"
        headers[.cacheControl] = "no-cache"
        headers[.connection] = "keep-alive"
        return Response(status: .ok, headers: headers, body: responseBody)
    }

    // MARK: - Cold-swap (v0.3.3)

    /// Errors surfaced by `ensureModelLoaded`. Internal — handler maps
    /// them onto OpenAI-style HTTP codes.
    private enum ModelSwapError: Error {
        case modelNotFound(id: String)
        case loadFailed(id: String, reason: String)
    }

    /// Cheap, side-effect-free check (A1): can `requestedID` be resolved
    /// to a model — either because it's already loaded, or the resolver
    /// (directly or via alias) knows about it on disk? No load, no lock —
    /// mirrors only the RESOLUTION half of `ensureModelLoaded` below.
    ///
    /// Used by the streaming responders to reject an unknown model with a
    /// real 404 BEFORE any response headers are sent, matching the
    /// non-streaming path (pre-fix, streaming requests for a nonexistent
    /// model got a 200 with an in-band SSE/NDJSON error frame instead — a
    /// regression introduced when SRV-2/SRV-3 moved the real swap inside
    /// the `ResponseBody` closure). This is a pre-flight, not a guarantee:
    /// a race where the model becomes unresolvable/evicted between this
    /// check and the real (locked) resolve+load inside the closure still
    /// surfaces as an in-band error frame — expected, and harmless, since
    /// SRV-2's atomicity is unchanged.
    private func canResolveModel(_ requestedID: String) async -> Bool {
        if await engineProvider()?.loadedModel?.id == requestedID {
            return true
        }
        if await modelResolver(requestedID) != nil {
            return true
        }
        if let aliasID = await ModelParametersStore().modelID(forAlias: requestedID) {
            return await modelResolver(aliasID) != nil
        }
        return false
    }

    /// Make sure the engine has `requestedID` loaded. No-op if it's
    /// already current. If another model is current, unload + load;
    /// if nothing is loaded, just load. If `requestedID` isn't on disk,
    /// throw `.modelNotFound`.
    ///
    /// Serialisation strategy (reviewer-chosen option `a`): awaits any
    /// in-flight swap before checking/starting its own. Two concurrent
    /// requests for the same newly-wanted model therefore share the
    /// same load — only one disk read, one memory bake-in.
    private func ensureModelLoaded(_ requestedID: String) async throws {
        // If a swap is already in flight, wait for it to finish first.
        // This either (a) lands our model for free or (b) lands a
        // different model that we then need to evict — both handled by
        // the check below.
        if let pending = loadInFlight {
            _ = try? await pending.value
            loadInFlight = nil
        }

        if await engineProvider()?.loadedModel?.id == requestedID {
            return
        }

        // Resolve `requestedID`. First try the direct resolver (id or
        // displayName). If that misses, treat `requestedID` as a
        // user-facing alias (v0.5.1): scan the per-model params files
        // for a matching alias, then resolve the backing directory id.
        let target: LocalModel
        if let resolved = await modelResolver(requestedID) {
            target = resolved
        } else if let aliasID = await ModelParametersStore().modelID(forAlias: requestedID),
                  let aliasTarget = await modelResolver(aliasID) {
            target = aliasTarget
        } else {
            throw ModelSwapError.modelNotFound(id: requestedID)
        }

        // An alias may point at the model that is already loaded; skip
        // the swap in that case (the id-equality early return above only
        // catches a request naming the directory id directly).
        if await engineProvider()?.loadedModel?.id == target.id {
            return
        }

        // Kick off the swap as a Task we can store for concurrent
        // callers to await. Errors propagate via the Task's value.
        // When a loadHook is installed (GUI path), route through it
        // so observable state (currentModel, status, callbacks) stays
        // in sync. Otherwise fall back to raw engine calls against
        // whatever `engineProvider` currently resolves to (CLI path,
        // where the provider is a fixed single engine — SRV-1).
        // A2d (SRV-2 generalized to drain-after-swap): a resident batched cohort
        // must drain before the model is swapped out from under its live rows.
        // No-op when no seam is installed (default) — zero regression. This runs
        // under the FIFO generation lock held by `beginGeneration`, and the batched
        // path never takes that lock, so a swap waits for rows while rows never
        // wait for the swap: no deadlock.
        await batchServing?.drainForModelChange()

        let hook = loadHook
        let provider = engineProvider
        let swapTask = Task {
            if let hook {
                try await hook(target)
            } else if let engine = await provider() {
                try? await engine.unload()
                try await engine.load(target)
            } else {
                throw EngineError.modelNotLoaded
            }
        }
        loadInFlight = swapTask
        do {
            try await swapTask.value
            loadInFlight = nil
        } catch {
            loadInFlight = nil
            throw ModelSwapError.loadFailed(
                id: requestedID,
                reason: error.localizedDescription
            )
        }
    }

    /// Per-model chat-template kwargs (v0.5.1) configured for `modelID`,
    /// or nil when none are set. Populates `GenerateRequest.templateKwargs`
    /// on every generation path so the Jinja template receives them as
    /// `additionalContext`. Alias-aware (A5): a request that names the model
    /// by its user-facing alias resolves to the backing directory id before
    /// the per-model params are loaded, mirroring the swap resolver above.
    private func templateKwargs(for modelID: String) async -> [String: JSONValue]? {
        let store = ModelParametersStore()
        let resolvedID = await store.modelID(forAlias: modelID) ?? modelID
        let params = await store.load(for: resolvedID)
        guard let kwargs = params.templateKwargs, !kwargs.isEmpty else { return nil }
        return kwargs
    }

    private func nonStreamingChatResponse(genRequest: GenerateRequest) async throws -> Response {
        // Serialise + cold-swap atomically (SRV-2): acquire the lock, swap
        // under it, and re-resolve the active engine (SRV-1). Release on
        // ALL exit paths (success, catch, throw).
        let engine: any InferenceEngine
        do {
            engine = try await beginGeneration(genRequest.model)
        } catch let err as ModelSwapError {
            return openAIStyleSwapErrorResponse(err)
        } catch {
            return errorResponse(status: .internalServerError, message: error.localizedDescription, code: "load_failed")
        }
        defer { Task { [weak self] in await self?.releaseGenerationLock() } }

        // POOL-3: mark the active model in-flight so a concurrent load
        // can't LRU-evict it mid-generation.
        let modelID = await engine.loadedModel?.id ?? genRequest.model
        await markInFlight(modelID, true)
        defer { Task { [weak self] in await self?.markInFlight(modelID, false) } }

        // Hop into the engine actor to call generate, then iterate the returned stream.
        let stream = await engine.generate(genRequest)
        let box = ChunkIteratorBox(stream)
        var fullText = ""
        var finishReason = "stop"
        var promptTokens = 0
        var completionTokens = 0
        // Tool calls the model requested this turn (v0.5). Populated from the
        // terminal chunk when generation ends in `.toolCalls`; empty otherwise.
        var capturedToolCalls: [ToolCallRequest] = []
        // Speculative decoding telemetry (D1). Populated from the terminal
        // chunk only when the request actually ran the speculative path AND
        // mlx-swift-lm reported counts for it; nil otherwise (never faked).
        var capturedSpeculativeDecoding: SpeculativeDecodingUsage?
        // Track E: per-token logprobs accumulated in emission order across all
        // chunks, flattened into the choice's `logprobs.content[]` below. Only
        // populated when the request set `logprobs` (the engine attaches them).
        var capturedLogprobs: [TokenLogprob] = []

        do {
            loop: while true {
                switch try await nextGenerationStep(box, stallTimeout: stallTimeoutSeconds) {
                case .finished:
                    break loop
                case .stalled:
                    // SRV-4: no chunk for `stallTimeoutSeconds` — fail loudly
                    // instead of hanging the client forever (issue #29).
                    return errorResponse(
                        status: .gatewayTimeout,
                        message: "Generation stalled: no output for over \(Int(stallTimeoutSeconds))s.",
                        code: "generation_stalled"
                    )
                case .chunk(let chunk):
                    fullText += chunk.text
                    if let usage = chunk.usage {
                        promptTokens = usage.promptTokens
                        completionTokens = usage.completionTokens
                    }
                    if let reason = chunk.finishReason {
                        finishReason = reason.rawValue
                    }
                    if let calls = chunk.toolCalls, !calls.isEmpty {
                        capturedToolCalls = calls
                    }
                    if let speculativeDecoding = chunk.speculativeDecoding {
                        capturedSpeculativeDecoding = speculativeDecoding
                    }
                    if let logprobs = chunk.logprobs {
                        capturedLogprobs.append(contentsOf: logprobs)
                    }
                }
            }
        } catch {
            return errorResponse(
                status: .internalServerError,
                message: error.localizedDescription,
                code: "engine_error"
            )
        }

        incrementTokens(completionTokens)

        let completionID = "chatcmpl-\(UUID().uuidString)"
        let timestamp = Int(Date().timeIntervalSince1970)

        // Split reasoning into its own field (DeepSeek / mlx-lm / LM Studio
        // convention) so external agents can filter the chain-of-thought
        // instead of receiving bare `<think>…</think>` inside `content`
        // (issue #30). Non-reasoning models are untouched: `reasoning` is
        // nil and `content` is the full text.
        let (reasoning, answer) = MessageSegmenter.splitReasoning(fullText)
        var message: [String: Any] = ["role": "assistant", "content": answer]
        if let reasoning {
            message["reasoning_content"] = reasoning
        }
        // OpenAI tool-call turn: attach the detected calls as `message.tool_calls`
        // (each with `arguments` as a JSON-encoded string). `finishReason` is
        // already "tool_calls" (carried on the terminal chunk). Per the OpenAI
        // shape, `content` is null when the assistant only called tools and
        // produced no text; any text the model did emit is preserved.
        if !capturedToolCalls.isEmpty {
            message["tool_calls"] = capturedToolCalls.map { openAIToolCallObject($0) }
            if answer.isEmpty {
                message["content"] = NSNull()
            }
        }

        var usage: [String: Any] = [
            "prompt_tokens": promptTokens,
            "completion_tokens": completionTokens,
            "total_tokens": promptTokens + completionTokens,
        ]
        // Non-standard OpenAI extension (D1), mirroring mlx-lm's Python
        // server: accepted/proposed draft-token counts for this turn. Only
        // present when speculative decoding actually ran — see
        // `capturedSpeculativeDecoding`'s doc comment above.
        if let speculativeDecoding = capturedSpeculativeDecoding {
            usage["speculative_decoding"] = [
                "proposed_tokens": speculativeDecoding.proposedTokens,
                "accepted_tokens": speculativeDecoding.acceptedTokens,
            ] as [String: Any]
        }

        var choice: [String: Any] = [
            "index": 0,
            "message": message,
            "finish_reason": finishReason,
        ]
        // Track E: OpenAI `choices[].logprobs` — present (with a `content` array)
        // only when the request asked for logprobs.
        if genRequest.parameters.logprobs {
            choice["logprobs"] = openAILogprobs(capturedLogprobs)
        }

        let body: [String: Any] = [
            "id": completionID,
            "object": "chat.completion",
            "created": timestamp,
            "model": genRequest.model,
            "choices": [choice],
            "usage": usage,
        ]
        return try jsonResponseAny(body)
    }

    private func streamingChatResponse(genRequest: GenerateRequest) async throws -> Response {
        // A1: cheap pre-flight resolve check (no load, no lock) — reject
        // an unknown model with a real 404 before any streaming headers
        // are sent. The real (locked) resolve+load still happens inside
        // the ResponseBody closure below; SRV-2 atomicity is unchanged.
        let modelIsResolvable = await canResolveModel(genRequest.model)
        if !modelIsResolvable {
            return openAIStyleSwapErrorResponse(.modelNotFound(id: genRequest.model))
        }

        // Lock-scope invariant (SRV-2/SRV-3): acquire, cold-swap, and
        // generate ALL happen inside the writer closure below, in the same
        // scope as the `defer`-guaranteed release — see the invariant
        // comment above `acquireGenerationLock()`. Headers are sent
        // immediately (below); a cold-swap failure discovered inside the
        // closure is reported as an in-band SSE error frame since the HTTP
        // status can no longer change by that point.
        let completionID = "chatcmpl-\(UUID().uuidString)"
        let timestamp = Int(Date().timeIntervalSince1970)
        let model = genRequest.model
        // Track E: whether to emit `choices[].logprobs` on delta frames.
        let wantsLogprobs = genRequest.parameters.logprobs
        let server = self

        let responseBody = ResponseBody { writer in
            var completionTokens = 0
            // Track E: per-token logprobs not yet attached to an emitted frame (a
            // chunk fully buffered as a partial reasoning tag carries its logprobs
            // forward until the next frame flushes).
            var pendingStreamLogprobs: [TokenLogprob] = []

            let engine: any InferenceEngine
            do {
                engine = try await server.beginGeneration(model)
            } catch {
                let msg = error.localizedDescription.replacingOccurrences(of: "\"", with: "\\\"")
                var buf = ByteBuffer()
                buf.writeString("data: {\"error\":{\"message\":\"\(msg)\"}}\n\n")
                try? await writer.write(buf)
                var doneBuf = ByteBuffer()
                doneBuf.writeString("data: [DONE]\n\n")
                try? await writer.write(doneBuf)
                try? await writer.finish(nil)
                return
            }
            defer { Task { await server.releaseGenerationLock() } }

            // POOL-3: mark the active model in-flight so a concurrent load
            // can't LRU-evict it mid-generation.
            let modelID = await engine.loadedModel?.id ?? model
            await server.markInFlight(modelID, true)
            defer { Task { await server.markInFlight(modelID, false) } }

            // Seed the streaming reasoning splitter: does the rendered prompt
            // open a <think> block the model will continue (qwen3's template
            // does)? This decides whether the first streamed token is
            // reasoning even though the opening tag never appears in the
            // stream (issue #30).
            let startInReasoning = await engine.promptOpensThinkBlock(genRequest)
            let stream = await engine.generate(genRequest)
            let stallTimeout = await server.stallTimeoutSeconds
            var splitter = ReasoningStreamSplitter(startInReasoning: startInReasoning)
            let box = ChunkIteratorBox(stream)
            do {
                loop: while true {
                    switch try await nextGenerationStep(box, stallTimeout: stallTimeout) {
                    case .finished:
                        break loop
                    case .stalled:
                        // SRV-4: no chunk for `stallTimeoutSeconds` — emit an
                        // in-band SSE error then close (issue #29).
                        let errPayload = "data: {\"error\":{\"message\":\"Generation stalled: no output for over \(Int(stallTimeout))s.\",\"code\":\"generation_stalled\"}}\n\n"
                        var buf = ByteBuffer()
                        buf.writeString(errPayload)
                        try? await writer.write(buf)
                        break loop
                    case .chunk(let chunk):
                        // P3-4: accumulate the engine's real completion-token
                        // count (delivered on the terminal chunk's usage), not
                        // the number of SSE/NDJSON frames.
                        if let usage = chunk.usage { completionTokens = usage.completionTokens }
                        // Track E: accumulate this chunk's logprobs BEFORE any
                        // `continue` (a fully-buffered partial-tag chunk still
                        // carries logprobs for its tokens), attached to the next
                        // emitted frame below.
                        if wantsLogprobs, let logprobs = chunk.logprobs {
                            pendingStreamLogprobs.append(contentsOf: logprobs)
                        }
                        let (reasoning, answer) = splitter.push(chunk.text)
                        var delta: [String: Any] = [:]
                        if !reasoning.isEmpty { delta["reasoning_content"] = reasoning }
                        if !answer.isEmpty { delta["content"] = answer }
                        if chunk.finishReason != nil {
                            // Flush any buffered tail into this terminal chunk.
                            let (rTail, aTail) = splitter.finish()
                            let r = (delta["reasoning_content"] as? String ?? "") + rTail
                            let a = (delta["content"] as? String ?? "") + aTail
                            if !r.isEmpty { delta["reasoning_content"] = r }
                            if !a.isEmpty { delta["content"] = a }
                        } else if delta.isEmpty {
                            // Chunk fully buffered as a partial tag — nothing to emit yet.
                            continue
                        }
                        // OpenAI tool-call streaming: when the terminal chunk carries
                        // detected calls, split them across their own SSE frames —
                        // mirroring the reasoning/answer split above (distinct concerns,
                        // distinct frames). Order: [optional buffered text/reasoning
                        // delta] → a `delta.tool_calls` frame → a final frame whose delta
                        // is empty and which carries `finish_reason:"tool_calls"`. Each
                        // call is emitted complete in one delta (see openAIToolCallDelta),
                        // not per-fragment argument streaming.
                        if chunk.finishReason == .toolCalls, let calls = chunk.toolCalls, !calls.isEmpty {
                            var toolFrames: [[String: Any]] = []
                            if !delta.isEmpty {
                                toolFrames.append(["index": 0, "delta": delta])
                            }
                            let toolCallDeltas = calls.enumerated().map {
                                openAIToolCallDelta(index: $0.offset, call: $0.element)
                            }
                            toolFrames.append([
                                "index": 0,
                                "delta": ["tool_calls": toolCallDeltas] as [String: Any],
                            ])
                            toolFrames.append([
                                "index": 0,
                                "delta": [String: Any](),
                                "finish_reason": FinishReason.toolCalls.rawValue,
                            ])
                            // Track E: the terminal tool-call chunk still carries any
                            // logprobs accumulated since the last emitted frame. Attach
                            // them to the FIRST frame emitted on this branch (then clear),
                            // mirroring the non-streaming path — the plain-frame attach
                            // below is unreachable here because of the `continue`.
                            if wantsLogprobs, !pendingStreamLogprobs.isEmpty, !toolFrames.isEmpty {
                                toolFrames[0]["logprobs"] = openAILogprobs(pendingStreamLogprobs)
                                pendingStreamLogprobs.removeAll()
                            }
                            for frameChoice in toolFrames {
                                let payload: [String: Any] = [
                                    "id": completionID,
                                    "object": "chat.completion.chunk",
                                    "created": timestamp,
                                    "model": model,
                                    "choices": [frameChoice],
                                ]
                                let jsonData = try JSONSerialization.data(withJSONObject: payload)
                                let jsonStr = String(decoding: jsonData, as: UTF8.self)
                                var buf = ByteBuffer()
                                buf.writeString("data: \(jsonStr)\n\n")
                                try await writer.write(buf)
                            }
                            continue
                        }
                        var choice: [String: Any] = ["index": 0, "delta": delta]
                        if let reason = chunk.finishReason {
                            choice["finish_reason"] = reason.rawValue
                        }
                        // Track E: attach any logprobs accumulated since the last
                        // emitted frame (OpenAI `choices[].logprobs`), then clear.
                        if wantsLogprobs, !pendingStreamLogprobs.isEmpty {
                            choice["logprobs"] = openAILogprobs(pendingStreamLogprobs)
                            pendingStreamLogprobs.removeAll()
                        }
                        let payload: [String: Any] = [
                            "id": completionID,
                            "object": "chat.completion.chunk",
                            "created": timestamp,
                            "model": model,
                            "choices": [choice],
                        ]
                        let jsonData = try JSONSerialization.data(withJSONObject: payload)
                        let jsonStr = String(decoding: jsonData, as: UTF8.self)
                        let sseChunk = "data: \(jsonStr)\n\n"
                        var buf = ByteBuffer()
                        buf.writeString(sseChunk)
                        try await writer.write(buf)
                    }
                }
            } catch {
                let msg = error.localizedDescription
                    .replacingOccurrences(of: "\"", with: "\\\"")
                let errPayload = "data: {\"error\":{\"message\":\"\(msg)\"}}\n\n"
                var buf = ByteBuffer()
                buf.writeString(errPayload)
                try? await writer.write(buf)
            }

            var doneBuf = ByteBuffer()
            doneBuf.writeString("data: [DONE]\n\n")
            try await writer.write(doneBuf)
            try await writer.finish(nil)

            await server.incrementTokens(completionTokens)
        }

        var headers = HTTPFields()
        headers[.contentType] = "text/event-stream"
        headers[.cacheControl] = "no-cache"
        headers[.connection] = "keep-alive"

        return Response(
            status: .ok,
            headers: headers,
            body: responseBody
        )
    }

    // MARK: - A2d batched-path responders

    /// Batched-path counterpart to `nonStreamingChatResponse`. The stream is
    /// already resolved by the seam (`BatchGenerationServing.submit`), so this
    /// does NOT acquire the FIFO generation lock — the scheduler's admission is
    /// the concurrency control, and bypassing the lock is the whole point of
    /// batching. It reuses the same stall watchdog (`nextGenerationStep`),
    /// in-flight refcount (`markInFlight`, POOL-3), reasoning split, and JSON body
    /// shape as the single path. The batched stream never carries tool calls or
    /// speculative-decoding telemetry (out of the v1 batch scope), so those
    /// branches are intentionally absent.
    private func batchNonStreamingChatResponse(
        genRequest: GenerateRequest,
        modelID: String,
        stream: AsyncThrowingStream<GenerateChunk, Error>
    ) async throws -> Response {
        // POOL-3: the "true" mark already happened in `handleChatCompletions`
        // BEFORE `submit` was called (MEDIUM#2 — admission is the start of
        // decode, so marking here would be too late). Only the "false" mark
        // belongs to this responder, exactly once, on every exit path.
        defer { Task { [weak self] in await self?.markInFlight(modelID, false) } }

        let box = ChunkIteratorBox(stream)
        var fullText = ""
        var finishReason = "stop"
        var promptTokens = 0
        var completionTokens = 0
        do {
            loop: while true {
                switch try await nextGenerationStep(box, stallTimeout: stallTimeoutSeconds) {
                case .finished:
                    break loop
                case .stalled:
                    // SRV-4: a wedged slot trips its own watchdog and fails loudly
                    // (per slot — a stalled row never hangs its siblings).
                    return errorResponse(
                        status: .gatewayTimeout,
                        message: "Generation stalled: no output for over \(Int(stallTimeoutSeconds))s.",
                        code: "generation_stalled"
                    )
                case .chunk(let chunk):
                    fullText += chunk.text
                    if let usage = chunk.usage {
                        promptTokens = usage.promptTokens
                        completionTokens = usage.completionTokens
                    }
                    if let reason = chunk.finishReason {
                        finishReason = reason.rawValue
                    }
                }
            }
        } catch {
            return errorResponse(
                status: .internalServerError,
                message: error.localizedDescription,
                code: "engine_error"
            )
        }

        incrementTokens(completionTokens)

        let completionID = "chatcmpl-\(UUID().uuidString)"
        let timestamp = Int(Date().timeIntervalSince1970)
        let (reasoning, answer) = MessageSegmenter.splitReasoning(fullText)
        var message: [String: Any] = ["role": "assistant", "content": answer]
        if let reasoning {
            message["reasoning_content"] = reasoning
        }
        let usage: [String: Any] = [
            "prompt_tokens": promptTokens,
            "completion_tokens": completionTokens,
            "total_tokens": promptTokens + completionTokens,
        ]
        let body: [String: Any] = [
            "id": completionID,
            "object": "chat.completion",
            "created": timestamp,
            "model": genRequest.model,
            "choices": [
                [
                    "index": 0,
                    "message": message,
                    "finish_reason": finishReason,
                ] as [String: Any]
            ],
            "usage": usage,
        ]
        return try jsonResponseAny(body)
    }

    /// Batched-path counterpart to `streamingChatResponse`. Streams the
    /// seam-resolved batch stream as OpenAI SSE frames WITHOUT the FIFO generation
    /// lock (the scheduler admits concurrently), reusing the same stall watchdog,
    /// in-flight refcount, reasoning-stream splitter, and `[DONE]` framing. There
    /// is no cold-swap-under-lock (the seam only accepts a request against the
    /// resident model) and no tool-call framing (out of v1 batch scope).
    private func batchStreamingChatResponse(
        genRequest: GenerateRequest,
        modelID: String,
        stream: AsyncThrowingStream<GenerateChunk, Error>
    ) -> Response {
        let completionID = "chatcmpl-\(UUID().uuidString)"
        let timestamp = Int(Date().timeIntervalSince1970)
        let model = genRequest.model
        let server = self

        let responseBody = ResponseBody { writer in
            var completionTokens = 0

            // POOL-3: the "true" mark already happened in `handleChatCompletions`
            // BEFORE `submit` was called (MEDIUM#2). Only the "false" mark
            // belongs here, exactly once, regardless of how this closure exits.
            defer { Task { await server.markInFlight(modelID, false) } }

            // The batched path does not seed `startInReasoning` from the rendered
            // prompt (that needs a `container.prepare` on the resident model, which
            // must be serialised against the live cohort — a construction-wave
            // concern). In-stream <think> tags are still split correctly.
            let stallTimeout = await server.stallTimeoutSeconds
            var splitter = ReasoningStreamSplitter(startInReasoning: false)
            let box = ChunkIteratorBox(stream)
            do {
                loop: while true {
                    switch try await nextGenerationStep(box, stallTimeout: stallTimeout) {
                    case .finished:
                        break loop
                    case .stalled:
                        // SRV-4: no chunk for the stall timeout — emit an in-band
                        // SSE error then close, without disturbing sibling slots.
                        let errPayload = "data: {\"error\":{\"message\":\"Generation stalled: no output for over \(Int(stallTimeout))s.\",\"code\":\"generation_stalled\"}}\n\n"
                        var buf = ByteBuffer()
                        buf.writeString(errPayload)
                        try? await writer.write(buf)
                        break loop
                    case .chunk(let chunk):
                        if let usage = chunk.usage { completionTokens = usage.completionTokens }
                        let (reasoning, answer) = splitter.push(chunk.text)
                        var delta: [String: Any] = [:]
                        if !reasoning.isEmpty { delta["reasoning_content"] = reasoning }
                        if !answer.isEmpty { delta["content"] = answer }
                        if chunk.finishReason != nil {
                            // Flush any buffered tail into this terminal chunk.
                            let (rTail, aTail) = splitter.finish()
                            let r = (delta["reasoning_content"] as? String ?? "") + rTail
                            let a = (delta["content"] as? String ?? "") + aTail
                            if !r.isEmpty { delta["reasoning_content"] = r }
                            if !a.isEmpty { delta["content"] = a }
                        } else if delta.isEmpty {
                            // Chunk fully buffered as a partial tag — nothing yet.
                            continue
                        }
                        var choice: [String: Any] = ["index": 0, "delta": delta]
                        if let reason = chunk.finishReason {
                            choice["finish_reason"] = reason.rawValue
                        }
                        let payload: [String: Any] = [
                            "id": completionID,
                            "object": "chat.completion.chunk",
                            "created": timestamp,
                            "model": model,
                            "choices": [choice],
                        ]
                        let jsonData = try JSONSerialization.data(withJSONObject: payload)
                        let jsonStr = String(decoding: jsonData, as: UTF8.self)
                        var buf = ByteBuffer()
                        buf.writeString("data: \(jsonStr)\n\n")
                        try await writer.write(buf)
                    }
                }
            } catch {
                let msg = error.localizedDescription.replacingOccurrences(of: "\"", with: "\\\"")
                let errPayload = "data: {\"error\":{\"message\":\"\(msg)\"}}\n\n"
                var buf = ByteBuffer()
                buf.writeString(errPayload)
                try? await writer.write(buf)
            }

            var doneBuf = ByteBuffer()
            doneBuf.writeString("data: [DONE]\n\n")
            try await writer.write(doneBuf)
            try await writer.finish(nil)

            await server.incrementTokens(completionTokens)
        }

        var headers = HTTPFields()
        headers[.contentType] = "text/event-stream"
        headers[.cacheControl] = "no-cache"
        headers[.connection] = "keep-alive"

        return Response(status: .ok, headers: headers, body: responseBody)
    }

    // MARK: - Anthropic Messages API compatibility

    /// Anthropic `POST /v1/messages`. Decodes the Anthropic wire format,
    /// maps it onto the same `GenerateRequest` the OpenAI path uses, runs
    /// the shared cold-swap, and dispatches to the Anthropic streaming or
    /// non-streaming responder. `system` is top-level (not a message turn)
    /// and `max_tokens` is required by Anthropic's spec.
    private func handleAnthropicMessages(
        request: Request,
        context: BasicRequestContext
    ) async throws -> Response {
        let buffer = try await request.body.collect(upTo: context.maxUploadSize)
        guard let data = buffer.getData(at: buffer.readerIndex, length: buffer.readableBytes) else {
            return anthropicErrorResponse(
                status: .badRequest,
                message: "Empty request body",
                type: "invalid_request_error"
            )
        }

        let req: AnthropicMessagesRequest
        do {
            req = try JSONDecoder().decode(AnthropicMessagesRequest.self, from: data)
        } catch {
            return anthropicErrorResponse(
                status: .badRequest,
                message: "Invalid JSON: \(error.localizedDescription)",
                type: "invalid_request_error"
            )
        }

        // Reject unsupported `tool_choice` modes explicitly — `any`/`tool` force
        // a call (which we can't guarantee) and `none` suppresses tools; rather
        // than silently ignore the directive we 400 (no silent degradation).
        if let choice = req.tool_choice, choice.type != "auto" {
            return anthropicErrorResponse(
                status: .badRequest,
                message: "Unsupported `tool_choice.type` '\(choice.type)': only 'auto' "
                    + "(or omitting `tool_choice`) is supported.",
                type: "invalid_request_error"
            )
        }

        // Anthropic carries the system prompt at the top level (not as a
        // message turn), so there's nothing to strip out of `messages` —
        // GenerateRequest.allMessages re-prepends this systemPrompt.
        let systemPrompt = req.system?.text

        // Map turns. Anthropic roles are only user / assistant; unknown roles are
        // dropped. Text blocks → `content`, image blocks → `images`. Tool blocks
        // are fully wired (agent-tools wave): assistant `tool_use` →
        // `ChatMessage.toolCalls`; each user `tool_result` → a `.tool` message
        // carrying `tool_use_id` (emitted before any residual user text so the
        // chat template's call ↔ result pairing holds). A malformed tool block is
        // a 400 (never a silent drop).
        let messages: [ChatMessage]
        do {
            messages = try decodeAnthropicMessages(req.messages)
        } catch let error as ToolHistoryDecodeError {
            return anthropicErrorResponse(
                status: .badRequest,
                message: error.message,
                type: "invalid_request_error"
            )
        }

        // Convert Anthropic tools to the same internal OpenAI-function-spec shape
        // the OpenAI path forwards, so both protocols feed the chat template
        // identically. Absent ⇒ nil (byte-for-byte unchanged for no-tools calls).
        let tools: [JSONValue]? = req.tools.map { $0.map { openAIToolSpec(fromAnthropic: $0) } }

        let params = GenerationParameters(
            temperature: req.temperature ?? 0.7,
            topP: req.top_p ?? 0.95,
            maxTokens: req.max_tokens,
            stream: req.stream ?? false
        )

        let genRequest = GenerateRequest(
            model: req.model,
            messages: messages,
            systemPrompt: systemPrompt,
            parameters: params,
            templateKwargs: await templateKwargs(for: req.model),
            tools: tools
        )

        // Cold-swap (SRV-2): resolved inside each responder, atomically with
        // the generation lock — see `beginGeneration`. Missing model → 404,
        // load failure → 500, both mapped by the responder itself.
        let wantsStream = req.stream ?? false
        if wantsStream {
            return try await anthropicStreamingResponse(genRequest: genRequest)
        } else {
            return try await anthropicNonStreamingResponse(genRequest: genRequest)
        }
    }

    /// Non-streaming Anthropic `/v1/messages` response. Accumulates the
    /// full generation, splits off reasoning (dropped in this MVP — only
    /// the answer is surfaced in `content[0]`), and emits Anthropic's
    /// message envelope.
    private func anthropicNonStreamingResponse(genRequest: GenerateRequest) async throws -> Response {
        // Serialise + cold-swap atomically (SRV-2): acquire the lock, swap
        // under it, and re-resolve the active engine (SRV-1). Release on
        // ALL exit paths (success, catch, throw).
        let engine: any InferenceEngine
        do {
            engine = try await beginGeneration(genRequest.model)
        } catch let err as ModelSwapError {
            return openAIStyleSwapErrorResponse(err)
        } catch {
            return anthropicErrorResponse(
                status: .internalServerError, message: error.localizedDescription, type: "api_error")
        }
        defer { Task { [weak self] in await self?.releaseGenerationLock() } }

        let modelID = await engine.loadedModel?.id ?? genRequest.model
        await markInFlight(modelID, true)
        defer { Task { [weak self] in await self?.markInFlight(modelID, false) } }

        let stream = await engine.generate(genRequest)
        let box = ChunkIteratorBox(stream)
        var fullText = ""
        var finishReason: FinishReason?
        var promptTokens = 0
        var completionTokens = 0
        // Tool calls the model requested this turn. Populated from the terminal
        // chunk when generation ends in `.toolCalls`; empty otherwise.
        var capturedToolCalls: [ToolCallRequest] = []

        do {
            loop: while true {
                switch try await nextGenerationStep(box, stallTimeout: stallTimeoutSeconds) {
                case .finished:
                    break loop
                case .stalled:
                    return anthropicErrorResponse(
                        status: .gatewayTimeout,
                        message: "Generation stalled: no output for over \(Int(stallTimeoutSeconds))s.",
                        type: "api_error"
                    )
                case .chunk(let chunk):
                    fullText += chunk.text
                    if let usage = chunk.usage {
                        promptTokens = usage.promptTokens
                        completionTokens = usage.completionTokens
                    }
                    if let reason = chunk.finishReason {
                        finishReason = reason
                    }
                    if let calls = chunk.toolCalls, !calls.isEmpty {
                        capturedToolCalls = calls
                    }
                }
            }
        } catch {
            return anthropicErrorResponse(
                status: .internalServerError,
                message: error.localizedDescription,
                type: "api_error"
            )
        }

        incrementTokens(completionTokens)

        // Reasoning is not surfaced as a separate block in this MVP — only
        // the answer text goes into content[0]. `splitReasoning` strips any
        // `<think>…</think>`; non-reasoning models keep their full text.
        let (_, answer) = MessageSegmenter.splitReasoning(fullText)

        // Content: the text block first — UNLESS the assistant only called
        // tools and produced no text, in which case real Anthropic omits the
        // empty text block entirely (a tool-only reply is the common case in
        // an agent loop) — then one `tool_use` block per detected call
        // (`input` is a decoded JSON object, not a string). With no tool calls
        // the single text block is always present, byte-for-byte unchanged
        // from the pre-tools response, and `stop_reason` stays `end_turn`;
        // `.toolCalls` maps to `tool_use` via `anthropicStopReason`.
        var content: [[String: Any]] = []
        if !answer.isEmpty || capturedToolCalls.isEmpty {
            content.append(["type": "text", "text": answer])
        }
        for call in capturedToolCalls {
            content.append(anthropicToolUseBlock(call))
        }

        let body: [String: Any] = [
            "id": "msg_\(UUID().uuidString)",
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": genRequest.model,
            "stop_reason": anthropicStopReason(finishReason),
            "stop_sequence": NSNull(),
            "usage": [
                "input_tokens": promptTokens,
                "output_tokens": completionTokens,
            ] as [String: Any],
        ]
        return try jsonResponseAny(body)
    }

    /// Streaming Anthropic `/v1/messages` — SSE with *named* events
    /// (`event: <name>\ndata: <json>\n\n`) in Anthropic's fixed order:
    /// message_start → content_block_start → content_block_delta* →
    /// content_block_stop → [tool_use content_block_start/delta/stop]* →
    /// message_delta → message_stop. Only the answer portion is streamed as
    /// `text_delta`; reasoning is dropped for this MVP (matching the
    /// non-streaming path). The index-0 text block opens LAZILY on the first
    /// non-empty answer delta and is entirely omitted for a tool-only reply
    /// (no free text at all) — matching real Anthropic, which never emits an
    /// empty text block — in which case the `tool_use` block(s) open at index
    /// 0 instead of 1; see `textBlockOpened` below. The generation lock is
    /// held for the whole body and released in `defer`.
    private func anthropicStreamingResponse(genRequest: GenerateRequest) async throws -> Response {
        // A1: cheap pre-flight resolve check (no load, no lock) — reject
        // an unknown model with a real 404 before any streaming headers
        // are sent. The real (locked) resolve+load still happens inside
        // the ResponseBody closure below; SRV-2 atomicity is unchanged.
        let modelIsResolvable = await canResolveModel(genRequest.model)
        if !modelIsResolvable {
            return openAIStyleSwapErrorResponse(.modelNotFound(id: genRequest.model))
        }

        // Lock-scope invariant (SRV-2/SRV-3): acquire, cold-swap, and
        // generate ALL happen inside the writer closure below.
        let messageID = "msg_\(UUID().uuidString)"
        let model = genRequest.model
        let server = self

        let responseBody = ResponseBody { writer in
            var completionTokens = 0
            var promptTokens = 0
            var finishReason: FinishReason?
            // Tool calls the model requested this turn — emitted as `tool_use`
            // content blocks after the text block closes. Empty unless the
            // terminal chunk ends in `.toolCalls`.
            var capturedToolCalls: [ToolCallRequest] = []
            // Whether the index-0 text content_block has been opened yet. Opening
            // is LAZY (on the first non-empty answer delta) rather than eager, so
            // a tool-only reply (no free text at all) never opens one — real
            // Anthropic omits the empty text block in that case. See the
            // pre-loop-exit handling below for the "opened at all?" resolution.
            var textBlockOpened = false

            let engine: any InferenceEngine
            do {
                engine = try await server.beginGeneration(model)
            } catch {
                let msg = error.localizedDescription.replacingOccurrences(of: "\"", with: "\\\"")
                var buf = ByteBuffer()
                buf.writeString("event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"\(msg)\"}}\n\n")
                try? await writer.write(buf)
                try? await writer.finish(nil)
                return
            }
            defer { Task { await server.releaseGenerationLock() } }

            let modelID = await engine.loadedModel?.id ?? model
            await server.markInFlight(modelID, true)
            defer { Task { await server.markInFlight(modelID, false) } }

            // Seed the reasoning splitter — does the rendered prompt open a
            // <think> block the model continues (qwen3)? Reasoning is dropped
            // for the Anthropic MVP, so this only affects which text counts as
            // the answer, never a separate surfaced block.
            let startInReasoning = await engine.promptOpensThinkBlock(genRequest)
            let stream = await engine.generate(genRequest)
            let stallTimeout = await server.stallTimeoutSeconds
            var splitter = ReasoningStreamSplitter(startInReasoning: startInReasoning)
            let box = ChunkIteratorBox(stream)

            do {
                // 1. message_start
                // Wire-format limitation (P3-3): real Anthropic reports the true
                // prompt token count in `message_start.usage.input_tokens`
                // because it tokenises the prompt up front. Our MLX engine only
                // surfaces prompt/completion counts on the TERMINAL generation
                // chunk (`GenerateChunk.usage`), which arrives after this event
                // must already be on the wire — Anthropic's fixed event order
                // requires `message_start` first, before any token flows.
                // Deferring `message_start` wouldn't help (the count is still
                // unknown at the first token), so we send 0 here and deliver the
                // real counts in the closing `message_delta.usage` below.
                try await writeAnthropicSSE(&writer, event: "message_start", payload: [
                    "type": "message_start",
                    "message": [
                        "id": messageID,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": model,
                        "stop_reason": NSNull(),
                        "stop_sequence": NSNull(),
                        "usage": [
                            "input_tokens": 0,
                            "output_tokens": 0,
                        ] as [String: Any],
                    ] as [String: Any],
                ])

                // 2/3. content_block_start for the text block (index 0), opened
                // LAZILY on the first non-empty answer delta — see
                // `textBlockOpened`'s doc comment — then a content_block_delta per
                // subsequent delta.
                stallLoop: while true {
                    switch try await nextGenerationStep(box, stallTimeout: stallTimeout) {
                    case .finished:
                        break stallLoop
                    case .stalled:
                        // SRV-4: no chunk for `stallTimeoutSeconds` — emit an
                        // in-band error event then close (issue #29).
                        var buf = ByteBuffer()
                        buf.writeString("event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"Generation stalled: no output for over \(Int(stallTimeout))s.\"}}\n\n")
                        try? await writer.write(buf)
                        break stallLoop
                    case .chunk(let chunk):
                        if let usage = chunk.usage {
                            completionTokens = usage.completionTokens
                            promptTokens = usage.promptTokens
                        }
                        let (_, answer) = splitter.push(chunk.text)
                        var text = answer
                        if let reason = chunk.finishReason {
                            finishReason = reason
                            // Flush any buffered tail into this terminal chunk.
                            let (_, aTail) = splitter.finish()
                            text += aTail
                        }
                        // Capture tool calls BEFORE the empty-text guard below: a
                        // tool-only terminal chunk carries no answer text but must
                        // still surface its `tool_use` blocks.
                        if let calls = chunk.toolCalls, !calls.isEmpty {
                            capturedToolCalls = calls
                        }
                        guard !text.isEmpty else { continue }
                        if !textBlockOpened {
                            try await writeAnthropicSSE(&writer, event: "content_block_start", payload: [
                                "type": "content_block_start",
                                "index": 0,
                                "content_block": [
                                    "type": "text",
                                    "text": "",
                                ] as [String: Any],
                            ])
                            textBlockOpened = true
                        }
                        try await writeAnthropicSSE(&writer, event: "content_block_delta", payload: [
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": [
                                "type": "text_delta",
                                "text": text,
                            ] as [String: Any],
                        ])
                    }
                }
            } catch {
                let msg = error.localizedDescription
                    .replacingOccurrences(of: "\"", with: "\\\"")
                var buf = ByteBuffer()
                buf.writeString("event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"\(msg)\"}}\n\n")
                try? await writer.write(buf)
            }

            // 4. content_block_stop for the text block. If it was never opened
            // (no non-empty delta arrived) AND there are no tool calls, still
            // open+close an empty one here — matching the pre-tools byte-for-byte
            // invariant for a content-less response. A tool-only reply (no text,
            // ≥1 tool call) skips the text block entirely; its tool_use block(s)
            // below open at index 0 instead of 1.
            if !textBlockOpened && capturedToolCalls.isEmpty {
                try? await writeAnthropicSSE(&writer, event: "content_block_start", payload: [
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": ["type": "text", "text": ""] as [String: Any],
                ])
                textBlockOpened = true
            }
            if textBlockOpened {
                try? await writeAnthropicSSE(&writer, event: "content_block_stop", payload: [
                    "type": "content_block_stop",
                    "index": 0,
                ])
            }

            // 4b. Tool-use blocks follow the text block (index 1…N) when one was
            // opened, or start at index 0 when the reply was tool-only (see
            // above). Detection is terminal, so each call is emitted whole: a
            // content_block_start (empty `input`), ONE `input_json_delta` frame
            // carrying the full arguments JSON, then a content_block_stop. No
            // tool calls ⇒ this loop is skipped and the event stream is
            // byte-for-byte unchanged.
            let toolBlockBaseIndex = textBlockOpened ? 1 : 0
            for (offset, call) in capturedToolCalls.enumerated() {
                let index = toolBlockBaseIndex + offset
                try? await writeAnthropicSSE(&writer, event: "content_block_start", payload: [
                    "type": "content_block_start",
                    "index": index,
                    "content_block": [
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": [String: Any](),
                    ] as [String: Any],
                ])
                try? await writeAnthropicSSE(&writer, event: "content_block_delta", payload: [
                    "type": "content_block_delta",
                    "index": index,
                    "delta": [
                        "type": "input_json_delta",
                        "partial_json": openAIToolArgumentsJSON(call.arguments),
                    ] as [String: Any],
                ])
                try? await writeAnthropicSSE(&writer, event: "content_block_stop", payload: [
                    "type": "content_block_stop",
                    "index": index,
                ])
            }

            // 5. message_delta — final stop reason + output token count.
            try? await writeAnthropicSSE(&writer, event: "message_delta", payload: [
                "type": "message_delta",
                "delta": [
                    "stop_reason": anthropicStopReason(finishReason),
                    "stop_sequence": NSNull(),
                ] as [String: Any],
                "usage": [
                    "input_tokens": promptTokens,
                    "output_tokens": completionTokens,
                ] as [String: Any],
            ])

            // 6. message_stop
            try? await writeAnthropicSSE(&writer, event: "message_stop", payload: [
                "type": "message_stop",
            ])
            try await writer.finish(nil)

            await server.incrementTokens(completionTokens)
        }

        var headers = HTTPFields()
        headers[.contentType] = "text/event-stream"
        headers[.cacheControl] = "no-cache"
        headers[.connection] = "keep-alive"

        return Response(
            status: .ok,
            headers: headers,
            body: responseBody
        )
    }

    private func handleLoadModel(
        request: Request,
        context: BasicRequestContext
    ) async throws -> Response {
        let buffer = try await request.body.collect(upTo: context.maxUploadSize)
        guard let data = buffer.getData(at: buffer.readerIndex, length: buffer.readableBytes) else {
            return errorResponse(
                status: .badRequest,
                message: "Empty body",
                code: "invalid_request_error"
            )
        }

        let loadReq: LoadModelRequest
        do {
            loadReq = try JSONDecoder().decode(LoadModelRequest.self, from: data)
        } catch {
            return errorResponse(
                status: .badRequest,
                message: "Invalid JSON: \(error.localizedDescription)",
                code: "invalid_request_error"
            )
        }

        let modelPath = loadReq.model_path
        let modelID = URL(fileURLWithPath: modelPath).lastPathComponent
        let model = LocalModel(
            id: modelID,
            displayName: modelID,
            directory: URL(fileURLWithPath: modelPath),
            sizeBytes: 0,
            format: .mlx,
            quantization: nil,
            parameterCount: nil,
            architecture: nil
        )

        guard let engine = await engineProvider() else {
            return errorResponse(
                status: .internalServerError,
                message: "No active engine to load into",
                code: "load_failed"
            )
        }

        // A2: take the generation lock around the mutating load — the same
        // swap-vs-generate hazard SRV-2 fixed for the generation endpoints,
        // reached here through a different door (`/x/models/load`). A throw
        // from acquire means the task was cancelled while parked → the lock
        // is NOT held, so we must not release on that path (mirrors the
        // embeddings/rerank handlers). Intended consequence: a manual load
        // now waits for an in-flight generation to finish.
        do {
            try await acquireGenerationLock()
        } catch {
            return errorResponse(
                status: .internalServerError,
                message: "Cancelled while waiting for the generation lock",
                code: "cancelled"
            )
        }
        // A2d: drain any resident batched cohort before the mutating load — the
        // same drain-before-model-change invariant the generation cold-swap uses,
        // reached here through the manual `/x/models/load` door. No-op without a
        // seam (default), so zero regression.
        await batchServing?.drainForModelChange()
        // Measure the load itself, not the time spent waiting for the lock.
        let start = Date()
        do {
            try await engine.load(model)
            releaseGenerationLock()
        } catch {
            releaseGenerationLock()
            return errorResponse(
                status: .internalServerError,
                message: error.localizedDescription,
                code: "load_failed"
            )
        }
        let loadTimeMs = Int(Date().timeIntervalSince(start) * 1000)

        return try jsonResponseAny([
            "status": "loaded",
            "model": modelID,
            "load_time_ms": loadTimeMs,
        ])
    }

    private func handleUnloadModel(
        request: Request,
        context: BasicRequestContext
    ) async throws -> Response {
        guard let engine = await engineProvider() else {
            return try jsonResponse(["status": "unloaded"])
        }

        // A2: same lock discipline as handleLoadModel — an unload mutates
        // shared engine/MLX state and must not race an in-flight generation.
        // Throw from acquire = lock NOT held = do not release (embeddings pattern).
        do {
            try await acquireGenerationLock()
        } catch {
            return errorResponse(
                status: .internalServerError,
                message: "Cancelled while waiting for the generation lock",
                code: "cancelled"
            )
        }
        // A2d: drain any resident batched cohort before the unload frees the
        // model out from under its live rows (the manual `/x/models/unload`
        // door). No-op without a seam (default), so zero regression.
        await batchServing?.drainForModelChange()
        do {
            try await engine.unload()
            releaseGenerationLock()
        } catch {
            releaseGenerationLock()
            return errorResponse(
                status: .internalServerError,
                message: error.localizedDescription,
                code: "unload_failed"
            )
        }
        return try jsonResponse(["status": "unloaded"])
    }

    // MARK: - Embeddings / rerank (v0.5.2)

    /// Thrown by `ensureEmbedderLoaded` when a request resolves to a real
    /// model that isn't an embedder (v0.5.2 follow-up). Kept distinct from
    /// `ModelSwapError` so the generation cold-swap's error mapping stays
    /// untouched; `ensureEmbedderOr404` maps it to a 400 `model_not_embedder`.
    private enum EmbedderKindError: Error {
        case notAnEmbedder(id: String, format: String)
    }

    /// Make sure `embeddingEngine` has `requestedID` resident. Resolves the
    /// model the same way `ensureModelLoaded` does (direct id/displayName,
    /// then user-facing alias), creating + loading a fresh `EmbeddingEngine`
    /// on a cold miss and swapping when a different embedder is requested.
    /// Throws `.modelNotFound` when the id isn't on disk, `.loadFailed` when
    /// the load itself fails, and `EmbedderKindError.notAnEmbedder` when the
    /// resolved model isn't an embedder (so /v1/embeddings + /v1/rerank can't
    /// be pointed at a chat model and return meaningless vectors).
    /// Resolve a request's `model` id to a `LocalModel` the same way the
    /// generation cold-swap does: direct id/displayName first, then a
    /// user-facing alias. Returns `nil` when nothing on disk matches. Shared
    /// by the embedder kind-gate and the rerank format router so both resolve
    /// identically.
    private func resolveModel(_ requestedID: String) async -> LocalModel? {
        if let resolved = await modelResolver(requestedID) {
            return resolved
        }
        if let aliasID = await ModelParametersStore().modelID(forAlias: requestedID),
           let aliasTarget = await modelResolver(aliasID) {
            return aliasTarget
        }
        return nil
    }

    private func ensureEmbedderLoaded(_ requestedID: String) async throws {
        // Same resolution order as the generation cold-swap.
        guard let target = await resolveModel(requestedID) else {
            throw ModelSwapError.modelNotFound(id: requestedID)
        }

        // P3-8: gate on model kind. A chat/VLM model resolves fine but the
        // embedder would produce meaningless vectors — reject it up front with
        // a clear 400 (mapped in `ensureEmbedderOr404`) rather than embedding
        // against the wrong model. `.embedder` is set by
        // `ModelLibraryManager.upgradeFormat` after the file-listing scan.
        guard target.format == .embedder else {
            throw EmbedderKindError.notAnEmbedder(id: requestedID, format: target.format.rawValue)
        }

        if let current = embeddingEngine, await current.loadedModel?.id == target.id {
            return
        }

        let newEngine = EmbeddingEngine()
        do {
            try await newEngine.load(target)
        } catch {
            throw ModelSwapError.loadFailed(id: requestedID, reason: error.localizedDescription)
        }
        embeddingEngine = newEngine
    }

    /// Make sure `rerankEngine` has `target` resident, cold-swapping when a
    /// different reranker is requested. A near-clone of `ensureEmbedderLoaded`,
    /// differing only in the engine type — and in that `target` arrives
    /// PRE-RESOLVED and already known `.reranker` (the format routing lives in
    /// `handleRerank`, so there's no inline kind-gate here). Throws
    /// `ModelSwapError.loadFailed` when the load itself fails (e.g. a
    /// weight-key mismatch under `verify: [.all]`, or a missing config).
    private func ensureRerankerLoaded(_ target: LocalModel) async throws {
        if let current = rerankEngine, await current.loadedModel?.id == target.id {
            return
        }
        let newEngine = RerankEngine()
        do {
            try await newEngine.load(target)
        } catch {
            throw ModelSwapError.loadFailed(id: target.id, reason: error.localizedDescription)
        }
        rerankEngine = newEngine
    }

    /// `POST /v1/embeddings` — OpenAI-compatible text embeddings. Cold-swaps
    /// the embedder to `req.model`, embeds the `input` (string or array), and
    /// returns `{ object:"list", data:[{object:"embedding", embedding, index}],
    /// model, usage }`.
    private func handleEmbeddings(
        request: Request,
        context: BasicRequestContext
    ) async throws -> Response {
        let buffer = try await request.body.collect(upTo: context.maxUploadSize)
        guard let data = buffer.getData(at: buffer.readerIndex, length: buffer.readableBytes) else {
            return errorResponse(
                status: .badRequest,
                message: "Empty body",
                code: "invalid_request_error"
            )
        }

        let req: EmbeddingsRequest
        do {
            req = try JSONDecoder().decode(EmbeddingsRequest.self, from: data)
        } catch {
            return errorResponse(
                status: .badRequest,
                message: "Invalid JSON: \(error.localizedDescription)",
                code: "invalid_request_error"
            )
        }

        if let failure = await ensureEmbedderOr404(req.model) {
            return failure
        }
        guard let embedder = embeddingEngine else {
            return errorResponse(
                status: .internalServerError,
                message: "Embedder not loaded",
                code: "load_failed"
            )
        }

        // Serialise MLX compute with generation — both touch global MLX
        // allocator state that isn't safe to share concurrently. The acquire
        // throws only on cancellation (v0.5.3 cancellation-aware waiters);
        // a throw means the lock is NOT held, so no release on that path.
        do {
            try await acquireGenerationLock()
        } catch {
            return errorResponse(
                status: .internalServerError,
                message: "Cancelled while waiting for the generation lock",
                code: "cancelled"
            )
        }
        let vectors: [[Float]]
        do {
            vectors = try await embedder.embed(req.input.values)
            releaseGenerationLock()
        } catch {
            releaseGenerationLock()
            return errorResponse(
                status: .internalServerError,
                message: "Embedding failed: \(error.localizedDescription)",
                code: "embedding_failed"
            )
        }

        let dataArray: [[String: Any]] = vectors.enumerated().map { index, vector in
            [
                "object": "embedding",
                "embedding": vector.map { Double($0) },
                "index": index,
            ]
        }
        // Token usage isn't tracked for embeddings in this MVP (a precise
        // count would need to be threaded out of `EmbeddingEngine.embed`).
        return try jsonResponseAny([
            "object": "list",
            "data": dataArray,
            "model": req.model,
            "usage": [
                "prompt_tokens": 0,
                "total_tokens": 0,
            ] as [String: Any],
        ])
    }

    /// `POST /v1/rerank` — ranks `documents` against `query`, routed by the
    /// resolved model's kind:
    /// - `.reranker` → a TRUE cross-encoder (`RerankEngine`) that scores each
    ///   `[query, document]` pair jointly (one forward pass over both spans).
    /// - `.embedder` → the bi-encoder cosine fallback (embed independently,
    ///   compare vectors — the documented approximation, kept for no-regression).
    /// - anything else → 400; unknown id → 404.
    ///
    /// Returns `{ results:[{index, relevance_score[, document]}], model }`,
    /// ordered by descending relevance. `relevance_score` is `sigmoid(logit)`
    /// (bounded 0..1) for the cross-encoder and raw cosine for the fallback;
    /// ranking is by the raw score in both, so the sigmoid never reorders.
    private func handleRerank(
        request: Request,
        context: BasicRequestContext
    ) async throws -> Response {
        let buffer = try await request.body.collect(upTo: context.maxUploadSize)
        guard let data = buffer.getData(at: buffer.readerIndex, length: buffer.readableBytes) else {
            return errorResponse(
                status: .badRequest,
                message: "Empty body",
                code: "invalid_request_error"
            )
        }

        let req: RerankRequest
        do {
            req = try JSONDecoder().decode(RerankRequest.self, from: data)
        } catch {
            return errorResponse(
                status: .badRequest,
                message: "Invalid JSON: \(error.localizedDescription)",
                code: "invalid_request_error"
            )
        }

        // Route by model kind. Resolve once here so the reranker branch can be
        // chosen BEFORE the embedder kind-gate (which would otherwise 400 a
        // reranker as "not an embedder").
        guard let target = await resolveModel(req.model) else {
            return errorResponse(
                status: .notFound,
                message: "Model not found: \(req.model). Download a reranker "
                    + "(e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) or an embedder "
                    + "(e.g. `bge-small-en-v1.5`) and check `macmlx list`.",
                code: "model_not_found"
            )
        }
        if target.format == .reranker {
            return try await rerankWithCrossEncoder(req: req, target: target)
        }
        guard target.format == .embedder else {
            return errorResponse(
                status: .badRequest,
                message: "Model \(req.model) is not a reranker or embedding model "
                    + "(kind: \(target.format.rawValue)). Use a cross-encoder reranker "
                    + "(e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) or an embedder for /v1/rerank.",
                code: "model_not_embedder"
            )
        }

        // `.embedder` → bi-encoder cosine fallback (documented approximation).
        // Reuse the proven embedder cold-swap; `target` is already `.embedder`
        // so this only loads (or reports a load failure), never 404s/400s here.
        if let failure = await ensureEmbedderOr404(req.model) {
            return failure
        }
        guard let embedder = embeddingEngine else {
            return errorResponse(
                status: .internalServerError,
                message: "Embedder not loaded",
                code: "load_failed"
            )
        }

        // Same lock discipline as /v1/embeddings: throw = not held = no release.
        do {
            try await acquireGenerationLock()
        } catch {
            return errorResponse(
                status: .internalServerError,
                message: "Cancelled while waiting for the generation lock",
                code: "cancelled"
            )
        }
        let embeddings: [[Float]]
        do {
            embeddings = try await embedder.embed([req.query] + req.documents)
            releaseGenerationLock()
        } catch {
            releaseGenerationLock()
            return errorResponse(
                status: .internalServerError,
                message: "Embedding failed: \(error.localizedDescription)",
                code: "embedding_failed"
            )
        }

        guard let queryVector = embeddings.first else {
            return errorResponse(
                status: .internalServerError,
                message: "Embedder returned no vectors",
                code: "embedding_failed"
            )
        }
        let documentVectors = Array(embeddings.dropFirst())
        let ranked = rerankByCosine(
            query: queryVector,
            documents: documentVectors,
            topN: req.top_n
        )
        // Cosine similarity is already the exposed relevance score (unchanged
        // from the MVP) — identity transform, plus optional echoed documents.
        let results = Self.rerankResults(
            ranked: ranked, documents: req.documents,
            returnDocuments: req.return_documents == true,
            scoreTransform: { Double($0) }
        )
        return try jsonResponseAny([
            "results": Self.rerankResultsJSON(results),
            "model": req.model,
        ])
    }

    /// The TRUE cross-encoder branch of `/v1/rerank` — cold-swaps the
    /// `.reranker` model resident, scores every `[query, doc]` pair jointly,
    /// then ranks by raw logit and exposes `sigmoid(logit)` as
    /// `relevance_score`. Mirrors the cosine branch's lock discipline
    /// (acquire → score → release; a throw on acquire means the lock is not
    /// held, so no release on that path).
    private func rerankWithCrossEncoder(
        req: RerankRequest, target: LocalModel
    ) async throws -> Response {
        do {
            try await ensureRerankerLoaded(target)
        } catch let err as ModelSwapError {
            if case .loadFailed(let id, let reason) = err {
                return errorResponse(
                    status: .internalServerError,
                    message: "Failed to load \(id): \(reason)",
                    code: "load_failed"
                )
            }
            return errorResponse(
                status: .internalServerError, message: err.localizedDescription, code: "load_failed"
            )
        } catch {
            return errorResponse(
                status: .internalServerError, message: error.localizedDescription, code: "load_failed"
            )
        }
        guard let reranker = rerankEngine else {
            return errorResponse(
                status: .internalServerError, message: "Reranker not loaded", code: "load_failed"
            )
        }

        do {
            try await acquireGenerationLock()
        } catch {
            return errorResponse(
                status: .internalServerError,
                message: "Cancelled while waiting for the generation lock",
                code: "cancelled"
            )
        }
        let scores: [Float]
        do {
            scores = try await reranker.score(query: req.query, documents: req.documents)
            releaseGenerationLock()
        } catch {
            releaseGenerationLock()
            return errorResponse(
                status: .internalServerError,
                message: "Rerank failed: \(error.localizedDescription)",
                code: "rerank_failed"
            )
        }

        let ranked = rankAndTruncate(scores: scores, topN: req.top_n)
        let results = Self.rerankResults(
            ranked: ranked, documents: req.documents,
            returnDocuments: req.return_documents == true,
            scoreTransform: rerankSigmoid
        )
        return try jsonResponseAny([
            "results": Self.rerankResultsJSON(results),
            "model": req.model,
        ])
    }

    /// A single `/v1/rerank` result: rank-ordered document `index`, its
    /// exposed `relevanceScore`, and (when requested) the echoed `document`.
    typealias RerankResult = (index: Int, relevanceScore: Double, document: String?)

    /// Assemble ranked rerank results — index + exposed relevance score +
    /// optional echoed document. Kept a `nonisolated static` PURE function
    /// (typed tuples, no `Any`, no live server / loaded model) so the
    /// ordering, score transform, and `return_documents` wiring are
    /// unit-testable in isolation. `scoreTransform` maps each raw score to the
    /// API `relevance_score` (`Double(_)` identity for cosine, `rerankSigmoid`
    /// for the cross-encoder). Out-of-range indices simply omit the document.
    nonisolated static func rerankResults(
        ranked: [(index: Int, score: Float)],
        documents: [String],
        returnDocuments: Bool,
        scoreTransform: (Float) -> Double
    ) -> [RerankResult] {
        ranked.map { entry in
            let document =
                (returnDocuments && entry.index >= 0 && entry.index < documents.count)
                ? documents[entry.index] : nil
            return (
                index: entry.index, relevanceScore: scoreTransform(entry.score), document: document
            )
        }
    }

    /// Serialize typed ``RerankResult``s into the endpoint's JSON array shape
    /// (`jsonResponseAny` takes `[String: Any]`). Omits `document` when nil.
    nonisolated static func rerankResultsJSON(_ results: [RerankResult]) -> [[String: Any]] {
        results.map { result in
            var item: [String: Any] = [
                "index": result.index,
                "relevance_score": result.relevanceScore,
            ]
            if let document = result.document {
                item["document"] = document
            }
            return item
        }
    }

    /// Shared cold-swap-or-error for the embeddings/rerank handlers. Returns
    /// a ready-made error `Response` (404 for an unknown model, 400 when the
    /// model isn't an embedder, 500 for a load failure) on failure, or `nil`
    /// once the embedder is resident.
    private func ensureEmbedderOr404(_ modelID: String) async -> Response? {
        do {
            try await ensureEmbedderLoaded(modelID)
            return nil
        } catch let err as EmbedderKindError {
            switch err {
            case .notAnEmbedder(let id, let format):
                // P3-8: resolved to a real model, but not an embedder — 400 so
                // callers don't get meaningless vectors from a chat/VLM model.
                return errorResponse(
                    status: .badRequest,
                    message: "Model \(id) is not an embedding model (kind: \(format)). "
                        + "Use an embedder (e.g. `bge-small-en-v1.5`) for /v1/embeddings and /v1/rerank.",
                    code: "model_not_embedder"
                )
            }
        } catch let err as ModelSwapError {
            switch err {
            case .modelNotFound(let id):
                return errorResponse(
                    status: .notFound,
                    message: "Model not found: \(id). Download an embedder (e.g. `bge-small-en-v1.5`) and check `macmlx list`.",
                    code: "model_not_found"
                )
            case .loadFailed(let id, let reason):
                return errorResponse(
                    status: .internalServerError,
                    message: "Failed to load \(id): \(reason)",
                    code: "load_failed"
                )
            }
        } catch {
            return errorResponse(
                status: .internalServerError,
                message: error.localizedDescription,
                code: "load_failed"
            )
        }
    }

    // MARK: JSON helpers

    private func jsonResponse(_ dict: [String: String]) throws -> Response {
        let data = try JSONSerialization.data(withJSONObject: dict)
        var headers = HTTPFields()
        headers[.contentType] = "application/json; charset=utf-8"
        return Response(
            status: .ok,
            headers: headers,
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }

    private func jsonResponseAny(_ dict: [String: Any]) throws -> Response {
        let data = try JSONSerialization.data(withJSONObject: dict)
        var headers = HTTPFields()
        headers[.contentType] = "application/json; charset=utf-8"
        return Response(
            status: .ok,
            headers: headers,
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }

    private func errorResponse(
        status: HTTPResponse.Status,
        message: String,
        code: String
    ) -> Response {
        let body: [String: Any] = [
            "error": [
                "message": message,
                "type": "invalid_request_error",
                "code": code,
            ] as [String: Any]
        ]
        let data = (try? JSONSerialization.data(withJSONObject: body)) ?? Data()
        var headers = HTTPFields()
        headers[.contentType] = "application/json; charset=utf-8"
        return Response(
            status: status,
            headers: headers,
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }

    /// Anthropic-shaped error envelope for a non-streaming `/v1/messages` HTTP
    /// error response: `{"type":"error","error":{"type":<type>,"message":<message>}}`.
    /// Mirrors the shape of the SSE in-band `event: error` frames
    /// `anthropicStreamingResponse` already emits — NOT the OpenAI
    /// `{"error":{"message",...}}` envelope `errorResponse` produces. Claude
    /// Code (the v0.6 benchmark target) speaks Anthropic's protocol natively
    /// and expects this shape from `/v1/messages`, including its request-level
    /// 400s. `type` is one of Anthropic's error-type strings
    /// (`invalid_request_error`, `api_error`, …); callers choose the one
    /// matching `status`.
    private func anthropicErrorResponse(
        status: HTTPResponse.Status,
        message: String,
        type: String
    ) -> Response {
        let body: [String: Any] = [
            "type": "error",
            "error": [
                "type": type,
                "message": message,
            ] as [String: Any],
        ]
        let data = (try? JSONSerialization.data(withJSONObject: body)) ?? Data()
        var headers = HTTPFields()
        headers[.contentType] = "application/json; charset=utf-8"
        return Response(
            status: status,
            headers: headers,
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }
}

// MARK: - Stall watchdog (SRV-4)

/// Outcome of racing a generation stream's `next()` against the stall
/// deadline. `.stalled` means `stallTimeout` elapsed with no new chunk —
/// an inter-chunk GAP, not total duration, so a long generation that keeps
/// producing tokens is never killed.
private enum GenerationStep {
    case chunk(GenerateChunk)
    case finished
    case stalled
}

/// Reference-type wrapper around a generation stream's iterator so it can
/// be driven from inside a `TaskGroup` child task — an `inout` local
/// iterator can't cross that boundary, and `AsyncThrowingStream.AsyncIterator`
/// is a struct. `@unchecked Sendable`: by construction only one `next()`
/// call is ever in flight on a given box at a time (`nextGenerationStep`
/// awaits it to completion — including via cancellation — before another
/// caller could invoke `next()` again).
private final class ChunkIteratorBox: @unchecked Sendable {
    private var iterator: AsyncThrowingStream<GenerateChunk, Error>.AsyncIterator
    init(_ stream: AsyncThrowingStream<GenerateChunk, Error>) {
        self.iterator = stream.makeAsyncIterator()
    }
    func next() async throws -> GenerateChunk? {
        try await iterator.next()
    }
}

/// Advance `box` by one chunk, racing it against `stallTimeout` seconds of
/// silence (SRV-4 / issue #29). Free function (not an actor method) so it
/// can run inside the `ResponseBody` writer closure, which executes outside
/// actor isolation — mirrors `writeAnthropicSSE` below. `stallTimeout <= 0`
/// disables the watchdog (waits with no timeout — pre-SRV-4 behaviour).
///
/// On `.stalled`, the still-pending `box.next()` child task is cancelled
/// via `group.cancelAll()`. Cancelling the task that's awaiting an
/// `AsyncThrowingStream`'s `next()` causes that stream to treat the
/// iteration as terminated and invoke its `onTermination` handler — which
/// is how `MLXSwiftEngine.generate` (POOL-2) learns to stop the underlying
/// generation instead of burning GPU on an abandoned request.
private func nextGenerationStep(
    _ box: ChunkIteratorBox,
    stallTimeout: TimeInterval
) async throws -> GenerationStep {
    // A4: clamp before the nanosecond conversion. `stallTimeout` traces
    // back to `SettingsManager.generationStallTimeoutSeconds` (a
    // user/settings-configurable `Int`) — an absurd or corrupted value
    // would make `stallTimeout * 1_000_000_000` exceed `UInt64.max`, and
    // `UInt64(_:)` on an out-of-range `Double` traps instead of clamping.
    // Floor at 0 (still "disabled" via the guard below); cap at 24h — a
    // generous ceiling no legitimate stall timeout should ever need.
    let clampedTimeout = min(max(stallTimeout, 0), 86_400)
    guard clampedTimeout > 0 else {
        if let chunk = try await box.next() { return .chunk(chunk) }
        return .finished
    }
    return try await withThrowingTaskGroup(of: GenerationStep.self) { group in
        group.addTask {
            if let chunk = try await box.next() { return .chunk(chunk) }
            return .finished
        }
        group.addTask {
            try await Task.sleep(nanoseconds: UInt64(clampedTimeout * 1_000_000_000))
            return .stalled
        }
        guard let first = try await group.next() else { return .finished }
        group.cancelAll()
        return first
    }
}

// MARK: - OpenAI tool-call serialization helpers

/// Serialise a detected tool call's arguments object into the JSON-ENCODED
/// STRING OpenAI puts in `function.arguments` (e.g. `"{\"city\":\"SF\"}"`),
/// NOT a nested JSON object. `toSendable()` unwraps each `JSONValue` to the
/// Foundation types `JSONSerialization` accepts. An arguments map that somehow
/// fails to serialise degrades to `"{}"` rather than failing the whole
/// response. Free function (not actor-isolated) so it runs inside the
/// `ResponseBody` streaming closure — mirrors `writeAnthropicSSE`.
private func openAIToolArgumentsJSON(_ arguments: [String: JSONValue]) -> String {
    let sendable = arguments.mapValues { $0.toSendable() }
    guard let data = try? JSONSerialization.data(withJSONObject: sendable) else {
        return "{}"
    }
    return String(decoding: data, as: UTF8.self)
}

/// One OpenAI `tool_calls[]` object for a non-streaming `message`:
/// `{"id","type":"function","function":{"name","arguments":<JSON string>}}`.
private func openAIToolCallObject(_ call: ToolCallRequest) -> [String: Any] {
    [
        "id": call.id,
        "type": "function",
        "function": [
            "name": call.name,
            "arguments": openAIToolArgumentsJSON(call.arguments),
        ] as [String: Any],
    ]
}

/// One OpenAI streaming `delta.tool_calls[]` object — the non-streaming object
/// plus the required `index`. We emit each call complete in a single delta
/// (name + full arguments at once), not OpenAI's per-fragment argument
/// streaming; a documented simplification that external agent loops, which
/// reassemble tool calls by `index`, handle transparently.
private func openAIToolCallDelta(index: Int, call: ToolCallRequest) -> [String: Any] {
    var object = openAIToolCallObject(call)
    object["index"] = index
    return object
}

/// One Anthropic `content[]` `tool_use` block: `{type,id,name,input}`. Unlike
/// OpenAI's `function.arguments` (a JSON-encoded STRING), Anthropic's `input` is
/// a DECODED JSON object — `toSendable()` unwraps each `JSONValue` to the
/// Foundation types `JSONSerialization` accepts. Free function (not
/// actor-isolated) so it also runs inside the streaming `ResponseBody` closure —
/// mirrors `openAIToolCallObject`.
private func anthropicToolUseBlock(_ call: ToolCallRequest) -> [String: Any] {
    [
        "type": "tool_use",
        "id": call.id,
        "name": call.name,
        "input": call.arguments.mapValues { $0.toSendable() },
    ]
}

// MARK: - OpenAI logprobs serialization (Track E)

/// One OpenAI `choices[].logprobs` object:
/// `{"content":[{token,logprob,bytes,top_logprobs:[{token,logprob,bytes}]}]}`.
/// `content` is empty when logprobs were requested but no tokens were generated.
/// Free function (not actor-isolated) so it also runs inside the streaming
/// `ResponseBody` closure — mirrors `openAIToolCallObject`.
private func openAILogprobs(_ tokens: [TokenLogprob]) -> [String: Any] {
    let content: [[String: Any]] = tokens.map { token in
        [
            "token": token.token,
            "logprob": token.logprob,
            "bytes": token.bytes ?? [],
            "top_logprobs": token.topLogprobs.map { alternative -> [String: Any] in
                [
                    "token": alternative.token,
                    "logprob": alternative.logprob,
                    "bytes": alternative.bytes ?? [],
                ]
            },
        ]
    }
    return ["content": content]
}

// MARK: - Anthropic SSE helpers

/// Serialise one Anthropic SSE frame: `event: <name>\ndata: <json>\n\n`.
/// Free function (not an actor method) so it can run inside the
/// `ResponseBody` writer closure, which executes outside actor isolation.
private func writeAnthropicSSE(
    _ writer: inout any ResponseBodyWriter,
    event: String,
    payload: [String: Any]
) async throws {
    let jsonData = try JSONSerialization.data(withJSONObject: payload)
    let jsonStr = String(decoding: jsonData, as: UTF8.self)
    var buf = ByteBuffer()
    buf.writeString("event: \(event)\ndata: \(jsonStr)\n\n")
    try await writer.write(buf)
}

/// Map macMLX's `FinishReason` onto Anthropic's `stop_reason` vocabulary.
/// Only an explicit length cap maps to `max_tokens`; everything else
/// (normal stop, engine error, or absent) reads as `end_turn`, since the
/// Anthropic message envelope has no dedicated error stop reason.
private func anthropicStopReason(_ reason: FinishReason?) -> String {
    switch reason {
    case .length: return "max_tokens"
    case .toolCalls: return "tool_use"
    case .stop, .error, .none: return "end_turn"
    }
}

// MARK: - Bearer auth middleware

/// Constant-time equality over two strings' UTF-8 bytes (SRV-6). Unlike
/// `String.==`, which returns the moment it hits a differing byte — leaking,
/// via response timing, how many leading bytes of a guessed token matched the
/// real key — this compares EVERY byte of equal-length inputs by
/// OR-accumulating their XOR, so the running time depends only on the length,
/// never on WHERE the first difference is. A length mismatch is allowed to
/// short-circuit: leaking the expected key's length is acceptable; leaking a
/// matched-prefix position is the content side channel being closed. Left at
/// the default `internal` access so it's unit-testable via `@testable import`.
func constantTimeEquals(_ a: String, _ b: String) -> Bool {
    let lhs = Array(a.utf8)
    let rhs = Array(b.utf8)
    guard lhs.count == rhs.count else { return false }
    var diff: UInt8 = 0
    for i in 0..<lhs.count {
        diff |= lhs[i] ^ rhs[i]
    }
    return diff == 0
}

/// Gates the HTTP surface behind a bearer token when the server is
/// configured with an API key (OpenAI convention:
/// `Authorization: Bearer <key>`). The `/health` and `/v1/health`
/// liveness probes stay open so orchestrators can check readiness
/// without the key; everything else gets 401 + a JSON error body
/// shaped like the server's other errors.
private struct BearerAuthMiddleware: RouterMiddleware {
    typealias Context = BasicRequestContext
    let apiKey: String

    func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        let path = request.uri.path
        if path == "/health" || path == "/v1/health" {
            return try await next(request, context)
        }
        // SRV-6: constant-time compare so we don't leak, via response timing,
        // how many leading bytes of a guessed token matched the real key.
        guard let provided = request.headers[.authorization],
              constantTimeEquals(provided, "Bearer \(apiKey)") else {
            return Self.unauthorized()
        }
        return try await next(request, context)
    }

    private static func unauthorized() -> Response {
        let body: [String: Any] = [
            "error": [
                "message": "Missing or invalid API key.",
                "type": "invalid_request_error",
                "code": "invalid_api_key",
            ] as [String: Any]
        ]
        let data = (try? JSONSerialization.data(withJSONObject: body)) ?? Data()
        var headers = HTTPFields()
        headers[.contentType] = "application/json; charset=utf-8"
        return Response(
            status: .unauthorized,
            headers: headers,
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }
}

// MARK: - Request logging middleware

/// Logs every incoming HTTP request at .debug level. 404 responses
/// are re-logged at .warning with the path, so a client hitting
/// an unmapped route (e.g. a mistyped `/v2/chat` the server doesn't
/// expose) produces a visible Logs-tab entry for debugging.
private struct RequestLoggingMiddleware: RouterMiddleware {
    typealias Context = BasicRequestContext
    func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        let method = request.method.rawValue
        let path = request.uri.path
        await LogManager.shared.debug(
            "→ \(method) \(path)",
            category: .http
        )
        do {
            let response = try await next(request, context)
            if response.status == .notFound {
                await LogManager.shared.warning(
                    "404 \(method) \(path) — unhandled route",
                    category: .http
                )
            }
            return response
        } catch let httpError as HTTPError where httpError.status == .notFound {
            await LogManager.shared.warning(
                "404 \(method) \(path) — unhandled route (thrown)",
                category: .http
            )
            throw httpError
        }
    }
}
