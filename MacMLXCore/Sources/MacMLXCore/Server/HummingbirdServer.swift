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
        let content: MultimodalContent
    }

    let model: String
    let messages: [Message]
    let stream: Bool?
    let temperature: Double?
    let top_p: Double?
    let max_tokens: Int?
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

/// `/v1/rerank` request body (Cohere/Jina-style shape). Scored with a
/// bi-encoder approximation — see `handleRerank`.
private struct RerankRequest: Decodable, Sendable {
    let model: String
    let query: String
    let documents: [String]
    let top_n: Int?
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

/// One Anthropic content block. `text` is set for `type:"text"`;
/// `source` is set for `type:"image"`.
private struct AnthropicBlock: Decodable, Sendable {
    let type: String            // "text" or "image"
    let text: String?
    let source: AnthropicImageSource?
}

/// Anthropic image block `source`. Only the base64 form is decoded
/// (`type:"base64"`); `media_type` is the IANA MIME and `data` is the
/// raw base64 payload (no data-URL prefix).
private struct AnthropicImageSource: Decodable, Sendable {
    let type: String            // "base64"
    let media_type: String
    let data: String
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
        stallTimeoutSeconds: TimeInterval = HummingbirdServer.defaultStallTimeoutSeconds
    ) {
        self.engineProvider = engineProvider
        self.modelResolver = modelResolver
        self.loadHook = loadHook
        self.inFlightHook = inFlightHook
        self.apiKey = apiKey
        self.stallTimeoutSeconds = stallTimeoutSeconds
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
        if let apiKey {
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
            return errorResponse(
                status: .badRequest,
                message: "Invalid JSON: \(error.localizedDescription)",
                code: "invalid_request_error"
            )
        }

        // Extract system prompt from the first system message. This also
        // removes it from the downstream messages array — GenerateRequest
        // re-prepends the systemPrompt via its allMessages computed
        // property, so leaving it in both places produces a duplicate
        // system turn, which Qwen3 / Gemma / other strict Jinja chat
        // templates reject with a TemplateException.
        let systemPrompt = chatReq.messages.first(where: { $0.role == "system" })?.content.text

        // Map the rest (user / assistant), dropping unknown roles and
        // the now-separated system turns. Multimodal `content` arrays
        // are split here: text parts → `content`, image_url data URLs
        // → `images` via `extractImages()`. See MultimodalContent.
        let messages: [ChatMessage] = chatReq.messages.compactMap { msg in
            guard let role = MessageRole(rawValue: msg.role), role != .system else { return nil }
            return ChatMessage(
                role: role,
                content: msg.content.text,
                images: msg.content.extractImages()
            )
        }

        let params = GenerationParameters(
            temperature: chatReq.temperature ?? 0.7,
            topP: chatReq.top_p ?? 0.95,
            maxTokens: chatReq.max_tokens ?? 2048,
            stream: chatReq.stream ?? false
        )

        let genRequest = GenerateRequest(
            model: chatReq.model,
            messages: messages,
            systemPrompt: systemPrompt,
            parameters: params,
            templateKwargs: await templateKwargs(for: chatReq.model)
        )

        // Cold-swap (v0.3.3) is no longer resolved here. SRV-2: the swap
        // must happen atomically with the generation lock, so each
        // responder below calls `beginGeneration(genRequest.model)` itself
        // (acquire → swap → re-resolve engine, all under the lock) instead
        // of us doing it here, unlocked, ahead of time.
        let wantsStream = chatReq.stream ?? false
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
            entries.append([
                "name": currentID,
                "model": currentID,
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
            guard let role = MessageRole(rawValue: msg.role), role != .system else { return nil }
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
            var chunkCount = 0

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
                        await server.incrementTokens(chunkCount)
                        return
                    case .chunk(let chunk):
                        chunkCount += 1
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
                await server.incrementTokens(chunkCount)
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
            await server.incrementTokens(chunkCount)
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
            var chunkCount = 0

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
                        await server.incrementTokens(chunkCount)
                        return
                    case .chunk(let chunk):
                        chunkCount += 1
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
                await server.incrementTokens(chunkCount)
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
            await server.incrementTokens(chunkCount)
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
            "usage": [
                "prompt_tokens": promptTokens,
                "completion_tokens": completionTokens,
                "total_tokens": promptTokens + completionTokens,
            ] as [String: Any],
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
        let server = self

        let responseBody = ResponseBody { writer in
            var chunkCount = 0

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
                        chunkCount += 1
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

            await server.incrementTokens(chunkCount)
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
            return errorResponse(
                status: .badRequest,
                message: "Empty request body",
                code: "invalid_request_error"
            )
        }

        let req: AnthropicMessagesRequest
        do {
            req = try JSONDecoder().decode(AnthropicMessagesRequest.self, from: data)
        } catch {
            return errorResponse(
                status: .badRequest,
                message: "Invalid JSON: \(error.localizedDescription)",
                code: "invalid_request_error"
            )
        }

        // Anthropic carries the system prompt at the top level (not as a
        // message turn), so there's nothing to strip out of `messages` —
        // GenerateRequest.allMessages re-prepends this systemPrompt.
        let systemPrompt = req.system?.text

        // Map turns. Anthropic roles are only user / assistant; unknown
        // roles are dropped. Text blocks → `content`, image blocks →
        // `images` via `extractImages()`. See AnthropicContent.
        let messages: [ChatMessage] = req.messages.compactMap { msg in
            guard let role = MessageRole(rawValue: msg.role), role != .system else { return nil }
            return ChatMessage(
                role: role,
                content: msg.content.text,
                images: msg.content.extractImages()
            )
        }

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
            templateKwargs: await templateKwargs(for: req.model)
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
            return errorResponse(status: .internalServerError, message: error.localizedDescription, code: "load_failed")
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

        do {
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
                    if let usage = chunk.usage {
                        promptTokens = usage.promptTokens
                        completionTokens = usage.completionTokens
                    }
                    if let reason = chunk.finishReason {
                        finishReason = reason
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

        // Reasoning is not surfaced as a separate block in this MVP — only
        // the answer text goes into content[0]. `splitReasoning` strips any
        // `<think>…</think>`; non-reasoning models keep their full text.
        let (_, answer) = MessageSegmenter.splitReasoning(fullText)

        let body: [String: Any] = [
            "id": "msg_\(UUID().uuidString)",
            "type": "message",
            "role": "assistant",
            "content": [
                ["type": "text", "text": answer] as [String: Any]
            ],
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
    /// content_block_stop → message_delta → message_stop. Only the answer
    /// portion is streamed as `text_delta`; reasoning is dropped for this
    /// MVP (matching the non-streaming path). The generation lock is held
    /// for the whole body and released in `defer`.
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
            var chunkCount = 0
            var completionTokens = 0
            var promptTokens = 0
            var finishReason: FinishReason?

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

                // 2. content_block_start (single text block at index 0)
                try await writeAnthropicSSE(&writer, event: "content_block_start", payload: [
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": [
                        "type": "text",
                        "text": "",
                    ] as [String: Any],
                ])

                // 3. content_block_delta per answer delta.
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
                        chunkCount += 1
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
                        guard !text.isEmpty else { continue }
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

            // 4. content_block_stop
            try? await writeAnthropicSSE(&writer, event: "content_block_stop", payload: [
                "type": "content_block_stop",
                "index": 0,
            ])

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

            await server.incrementTokens(chunkCount)
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
        let start = Date()
        do {
            try await engine.load(model)
        } catch {
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
        do {
            try await engine.unload()
        } catch {
            return errorResponse(
                status: .internalServerError,
                message: error.localizedDescription,
                code: "unload_failed"
            )
        }
        return try jsonResponse(["status": "unloaded"])
    }

    // MARK: - Embeddings / rerank (v0.5.2)

    /// Make sure `embeddingEngine` has `requestedID` resident. Resolves the
    /// model the same way `ensureModelLoaded` does (direct id/displayName,
    /// then user-facing alias), creating + loading a fresh `EmbeddingEngine`
    /// on a cold miss and swapping when a different embedder is requested.
    /// Throws `.modelNotFound` when the id isn't on disk, `.loadFailed` when
    /// the load itself fails.
    private func ensureEmbedderLoaded(_ requestedID: String) async throws {
        // Same resolution order as the generation cold-swap.
        let target: LocalModel
        if let resolved = await modelResolver(requestedID) {
            target = resolved
        } else if let aliasID = await ModelParametersStore().modelID(forAlias: requestedID),
                  let aliasTarget = await modelResolver(aliasID) {
            target = aliasTarget
        } else {
            throw ModelSwapError.modelNotFound(id: requestedID)
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
        // allocator state that isn't safe to share concurrently.
        await acquireGenerationLock()
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

    /// `POST /v1/rerank` — bi-encoder rerank MVP. Embeds `[query] + documents`
    /// with the same embedder and ranks documents by cosine similarity to the
    /// query. This is an approximation; a true cross-encoder reranker is a
    /// from-scratch follow-up (see `rerankByCosine`). Returns
    /// `{ results:[{index, relevance_score}], model }`.
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

        await acquireGenerationLock()
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

        let results: [[String: Any]] = ranked.map { entry in
            [
                "index": entry.index,
                "relevance_score": Double(entry.score),
            ]
        }
        return try jsonResponseAny([
            "results": results,
            "model": req.model,
        ])
    }

    /// Shared cold-swap-or-error for the embeddings/rerank handlers. Returns
    /// a ready-made error `Response` (404 for an unknown model, 500 for a
    /// load failure) on failure, or `nil` once the embedder is resident.
    private func ensureEmbedderOr404(_ modelID: String) async -> Response? {
        do {
            try await ensureEmbedderLoaded(modelID)
            return nil
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
    case .stop, .error, .none: return "end_turn"
    }
}

// MARK: - Bearer auth middleware

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
        guard request.headers[.authorization] == "Bearer \(apiKey)" else {
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
/// an unmapped route (e.g. `/v1/completions` which we don't support)
/// produces a visible Logs-tab entry for debugging.
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
