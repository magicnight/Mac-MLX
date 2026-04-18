import Foundation
import HTTPTypes
import Hummingbird
import NIOCore
import NIOPosix
import ServiceLifecycle

// MARK: - OpenAI-compatible request/response types

/// OpenAI-compatible chat completion request body.
private struct ChatCompletionRequest: Decodable, Sendable {
    struct Message: Decodable, Sendable {
        let role: String
        let content: String
    }

    let model: String
    let messages: [Message]
    let stream: Bool?
    let temperature: Double?
    let top_p: Double?
    let max_tokens: Int?
}

/// Request body for `/x/models/load`.
private struct LoadModelRequest: Decodable, Sendable {
    let model_path: String
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

    // MARK: State

    private let engine: any InferenceEngine

    /// Caller-supplied resolver for cold-swap. Defaults to returning nil
    /// (i.e. cold-swap disabled — server behaves as pre-v0.3.3: only
    /// the explicitly-loaded model can answer). The CLI's
    /// `ServeCommand` wires this up against `ModelLibraryManager.scan`.
    private let modelResolver: ModelResolver

    /// In-flight cold-swap Task, if any. Guards against thrashing when
    /// two requests for different models arrive at once: the second one
    /// awaits the first's completion before checking whether it can
    /// reuse the newly-loaded model or needs its own swap.
    private var loadInFlight: Task<Void, Error>?

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

    /// Create a server without cold-swap support — a chat completion
    /// request whose `model` field doesn't match the engine's loaded
    /// model will fail at the engine layer.
    public init(engine: any InferenceEngine) {
        self.engine = engine
        self.modelResolver = { _ in nil }
    }

    /// Create a server with cold-swap support — chat completion
    /// requests naming a different model than currently loaded will
    /// trigger an unload + load before the request proceeds.
    public init(
        engine: any InferenceEngine,
        modelResolver: @escaping ModelResolver
    ) {
        self.engine = engine
        self.modelResolver = modelResolver
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
        let loaded = await engine.loadedModel
        var data: [[String: String]] = []
        if let model = loaded {
            data = [["id": model.id, "object": "model", "owned_by": "local"]]
        }
        let body: [String: Any] = ["object": "list", "data": data]
        return try jsonResponseAny(body)
    }

    private func handleStatus() async throws -> Response {
        let loaded = await engine.loadedModel
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

        // Map messages, dropping unknown roles.
        let messages: [ChatMessage] = chatReq.messages.compactMap { msg in
            guard let role = MessageRole(rawValue: msg.role) else { return nil }
            return ChatMessage(role: role, content: msg.content)
        }

        // Extract system prompt from the first system message.
        let systemPrompt = chatReq.messages.first(where: { $0.role == "system" })?.content

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
            parameters: params
        )

        // Cold-swap (v0.3.3). If the request names a different model than
        // is currently loaded, try to resolve + load it on the fly. A
        // missing model → 404, a load failure → 500. Concurrent requests
        // for different models serialise on `loadInFlight`.
        do {
            try await ensureModelLoaded(chatReq.model)
        } catch let err as ModelSwapError {
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
        } catch {
            return errorResponse(
                status: .internalServerError,
                message: error.localizedDescription,
                code: "load_failed"
            )
        }

        let wantsStream = chatReq.stream ?? false
        if wantsStream {
            return try await streamingChatResponse(genRequest: genRequest)
        } else {
            return try await nonStreamingChatResponse(genRequest: genRequest)
        }
    }

    // MARK: - Cold-swap (v0.3.3)

    /// Errors surfaced by `ensureModelLoaded`. Internal — handler maps
    /// them onto OpenAI-style HTTP codes.
    private enum ModelSwapError: Error {
        case modelNotFound(id: String)
        case loadFailed(id: String, reason: String)
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

        if await engine.loadedModel?.id == requestedID {
            return
        }

        guard let target = await modelResolver(requestedID) else {
            throw ModelSwapError.modelNotFound(id: requestedID)
        }

        // Kick off the swap as a Task we can store for concurrent
        // callers to await. Errors propagate via the Task's value.
        let swapTask = Task { [engine] in
            try? await engine.unload()
            try await engine.load(target)
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

    private func nonStreamingChatResponse(genRequest: GenerateRequest) async throws -> Response {
        // Hop into the engine actor to call generate, then iterate the returned stream.
        let stream = await engine.generate(genRequest)
        var fullText = ""
        var finishReason = "stop"
        var promptTokens = 0
        var completionTokens = 0

        do {
            for try await chunk in stream {
                fullText += chunk.text
                if let usage = chunk.usage {
                    promptTokens = usage.promptTokens
                    completionTokens = usage.completionTokens
                }
                if let reason = chunk.finishReason {
                    finishReason = reason.rawValue
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
        let body: [String: Any] = [
            "id": completionID,
            "object": "chat.completion",
            "created": timestamp,
            "model": genRequest.model,
            "choices": [
                [
                    "index": 0,
                    "message": [
                        "role": "assistant",
                        "content": fullText,
                    ] as [String: Any],
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
        // Hop into the engine actor to call generate, then stream the result.
        let stream = await engine.generate(genRequest)
        let completionID = "chatcmpl-\(UUID().uuidString)"
        let timestamp = Int(Date().timeIntervalSince1970)
        let model = genRequest.model
        let server = self

        let responseBody = ResponseBody { writer in
            var chunkCount = 0
            do {
                for try await chunk in stream {
                    chunkCount += 1
                    let delta: [String: Any] = ["content": chunk.text]
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
