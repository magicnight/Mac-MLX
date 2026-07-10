import Foundation
import Testing
@testable import MacMLXCore

// MARK: - HummingbirdServer Tests
//
// Each test starts a server on a unique high port to avoid interference
// when run in parallel. Servers are stopped with `await server.stop()` at
// the end of each test (not fire-and-forget defer) so ports are released
// before the test returns.
//
// Port assignments (19_000 range, spaced by 10 to allow port-retry headroom):
//   healthEndpointReturnsOk             : 19_000
//   modelsEndpointEmptyWhenNoModelLoaded: 19_010
//   modelsEndpointListsLoadedModel      : 19_020
//   chatCompletionsNonStreaming         : 19_030
//   portRetrySucceedsWhenPreferredBusy  : 19_100
//
// v0.5.3 stability wave (SRV-1/SRV-2/SRV-3b/SRV-4):
//   srv1EngineProviderReflectsSwapNotFrozenReference : 19_510
//   srv2ConcurrentDifferentModelRequestsEachGetTheirOwnModel : 19_600
//   srv3bCancelledParkedWaiterDoesNotDeadlock (no server — lock-only)
//   srv4StallWatchdogTimesOutAndReleasesLock : 19_700
//
// v0.5.3 review follow-up (A1):
//   coldSwapStreamingReturns404WhenModelMissingBeforeHeadersSent : 19_220

@Suite("HummingbirdServer")
struct HummingbirdServerTests {

    // MARK: Helpers

    private func makeServer() -> HummingbirdServer {
        HummingbirdServer(engine: StubInferenceEngine(engineID: .mlxSwift))
    }

    private func get(_ url: URL) async throws -> (Data, HTTPURLResponse) {
        let (data, response) = try await URLSession.shared.data(from: url)
        let http = try #require(response as? HTTPURLResponse)
        return (data, http)
    }

    private func post(_ url: URL, body: some Encodable) async throws -> (Data, HTTPURLResponse) {
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONEncoder().encode(body)
        let (data, response) = try await URLSession.shared.data(for: req)
        let http = try #require(response as? HTTPURLResponse)
        return (data, http)
    }

    private func postRaw(_ url: URL, jsonObject: Any) async throws -> (Data, HTTPURLResponse) {
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: jsonObject)
        let (data, response) = try await URLSession.shared.data(for: req)
        let http = try #require(response as? HTTPURLResponse)
        return (data, http)
    }

    // MARK: Tests

    @Test
    func healthEndpointReturnsOk() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 19_000)

        let url = URL(string: "http://127.0.0.1:\(port)/health")!
        let (data, response) = try await get(url)

        await server.stop()

        #expect(response.statusCode == 200)
        let json = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: String]
        )
        #expect(json["status"] == "ok")
    }

    // MARK: API-key auth (v0.5.1)
    //   protectedRouteRejectsMissingKey : 19_300
    //   protectedRouteRejectsWrongKey   : 19_310
    //   protectedRouteAcceptsCorrectKey : 19_320
    //   openServerNeedsNoKey            : 19_330
    //   healthProbeStaysOpenWithKey     : 19_340

    private func keyedServer(_ key: String) -> HummingbirdServer {
        HummingbirdServer(engine: StubInferenceEngine(engineID: .mlxSwift), apiKey: key)
    }

    @Test
    func protectedRouteRejectsMissingKey() async throws {
        let server = keyedServer("s3cret")
        let port = try await server.start(preferredPort: 19_300)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/models")!
        let (_, response) = try await get(url)
        await server.stop()
        #expect(response.statusCode == 401)
    }

    @Test
    func protectedRouteRejectsWrongKey() async throws {
        let server = keyedServer("s3cret")
        let port = try await server.start(preferredPort: 19_310)
        var req = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/models")!)
        req.setValue("Bearer wrong", forHTTPHeaderField: "Authorization")
        let (_, response) = try await URLSession.shared.data(for: req)
        await server.stop()
        let http = try #require(response as? HTTPURLResponse)
        #expect(http.statusCode == 401)
    }

    @Test
    func protectedRouteAcceptsCorrectKey() async throws {
        let server = keyedServer("s3cret")
        let port = try await server.start(preferredPort: 19_320)
        var req = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/models")!)
        req.setValue("Bearer s3cret", forHTTPHeaderField: "Authorization")
        let (_, response) = try await URLSession.shared.data(for: req)
        await server.stop()
        let http = try #require(response as? HTTPURLResponse)
        #expect(http.statusCode == 200)
    }

    @Test
    func openServerNeedsNoKey() async throws {
        let server = makeServer()  // no apiKey → open
        let port = try await server.start(preferredPort: 19_330)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/models")!
        let (_, response) = try await get(url)
        await server.stop()
        #expect(response.statusCode == 200)
    }

    @Test
    func healthProbeStaysOpenWithKey() async throws {
        let server = keyedServer("s3cret")
        let port = try await server.start(preferredPort: 19_340)
        let url = URL(string: "http://127.0.0.1:\(port)/health")!
        let (_, response) = try await get(url)
        await server.stop()
        #expect(response.statusCode == 200)
    }

    /// SRV-6: the bearer-token check must be a constant-time byte comparison,
    /// not a short-circuiting `String.==` (a timing side channel that leaks
    /// how many leading bytes of a guess matched). Exercises the extracted
    /// `constantTimeEquals` helper directly (internal, visible via
    /// `@testable import`): equal → true; a difference at the FIRST byte, at
    /// the LAST byte, and a length mismatch → all false.
    @Test
    func srv6ConstantTimeEqualsRejectsAnyDifference() {
        // Byte-identical → equal.
        #expect(constantTimeEquals("Bearer s3cret", "Bearer s3cret"))
        #expect(constantTimeEquals("", ""))
        // Differ at the very first byte → not equal (must not early-out true).
        #expect(!constantTimeEquals("Xearer s3cret", "Bearer s3cret"))
        // Differ only at the very last byte → not equal (the whole length is
        // still compared).
        #expect(!constantTimeEquals("Bearer s3creT", "Bearer s3cret"))
        // Different lengths → not equal (allowed to short-circuit on length).
        #expect(!constantTimeEquals("Bearer s3cret", "Bearer s3cret2"))
        #expect(!constantTimeEquals("", "x"))
        // Multibyte (UTF-8) content is compared by bytes, not scalars.
        #expect(constantTimeEquals("Bearer kéy", "Bearer kéy"))
        #expect(!constantTimeEquals("Bearer kéy", "Bearer key"))
    }

    @Test
    func modelsEndpointEmptyWhenNoModelLoaded() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 19_010)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/models")!
        let (data, response) = try await get(url)

        await server.stop()

        #expect(response.statusCode == 200)
        let json = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        #expect(json["object"] as? String == "list")
        let dataArray = try #require(json["data"] as? [[String: String]])
        #expect(dataArray.isEmpty)
    }

    @Test
    func modelsEndpointListsLoadedModel() async throws {
        let engine = StubInferenceEngine(engineID: .mlxSwift)
        let model = LocalModel(
            id: "test-model",
            displayName: "Test Model",
            directory: URL(filePath: "/tmp"),
            sizeBytes: 0,
            format: .mlx,
            quantization: nil,
            parameterCount: nil,
            architecture: nil
        )
        try await engine.load(model)

        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 19_020)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/models")!
        let (data, response) = try await get(url)

        await server.stop()

        #expect(response.statusCode == 200)
        let json = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        let dataArray = try #require(json["data"] as? [[String: Any]])
        #expect(dataArray.count == 1)
        #expect(dataArray[0]["id"] as? String == "test-model")
    }

    @Test
    func chatCompletionsNonStreaming() async throws {
        let engine = StubInferenceEngine(engineID: .mlxSwift)
        let model = LocalModel(
            id: "stub-model",
            displayName: "Stub",
            directory: URL(filePath: "/tmp"),
            sizeBytes: 0,
            format: .mlx,
            quantization: nil,
            parameterCount: nil,
            architecture: nil
        )
        try await engine.load(model)

        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 19_030)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let body: [String: Any] = [
            "model": "stub-model",
            "messages": [["role": "user", "content": "Hello"]],
            "stream": false,
        ]
        let (data, response) = try await postRaw(url, jsonObject: body)

        await server.stop()

        #expect(response.statusCode == 200)
        let json = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        #expect(json["object"] as? String == "chat.completion")
        let choices = try #require(json["choices"] as? [[String: Any]])
        #expect(choices.count == 1)
        let message = try #require(choices[0]["message"] as? [String: String])
        let content = try #require(message["content"])
        // StubInferenceEngine yields "stub-" + "response"
        #expect(content == "stub-response")
    }

    /// Stub engine that records the `GenerateRequest` it was asked to
    /// generate, so a test can inspect exactly which messages/roles survived
    /// server-side decoding. Unlike `StubInferenceEngine` (fixed response, no
    /// capture) or `EchoingStubEngine` (echoes the model id, not the request).
    private actor CapturingStubEngine: InferenceEngine {
        nonisolated let engineID: EngineID = .mlxSwift
        private(set) var status: EngineStatus = .idle
        private(set) var loadedModel: LocalModel?
        let version = "capturing-1"

        private(set) var capturedRequest: GenerateRequest?

        func load(_ model: LocalModel) async throws {
            status = .loading(model: model.id)
            loadedModel = model
            status = .ready(model: model.id)
        }

        func unload() async throws {
            loadedModel = nil
            status = .idle
        }

        nonisolated func generate(_ request: GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error> {
            AsyncThrowingStream { continuation in
                Task { [weak self] in
                    guard let self else { continuation.finish(); return }
                    await self.capture(request)
                    continuation.yield(GenerateChunk(
                        text: "stub-response",
                        finishReason: .stop,
                        usage: TokenUsage(promptTokens: 1, completionTokens: 2)
                    ))
                    continuation.finish()
                }
            }
        }

        private func capture(_ request: GenerateRequest) {
            capturedRequest = request
        }

        func healthCheck() async -> Bool { true }
    }

    /// Agent-tools wave: OpenAI tool history is fully decoded. A `role:"tool"`
    /// turn becomes a `.tool` `ChatMessage` carrying its `tool_call_id`, and the
    /// preceding assistant `tool_calls` array (with `content:null`) becomes that
    /// message's `toolCalls` — so the chat template's call ↔ result pairing is
    /// satisfied instead of the turns being silently dropped. Supersedes the
    /// pre-wave-2 drop guard.
    @Test
    func chatCompletionsDecodesOpenAIToolHistory() async throws {
        let engine = CapturingStubEngine()
        let model = LocalModel(
            id: "stub-model",
            displayName: "Stub",
            directory: URL(filePath: "/tmp"),
            sizeBytes: 0,
            format: .mlx,
            quantization: nil,
            parameterCount: nil,
            architecture: nil
        )
        try await engine.load(model)

        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 19_040)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let body: [String: Any] = [
            "model": "stub-model",
            "messages": [
                ["role": "user", "content": "What's the weather in Paris?"],
                [
                    "role": "assistant",
                    "content": NSNull(),
                    "tool_calls": [[
                        "id": "call_abc123",
                        "type": "function",
                        "function": [
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Paris\"}",
                        ],
                    ]],
                ],
                ["role": "tool", "tool_call_id": "call_abc123", "content": "22C, sunny"],
            ],
            "stream": false,
        ]
        let (_, response) = try await postRaw(url, jsonObject: body)

        await server.stop()

        #expect(response.statusCode == 200)
        let messages = try #require(await engine.capturedRequest?.messages)
        #expect(messages.count == 3)
        #expect(messages.map(\.role) == [.user, .assistant, .tool])

        // Assistant turn carries the decoded tool call (arguments parsed from the
        // JSON-string wire form into an object).
        let calls = try #require(messages[1].toolCalls)
        #expect(calls.count == 1)
        #expect(calls[0].id == "call_abc123")
        #expect(calls[0].name == "get_weather")
        #expect(calls[0].arguments["city"] == .string("Paris"))

        // Tool result pairs on the same id.
        #expect(messages[2].role == .tool)
        #expect(messages[2].toolCallID == "call_abc123")
        #expect(messages[2].content == "22C, sunny")
    }

    /// A malformed `tool_calls` arguments string is a 400 — never silently
    /// dropped or coerced to empty args (project rule: no silent degradation).
    /// The engine is never invoked.
    @Test
    func chatCompletionsMalformedToolArgumentsReturns400() async throws {
        let engine = CapturingStubEngine()
        try await engine.load(fixtureModel(id: "stub-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 20_000)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let body: [String: Any] = [
            "model": "stub-model",
            "messages": [[
                "role": "assistant",
                "content": NSNull(),
                "tool_calls": [[
                    "id": "call_x",
                    "type": "function",
                    "function": ["name": "get_weather", "arguments": "not-json"],
                ]],
            ]],
            "stream": false,
        ]
        let (_, response) = try await postRaw(url, jsonObject: body)

        await server.stop()

        #expect(response.statusCode == 400)
        #expect(await engine.capturedRequest == nil)
    }

    /// A `role:"tool"` message with a missing `tool_call_id` is a 400 — the
    /// exact half-formed shape (`toolCallID: nil`) the original pre-wave-2 guard
    /// existed to avoid; symmetric with the Anthropic `tool_result` guard.
    @Test
    func chatCompletionsMissingToolCallIDReturns400() async throws {
        let engine = CapturingStubEngine()
        try await engine.load(fixtureModel(id: "stub-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 20_001)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let body: [String: Any] = [
            "model": "stub-model",
            "messages": [
                ["role": "user", "content": "Weather?"],
                ["role": "tool", "content": "22C, sunny"],
            ],
            "stream": false,
        ]
        let (_, response) = try await postRaw(url, jsonObject: body)

        await server.stop()

        #expect(response.statusCode == 400)
        #expect(await engine.capturedRequest == nil)
    }

    /// A `tool_calls` entry with an empty function name is a 400 (the name key
    /// must be present for `Decodable` to succeed — an EMPTY string is the way to
    /// reach the post-decode guard rather than the generic "Invalid JSON" 400).
    @Test
    func chatCompletionsMissingFunctionNameReturns400() async throws {
        let engine = CapturingStubEngine()
        try await engine.load(fixtureModel(id: "stub-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 20_002)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let body: [String: Any] = [
            "model": "stub-model",
            "messages": [[
                "role": "assistant",
                "content": NSNull(),
                "tool_calls": [[
                    "id": "call_x",
                    "type": "function",
                    "function": ["name": "", "arguments": "{}"],
                ]],
            ]],
            "stream": false,
        ]
        let (_, response) = try await postRaw(url, jsonObject: body)

        await server.stop()

        #expect(response.statusCode == 400)
        #expect(await engine.capturedRequest == nil)
    }

    /// A `tool` message answering a `tool_call_id` that no preceding assistant
    /// `tool_calls` entry ever issued is an ORPHAN result — a 400, not a
    /// silently-accepted turn that would reach the chat template unpaired.
    @Test
    func chatCompletionsOrphanToolResultReturns400() async throws {
        let engine = CapturingStubEngine()
        try await engine.load(fixtureModel(id: "stub-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 20_003)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let body: [String: Any] = [
            "model": "stub-model",
            "messages": [
                ["role": "user", "content": "Weather?"],
                ["role": "tool", "tool_call_id": "call_never_issued", "content": "22C"],
            ],
            "stream": false,
        ]
        let (_, response) = try await postRaw(url, jsonObject: body)

        await server.stop()

        #expect(response.statusCode == 400)
        #expect(await engine.capturedRequest == nil)
    }

    /// Two assistant `tool_calls` entries (in different turns) reusing the same
    /// `id` is a 400 — ids must be unique across the whole history, not just
    /// within one turn.
    @Test
    func chatCompletionsDuplicateToolCallIDReturns400() async throws {
        let engine = CapturingStubEngine()
        try await engine.load(fixtureModel(id: "stub-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 20_004)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let toolCallEntry: [String: Any] = [
            "id": "call_dup",
            "type": "function",
            "function": ["name": "get_weather", "arguments": "{\"city\":\"Paris\"}"],
        ]
        let body: [String: Any] = [
            "model": "stub-model",
            "messages": [
                ["role": "user", "content": "Weather?"],
                ["role": "assistant", "content": NSNull(), "tool_calls": [toolCallEntry]],
                ["role": "tool", "tool_call_id": "call_dup", "content": "22C"],
                ["role": "user", "content": "And tomorrow?"],
                ["role": "assistant", "content": NSNull(), "tool_calls": [toolCallEntry]],
            ],
            "stream": false,
        ]
        let (_, response) = try await postRaw(url, jsonObject: body)

        await server.stop()

        #expect(response.statusCode == 400)
        #expect(await engine.capturedRequest == nil)
    }

    /// A SECOND `tool` message answering an `id` that already has a result is a
    /// 400 — distinct from the orphan case (the id WAS issued, but is already
    /// consumed).
    @Test
    func chatCompletionsToolResultAnsweringAlreadyConsumedCallReturns400() async throws {
        let engine = CapturingStubEngine()
        try await engine.load(fixtureModel(id: "stub-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 20_005)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let body: [String: Any] = [
            "model": "stub-model",
            "messages": [
                ["role": "user", "content": "Weather?"],
                [
                    "role": "assistant", "content": NSNull(),
                    "tool_calls": [[
                        "id": "call_once",
                        "type": "function",
                        "function": ["name": "get_weather", "arguments": "{\"city\":\"Paris\"}"],
                    ]],
                ],
                ["role": "tool", "tool_call_id": "call_once", "content": "22C"],
                ["role": "tool", "tool_call_id": "call_once", "content": "22C again"],
            ],
            "stream": false,
        ]
        let (_, response) = try await postRaw(url, jsonObject: body)

        await server.stop()

        #expect(response.statusCode == 400)
        #expect(await engine.capturedRequest == nil)
    }

    /// `tools: []` (present but empty) behaves exactly like an absent `tools`
    /// field: `GenerateRequest.tools` is empty and the response is a plain
    /// assistant message with no `tool_calls` — the engine's tool-call detector
    /// gates on `tools?.isEmpty == false` (see `ChatCompletionRequest.tools`
    /// doc), so `[]` and `nil` are behaviorally identical.
    @Test
    func chatCompletionsEmptyToolsArrayBehavesLikeAbsent() async throws {
        let engine = CapturingStubEngine()
        try await engine.load(fixtureModel(id: "stub-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 20_006)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let body: [String: Any] = [
            "model": "stub-model",
            "messages": [["role": "user", "content": "Hello"]],
            "stream": false,
            "tools": [],
        ]
        let (data, response) = try await postRaw(url, jsonObject: body)

        await server.stop()

        #expect(response.statusCode == 200)
        let tools = await engine.capturedRequest?.tools
        #expect(tools?.isEmpty ?? true)
        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let choices = try #require(json["choices"] as? [[String: Any]])
        let message = try #require(choices[0]["message"] as? [String: Any])
        #expect(message["content"] as? String == "stub-response")
        #expect(message["tool_calls"] == nil)
    }

    /// Regression invariant (no tools in play): a plain MULTI-TURN conversation
    /// with NO tool_calls/tool messages decodes through `decodeOpenAIMessages`
    /// (and its `validateToolCallPairing` pass, which is unconditional) exactly
    /// as it did before this wave — same message count, roles, and content,
    /// byte for byte, with no tool fields set. Guards against the tool-history
    /// decode/pairing-validation path added this wave corrupting ordinary,
    /// tool-free chat. Anthropic's analogous guard is
    /// `anthropicMessagesWithoutToolsUnchanged`.
    @Test
    func chatCompletionsPlainMultiTurnHistoryUnchanged() async throws {
        let engine = CapturingStubEngine()
        try await engine.load(fixtureModel(id: "stub-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 20_007)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let body: [String: Any] = [
            "model": "stub-model",
            "messages": [
                ["role": "user", "content": "What's the weather?"],
                ["role": "assistant", "content": "I don't have that information."],
                ["role": "user", "content": "OK, never mind."],
            ],
            "stream": false,
        ]
        let (_, response) = try await postRaw(url, jsonObject: body)

        await server.stop()

        #expect(response.statusCode == 200)
        let messages = try #require(await engine.capturedRequest?.messages)
        #expect(messages.map(\.role) == [.user, .assistant, .user])
        #expect(messages.map(\.content) == [
            "What's the weather?", "I don't have that information.", "OK, never mind.",
        ])
        #expect(messages.allSatisfy { $0.toolCalls == nil && $0.toolCallID == nil })
    }

    // MARK: - Tool pass-through (v0.5, OpenAI /v1/chat/completions)
    //
    // The server is the SERVER half of tool-calling: it (1) forwards the
    // request's `tools` into `GenerateRequest.tools` (which the chat template
    // consumes) and (2) serialises detected `tool_calls` back in OpenAI shape.
    // It runs NO agent loop — external clients do. These tests use stub engines
    // so no model is needed.
    //
    // Port assignments (spaced by 10):
    //   chatCompletionsToolsPopulateGenerateRequestTools : 19_800
    //   chatCompletionsNonStreamingEmitsToolCalls        : 19_810
    //   chatCompletionsStreamingEmitsToolCallDelta       : 19_820
    //   chatCompletionsToolChoiceNoneOmitsTools          : 19_830
    //   chatCompletionsWithoutToolsFieldDecodesUnchanged : 19_840

    /// Stub engine that ends a generation in a tool call with no accompanying
    /// text — the shape `MLXSwiftEngine` produces on the terminal chunk when
    /// `ToolCallProcessor` drains a call. Lets the server's OpenAI tool_calls
    /// serialization be exercised without a real model.
    private actor ToolCallStubEngine: InferenceEngine {
        nonisolated let engineID: EngineID = .mlxSwift
        private(set) var status: EngineStatus = .idle
        private(set) var loadedModel: LocalModel?
        let version = "toolcall-1"

        func load(_ model: LocalModel) async throws {
            status = .loading(model: model.id)
            loadedModel = model
            status = .ready(model: model.id)
        }

        func unload() async throws {
            loadedModel = nil
            status = .idle
        }

        nonisolated func generate(_ request: GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error> {
            AsyncThrowingStream { continuation in
                let call = ToolCallRequest(
                    id: "call_abc123",
                    name: "get_weather",
                    arguments: ["city": .string("Paris")]
                )
                continuation.yield(GenerateChunk(
                    text: "",
                    finishReason: .toolCalls,
                    usage: TokenUsage(promptTokens: 3, completionTokens: 4),
                    toolCalls: [call]
                ))
                continuation.finish()
            }
        }

        func healthCheck() async -> Bool { true }
    }

    /// Stub engine that streams SOME free text before ending the turn in a
    /// tool call — unlike `ToolCallStubEngine` (tool-only, empty text
    /// throughout). Lets a MIXED text+tool_use reply be exercised, distinct
    /// from the tool-only special case (`ToolCallStubEngine`'s empty text block
    /// is omitted entirely on the Anthropic surface — see
    /// `anthropicMessagesStreamingEmitsToolUse` / `…NonStreamingEmitsToolUse`).
    private actor TextThenToolCallStubEngine: InferenceEngine {
        nonisolated let engineID: EngineID = .mlxSwift
        private(set) var status: EngineStatus = .idle
        private(set) var loadedModel: LocalModel?
        let version = "text-then-toolcall-1"

        func load(_ model: LocalModel) async throws {
            status = .loading(model: model.id)
            loadedModel = model
            status = .ready(model: model.id)
        }

        func unload() async throws {
            loadedModel = nil
            status = .idle
        }

        nonisolated func generate(_ request: GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error> {
            AsyncThrowingStream { continuation in
                continuation.yield(GenerateChunk(text: "Let me check.", finishReason: nil))
                let call = ToolCallRequest(
                    id: "call_abc123",
                    name: "get_weather",
                    arguments: ["city": .string("Paris")]
                )
                continuation.yield(GenerateChunk(
                    text: "",
                    finishReason: .toolCalls,
                    usage: TokenUsage(promptTokens: 3, completionTokens: 4),
                    toolCalls: [call]
                ))
                continuation.finish()
            }
        }

        func healthCheck() async -> Bool { true }
    }

    /// A single OpenAI `tools` entry `{"type":"function","function":{...}}`,
    /// as a JSON-object body clients send.
    private var weatherToolSpec: [String: Any] {
        [
            "type": "function",
            "function": [
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": [
                    "type": "object",
                    "properties": ["city": ["type": "string"]],
                    "required": ["city"],
                ],
            ],
        ]
    }

    /// A request carrying `tools` populates `GenerateRequest.tools` verbatim,
    /// so the chat template can offer them to the model.
    @Test
    func chatCompletionsToolsPopulateGenerateRequestTools() async throws {
        let engine = CapturingStubEngine()
        try await engine.load(fixtureModel(id: "stub-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 19_800)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let body: [String: Any] = [
            "model": "stub-model",
            "messages": [["role": "user", "content": "Weather in Paris?"]],
            "stream": false,
            "tools": [weatherToolSpec],
        ]
        let (_, response) = try await postRaw(url, jsonObject: body)

        await server.stop()

        #expect(response.statusCode == 200)
        let tools = try #require(await engine.capturedRequest?.tools)
        #expect(tools.count == 1)
        // Forwarded verbatim: the element is the OpenAI function spec object.
        guard case .object(let spec) = tools[0] else {
            Issue.record("tool spec did not decode as a JSON object")
            return
        }
        #expect(spec["type"] == .string("function"))
        guard case .object(let fn)? = spec["function"] else {
            Issue.record("function payload missing")
            return
        }
        #expect(fn["name"] == .string("get_weather"))
    }

    /// A terminal chunk carrying `toolCalls` is serialised into the OpenAI
    /// non-streaming `message.tool_calls` shape: id / type / function.name and
    /// `arguments` as a JSON-ENCODED STRING, with `finish_reason:"tool_calls"`
    /// and `content` null (only tool calls, no text).
    @Test
    func chatCompletionsNonStreamingEmitsToolCalls() async throws {
        let engine = ToolCallStubEngine()
        try await engine.load(fixtureModel(id: "tool-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 19_810)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let body: [String: Any] = [
            "model": "tool-model",
            "messages": [["role": "user", "content": "Weather in Paris?"]],
            "stream": false,
            "tools": [weatherToolSpec],
        ]
        let (data, response) = try await postRaw(url, jsonObject: body)

        await server.stop()

        #expect(response.statusCode == 200)
        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let choices = try #require(json["choices"] as? [[String: Any]])
        #expect(choices.count == 1)
        #expect(choices[0]["finish_reason"] as? String == "tool_calls")

        let message = try #require(choices[0]["message"] as? [String: Any])
        // content is JSON null when the assistant only called tools.
        let content = try #require(message["content"])
        #expect(content is NSNull)

        let toolCalls = try #require(message["tool_calls"] as? [[String: Any]])
        #expect(toolCalls.count == 1)
        #expect(toolCalls[0]["id"] as? String == "call_abc123")
        #expect(toolCalls[0]["type"] as? String == "function")
        let fn = try #require(toolCalls[0]["function"] as? [String: Any])
        #expect(fn["name"] as? String == "get_weather")
        // arguments is a JSON-encoded STRING, not a nested object.
        let argsString = try #require(fn["arguments"] as? String)
        let argsObj = try #require(
            try JSONSerialization.jsonObject(with: Data(argsString.utf8)) as? [String: Any]
        )
        #expect(argsObj["city"] as? String == "Paris")
    }

    /// Streaming: a `delta` frame carries the tool_call (with `index`), and a
    /// later frame carries `finish_reason:"tool_calls"`.
    @Test
    func chatCompletionsStreamingEmitsToolCallDelta() async throws {
        let engine = ToolCallStubEngine()
        try await engine.load(fixtureModel(id: "tool-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 19_820)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let body: [String: Any] = [
            "model": "tool-model",
            "messages": [["role": "user", "content": "Weather in Paris?"]],
            "stream": true,
            "tools": [weatherToolSpec],
        ]
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await URLSession.shared.data(for: req)
        let http = try #require(response as? HTTPURLResponse)

        await server.stop()

        #expect(http.statusCode == 200)
        let frames = sseFrames(data)

        var sawToolCallDelta = false
        var sawToolCallsFinish = false
        var toolCallID: String?
        var toolCallName: String?
        var toolCallArgs: String?
        for frame in frames {
            guard let choices = frame["choices"] as? [[String: Any]],
                  let choice = choices.first
            else { continue }
            if let delta = choice["delta"] as? [String: Any],
               let calls = delta["tool_calls"] as? [[String: Any]],
               let call = calls.first {
                sawToolCallDelta = true
                #expect(call["index"] as? Int == 0)
                #expect(call["type"] as? String == "function")
                toolCallID = call["id"] as? String
                if let fn = call["function"] as? [String: Any] {
                    toolCallName = fn["name"] as? String
                    toolCallArgs = fn["arguments"] as? String
                }
            }
            if choice["finish_reason"] as? String == "tool_calls" {
                sawToolCallsFinish = true
            }
        }
        #expect(sawToolCallDelta)
        #expect(sawToolCallsFinish)
        #expect(toolCallID == "call_abc123")
        #expect(toolCallName == "get_weather")
        let argsString = try #require(toolCallArgs)
        let argsObj = try #require(
            try JSONSerialization.jsonObject(with: Data(argsString.utf8)) as? [String: Any]
        )
        #expect(argsObj["city"] as? String == "Paris")
    }

    /// `tool_choice:"none"` suppresses tools — `GenerateRequest.tools` stays nil
    /// even though `tools` was supplied.
    @Test
    func chatCompletionsToolChoiceNoneOmitsTools() async throws {
        let engine = CapturingStubEngine()
        try await engine.load(fixtureModel(id: "stub-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 19_830)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let body: [String: Any] = [
            "model": "stub-model",
            "messages": [["role": "user", "content": "Weather in Paris?"]],
            "stream": false,
            "tools": [weatherToolSpec],
            "tool_choice": "none",
        ]
        let (_, response) = try await postRaw(url, jsonObject: body)

        await server.stop()

        #expect(response.statusCode == 200)
        #expect(await engine.capturedRequest?.tools == nil)
    }

    /// Back-compat: a request with NO `tools` field decodes fine and behaves
    /// exactly as before — `GenerateRequest.tools` is nil and the response is a
    /// plain assistant message.
    @Test
    func chatCompletionsWithoutToolsFieldDecodesUnchanged() async throws {
        let engine = CapturingStubEngine()
        try await engine.load(fixtureModel(id: "stub-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 19_840)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let body: [String: Any] = [
            "model": "stub-model",
            "messages": [["role": "user", "content": "Hello"]],
            "stream": false,
        ]
        let (data, response) = try await postRaw(url, jsonObject: body)

        await server.stop()

        #expect(response.statusCode == 200)
        #expect(await engine.capturedRequest?.tools == nil)
        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let choices = try #require(json["choices"] as? [[String: Any]])
        let message = try #require(choices[0]["message"] as? [String: Any])
        #expect(message["content"] as? String == "stub-response")
        #expect(message["tool_calls"] == nil)
    }

    // MARK: - Speculative decoding pass-through (D1, OpenAI /v1/chat/completions)
    //
    // NOT OpenAI-standard fields — mirrors mlx-lm's Python server `draft_model`
    // / `num_draft_tokens` request fields. These tests use stub engines so no
    // real model (or draft model) is needed; the real draft-model load/swap
    // state machine is covered by `MLXSwiftEngineSpeculativeDecodingTests`,
    // and real on/off token parity by the gated
    // `SpeculativeDecodingModelTests`.
    //
    // Port assignments (spaced by 10):
    //   chatCompletionsDraftModelAndNumDraftTokensPopulateGenerateRequest : 19_850
    //   chatCompletionsNumDraftTokensClampsOutOfRangeValue                : 19_860
    //   chatCompletionsNonStreamingSurfacesSpeculativeDecodingUsage       : 19_870
    //   chatCompletionsWithoutDraftModelOmitsSpeculativeDecodingUsage     : 19_880

    /// A request carrying `draft_model`/`num_draft_tokens` populates
    /// `GenerateRequest.draftModelID`/`numDraftTokens` verbatim.
    @Test
    func chatCompletionsDraftModelAndNumDraftTokensPopulateGenerateRequest() async throws {
        let engine = CapturingStubEngine()
        try await engine.load(fixtureModel(id: "stub-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 19_850)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let body: [String: Any] = [
            "model": "stub-model",
            "messages": [["role": "user", "content": "Hi"]],
            "stream": false,
            "draft_model": "small-draft-model",
            "num_draft_tokens": 5,
        ]
        let (_, response) = try await postRaw(url, jsonObject: body)

        await server.stop()

        #expect(response.statusCode == 200)
        let captured = try #require(await engine.capturedRequest)
        #expect(captured.draftModelID == "small-draft-model")
        #expect(captured.numDraftTokens == 5)
    }

    /// `num_draft_tokens` outside 1...8 is clamped (via `GenerateRequest`)
    /// before it ever reaches the engine.
    @Test
    func chatCompletionsNumDraftTokensClampsOutOfRangeValue() async throws {
        let engine = CapturingStubEngine()
        try await engine.load(fixtureModel(id: "stub-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 19_860)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let body: [String: Any] = [
            "model": "stub-model",
            "messages": [["role": "user", "content": "Hi"]],
            "stream": false,
            "draft_model": "small-draft-model",
            "num_draft_tokens": 99,
        ]
        let (_, response) = try await postRaw(url, jsonObject: body)

        await server.stop()

        #expect(response.statusCode == 200)
        #expect(await engine.capturedRequest?.numDraftTokens == 8)
    }

    /// Stub engine whose terminal chunk carries speculative decoding
    /// telemetry — the shape `MLXSwiftEngine` produces when mlx-swift-lm
    /// reports accepted/proposed counts for a speculative generation. Lets
    /// the server's non-standard `usage.speculative_decoding` pass-through be
    /// exercised without a real model or a real draft model.
    private actor SpeculativeStubEngine: InferenceEngine {
        nonisolated let engineID: EngineID = .mlxSwift
        private(set) var status: EngineStatus = .idle
        private(set) var loadedModel: LocalModel?
        let version = "spec-1"

        func load(_ model: LocalModel) async throws {
            status = .loading(model: model.id)
            loadedModel = model
            status = .ready(model: model.id)
        }

        func unload() async throws {
            loadedModel = nil
            status = .idle
        }

        nonisolated func generate(_ request: GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error> {
            AsyncThrowingStream { continuation in
                continuation.yield(GenerateChunk(
                    text: "spec-response",
                    finishReason: .stop,
                    usage: TokenUsage(promptTokens: 5, completionTokens: 6),
                    speculativeDecoding: SpeculativeDecodingUsage(proposedTokens: 6, acceptedTokens: 4)
                ))
                continuation.finish()
            }
        }

        func healthCheck() async -> Bool { true }
    }

    /// The non-streaming response surfaces accepted/proposed draft-token
    /// counts as a non-standard `usage.speculative_decoding` sub-object when
    /// the terminal chunk carries them.
    @Test
    func chatCompletionsNonStreamingSurfacesSpeculativeDecodingUsage() async throws {
        let engine = SpeculativeStubEngine()
        try await engine.load(fixtureModel(id: "stub-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 19_870)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let body: [String: Any] = [
            "model": "stub-model",
            "messages": [["role": "user", "content": "Hi"]],
            "stream": false,
            "draft_model": "small-draft-model",
        ]
        let (data, response) = try await postRaw(url, jsonObject: body)

        await server.stop()

        #expect(response.statusCode == 200)
        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let usage = try #require(json["usage"] as? [String: Any])
        #expect(usage["prompt_tokens"] as? Int == 5)
        #expect(usage["completion_tokens"] as? Int == 6)
        let spec = try #require(usage["speculative_decoding"] as? [String: Any])
        #expect(spec["proposed_tokens"] as? Int == 6)
        #expect(spec["accepted_tokens"] as? Int == 4)
    }

    /// Back-compat / no-fake-fields: a generation that never ran the
    /// speculative path (terminal chunk's `speculativeDecoding` is nil, as
    /// with every existing stub/engine before D1) omits
    /// `usage.speculative_decoding` entirely rather than emitting a
    /// zero-filled placeholder.
    @Test
    func chatCompletionsWithoutDraftModelOmitsSpeculativeDecodingUsage() async throws {
        let engine = CapturingStubEngine()
        try await engine.load(fixtureModel(id: "stub-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 19_880)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let body: [String: Any] = [
            "model": "stub-model",
            "messages": [["role": "user", "content": "Hi"]],
            "stream": false,
        ]
        let (data, response) = try await postRaw(url, jsonObject: body)

        await server.stop()

        #expect(response.statusCode == 200)
        #expect(await engine.capturedRequest?.draftModelID == nil)
        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let usage = try #require(json["usage"] as? [String: Any])
        #expect(usage["speculative_decoding"] == nil)
    }

    /// Split an SSE response body into decoded `data:` JSON frames, dropping the
    /// terminal `[DONE]` sentinel. Used by the streaming tool-call test.
    private func sseFrames(_ data: Data) -> [[String: Any]] {
        let text = String(decoding: data, as: UTF8.self)
        return text.components(separatedBy: "\n\n").compactMap { block in
            guard let range = block.range(of: "data: ") else { return nil }
            let payload = String(block[range.upperBound...])
            guard payload != "[DONE]",
                  let obj = try? JSONSerialization.jsonObject(with: Data(payload.utf8)) as? [String: Any]
            else { return nil }
            return obj
        }
    }

    // MARK: - Anthropic Messages API (v0.5.1)
    //
    // Port assignments (spaced by 10):
    //   anthropicMessagesNonStreaming  : 19_400
    //   anthropicMessagesSystemTopLevel: 19_410
    //   anthropicMessagesStreaming     : 19_420

    /// Build a stub engine with a model already loaded, matching the
    /// setup `chatCompletionsNonStreaming` uses.
    private func loadedStubServer(modelID: String = "stub-model") async throws -> HummingbirdServer {
        let engine = StubInferenceEngine(engineID: .mlxSwift)
        let model = LocalModel(
            id: modelID,
            displayName: "Stub",
            directory: URL(filePath: "/tmp"),
            sizeBytes: 0,
            format: .mlx,
            quantization: nil,
            parameterCount: nil,
            architecture: nil
        )
        try await engine.load(model)
        return HummingbirdServer(engine: engine)
    }

    @Test
    func anthropicMessagesNonStreaming() async throws {
        let server = try await loadedStubServer()
        let port = try await server.start(preferredPort: 19_400)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/messages")!
        let body: [String: Any] = [
            "model": "stub-model",
            "max_tokens": 64,
            "messages": [["role": "user", "content": "Hello"]],
        ]
        let (data, response) = try await postRaw(url, jsonObject: body)

        await server.stop()

        #expect(response.statusCode == 200)
        let json = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        #expect(json["type"] as? String == "message")
        let content = try #require(json["content"] as? [[String: Any]])
        #expect(content.count == 1)
        // StubInferenceEngine yields "stub-" + "response"
        #expect(content[0]["text"] as? String == "stub-response")
        let usage = try #require(json["usage"] as? [String: Any])
        #expect(usage["input_tokens"] as? Int == 1)
        #expect(usage["output_tokens"] as? Int == 2)
        #expect(json["stop_reason"] as? String == "end_turn")
    }

    @Test
    func anthropicMessagesSystemTopLevel() async throws {
        let server = try await loadedStubServer()
        let port = try await server.start(preferredPort: 19_410)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/messages")!
        let body: [String: Any] = [
            "model": "stub-model",
            "max_tokens": 64,
            "system": "You are terse.",
            "messages": [["role": "user", "content": "Hello"]],
        ]
        let (data, response) = try await postRaw(url, jsonObject: body)

        await server.stop()

        #expect(response.statusCode == 200)
        let json = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        let content = try #require(json["content"] as? [[String: Any]])
        #expect(content[0]["text"] as? String == "stub-response")
    }

    @Test
    func anthropicMessagesStreaming() async throws {
        let server = try await loadedStubServer()
        let port = try await server.start(preferredPort: 19_420)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/messages")!
        let body: [String: Any] = [
            "model": "stub-model",
            "max_tokens": 64,
            "stream": true,
            "messages": [["role": "user", "content": "Hello"]],
        ]
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await URLSession.shared.data(for: req)
        let http = try #require(response as? HTTPURLResponse)

        await server.stop()

        #expect(http.statusCode == 200)
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains("event: message_start"))
        #expect(text.contains("event: content_block_delta"))
        #expect(text.contains("text_delta"))
        #expect(text.contains("event: message_stop"))
    }

    // MARK: - Anthropic tool loop (agent-tools wave)
    //
    // Claude Code speaks this protocol natively. These prove request decode
    // (tools + tool_use/tool_result history) and response encode (tool_use
    // content blocks, non-streaming + streaming) with stub engines — no model.
    //
    // Port assignments (spaced by 10): 20_010 … 20_080.

    /// Parse an Anthropic named-event SSE body into `(event, json)` pairs.
    private func anthropicEvents(_ data: Data) -> [(event: String, json: [String: Any])] {
        let text = String(decoding: data, as: UTF8.self)
        return text.components(separatedBy: "\n\n").compactMap { block in
            guard let eRange = block.range(of: "event: "),
                  let dRange = block.range(of: "\ndata: ")
            else { return nil }
            let event = String(block[eRange.upperBound..<dRange.lowerBound])
            let payload = String(block[dRange.upperBound...])
            guard let obj = try? JSONSerialization.jsonObject(with: Data(payload.utf8)) as? [String: Any]
            else { return nil }
            return (event, obj)
        }
    }

    /// A single Anthropic `tools[]` entry `{name, description, input_schema}`.
    private var anthropicWeatherTool: [String: Any] {
        [
            "name": "get_weather",
            "description": "Get the weather for a city",
            "input_schema": [
                "type": "object",
                "properties": ["city": ["type": "string"]],
                "required": ["city"],
            ],
        ]
    }

    /// Anthropic `tools` convert to the internal OpenAI-function-spec shape that
    /// lands verbatim in `GenerateRequest.tools` (`input_schema` → `parameters`).
    @Test
    func anthropicMessagesToolsPopulateGenerateRequestTools() async throws {
        let engine = CapturingStubEngine()
        try await engine.load(fixtureModel(id: "stub-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 20_010)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/messages")!
        let body: [String: Any] = [
            "model": "stub-model",
            "max_tokens": 64,
            "messages": [["role": "user", "content": "Weather in Paris?"]],
            "tools": [anthropicWeatherTool],
        ]
        let (_, response) = try await postRaw(url, jsonObject: body)
        await server.stop()

        #expect(response.statusCode == 200)
        let tools = try #require(await engine.capturedRequest?.tools)
        #expect(tools.count == 1)
        guard case .object(let spec) = tools[0] else {
            Issue.record("tool spec did not decode as a JSON object")
            return
        }
        #expect(spec["type"] == .string("function"))
        guard case .object(let fn)? = spec["function"] else {
            Issue.record("function payload missing")
            return
        }
        #expect(fn["name"] == .string("get_weather"))
        // input_schema maps to function.parameters verbatim.
        guard case .object(let params)? = fn["parameters"] else {
            Issue.record("parameters missing")
            return
        }
        #expect(params["type"] == .string("object"))
    }

    /// Full Anthropic tool history decodes with pairing: an assistant turn's
    /// `tool_use` block → `ChatMessage.toolCalls`, and a following user turn's
    /// `tool_result` block → a `.tool` message (emitted before the residual user
    /// text, which is NOT lost). tool_result content given as an array of text
    /// blocks flattens correctly.
    @Test
    func anthropicMessagesDecodeToolHistory() async throws {
        let engine = CapturingStubEngine()
        try await engine.load(fixtureModel(id: "stub-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 20_020)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/messages")!
        let body: [String: Any] = [
            "model": "stub-model",
            "max_tokens": 64,
            "messages": [
                ["role": "user", "content": "Weather in Paris?"],
                [
                    "role": "assistant",
                    "content": [
                        ["type": "text", "text": "Let me check."],
                        [
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "get_weather",
                            "input": ["city": "Paris"],
                        ],
                    ],
                ],
                [
                    "role": "user",
                    "content": [
                        [
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [["type": "text", "text": "22C and sunny"]],
                        ],
                        ["type": "text", "text": "Is that warm?"],
                    ],
                ],
            ],
        ]
        let (_, response) = try await postRaw(url, jsonObject: body)
        await server.stop()

        #expect(response.statusCode == 200)
        let messages = try #require(await engine.capturedRequest?.messages)
        // user, assistant(+toolCalls), tool(result), user(residual text).
        #expect(messages.map(\.role) == [.user, .assistant, .tool, .user])

        let calls = try #require(messages[1].toolCalls)
        #expect(calls.count == 1)
        #expect(calls[0].id == "toolu_1")
        #expect(calls[0].name == "get_weather")
        #expect(calls[0].arguments["city"] == .string("Paris"))
        #expect(messages[1].content == "Let me check.")

        #expect(messages[2].toolCallID == "toolu_1")
        #expect(messages[2].content == "22C and sunny")

        // The user text in the same turn as the tool_result is preserved.
        #expect(messages[3].content == "Is that warm?")
    }

    /// Non-streaming, TOOL-ONLY reply (no free text — `ToolCallStubEngine` yields
    /// `text: ""`): the empty text block is OMITTED entirely (real Anthropic never
    /// emits one), so `content[0]` IS the `tool_use` block, whose `input` is a
    /// DECODED JSON object (not a string), with `stop_reason:"tool_use"`.
    @Test
    func anthropicMessagesNonStreamingEmitsToolUse() async throws {
        let engine = ToolCallStubEngine()
        try await engine.load(fixtureModel(id: "tool-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 20_040)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/messages")!
        let body: [String: Any] = [
            "model": "tool-model",
            "max_tokens": 64,
            "messages": [["role": "user", "content": "Weather in Paris?"]],
            "tools": [anthropicWeatherTool],
        ]
        let (data, response) = try await postRaw(url, jsonObject: body)
        await server.stop()

        #expect(response.statusCode == 200)
        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(json["stop_reason"] as? String == "tool_use")
        let content = try #require(json["content"] as? [[String: Any]])
        // No empty text block — content is JUST the tool_use block, at index 0.
        #expect(content.count == 1)
        #expect(content.contains { $0["type"] as? String == "text" } == false)
        let toolBlock = try #require(content.first { $0["type"] as? String == "tool_use" })
        #expect(toolBlock["id"] as? String == "call_abc123")
        #expect(toolBlock["name"] as? String == "get_weather")
        // input is an OBJECT, not a JSON string.
        let input = try #require(toolBlock["input"] as? [String: Any])
        #expect(input["city"] as? String == "Paris")
    }

    /// Streaming, TOOL-ONLY reply (no free text): the text content_block never
    /// opens (real Anthropic never emits an empty one), so the `tool_use` block
    /// streams at index 0 — content_block_start → one `input_json_delta` (full
    /// args) → content_block_stop — and `message_delta` carries
    /// `stop_reason:"tool_use"`.
    @Test
    func anthropicMessagesStreamingEmitsToolUse() async throws {
        let engine = ToolCallStubEngine()
        try await engine.load(fixtureModel(id: "tool-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 20_050)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/messages")!
        let body: [String: Any] = [
            "model": "tool-model",
            "max_tokens": 64,
            "stream": true,
            "messages": [["role": "user", "content": "Weather in Paris?"]],
            "tools": [anthropicWeatherTool],
        ]
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await URLSession.shared.data(for: req)
        let http = try #require(response as? HTTPURLResponse)
        await server.stop()

        #expect(http.statusCode == 200)
        let events = anthropicEvents(data)

        // No text content_block_start at all (tool-only reply).
        #expect(events.contains {
            $0.event == "content_block_start"
                && (($0.json["content_block"] as? [String: Any])?["type"] as? String) == "text"
        } == false)
        // No text_delta either.
        #expect(events.contains {
            $0.event == "content_block_delta"
                && (($0.json["delta"] as? [String: Any])?["type"] as? String) == "text_delta"
        } == false)

        // tool_use content_block_start at index 0 (no text block precedes it).
        let toolStart = try #require(events.first {
            $0.event == "content_block_start"
                && (($0.json["content_block"] as? [String: Any])?["type"] as? String) == "tool_use"
        })
        #expect(toolStart.json["index"] as? Int == 0)
        let cb = try #require(toolStart.json["content_block"] as? [String: Any])
        #expect(cb["id"] as? String == "call_abc123")
        #expect(cb["name"] as? String == "get_weather")

        // input_json_delta carrying the full arguments JSON.
        let inputDelta = try #require(events.first {
            $0.event == "content_block_delta"
                && (($0.json["delta"] as? [String: Any])?["type"] as? String) == "input_json_delta"
        })
        let partial = try #require((inputDelta.json["delta"] as? [String: Any])?["partial_json"] as? String)
        let argsObj = try #require(
            try JSONSerialization.jsonObject(with: Data(partial.utf8)) as? [String: Any])
        #expect(argsObj["city"] as? String == "Paris")

        // message_delta carries the tool_use stop reason.
        let msgDelta = try #require(events.first { $0.event == "message_delta" })
        #expect((msgDelta.json["delta"] as? [String: Any])?["stop_reason"] as? String == "tool_use")
    }

    /// Mixed reply: the model emits SOME free text before/around a tool call
    /// (unlike `ToolCallStubEngine`, which is tool-only). The text block DOES
    /// open in this case, so the tool_use block streams at index 1 — proving the
    /// index-0 special case above is specifically about the tool-ONLY reply, not
    /// tool calls in general.
    @Test
    func anthropicMessagesStreamingMixedTextAndToolUseIndexesToolBlockAfterText() async throws {
        let engine = TextThenToolCallStubEngine()
        try await engine.load(fixtureModel(id: "tool-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 20_045)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/messages")!
        let body: [String: Any] = [
            "model": "tool-model",
            "max_tokens": 64,
            "stream": true,
            "messages": [["role": "user", "content": "Weather in Paris?"]],
            "tools": [anthropicWeatherTool],
        ]
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await URLSession.shared.data(for: req)
        let http = try #require(response as? HTTPURLResponse)
        await server.stop()

        #expect(http.statusCode == 200)
        let events = anthropicEvents(data)

        #expect(events.contains {
            $0.event == "content_block_start"
                && (($0.json["content_block"] as? [String: Any])?["type"] as? String) == "text"
        })
        let toolStart = try #require(events.first {
            $0.event == "content_block_start"
                && (($0.json["content_block"] as? [String: Any])?["type"] as? String) == "tool_use"
        })
        #expect(toolStart.json["index"] as? Int == 1)
    }

    /// An unsupported `tool_choice` mode (anything but `auto`) is a 400 — we
    /// don't silently ignore a force/suppress directive we can't honor.
    @Test
    func anthropicMessagesToolChoiceNonAutoReturns400() async throws {
        let engine = CapturingStubEngine()
        try await engine.load(fixtureModel(id: "stub-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 20_060)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/messages")!
        let body: [String: Any] = [
            "model": "stub-model",
            "max_tokens": 64,
            "messages": [["role": "user", "content": "Weather in Paris?"]],
            "tools": [anthropicWeatherTool],
            "tool_choice": ["type": "any"],
        ]
        let (_, response) = try await postRaw(url, jsonObject: body)
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(await engine.capturedRequest == nil)
    }

    /// A malformed `tool_use` block (`input` not a JSON object) is a 400 — never
    /// silently dropped.
    @Test
    func anthropicMessagesMalformedToolUseReturns400() async throws {
        let engine = CapturingStubEngine()
        try await engine.load(fixtureModel(id: "stub-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 20_070)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/messages")!
        let body: [String: Any] = [
            "model": "stub-model",
            "max_tokens": 64,
            "messages": [[
                "role": "assistant",
                "content": [[
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": "not-an-object",
                ]],
            ]],
        ]
        let (_, response) = try await postRaw(url, jsonObject: body)
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(await engine.capturedRequest == nil)
    }

    /// A `tool_use` block with a missing `id` is a 400 (a synthesised id would
    /// mispair with the following `tool_result`, so — unlike the OpenAI call id
    /// — this is never defaulted).
    @Test
    func anthropicMessagesMissingToolUseIDReturns400() async throws {
        let engine = CapturingStubEngine()
        try await engine.load(fixtureModel(id: "stub-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 20_011)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/messages")!
        let body: [String: Any] = [
            "model": "stub-model",
            "max_tokens": 64,
            "messages": [[
                "role": "assistant",
                "content": [[
                    "type": "tool_use",
                    "name": "get_weather",
                    "input": ["city": "Paris"],
                ]],
            ]],
        ]
        let (_, response) = try await postRaw(url, jsonObject: body)
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(await engine.capturedRequest == nil)
    }

    /// A `tool_result` block with a missing `tool_use_id` is a 400 — the
    /// correlator the chat template pairs on, so admitting one without it would
    /// reach the template unpaired.
    @Test
    func anthropicMessagesMissingToolResultIDReturns400() async throws {
        let engine = CapturingStubEngine()
        try await engine.load(fixtureModel(id: "stub-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 20_012)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/messages")!
        let body: [String: Any] = [
            "model": "stub-model",
            "max_tokens": 64,
            "messages": [
                ["role": "user", "content": "Weather in Paris?"],
                [
                    "role": "assistant",
                    "content": [[
                        "type": "tool_use", "id": "toolu_1", "name": "get_weather",
                        "input": ["city": "Paris"],
                    ]],
                ],
                [
                    "role": "user",
                    "content": [["type": "tool_result", "content": "22C and sunny"]],
                ],
            ],
        ]
        let (_, response) = try await postRaw(url, jsonObject: body)
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(await engine.capturedRequest == nil)
    }

    /// A `tool_result` answering a `tool_use_id` that no preceding assistant
    /// `tool_use` block ever issued is an ORPHAN result — a 400, mirroring the
    /// OpenAI orphan-result guard.
    @Test
    func anthropicMessagesOrphanToolResultReturns400() async throws {
        let engine = CapturingStubEngine()
        try await engine.load(fixtureModel(id: "stub-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 20_013)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/messages")!
        let body: [String: Any] = [
            "model": "stub-model",
            "max_tokens": 64,
            "messages": [
                ["role": "user", "content": "Weather in Paris?"],
                [
                    "role": "user",
                    "content": [[
                        "type": "tool_result", "tool_use_id": "toolu_never_issued",
                        "content": "22C and sunny",
                    ]],
                ],
            ],
        ]
        let (_, response) = try await postRaw(url, jsonObject: body)
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(await engine.capturedRequest == nil)
    }

    /// Two assistant `tool_use` blocks (in different turns) reusing the same
    /// `id` is a 400 — ids must be unique across the whole history, mirroring
    /// the OpenAI duplicate-call-id guard.
    @Test
    func anthropicMessagesDuplicateToolUseIDReturns400() async throws {
        let engine = CapturingStubEngine()
        try await engine.load(fixtureModel(id: "stub-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 20_014)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/messages")!
        let toolUseBlock: [String: Any] = [
            "type": "tool_use", "id": "toolu_dup", "name": "get_weather",
            "input": ["city": "Paris"],
        ]
        let body: [String: Any] = [
            "model": "stub-model",
            "max_tokens": 64,
            "messages": [
                ["role": "user", "content": "Weather in Paris?"],
                ["role": "assistant", "content": [toolUseBlock]],
                [
                    "role": "user",
                    "content": [
                        ["type": "tool_result", "tool_use_id": "toolu_dup", "content": "22C"]
                    ],
                ],
                ["role": "user", "content": "And tomorrow?"],
                ["role": "assistant", "content": [toolUseBlock]],
            ],
        ]
        let (_, response) = try await postRaw(url, jsonObject: body)
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(await engine.capturedRequest == nil)
    }

    /// A SECOND `tool_result` answering a `tool_use_id` that already has a
    /// result is a 400 — distinct from the orphan case (the id WAS issued, but
    /// is already consumed), mirroring the OpenAI already-consumed guard.
    @Test
    func anthropicMessagesToolResultAnsweringAlreadyConsumedCallReturns400() async throws {
        let engine = CapturingStubEngine()
        try await engine.load(fixtureModel(id: "stub-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 20_015)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/messages")!
        let body: [String: Any] = [
            "model": "stub-model",
            "max_tokens": 64,
            "messages": [
                ["role": "user", "content": "Weather in Paris?"],
                [
                    "role": "assistant",
                    "content": [[
                        "type": "tool_use", "id": "toolu_once", "name": "get_weather",
                        "input": ["city": "Paris"],
                    ]],
                ],
                [
                    "role": "user",
                    "content": [
                        ["type": "tool_result", "tool_use_id": "toolu_once", "content": "22C"],
                        ["type": "tool_result", "tool_use_id": "toolu_once", "content": "22C again"],
                    ],
                ],
            ],
        ]
        let (_, response) = try await postRaw(url, jsonObject: body)
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(await engine.capturedRequest == nil)
    }

    /// Regression: an Anthropic request WITHOUT tools decodes exactly as before —
    /// `GenerateRequest.tools` is nil and the response is a single text block
    /// with `stop_reason:"end_turn"`.
    @Test
    func anthropicMessagesWithoutToolsUnchanged() async throws {
        let engine = CapturingStubEngine()
        try await engine.load(fixtureModel(id: "stub-model"))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 20_080)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/messages")!
        let body: [String: Any] = [
            "model": "stub-model",
            "max_tokens": 64,
            "messages": [["role": "user", "content": "Hello"]],
        ]
        let (data, response) = try await postRaw(url, jsonObject: body)
        await server.stop()

        #expect(response.statusCode == 200)
        #expect(await engine.capturedRequest?.tools == nil)
        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let content = try #require(json["content"] as? [[String: Any]])
        #expect(content.count == 1)
        #expect(content[0]["type"] as? String == "text")
        #expect(json["stop_reason"] as? String == "end_turn")
    }

    // MARK: - Cold-swap (v0.3.3)
    //
    // Port assignments for these tests:
    //   coldSwapLoadsResolvedModel           : 19_200
    //   coldSwapReturns404WhenModelMissing   : 19_210

    /// Helper — build a LocalModel backed by a throwaway path.
    private func fixtureModel(id: String) -> LocalModel {
        LocalModel(
            id: id, displayName: id,
            directory: URL(filePath: "/tmp/\(id)"),
            sizeBytes: 0, format: .mlx,
            quantization: nil, parameterCount: nil, architecture: nil
        )
    }

    @Test
    func coldSwapLoadsResolvedModel() async throws {
        // Engine starts with no model loaded. Resolver knows about "other-model"
        // and can produce a LocalModel for it. A chat request naming
        // "other-model" should trigger a load and succeed.
        let engine = StubInferenceEngine(engineID: .mlxSwift)
        let resolver: HummingbirdServer.ModelResolver = { [fixtureModel] id in
            id == "other-model" ? fixtureModel("other-model") : nil
        }
        let server = HummingbirdServer(engine: engine, modelResolver: resolver)
        let port = try await server.start(preferredPort: 19_200)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let body: [String: Any] = [
            "model": "other-model",
            "messages": [["role": "user", "content": "Hi"]],
            "stream": false,
        ]
        let (data, response) = try await postRaw(url, jsonObject: body)
        let loaded = await engine.loadedModel?.id

        await server.stop()

        #expect(response.statusCode == 200)
        #expect(loaded == "other-model", "engine should have loaded the resolver's model")
        let json = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        #expect(json["object"] as? String == "chat.completion")
    }

    @Test
    func coldSwapReturns404WhenModelMissing() async throws {
        // Engine starts with no model loaded. Resolver always returns nil
        // (simulates a typo or not-downloaded model). Chat request should
        // come back as OpenAI-style 404 `model_not_found`.
        let engine = StubInferenceEngine(engineID: .mlxSwift)
        let resolver: HummingbirdServer.ModelResolver = { _ in nil }
        let server = HummingbirdServer(engine: engine, modelResolver: resolver)
        let port = try await server.start(preferredPort: 19_210)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let body: [String: Any] = [
            "model": "never-existed",
            "messages": [["role": "user", "content": "Hi"]],
            "stream": false,
        ]
        let (data, response) = try await postRaw(url, jsonObject: body)

        await server.stop()

        #expect(response.statusCode == 404)
        let json = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        let err = try #require(json["error"] as? [String: Any])
        #expect(err["code"] as? String == "model_not_found")
    }

    /// A1 (review follow-up): a STREAMING request for an unknown model
    /// must still get a real 404 — not a 200 with the error folded into
    /// an in-band SSE frame. SRV-2/SRV-3 moved the real resolve+load
    /// inside the `ResponseBody` writer closure (so swap-under-lock has
    /// the right scope), which silently swallowed the pre-existing
    /// 404-before-headers behaviour for unresolvable models on the
    /// streaming path. `canResolveModel`'s cheap pre-flight (no load, no
    /// lock) restores it without reintroducing the swap-outside-the-lock
    /// bug: the real load still only happens inside the locked closure.
    @Test
    func coldSwapStreamingReturns404WhenModelMissingBeforeHeadersSent() async throws {
        let engine = StubInferenceEngine(engineID: .mlxSwift)
        let resolver: HummingbirdServer.ModelResolver = { _ in nil }
        let server = HummingbirdServer(engine: engine, modelResolver: resolver)
        let port = try await server.start(preferredPort: 19_220)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let body: [String: Any] = [
            "model": "never-existed",
            "messages": [["role": "user", "content": "Hi"]],
            "stream": true,
        ]
        let (data, response) = try await postRaw(url, jsonObject: body)

        await server.stop()

        #expect(response.statusCode == 404, "unknown model on a streaming request must 404 before headers, not 200")
        let json = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        let err = try #require(json["error"] as? [String: Any])
        #expect(err["code"] as? String == "model_not_found")
    }

    @Test
    func portRetrySucceedsWhenPreferredPortBusy() async throws {
        // Occupy 19_100.
        let blocker = HummingbirdServer(engine: StubInferenceEngine(engineID: .mlxSwift))
        let occupiedPort = try await blocker.start(preferredPort: 19_100)

        // Start a second server at the same port — it should land on 19_101..19_120.
        let server2 = HummingbirdServer(engine: StubInferenceEngine(engineID: .mlxSwift))
        let actualPort = try await server2.start(preferredPort: occupiedPort)

        // Verify the second server actually serves requests.
        let url = URL(string: "http://127.0.0.1:\(actualPort)/health")!
        let (_, healthResponse) = try await get(url)

        // Stop both servers before assertions to ensure cleanup.
        await server2.stop()
        await blocker.stop()

        #expect(actualPort != occupiedPort)
        #expect(actualPort > occupiedPort)
        #expect(actualPort <= occupiedPort + 20)
        #expect(healthResponse.statusCode == 200)
    }

    // MARK: - v0.5.3 stability wave regression tests

    /// Stub engine whose `generate()` echoes its OWN loaded model id in
    /// the response text, unlike `StubInferenceEngine` (whose fixed
    /// "stub-response" text is depended on by many other tests above and
    /// is left unchanged). Lets a test prove WHICH engine instance
    /// actually answered a request (SRV-1), and reproduce the wrong-model
    /// race (SRV-2) via an optional per-generation delay. `hangAfterFirstChunk`
    /// simulates a true stall (SRV-4) — one chunk, then never finishes.
    private actor EchoingStubEngine: InferenceEngine {
        nonisolated let engineID: EngineID = .mlxSwift
        private(set) var status: EngineStatus = .idle
        private(set) var loadedModel: LocalModel?
        let version = "echo-1"

        private let chunkDelayNanos: UInt64
        private let hangAfterFirstChunk: Bool

        init(chunkDelayNanos: UInt64 = 0, hangAfterFirstChunk: Bool = false) {
            self.chunkDelayNanos = chunkDelayNanos
            self.hangAfterFirstChunk = hangAfterFirstChunk
        }

        func load(_ model: LocalModel) async throws {
            status = .loading(model: model.id)
            loadedModel = model
            status = .ready(model: model.id)
        }

        func unload() async throws {
            loadedModel = nil
            status = .idle
        }

        nonisolated func generate(_ request: GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error> {
            AsyncThrowingStream { continuation in
                Task { [weak self] in
                    guard let self else { continuation.finish(); return }
                    // Snapshot the model id up front — mirrors
                    // MLXSwiftEngine.runGeneration's `let support =
                    // loadedSupport` snapshot-at-start semantics, so a
                    // concurrent swap during `chunkDelayNanos` reproduces
                    // the same "answers from the wrong model" race SRV-2
                    // guards against.
                    let modelID = await self.loadedModel?.id ?? "none"
                    if self.chunkDelayNanos > 0 {
                        try? await Task.sleep(nanoseconds: self.chunkDelayNanos)
                    }
                    continuation.yield(GenerateChunk(text: "echo:\(modelID)"))
                    if self.hangAfterFirstChunk {
                        // True stall: never yield again, never finish.
                        try? await Task.sleep(nanoseconds: UInt64.max / 2)
                        return
                    }
                    continuation.yield(GenerateChunk(
                        text: "",
                        finishReason: .stop,
                        usage: TokenUsage(promptTokens: 1, completionTokens: 1)
                    ))
                    continuation.finish()
                }
            }
        }

        func healthCheck() async -> Bool { true }
    }

    /// Extract `choices[0].message.content` from a `/v1/chat/completions`
    /// non-streaming JSON body. Shared by the tests below.
    private func chatContent(_ data: Data) throws -> String {
        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let choices = try #require(json["choices"] as? [[String: Any]])
        let message = try #require(choices.first?["message"] as? [String: String])
        return try #require(message["content"])
    }

    /// SRV-1 (CRITICAL): the server must re-resolve the active engine on
    /// every request via `engineProvider`, not answer from a frozen
    /// reference captured once at construction. Simulates the GUI's
    /// `ModelPool`, which mints a brand-new `MLXSwiftEngine` per model —
    /// `loadHook` swaps an actor-boxed "active engine" to a NEW instance,
    /// and the response must come from that new instance.
    @Test
    func srv1EngineProviderReflectsSwapNotFrozenReference() async throws {
        let engineA = EchoingStubEngine()
        try await engineA.load(fixtureModel(id: "model-a"))

        let box = ActiveEngineBox(engineA)
        let loadHook: HummingbirdServer.LoadHook = { model in
            // A brand new instance — never mutates engineA in place.
            let engineB = EchoingStubEngine()
            try await engineB.load(model)
            await box.set(engineB)
        }
        let modelB = fixtureModel(id: "model-b")
        let resolver: HummingbirdServer.ModelResolver = { id in
            id == "model-b" ? modelB : nil
        }
        let server = HummingbirdServer(
            engineProvider: { await box.current },
            modelResolver: resolver,
            loadHook: loadHook
        )
        let port = try await server.start(preferredPort: 19_510)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let body: [String: Any] = [
            "model": "model-b",
            "messages": [["role": "user", "content": "Hi"]],
            "stream": false,
        ]
        let (data, response) = try await postRaw(url, jsonObject: body)
        await server.stop()

        #expect(response.statusCode == 200)
        let content = try chatContent(data)
        // A frozen `engine` reference (pre-SRV-1) would still answer from
        // engine A ("echo:model-a"); the fix re-resolves via
        // `engineProvider` AFTER the swap, so the response must reflect
        // the newly-resolved engine B.
        #expect(content == "echo:model-b", "response must come from the newly-resolved engine, not a stale one; got: \(content)")
    }

    /// SRV-2 (CRITICAL): model swap must be mutually exclusive with
    /// generation. Two concurrent requests naming DIFFERENT models must
    /// each be answered by their OWN requested model — never by whatever
    /// model the other request's swap happened to leave loaded mid-flight.
    /// `chunkDelayNanos` widens the race window so the pre-fix bug (swap
    /// running unlocked, ahead of an in-flight generation) would reliably
    /// reproduce wrong-model output without the fix.
    @Test
    func srv2ConcurrentDifferentModelRequestsEachGetTheirOwnModel() async throws {
        let engine = EchoingStubEngine(chunkDelayNanos: 150_000_000)  // 150ms
        try await engine.load(fixtureModel(id: "model-a"))

        let modelA = fixtureModel(id: "model-a")
        let modelB = fixtureModel(id: "model-b")
        let resolver: HummingbirdServer.ModelResolver = { id in
            id == "model-a" ? modelA : (id == "model-b" ? modelB : nil)
        }
        let server = HummingbirdServer(engineProvider: { engine }, modelResolver: resolver)
        let port = try await server.start(preferredPort: 19_600)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        func body(_ model: String) -> [String: Any] {
            ["model": model, "messages": [["role": "user", "content": "Hi"]], "stream": false]
        }

        async let respA = postRaw(url, jsonObject: body("model-a"))
        async let respB = postRaw(url, jsonObject: body("model-b"))
        let (dataA, _) = try await respA
        let (dataB, _) = try await respB
        await server.stop()

        let contentA = try chatContent(dataA)
        let contentB = try chatContent(dataB)
        #expect(contentA == "echo:model-a", "request naming model-a must be answered by model-a; got: \(contentA)")
        #expect(contentB == "echo:model-b", "request naming model-b must be answered by model-b; got: \(contentB)")
    }

    /// SRV-3b: a cancelled parked waiter must not deadlock the generation
    /// lock. Exercises `acquireGenerationLock`/`releaseGenerationLock`
    /// directly (both `internal`, visible via `@testable import`) since
    /// the scenario is about the lock PRIMITIVE's cancellation-awareness,
    /// not the HTTP layer: generation 1 holds the lock, generation 2
    /// parks behind it and is then cancelled, generation 1 releases, and
    /// generation 3 must still acquire promptly — the pre-fix
    /// `withCheckedContinuation` let a cancelled waiter receive ownership
    /// later and never release, deadlocking every subsequent generation.
    @Test
    func srv3bCancelledParkedWaiterDoesNotDeadlock() async throws {
        let server = makeServer()

        // Generation 1 acquires the lock.
        try await server.acquireGenerationLock()

        // Generation 2 parks behind it.
        let waiter2 = Task {
            try await server.acquireGenerationLock()
        }
        // Give it a moment to actually enqueue before cancelling.
        try await Task.sleep(nanoseconds: 100_000_000)
        waiter2.cancel()

        do {
            try await waiter2.value
            Issue.record("a cancelled parked waiter must not receive lock ownership")
        } catch is CancellationError {
            // Expected — the waiter was removed from the queue instead of
            // silently becoming the owner.
        }

        // Generation 1 finishes and releases.
        await server.releaseGenerationLock()

        // Generation 3 must acquire PROMPTLY. Race against a short ceiling
        // rather than trusting a bare `await` — under the pre-fix bug this
        // would hang forever (GPU idle, every request queued behind a
        // lock nobody will ever release).
        let acquired = await withTaskGroup(of: Bool.self) { group in
            group.addTask {
                (try? await server.acquireGenerationLock()) != nil
            }
            group.addTask {
                try? await Task.sleep(nanoseconds: 2_000_000_000)
                return false
            }
            let first = await group.next() ?? false
            group.cancelAll()
            return first
        }
        #expect(acquired, "a later acquire must succeed promptly — no deadlock from the cancelled waiter")
    }

    /// SRV-4 (HIGH, issue #29 root cause): a stall (no chunk for the
    /// configured timeout) must fail loudly with an HTTP 504 instead of
    /// hanging the client forever, AND must release the generation lock
    /// (SRV-3) so a follow-up request truly succeeds rather than queuing
    /// behind a leaked lock.
    @Test
    func srv4StallWatchdogTimesOutAndReleasesLock() async throws {
        let stallingEngine = EchoingStubEngine(hangAfterFirstChunk: true)
        try await stallingEngine.load(fixtureModel(id: "stall-model"))
        let box = ActiveEngineBox(stallingEngine)

        // Short test-configured stall timeout (1s) so the test stays fast.
        let server = HummingbirdServer(engineProvider: { await box.current }, stallTimeoutSeconds: 1)
        let port = try await server.start(preferredPort: 19_700)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let stallBody: [String: Any] = [
            "model": "stall-model",
            "messages": [["role": "user", "content": "Hi"]],
            "stream": false,
        ]

        let start = Date()
        let (data, response) = try await postRaw(url, jsonObject: stallBody)
        let elapsed = Date().timeIntervalSince(start)

        #expect(response.statusCode == 504)
        #expect(elapsed < 10, "stall watchdog should fire close to the configured 1s timeout, took \(elapsed)s")
        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let err = try #require(json["error"] as? [String: Any])
        #expect(err["code"] as? String == "generation_stalled")

        // Swap in a well-behaved engine and prove a follow-up request
        // truly SUCCEEDS — the generation lock from the stalled request
        // must have been released, not leaked.
        let healthyEngine = EchoingStubEngine()
        try await healthyEngine.load(fixtureModel(id: "ok-model"))
        await box.set(healthyEngine)

        let okBody: [String: Any] = [
            "model": "ok-model",
            "messages": [["role": "user", "content": "Hi"]],
            "stream": false,
        ]
        let (okData, okResponse) = try await postRaw(url, jsonObject: okBody)
        await server.stop()

        #expect(okResponse.statusCode == 200)
        let content = try chatContent(okData)
        #expect(content == "echo:ok-model")
    }

    // MARK: - v0.5.3 P3 cleanup wave
    //
    // Port assignments (spaced by 10):
    //   legacyCompletionsPromptOnlyReturns200                    : 19_050
    //   legacyCompletionsPromptOnlyStreamingReturns200           : 19_060
    //   streamingTokenCounterReflectsCompletionTokensNotFrames   : 19_070
    //   ollamaTagsReportsConfiguredAlias                         : 19_080

    /// Stub that streams a fixed number of text chunks then a terminal usage
    /// chunk, so a streaming test can prove the telemetry counter reflects the
    /// engine's reported `completionTokens` (K) rather than the SSE frame count
    /// (P3-4). `textChunks` and `reportedCompletionTokens` are deliberately
    /// distinct so the two are observable. Mirrors `EchoingStubEngine`'s
    /// synchronous-`let` access pattern from the nonisolated `generate`.
    private actor CountingStubEngine: InferenceEngine {
        nonisolated let engineID: EngineID = .mlxSwift
        private(set) var status: EngineStatus = .idle
        private(set) var loadedModel: LocalModel?
        let version = "counting-1"

        private let textChunks: Int
        private let reportedCompletionTokens: Int

        init(textChunks: Int, reportedCompletionTokens: Int) {
            self.textChunks = textChunks
            self.reportedCompletionTokens = reportedCompletionTokens
        }

        func load(_ model: LocalModel) async throws {
            status = .loading(model: model.id)
            loadedModel = model
            status = .ready(model: model.id)
        }

        func unload() async throws {
            loadedModel = nil
            status = .idle
        }

        nonisolated func generate(_ request: GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error> {
            AsyncThrowingStream { continuation in
                Task { [weak self] in
                    guard let self else { continuation.finish(); return }
                    for i in 0..<self.textChunks {
                        continuation.yield(GenerateChunk(text: "tok\(i) "))
                    }
                    continuation.yield(GenerateChunk(
                        text: "",
                        finishReason: .stop,
                        usage: TokenUsage(promptTokens: 3, completionTokens: self.reportedCompletionTokens)
                    ))
                    continuation.finish()
                }
            }
        }

        func healthCheck() async -> Bool { true }
    }

    /// P3-1: a legacy text-completions body (`{"model","prompt",...}` with NO
    /// `messages`) POSTed to `/v1/completions` must be accepted and answered
    /// with the same chat-format response — not rejected with a 400 for the
    /// missing `messages` field.
    @Test
    func legacyCompletionsPromptOnlyReturns200() async throws {
        let server = try await loadedStubServer(modelID: "stub-model")
        let port = try await server.start(preferredPort: 19_050)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/completions")!
        let body: [String: Any] = [
            "model": "stub-model",
            "prompt": "Hello",
            "max_tokens": 16,
            "stream": false,
        ]
        let (data, response) = try await postRaw(url, jsonObject: body)
        await server.stop()

        #expect(response.statusCode == 200)
        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(json["object"] as? String == "chat.completion")
        let choices = try #require(json["choices"] as? [[String: Any]])
        #expect(!choices.isEmpty)
        // The `prompt` was wrapped into a single user turn, so the stub answers
        // exactly as it does for a normal chat request.
        let message = try #require(choices.first?["message"] as? [String: String])
        #expect(message["content"] == "stub-response")
    }

    /// P3-1: the same legacy prompt-only body with `stream:true` must stream a
    /// normal chat-completion SSE sequence (not 400).
    @Test
    func legacyCompletionsPromptOnlyStreamingReturns200() async throws {
        let server = try await loadedStubServer(modelID: "stub-model")
        let port = try await server.start(preferredPort: 19_060)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/completions")!
        let body: [String: Any] = [
            "model": "stub-model",
            "prompt": "Hello",
            "stream": true,
        ]
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await URLSession.shared.data(for: req)
        let http = try #require(response as? HTTPURLResponse)
        await server.stop()

        #expect(http.statusCode == 200)
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains("chat.completion.chunk"))
        #expect(text.contains("[DONE]"))
    }

    /// P3-4: on a STREAMING request the token telemetry counter must grow by
    /// the engine's reported `completionTokens` (K), not by the number of SSE
    /// frames (N = text chunks + terminal chunk). The stub emits N frames whose
    /// K differs from N, so the pre-fix `incrementTokens(chunkCount)` (frames)
    /// and the fix are observably different.
    @Test
    func streamingTokenCounterReflectsCompletionTokensNotFrames() async throws {
        let textChunks = 3
        let reportedK = 7  // != frame count (textChunks + 1 = 4)
        let engine = CountingStubEngine(textChunks: textChunks, reportedCompletionTokens: reportedK)
        try await engine.load(fixtureModel(id: "count-model"))
        let server = HummingbirdServer(engineProvider: { engine })
        let port = try await server.start(preferredPort: 19_070)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let body: [String: Any] = [
            "model": "count-model",
            "messages": [["role": "user", "content": "Hi"]],
            "stream": true,
        ]
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (_, response) = try await URLSession.shared.data(for: req)
        let http = try #require(response as? HTTPURLResponse)
        #expect(http.statusCode == 200)

        // The counter is bumped on the server AFTER the body closure finishes
        // writing, which can land just after the client receives [DONE]. Poll
        // /x/status until it settles (non-zero), then assert the exact value.
        let statusURL = URL(string: "http://127.0.0.1:\(port)/x/status")!
        var counted = 0
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline {
            let (data, _) = try await get(statusURL)
            if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let c = json["tokens_generated_total"] as? Int, c > 0 {
                counted = c
                break
            }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        await server.stop()

        #expect(
            counted == reportedK,
            "telemetry must count completion tokens (K=\(reportedK)), not SSE frames (N=\(textChunks + 1)); got \(counted)"
        )
    }

    /// P3-2: `/api/tags` must report the user-facing alias (when configured),
    /// matching `/v1/models` — not the raw directory id.
    @Test
    func ollamaTagsReportsConfiguredAlias() async throws {
        // Unique id + alias so this never collides with a real user override in
        // the shared ~/.mac-mlx/model-params dir the server reads; removed below.
        let modelID = "macmlx-test-\(UUID().uuidString)"
        let alias = "friendly-\(UUID().uuidString)"
        let store = ModelParametersStore()  // default dir — the same store the server reads
        try await store.save(ModelParameters(alias: alias), for: modelID)

        let engine = StubInferenceEngine(engineID: .mlxSwift)
        let model = LocalModel(
            id: modelID, displayName: modelID,
            directory: URL(filePath: "/tmp"), sizeBytes: 0, format: .mlx,
            quantization: nil, parameterCount: nil, architecture: nil
        )
        try await engine.load(model)
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 19_080)

        let url = URL(string: "http://127.0.0.1:\(port)/api/tags")!
        let (data, response) = try await get(url)

        await server.stop()
        await store.reset(for: modelID)  // clean up the override file regardless of assertions

        #expect(response.statusCode == 200)
        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let models = try #require(json["models"] as? [[String: Any]])
        #expect(models.count == 1)
        #expect(models.first?["name"] as? String == alias, "/api/tags must report the configured alias, not the raw dir id")
        #expect(models.first?["model"] as? String == alias)
    }
}

/// Actor box holding a mutable "active engine" reference — used by the
/// SRV-1/SRV-4 tests above to simulate `EngineCoordinator.activeEngine`
/// (a computed property that can resolve to a DIFFERENT engine instance
/// after a cold-swap, unlike a frozen `let engine` captured once).
private actor ActiveEngineBox {
    var current: any InferenceEngine
    init(_ engine: any InferenceEngine) { current = engine }
    func set(_ engine: any InferenceEngine) { current = engine }
}
