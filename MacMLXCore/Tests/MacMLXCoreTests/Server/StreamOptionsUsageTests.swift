import Foundation
import Testing
@testable import MacMLXCore

/// OpenAI `stream_options.include_usage` (B1 of the Foundation Models work) and
/// the prompt-cache figures that ride on the same `TokenUsage` extension —
/// `prompt_tokens_details.cached_tokens` on the OpenAI shapes and
/// `cache_read_input_tokens` on the Anthropic one.
///
/// Port assignments (spaced by 10, continuing HummingbirdServerTests' range):
///   includeUsageEmitsUsageChunkBeforeDone     : 20_200
///   withoutIncludeUsageTheStreamIsUnchanged   : 20_210
///   includeUsageIsIgnoredWhenNotStreaming     : 20_220
///   cachedTokensReportedWhenTheEngineKnows    : 20_230
///   cachedTokensRideTheUsageChunk             : 20_240
///   anthropicReportsCacheReadInputTokens      : 20_250
///   anthropicOmitsCacheReadWhenUnknown        : 20_260
///   legacyCompletionsHonorIncludeUsage        : 20_270
///   stallLeavesNoUsageChunk                   : 20_280
///   errorLeavesNoUsageChunk                   : 20_290
///   toolCallFramesCarryUsageNull              : 20_340
///   originHeaderWithholdsCacheFigures         : 20_350
///   maxCompletionTokensIsHonored              : 20_360
@Suite("StreamOptionsUsage")
struct StreamOptionsUsageTests {

    // MARK: Helpers

    private func fixtureModel(id: String) -> LocalModel {
        LocalModel(
            id: id, displayName: id, directory: URL(filePath: "/tmp"), sizeBytes: 0,
            format: .mlx, quantization: nil, parameterCount: nil, architecture: nil)
    }

    private func loadedServer(
        _ engine: StubInferenceEngine, modelID: String = "stub-model"
    ) async throws -> HummingbirdServer {
        try await engine.load(fixtureModel(id: modelID))
        return HummingbirdServer(engineProvider: { engine })
    }

    private func postRaw(
        _ url: URL, jsonObject: Any, headers: [String: String] = [:]
    ) async throws -> (Data, HTTPURLResponse) {
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        for (name, value) in headers { req.setValue(value, forHTTPHeaderField: name) }
        req.httpBody = try JSONSerialization.data(withJSONObject: jsonObject)
        let (data, response) = try await URLSession.shared.data(for: req)
        return (data, try #require(response as? HTTPURLResponse))
    }

    /// Decoded `data:` frames with `[DONE]` dropped, in wire order.
    private func frames(_ data: Data) throws -> [[String: Any]] {
        try ssePayloads(data).filter { $0 != "[DONE]" }.map(jsonObject)
    }

    private func hasUsageChunk(_ frames: [[String: Any]]) -> Bool {
        frames.contains { ($0["choices"] as? [Any])?.isEmpty == true }
    }

    /// Every `data:` payload in wire order, the `[DONE]` sentinel included as
    /// the literal string so a test can assert what comes last.
    private func ssePayloads(_ data: Data) -> [String] {
        String(decoding: data, as: UTF8.self)
            .components(separatedBy: "\n\n")
            .compactMap { block in
                guard let range = block.range(of: "data: ") else { return nil }
                return String(block[range.upperBound...])
            }
    }

    private func jsonObject(_ payload: String) throws -> [String: Any] {
        try #require(try JSONSerialization.jsonObject(with: Data(payload.utf8)) as? [String: Any])
    }

    private func chatBody(stream: Bool, streamOptions: [String: Any]? = nil) -> [String: Any] {
        var body: [String: Any] = [
            "model": "stub-model",
            "messages": [["role": "user", "content": "Hi"]],
            "stream": stream,
        ]
        if let streamOptions { body["stream_options"] = streamOptions }
        return body
    }

    // MARK: OpenAI chat completions

    /// With `include_usage`, the stream ends with one usage-only chunk — empty
    /// `choices`, the whole request's `usage` — immediately before `[DONE]`,
    /// and every other chunk carries `usage: null`. An unknown key inside
    /// `stream_options` is ignored, as OpenAI ignores it. The stub reports no
    /// cache figure, so `prompt_tokens_details` must be absent rather than a
    /// fabricated zero.
    @Test
    func includeUsageEmitsUsageChunkBeforeDone() async throws {
        let server = try await loadedServer(StubInferenceEngine(engineID: .mlxSwift))
        let port = try await server.start(preferredPort: 20_200)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let (data, response) = try await postRaw(
            url,
            jsonObject: chatBody(stream: true, streamOptions: ["include_usage": true, "future_option": 1]))
        await server.stop()
        #expect(response.statusCode == 200)

        let payloads = ssePayloads(data)
        #expect(payloads.last == "[DONE]", "the sentinel still closes the stream")
        let frames = try payloads.dropLast().map(jsonObject)
        #expect(frames.count >= 2, "at least one content chunk plus the usage chunk")

        let usageFrame = try #require(frames.last)
        #expect(usageFrame["object"] as? String == "chat.completion.chunk")
        #expect((usageFrame["choices"] as? [Any])?.isEmpty == true, "the usage chunk has an empty choices array")
        let usage = try #require(usageFrame["usage"] as? [String: Any])
        #expect(usage["prompt_tokens"] as? Int == 1)
        #expect(usage["completion_tokens"] as? Int == 2)
        #expect(usage["total_tokens"] as? Int == 3)
        #expect(usage["prompt_tokens_details"] == nil)

        for frame in frames.dropLast() {
            #expect(frame["usage"] is NSNull, "every non-final chunk carries usage: null")
            #expect(!((frame["choices"] as? [Any]) ?? []).isEmpty, "content chunks keep their choices")
        }
    }

    /// Without `include_usage` the wire is byte-for-byte what it was: no
    /// `usage` key on any chunk and no usage-only chunk.
    @Test
    func withoutIncludeUsageTheStreamIsUnchanged() async throws {
        let server = try await loadedServer(StubInferenceEngine(engineID: .mlxSwift))
        let port = try await server.start(preferredPort: 20_210)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let (data, response) = try await postRaw(url, jsonObject: chatBody(stream: true))
        await server.stop()
        #expect(response.statusCode == 200)

        let payloads = ssePayloads(data)
        #expect(payloads.last == "[DONE]")
        let frames = try payloads.dropLast().map(jsonObject)
        #expect(!frames.isEmpty)
        for frame in frames {
            #expect(frame["usage"] == nil, "no usage key unless the client asked")
            #expect(!((frame["choices"] as? [Any]) ?? []).isEmpty, "no usage-only chunk unless the client asked")
        }
    }

    /// `stream_options` on a non-streaming request is ignored: the body already
    /// carries `usage`, and the request is not rejected.
    @Test
    func includeUsageIsIgnoredWhenNotStreaming() async throws {
        let server = try await loadedServer(StubInferenceEngine(engineID: .mlxSwift))
        let port = try await server.start(preferredPort: 20_220)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let (data, response) = try await postRaw(
            url, jsonObject: chatBody(stream: false, streamOptions: ["include_usage": true]))
        await server.stop()
        #expect(response.statusCode == 200)

        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(json["object"] as? String == "chat.completion")
        let usage = try #require(json["usage"] as? [String: Any])
        #expect(usage["total_tokens"] as? Int == 3)
        #expect(usage["prompt_tokens_details"] == nil)
    }

    /// When the engine reports how much of the prompt the cache served, the
    /// non-streaming body carries it as `prompt_tokens_details.cached_tokens`.
    @Test
    func cachedTokensReportedWhenTheEngineKnows() async throws {
        let server = try await loadedServer(
            StubInferenceEngine(engineID: .mlxSwift, promptTokens: 10, cachedPromptTokens: 5))
        let port = try await server.start(preferredPort: 20_230)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let (data, response) = try await postRaw(url, jsonObject: chatBody(stream: false))
        await server.stop()
        #expect(response.statusCode == 200)

        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let usage = try #require(json["usage"] as? [String: Any])
        let details = try #require(usage["prompt_tokens_details"] as? [String: Any])
        #expect(details["cached_tokens"] as? Int == 5)
        #expect(usage["prompt_tokens"] as? Int == 10, "OpenAI's prompt_tokens includes the cached tokens")
    }

    /// The same figure rides the streaming usage chunk.
    @Test
    func cachedTokensRideTheUsageChunk() async throws {
        let server = try await loadedServer(
            StubInferenceEngine(engineID: .mlxSwift, cachedPromptTokens: 0))
        let port = try await server.start(preferredPort: 20_240)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let (data, response) = try await postRaw(
            url, jsonObject: chatBody(stream: true, streamOptions: ["include_usage": true]))
        await server.stop()
        #expect(response.statusCode == 200)

        let frames = try ssePayloads(data).dropLast().map(jsonObject)
        let usage = try #require(frames.last?["usage"] as? [String: Any])
        let details = try #require(usage["prompt_tokens_details"] as? [String: Any])
        #expect(details["cached_tokens"] as? Int == 0, "a reported miss is a real zero, not an omission")
    }

    // MARK: Anthropic messages

    private func anthropicBody(stream: Bool) -> [String: Any] {
        [
            "model": "stub-model",
            "max_tokens": 64,
            "messages": [["role": "user", "content": "Hi"]],
            "stream": stream,
        ]
    }

    /// `/v1/messages` reports the cache figure as `cache_read_input_tokens` and,
    /// as Anthropic counts, takes it OUT of `input_tokens` — a client that sums
    /// the two (Claude Code's context meter does) gets the prompt back exactly
    /// once. Both the non-streaming body and the streaming `message_delta`.
    @Test
    func anthropicReportsCacheReadInputTokens() async throws {
        let server = try await loadedServer(
            StubInferenceEngine(engineID: .mlxSwift, promptTokens: 10, cachedPromptTokens: 7))
        let port = try await server.start(preferredPort: 20_250)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/messages")!

        let (data, response) = try await postRaw(url, jsonObject: anthropicBody(stream: false))
        #expect(response.statusCode == 200)
        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let usage = try #require(json["usage"] as? [String: Any])
        #expect(usage["input_tokens"] as? Int == 3, "the uncached remainder")
        #expect(usage["cache_read_input_tokens"] as? Int == 7)

        let (streamed, streamedResponse) = try await postRaw(url, jsonObject: anthropicBody(stream: true))
        await server.stop()
        #expect(streamedResponse.statusCode == 200)
        let text = String(decoding: streamed, as: UTF8.self)
        let delta = try #require(
            text.components(separatedBy: "\n\n").first { $0.hasPrefix("event: message_delta") })
        let deltaPayload = try #require(delta.components(separatedBy: "data: ").last)
        let deltaJSON = try jsonObject(deltaPayload)
        let deltaUsage = try #require(deltaJSON["usage"] as? [String: Any])
        #expect(deltaUsage["input_tokens"] as? Int == 3)
        #expect(deltaUsage["cache_read_input_tokens"] as? Int == 7)
    }

    /// No figure from the engine means no `cache_read_input_tokens` key.
    @Test
    func anthropicOmitsCacheReadWhenUnknown() async throws {
        let server = try await loadedServer(StubInferenceEngine(engineID: .mlxSwift))
        let port = try await server.start(preferredPort: 20_260)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/messages")!
        let (data, response) = try await postRaw(url, jsonObject: anthropicBody(stream: false))
        await server.stop()
        #expect(response.statusCode == 200)
        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let usage = try #require(json["usage"] as? [String: Any])
        #expect(usage["cache_read_input_tokens"] == nil)
        #expect(usage["input_tokens"] as? Int == 1, "with no figure the full prompt stays in input_tokens")
    }

    // MARK: Legacy completions alias

    /// The legacy prompt-only `/v1/completions` body streams through the same
    /// responder, so `stream_options.include_usage` works there too.
    @Test
    func legacyCompletionsHonorIncludeUsage() async throws {
        let server = try await loadedServer(StubInferenceEngine(engineID: .mlxSwift))
        let port = try await server.start(preferredPort: 20_270)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/completions")!
        let body: [String: Any] = [
            "model": "stub-model",
            "prompt": "Hello",
            "stream": true,
            "stream_options": ["include_usage": true],
        ]
        let (data, response) = try await postRaw(url, jsonObject: body)
        await server.stop()
        #expect(response.statusCode == 200)

        let payloads = ssePayloads(data)
        #expect(payloads.last == "[DONE]")
        let frames = try payloads.dropLast().map(jsonObject)
        let usage = try #require(frames.last?["usage"] as? [String: Any])
        #expect(usage["total_tokens"] as? Int == 3)
        #expect((frames.last?["choices"] as? [Any])?.isEmpty == true)
    }

    // MARK: Exit paths

    /// Yields one chunk, then either hangs or throws, and never sends a
    /// terminal usage chunk — the two ways a stream ends abnormally.
    private actor AbnormalStubEngine: InferenceEngine {
        enum Ending { case hang, throwing }
        nonisolated let engineID: EngineID = .mlxSwift
        private(set) var status: EngineStatus = .idle
        private(set) var loadedModel: LocalModel?
        let version = "abnormal-1"
        private let ending: Ending

        init(ending: Ending) { self.ending = ending }

        func load(_ model: LocalModel) async throws {
            status = .loading(model: model.id)
            loadedModel = model
            status = .ready(model: model.id)
        }

        func unload() async throws {
            loadedModel = nil
            status = .idle
        }

        struct Boom: Error {}

        nonisolated func generate(_ request: GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error> {
            let ending = self.ending
            return AsyncThrowingStream { continuation in
                // Usage is attached to the chunk that DOES arrive, so the
                // no-usage-chunk rule below rests on how the stream ended, not
                // on a missing figure.
                continuation.yield(GenerateChunk(
                    text: "partial", usage: TokenUsage(promptTokens: 1, completionTokens: 1)))
                switch ending {
                case .throwing:
                    continuation.finish(throwing: Boom())
                case .hang:
                    Task {
                        try? await Task.sleep(nanoseconds: 30_000_000_000)
                        continuation.finish()
                    }
                }
            }
        }

        func healthCheck() async -> Bool { true }
    }

    /// A stall (the SRV-4 watchdog) ends the stream with an error frame; no
    /// usage chunk may follow it, and `[DONE]` is still last.
    @Test
    func stallLeavesNoUsageChunk() async throws {
        let engine = AbnormalStubEngine(ending: .hang)
        try await engine.load(fixtureModel(id: "stub-model"))
        let server = HummingbirdServer(engineProvider: { engine }, stallTimeoutSeconds: 1)
        let port = try await server.start(preferredPort: 20_280)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let (data, response) = try await postRaw(
            url, jsonObject: chatBody(stream: true, streamOptions: ["include_usage": true]))
        await server.stop()
        #expect(response.statusCode == 200)

        let payloads = ssePayloads(data)
        #expect(payloads.last == "[DONE]")
        let frames = try frames(data)
        #expect(frames.contains { (($0["error"] as? [String: Any])?["code"] as? String) == "generation_stalled" })
        #expect(!hasUsageChunk(frames), "no usage chunk after a stall frame")
    }

    /// An engine error ends the stream with an error frame; same rule.
    @Test
    func errorLeavesNoUsageChunk() async throws {
        let engine = AbnormalStubEngine(ending: .throwing)
        try await engine.load(fixtureModel(id: "stub-model"))
        let server = HummingbirdServer(engineProvider: { engine })
        let port = try await server.start(preferredPort: 20_290)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let (data, response) = try await postRaw(
            url, jsonObject: chatBody(stream: true, streamOptions: ["include_usage": true]))
        await server.stop()
        #expect(response.statusCode == 200)

        let payloads = ssePayloads(data)
        #expect(payloads.last == "[DONE]")
        let frames = try frames(data)
        #expect(frames.contains { $0["error"] != nil })
        #expect(!hasUsageChunk(frames), "no usage chunk after an error frame")
    }

    /// Ends in a tool call, so the server emits the tool-call frames (a separate
    /// code path from plain content frames).
    private actor ToolCallStubEngine: InferenceEngine {
        nonisolated let engineID: EngineID = .mlxSwift
        private(set) var status: EngineStatus = .idle
        private(set) var loadedModel: LocalModel?
        let version = "tool-1"

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
                let call = ToolCallRequest(id: "call_1", name: "get_weather", arguments: ["city": .string("Paris")])
                continuation.yield(GenerateChunk(text: "Let me check."))
                continuation.yield(GenerateChunk(
                    text: "",
                    finishReason: .toolCalls,
                    usage: TokenUsage(promptTokens: 3, completionTokens: 4),
                    toolCalls: [call]))
                continuation.finish()
            }
        }

        func healthCheck() async -> Bool { true }
    }

    /// Tool-call frames carry `usage: null` like every other chunk, and the
    /// usage chunk still comes last with the request's counts.
    @Test
    func toolCallFramesCarryUsageNull() async throws {
        let engine = ToolCallStubEngine()
        try await engine.load(fixtureModel(id: "stub-model"))
        let server = HummingbirdServer(engineProvider: { engine })
        let port = try await server.start(preferredPort: 20_340)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        var body = chatBody(stream: true, streamOptions: ["include_usage": true])
        body["tools"] = [["type": "function", "function": ["name": "get_weather", "parameters": ["type": "object"]]]]
        let (data, response) = try await postRaw(url, jsonObject: body)
        await server.stop()
        #expect(response.statusCode == 200)

        let frames = try frames(data)
        let toolFrame = frames.first { frame in
            (((frame["choices"] as? [[String: Any]])?.first?["delta"] as? [String: Any])?["tool_calls"]) != nil
        }
        #expect(toolFrame != nil, "a tool_calls delta frame was emitted")
        for frame in frames.dropLast() {
            #expect(frame["usage"] is NSNull, "every non-final frame carries usage: null")
        }
        let usageFrame = try #require(frames.last)
        #expect((usageFrame["choices"] as? [Any])?.isEmpty == true)
        #expect((usageFrame["usage"] as? [String: Any])?["total_tokens"] as? Int == 7)
    }

    // MARK: Cache figure exposure

    /// A request carrying `Origin` (a cross-origin browser caller) gets the
    /// counts but not the cache figure, on every shape that carries one: the
    /// figure is a prefix oracle against the cache and the server answers such
    /// callers without credentials.
    @Test
    func originHeaderWithholdsCacheFigures() async throws {
        let server = try await loadedServer(
            StubInferenceEngine(engineID: .mlxSwift, promptTokens: 10, cachedPromptTokens: 4))
        let port = try await server.start(preferredPort: 20_350)
        let origin = ["Origin": "https://evil.example"]

        let chatURL = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let (plain, plainResponse) = try await postRaw(chatURL, jsonObject: chatBody(stream: false), headers: origin)
        #expect(plainResponse.statusCode == 200)
        let plainUsage = try #require((try JSONSerialization.jsonObject(with: plain) as? [String: Any])?["usage"] as? [String: Any])
        #expect(plainUsage["prompt_tokens"] as? Int == 10)
        #expect(plainUsage["prompt_tokens_details"] == nil)

        let (streamed, streamedResponse) = try await postRaw(
            chatURL, jsonObject: chatBody(stream: true, streamOptions: ["include_usage": true]), headers: origin)
        #expect(streamedResponse.statusCode == 200)
        let streamedUsage = try #require(try frames(streamed).last?["usage"] as? [String: Any])
        #expect(streamedUsage["total_tokens"] as? Int == 12)
        #expect(streamedUsage["prompt_tokens_details"] == nil)

        let messagesURL = URL(string: "http://127.0.0.1:\(port)/v1/messages")!
        let (anthropic, anthropicResponse) = try await postRaw(messagesURL, jsonObject: anthropicBody(stream: false), headers: origin)
        await server.stop()
        #expect(anthropicResponse.statusCode == 200)
        let anthropicUsage = try #require((try JSONSerialization.jsonObject(with: anthropic) as? [String: Any])?["usage"] as? [String: Any])
        #expect(anthropicUsage["input_tokens"] as? Int == 10, "the full prompt, since nothing is split out")
        #expect(anthropicUsage["cache_read_input_tokens"] == nil)
    }

    // MARK: max_completion_tokens

    private actor CapturingStubEngine: InferenceEngine {
        nonisolated let engineID: EngineID = .mlxSwift
        private(set) var status: EngineStatus = .idle
        private(set) var loadedModel: LocalModel?
        let version = "capturing-1"
        private(set) var capturedMaxTokens: [Int] = []

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
                    await self.record(request.parameters.maxTokens)
                    continuation.yield(GenerateChunk(
                        text: "ok", finishReason: .stop,
                        usage: TokenUsage(promptTokens: 1, completionTokens: 1)))
                    continuation.finish()
                }
            }
        }

        private func record(_ maxTokens: Int) { capturedMaxTokens.append(maxTokens) }

        func healthCheck() async -> Bool { true }
    }

    /// `max_completion_tokens` (what Apple's client sends) sets the budget;
    /// `max_tokens` wins when both are present.
    @Test
    func maxCompletionTokensIsHonored() async throws {
        let engine = CapturingStubEngine()
        try await engine.load(fixtureModel(id: "stub-model"))
        let server = HummingbirdServer(engineProvider: { engine })
        let port = try await server.start(preferredPort: 20_360)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!

        var onlyNew = chatBody(stream: false)
        onlyNew["max_completion_tokens"] = 7
        _ = try await postRaw(url, jsonObject: onlyNew)

        var both = chatBody(stream: false)
        both["max_tokens"] = 9
        both["max_completion_tokens"] = 7
        _ = try await postRaw(url, jsonObject: both)
        await server.stop()

        #expect(await engine.capturedMaxTokens == [7, 9])
    }
}
