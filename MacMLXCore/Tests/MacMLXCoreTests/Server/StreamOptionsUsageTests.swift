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

    private func postRaw(_ url: URL, jsonObject: Any) async throws -> (Data, HTTPURLResponse) {
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: jsonObject)
        let (data, response) = try await URLSession.shared.data(for: req)
        return (data, try #require(response as? HTTPURLResponse))
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
            StubInferenceEngine(engineID: .mlxSwift, cachedPromptTokens: 5))
        let port = try await server.start(preferredPort: 20_230)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let (data, response) = try await postRaw(url, jsonObject: chatBody(stream: false))
        await server.stop()
        #expect(response.statusCode == 200)

        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let usage = try #require(json["usage"] as? [String: Any])
        let details = try #require(usage["prompt_tokens_details"] as? [String: Any])
        #expect(details["cached_tokens"] as? Int == 5)
        #expect(usage["prompt_tokens"] as? Int == 1, "the cached figure is a detail, not a correction to prompt_tokens")
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

    /// `/v1/messages` reports the cache figure as `cache_read_input_tokens`,
    /// both in the non-streaming body and on the streaming `message_delta`.
    @Test
    func anthropicReportsCacheReadInputTokens() async throws {
        let server = try await loadedServer(
            StubInferenceEngine(engineID: .mlxSwift, cachedPromptTokens: 5))
        let port = try await server.start(preferredPort: 20_250)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/messages")!

        let (data, response) = try await postRaw(url, jsonObject: anthropicBody(stream: false))
        #expect(response.statusCode == 200)
        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let usage = try #require(json["usage"] as? [String: Any])
        #expect(usage["input_tokens"] as? Int == 1)
        #expect(usage["cache_read_input_tokens"] as? Int == 5)

        let (streamed, streamedResponse) = try await postRaw(url, jsonObject: anthropicBody(stream: true))
        await server.stop()
        #expect(streamedResponse.statusCode == 200)
        let text = String(decoding: streamed, as: UTF8.self)
        let delta = try #require(
            text.components(separatedBy: "\n\n").first { $0.hasPrefix("event: message_delta") })
        let deltaPayload = try #require(delta.components(separatedBy: "data: ").last)
        let deltaJSON = try jsonObject(deltaPayload)
        let deltaUsage = try #require(deltaJSON["usage"] as? [String: Any])
        #expect(deltaUsage["cache_read_input_tokens"] as? Int == 5)
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
}
