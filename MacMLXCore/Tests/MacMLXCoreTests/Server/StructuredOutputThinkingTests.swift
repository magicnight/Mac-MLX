import Foundation
import Testing
@testable import MacMLXCore

/// Structured output on a thinking model, at the server boundary. Found by
/// driving a live Qwen3.6 checkpoint through Apple's Foundation Models client:
/// the rendered prompt opens a `<think>` block, the constraint forces JSON from
/// the first byte, and the stream was classified as reasoning throughout.
///
/// Port assignments (spaced by 10):
///   constrainedStreamIsContentEvenWhenPromptOpensThink : 20_300
///   unconstrainedStreamStillHonorsThePromptSeed        : 20_310
///   constrainedRequestRendersWithThinkingOff           : 20_320
///   unconstrainedRequestKeepsItsKwargs                 : 20_330
///   closingThinkTagInsideConstrainedStringIsData       : 20_370
///   openingThinkTagInsideConstrainedStringIsData       : 20_380
///   closingThinkTagInsideConstrainedStringNonStreaming : 20_390
///   openingThinkTagInsideConstrainedStringNonStreaming : 20_400
@Suite("StructuredOutputThinking")
struct StructuredOutputThinkingTests {

    /// An engine whose rendered prompt opens a think block (as Qwen3's template
    /// does) and that answers with a JSON object, as a constrained generation
    /// would. Records the request it was given.
    private actor ThinkOpeningStubEngine: InferenceEngine {
        nonisolated let engineID: EngineID = .mlxSwift
        private(set) var status: EngineStatus = .idle
        private(set) var loadedModel: LocalModel?
        let version = "think-opening-1"
        private(set) var capturedRequest: GenerateRequest?
        /// The text the stub streams, chunk by chunk; the last chunk is terminal.
        private let chunks: [String]

        init(chunks: [String] = ["{\"label\":", "\"positive\"}"]) { self.chunks = chunks }

        func load(_ model: LocalModel) async throws {
            status = .loading(model: model.id)
            loadedModel = model
            status = .ready(model: model.id)
        }

        func unload() async throws {
            loadedModel = nil
            status = .idle
        }

        func promptOpensThinkBlock(_ request: GenerateRequest) async -> Bool { true }

        nonisolated func generate(_ request: GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error> {
            AsyncThrowingStream { continuation in
                Task { [weak self] in
                    guard let self else { continuation.finish(); return }
                    await self.capture(request)
                    let chunks = self.chunks
                    for piece in chunks.dropLast() {
                        continuation.yield(GenerateChunk(text: piece))
                    }
                    continuation.yield(GenerateChunk(
                        text: chunks.last ?? "",
                        finishReason: .stop,
                        usage: TokenUsage(promptTokens: 1, completionTokens: 2)))
                    continuation.finish()
                }
            }
        }

        private func capture(_ request: GenerateRequest) { capturedRequest = request }

        func healthCheck() async -> Bool { true }
    }

    private func loadedServer(_ engine: ThinkOpeningStubEngine) async throws -> HummingbirdServer {
        try await engine.load(LocalModel(
            id: "stub-model", displayName: "stub-model", directory: URL(filePath: "/tmp"), sizeBytes: 0,
            format: .mlx, quantization: nil, parameterCount: nil, architecture: nil))
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

    private func deltas(_ data: Data) throws -> [[String: Any]] {
        try String(decoding: data, as: UTF8.self)
            .components(separatedBy: "\n\n")
            .compactMap { block -> String? in
                guard let range = block.range(of: "data: ") else { return nil }
                let payload = String(block[range.upperBound...])
                return payload == "[DONE]" ? nil : payload
            }
            .map { try #require(try JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any]) }
            .compactMap { ($0["choices"] as? [[String: Any]])?.first?["delta"] as? [String: Any] }
    }

    private var flatSchema: [String: Any] {
        [
        "type": "json_schema",
        "json_schema": [
            "name": "Sentiment",
            "strict": true,
            "schema": [
                "type": "object",
                "properties": ["label": ["type": "string"]],
                "required": ["label"],
                "additionalProperties": false,
            ] as [String: Any],
        ] as [String: Any],
        ]
    }

    private func body(stream: Bool, constrained: Bool) -> [String: Any] {
        var body: [String: Any] = [
            "model": "stub-model",
            "messages": [["role": "user", "content": "Classify this."]],
            "stream": stream,
        ]
        if constrained { body["response_format"] = flatSchema }
        return body
    }

    /// The defect: with the prompt opening a think block, a constrained stream
    /// was seeded as reasoning and the JSON answer went out as
    /// `reasoning_content`. It must be `content`, all of it.
    @Test
    func constrainedStreamIsContentEvenWhenPromptOpensThink() async throws {
        let server = try await loadedServer(ThinkOpeningStubEngine())
        let port = try await server.start(preferredPort: 20_300)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let (data, response) = try await postRaw(url, jsonObject: body(stream: true, constrained: true))
        await server.stop()
        #expect(response.statusCode == 200)

        let deltas = try deltas(data)
        let content = deltas.compactMap { $0["content"] as? String }.joined()
        let reasoning = deltas.compactMap { $0["reasoning_content"] as? String }.joined()
        #expect(content == "{\"label\":\"positive\"}")
        #expect(reasoning.isEmpty, "a constrained generation is never reasoning")
    }

    /// The seed still works where it should: the same engine, unconstrained,
    /// streams the same text as reasoning because the prompt opened a think
    /// block and no `</think>` ever arrived.
    @Test
    func unconstrainedStreamStillHonorsThePromptSeed() async throws {
        let server = try await loadedServer(ThinkOpeningStubEngine())
        let port = try await server.start(preferredPort: 20_310)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let (data, response) = try await postRaw(url, jsonObject: body(stream: true, constrained: false))
        await server.stop()
        #expect(response.statusCode == 200)

        let deltas = try deltas(data)
        let content = deltas.compactMap { $0["content"] as? String }.joined()
        let reasoning = deltas.compactMap { $0["reasoning_content"] as? String }.joined()
        #expect(reasoning == "{\"label\":\"positive\"}")
        #expect(content.isEmpty)
    }

    /// A constrained request renders its template with `enable_thinking` off,
    /// so a Qwen3-style template puts the model in answer mode.
    @Test
    func constrainedRequestRendersWithThinkingOff() async throws {
        let engine = ThinkOpeningStubEngine()
        let server = try await loadedServer(engine)
        let port = try await server.start(preferredPort: 20_320)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let (_, response) = try await postRaw(url, jsonObject: body(stream: false, constrained: true))
        await server.stop()
        #expect(response.statusCode == 200)

        let kwargs = try #require(await engine.capturedRequest?.templateKwargs)
        #expect(kwargs["enable_thinking"] == .bool(false))
    }

    /// An unconstrained request is untouched: no per-model kwargs are configured
    /// for the stub, so none are invented.
    @Test
    func unconstrainedRequestKeepsItsKwargs() async throws {
        let engine = ThinkOpeningStubEngine()
        let server = try await loadedServer(engine)
        let port = try await server.start(preferredPort: 20_330)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let (_, response) = try await postRaw(url, jsonObject: body(stream: false, constrained: false))
        await server.stop()
        #expect(response.statusCode == 200)
        #expect(await engine.capturedRequest?.templateKwargs == nil)
    }

    /// The pure merge: configured kwargs pass through, the switch is added only
    /// under a constraint, and an explicit `enable_thinking: true` is overridden
    /// there because the constraint could not honour it anyway.
    @Test
    func templateKwargsMerge() {
        let configured: [String: JSONValue] = ["foo": .int(1), "enable_thinking": .bool(true)]
        #expect(HummingbirdServer.effectiveTemplateKwargs(configured, constrainedBy: nil) == configured)
        #expect(HummingbirdServer.effectiveTemplateKwargs(nil, constrainedBy: nil) == nil)
        #expect(
            HummingbirdServer.effectiveTemplateKwargs(nil, constrainedBy: .jsonObject)
                == ["enable_thinking": .bool(false)])
        #expect(
            HummingbirdServer.effectiveTemplateKwargs(configured, constrainedBy: .jsonObject)
                == ["foo": .int(1), "enable_thinking": .bool(false)])
    }

    // MARK: Think tags inside a constrained string value are data

    private static let closingTagRaw = "{\"notes\":\"step 1 done</think> final\"}"
    private static let openingTagRaw = "{\"notes\":\"<think>plan\"}"

    /// Under a constraint the splitter is bypassed entirely, so a closing tag
    /// (which the splitter would strip) or an opening one (which would divert
    /// the rest of the answer into reasoning) inside a JSON string value stays
    /// in `content` untouched. One server per call: a fresh server on a port a
    /// stopped one just vacated can inherit a stale keep-alive connection.
    private func assertConstrainedStreamKeeps(_ raw: String, port: Int) async throws {
        let split = raw.index(raw.startIndex, offsetBy: 12)
        let server = try await loadedServer(
            ThinkOpeningStubEngine(chunks: [String(raw[..<split]), String(raw[split...])]))
        let boundPort = try await server.start(preferredPort: port)
        let url = URL(string: "http://127.0.0.1:\(boundPort)/v1/chat/completions")!
        let (data, response) = try await postRaw(url, jsonObject: body(stream: true, constrained: true))
        await server.stop()
        #expect(response.statusCode == 200)
        let deltas = try deltas(data)
        #expect(deltas.compactMap { $0["content"] as? String }.joined() == raw)
        #expect(deltas.compactMap { $0["reasoning_content"] as? String }.joined().isEmpty)
    }

    private func assertConstrainedBodyKeeps(_ raw: String, port: Int) async throws {
        let server = try await loadedServer(ThinkOpeningStubEngine(chunks: [raw]))
        let boundPort = try await server.start(preferredPort: port)
        let url = URL(string: "http://127.0.0.1:\(boundPort)/v1/chat/completions")!
        let (data, response) = try await postRaw(url, jsonObject: body(stream: false, constrained: true))
        await server.stop()
        #expect(response.statusCode == 200)
        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let message = try #require((json["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any])
        #expect(message["content"] as? String == raw)
        #expect(message["reasoning_content"] == nil)
    }

    @Test
    func closingThinkTagInsideConstrainedStringIsData() async throws {
        try await assertConstrainedStreamKeeps(Self.closingTagRaw, port: 20_370)
    }

    @Test
    func openingThinkTagInsideConstrainedStringIsData() async throws {
        try await assertConstrainedStreamKeeps(Self.openingTagRaw, port: 20_380)
    }

    /// The non-streaming path used to split the answer into invalid JSON plus a
    /// `reasoning_content` tail.
    @Test
    func closingThinkTagInsideConstrainedStringNonStreaming() async throws {
        try await assertConstrainedBodyKeeps(Self.closingTagRaw, port: 20_390)
    }

    @Test
    func openingThinkTagInsideConstrainedStringNonStreaming() async throws {
        try await assertConstrainedBodyKeeps(Self.openingTagRaw, port: 20_400)
    }
}
