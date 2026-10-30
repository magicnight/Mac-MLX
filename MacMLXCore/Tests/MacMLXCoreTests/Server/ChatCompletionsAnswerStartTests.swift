import Foundation
import Testing
@testable import MacMLXCore

/// Where a chat-completions answer starts on a thinking model. Found by
/// driving Qwen3.6 through Apple's Foundation Models client: the model emits
/// blank lines after `</think>`, they went out as `content`, and on a tool-call
/// turn Apple's client stored them as a response of their own, then replayed
/// the reasoning on that blank message and the tool calls on a second
/// assistant message. The answer now starts at its first non-whitespace text,
/// as it already does on `/v1/messages` and `/v1/responses`.
///
/// Port assignments (spaced by 10):
///   streamedAnswerStartsAtItsFirstNonWhitespaceText : 23_300
///   streamedToolOnlyTurnCarriesNoContent            : 23_310
///   answerKeepsWhitespaceAfterItsFirstText          : 23_320
///   nonStreamingAnswerStartsAtItsFirstText          : 23_330
///   nonStreamingToolOnlyTurnHasNullContent          : 23_340
@Suite("ChatCompletionsAnswerStart")
struct ChatCompletionsAnswerStartTests {

    /// An engine whose rendered prompt opens a think block, as Qwen3's template
    /// does, and that streams `chunks`; the last one is terminal and carries
    /// `toolCalls` when given.
    private actor ThinkingStubEngine: InferenceEngine {
        nonisolated let engineID: EngineID = .mlxSwift
        private(set) var status: EngineStatus = .idle
        private(set) var loadedModel: LocalModel?
        let version = "answer-start-1"
        private let chunks: [String]
        private let toolCalls: [ToolCallRequest]?

        init(chunks: [String], toolCalls: [ToolCallRequest]? = nil) {
            self.chunks = chunks
            self.toolCalls = toolCalls
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

        func promptOpensThinkBlock(_ request: GenerateRequest) async -> Bool { true }

        nonisolated func generate(_ request: GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error> {
            AsyncThrowingStream { continuation in
                Task { [weak self] in
                    guard let self else { continuation.finish(); return }
                    let chunks = self.chunks
                    let toolCalls = self.toolCalls
                    for piece in chunks.dropLast() {
                        continuation.yield(GenerateChunk(text: piece))
                    }
                    continuation.yield(GenerateChunk(
                        text: chunks.last ?? "",
                        finishReason: toolCalls == nil ? .stop : .toolCalls,
                        usage: TokenUsage(promptTokens: 1, completionTokens: 2),
                        toolCalls: toolCalls))
                    continuation.finish()
                }
            }
        }

        func healthCheck() async -> Bool { true }
    }

    private let weatherCall = ToolCallRequest(
        id: "call_1", name: "get_weather", arguments: ["city": .string("Oslo")])

    private func serve(_ engine: ThinkingStubEngine, port: Int) async throws -> (HummingbirdServer, URL) {
        try await engine.load(LocalModel(
            id: "stub-model", displayName: "stub-model", directory: URL(filePath: "/tmp"), sizeBytes: 0,
            format: .mlx, quantization: nil, parameterCount: nil, architecture: nil))
        let server = HummingbirdServer(engineProvider: { engine })
        let bound = try await server.start(preferredPort: port)
        return (server, try #require(URL(string: "http://127.0.0.1:\(bound)/v1/chat/completions")))
    }

    private func post(_ url: URL, stream: Bool) async throws -> Data {
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: [
            "model": "stub-model",
            "messages": [["role": "user", "content": "Weather in Oslo?"]],
            "stream": stream,
        ] as [String: Any])
        let (data, response) = try await URLSession.shared.data(for: req)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        return data
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

    private func message(_ data: Data) throws -> [String: Any] {
        let object = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let choice = try #require((object["choices"] as? [[String: Any]])?.first)
        return try #require(choice["message"] as? [String: Any])
    }

    @Test
    func streamedAnswerStartsAtItsFirstNonWhitespaceText() async throws {
        let engine = ThinkingStubEngine(chunks: ["Capital of France.", "</think>", "\n\n", " Par", "is"])
        let (server, url) = try await serve(engine, port: 23_300)
        let data = try await post(url, stream: true)
        await server.stop()

        let deltas = try deltas(data)
        #expect(deltas.compactMap { $0["content"] as? String }.joined() == "Paris")
        #expect(deltas.compactMap { $0["reasoning_content"] as? String }.joined() == "Capital of France.")
        // No frame carries whitespace-only content.
        #expect(!deltas.contains { ($0["content"] as? String)?.allSatisfy(\.isWhitespace) == true })
    }

    @Test
    func streamedToolOnlyTurnCarriesNoContent() async throws {
        let engine = ThinkingStubEngine(chunks: ["Use the tool.", "</think>\n\n", ""], toolCalls: [weatherCall])
        let (server, url) = try await serve(engine, port: 23_310)
        let data = try await post(url, stream: true)
        await server.stop()

        let deltas = try deltas(data)
        #expect(!deltas.contains { $0["content"] != nil })
        #expect(deltas.contains { $0["tool_calls"] != nil })
        #expect(deltas.compactMap { $0["reasoning_content"] as? String }.joined() == "Use the tool.")
    }

    @Test
    func answerKeepsWhitespaceAfterItsFirstText() async throws {
        let engine = ThinkingStubEngine(chunks: ["Hmm.", "</think>", "\n\n", "Hi", "\n\nthere "])
        let (server, url) = try await serve(engine, port: 23_320)
        let data = try await post(url, stream: true)
        await server.stop()

        #expect(try deltas(data).compactMap { $0["content"] as? String }.joined() == "Hi\n\nthere ")
    }

    @Test
    func nonStreamingAnswerStartsAtItsFirstText() async throws {
        let engine = ThinkingStubEngine(chunks: ["Capital of France.", "</think>", "\n\n", "Paris"])
        let (server, url) = try await serve(engine, port: 23_330)
        let data = try await post(url, stream: false)
        await server.stop()

        let message = try message(data)
        #expect(message["content"] as? String == "Paris")
        #expect(message["reasoning_content"] as? String == "Capital of France.")
    }

    @Test
    func nonStreamingToolOnlyTurnHasNullContent() async throws {
        let engine = ThinkingStubEngine(chunks: ["Use the tool.", "</think>\n\n", ""], toolCalls: [weatherCall])
        let (server, url) = try await serve(engine, port: 23_340)
        let data = try await post(url, stream: false)
        await server.stop()

        let message = try message(data)
        #expect(message["content"] is NSNull)
        #expect((message["tool_calls"] as? [[String: Any]])?.count == 1)
    }
}
