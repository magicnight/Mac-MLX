import Foundation
import Testing

@testable import MacMLXCore

import enum MacMLXCore.JSONValue

// MARK: - Tool key order through the endpoints
//
// A real HummingbirdServer over HTTP, backed by a stub engine that records the
// request it is asked to generate. No model, no Metal. Each endpoint that
// takes tools must hand the engine the client's schema with its keys in the
// order of the request body: `/v1/chat/completions` forwards the spec as sent,
// `/v1/messages` and `/v1/responses` build the OpenAI spec around the client's
// schema.
//
// Port assignments (24_300 range, spaced by 30 for port-retry headroom):
//   chatCompletionsKeepTheClientsOrder : 24_300
//   anthropicMessagesKeepTheSchemaOrder : 24_330
//   responsesKeepTheSchemaOrder        : 24_360

@Suite("ToolKeyOrderEndpoint")
struct ToolKeyOrderEndpointTests {

    private actor RecordingEngine: InferenceEngine {
        nonisolated let engineID: EngineID = .mlxSwift
        private(set) var status: EngineStatus = .idle
        private(set) var loadedModel: LocalModel?
        let version = "tool-order-stub-1"
        private(set) var capturedRequest: GenerateRequest?

        func load(_ model: LocalModel) async throws {
            loadedModel = model
            status = .ready(model: model.id)
        }

        func unload() async throws {
            loadedModel = nil
            status = .idle
        }

        nonisolated func generate(_ request: GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error> {
            AsyncThrowingStream { continuation in
                Task {
                    await self.capture(request)
                    continuation.yield(GenerateChunk(text: "ok", finishReason: .stop))
                    continuation.finish()
                }
            }
        }

        private func capture(_ request: GenerateRequest) { capturedRequest = request }

        func healthCheck() async -> Bool { true }
    }

    private static let model = "stub-model"

    /// A schema in an order that is neither sorted nor any convention's.
    private static let schema =
        #"{"type":"object","required":["zeta"],"properties":{"zeta":{"type":"string","description":"z"},"alpha":{"enum":["b","a"],"type":"string"}},"additionalProperties":false}"#
    private static let schemaOrder =
        "{type:object,required:[zeta],properties:{zeta:{type:string,description:z},alpha:{enum:[b,a],type:string}},"
        + "additionalProperties:false}"

    /// Posts `body` to `path` and returns the tools the engine was handed.
    private func toolsReachingTheEngine(path: String, body: String, port: Int) async throws -> [JSONValue] {
        let engine = RecordingEngine()
        try await engine.load(
            LocalModel(
                id: Self.model, displayName: "Stub", directory: URL(filePath: "/tmp"), sizeBytes: 0,
                format: .mlx, quantization: nil, parameterCount: nil, architecture: nil))
        let server = HummingbirdServer(engine: engine)
        let boundPort = try await server.start(preferredPort: port)
        defer { Task { await server.stop() } }

        var request = URLRequest(url: try #require(URL(string: "http://127.0.0.1:\(boundPort)\(path)")))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data(body.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        let http = try #require(response as? HTTPURLResponse)
        #expect(http.statusCode == 200, "\(String(decoding: data, as: UTF8.self))")
        return try #require(await engine.capturedRequest?.tools)
    }

    @Test
    func chatCompletionsKeepTheClientsOrder() async throws {
        let tools = try await toolsReachingTheEngine(
            path: "/v1/chat/completions",
            body: #"{"messages":[{"role":"user","content":"hi"}],"tools":[{"function":{"parameters":"#
                + Self.schema + #","name":"f","description":"d"},"type":"function"}],"model":"stub-model"}"#,
            port: 24_300)
        #expect(
            tools.map(\.orderedDescription)
                == ["{function:{parameters:\(Self.schemaOrder),name:f,description:d},type:function}"])
    }

    @Test
    func anthropicMessagesKeepTheSchemaOrder() async throws {
        let tools = try await toolsReachingTheEngine(
            path: "/v1/messages",
            body: #"{"tools":[{"input_schema":"# + Self.schema
                + #","description":"d","name":"f"}],"max_tokens":16,"model":"stub-model","#
                + #""messages":[{"role":"user","content":"hi"}]}"#,
            port: 24_330)
        #expect(
            tools.map(\.orderedDescription)
                == ["{type:function,function:{name:f,description:d,parameters:\(Self.schemaOrder)}}"])
    }

    @Test
    func responsesKeepTheSchemaOrder() async throws {
        let tools = try await toolsReachingTheEngine(
            path: "/v1/responses",
            body: #"{"tools":[{"parameters":"# + Self.schema
                + #","type":"function","description":"d","name":"f"}],"input":"hi","model":"stub-model"}"#,
            port: 24_360)
        #expect(
            tools.map(\.orderedDescription)
                == ["{type:function,function:{name:f,description:d,parameters:\(Self.schemaOrder)}}"])
    }
}
