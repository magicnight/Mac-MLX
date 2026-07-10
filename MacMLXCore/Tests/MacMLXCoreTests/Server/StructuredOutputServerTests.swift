import Foundation
import Testing

@testable import MacMLXCore

// MARK: - Structured-output server wiring (Track C)
//
// HTTP-level proof that `/v1/chat/completions` decodes OpenAI `response_format`,
// rejects unsupported schema features with a 400 BEFORE any generation, and
// accepts the supported shapes. Uses the stub engine — no model, no Metal — so
// it runs under a plain `swift test`.

@Suite("StructuredOutputServer")
struct StructuredOutputServerTests {

    private func loadedStubServer() async throws -> (HummingbirdServer, StubInferenceEngine) {
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
        return (HummingbirdServer(engine: engine), engine)
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

    @Test
    func rejectsUnsupportedNestedSchemaWith400() async throws {
        let (server, _) = try await loadedStubServer()
        let port = try await server.start(preferredPort: 19_910)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!

        let body: [String: Any] = [
            "model": "stub-model",
            "messages": [["role": "user", "content": "hi"]],
            "response_format": [
                "type": "json_schema",
                "json_schema": [
                    "name": "Nested",
                    "schema": [
                        "type": "object",
                        "properties": ["address": ["type": "object"]],
                    ],
                ],
            ],
        ]
        let (data, response) = try await postRaw(url, jsonObject: body)
        await server.stop()

        #expect(response.statusCode == 400)
        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let error = try #require(json["error"] as? [String: Any])
        let message = try #require(error["message"] as? String)
        #expect(message.contains("unsupported schema feature"))
        #expect(message.contains("nested object"))
    }

    @Test
    func acceptsJsonObjectResponseFormat() async throws {
        let (server, _) = try await loadedStubServer()
        let port = try await server.start(preferredPort: 19_920)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!

        let body: [String: Any] = [
            "model": "stub-model",
            "messages": [["role": "user", "content": "hi"]],
            "stream": false,
            "response_format": ["type": "json_object"],
        ]
        let (data, response) = try await postRaw(url, jsonObject: body)
        await server.stop()

        // The stub engine does not constrain, but the request must be accepted
        // and mapped — proving the decode wiring never rejects a supported shape.
        #expect(response.statusCode == 200)
        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(json["object"] as? String == "chat.completion")
    }

    @Test
    func acceptsMissingResponseFormat() async throws {
        // Zero-regression: a request with no response_format is unchanged.
        let (server, _) = try await loadedStubServer()
        let port = try await server.start(preferredPort: 19_930)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!

        let body: [String: Any] = [
            "model": "stub-model",
            "messages": [["role": "user", "content": "hi"]],
            "stream": false,
        ]
        let (data, response) = try await postRaw(url, jsonObject: body)
        await server.stop()

        #expect(response.statusCode == 200)
        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(json["object"] as? String == "chat.completion")
    }
}
