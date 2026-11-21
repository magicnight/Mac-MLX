import Foundation
import Testing

@testable import MacMLXCore

// MARK: - Deeply nested request bodies
//
// A real HummingbirdServer over HTTP, backed by the stub engine. No model, no
// Metal. A tool schema nested a few hundred levels used to take the process
// down: `JSONValue` decodes by recursion, on a thread with a 512 KB stack. Each
// endpoint that takes a free-form schema must answer such a body with a 400
// and go on serving.
//
// Port assignment (24_300 range, spaced by 30 for port-retry headroom):
//   aDeeplyNestedSchemaIsRejectedAndTheServerGoesOn : 24_390

@Suite("DeepNestingEndpoint")
struct DeepNestingEndpointTests {

    private static let model = "stub-model"

    private func post(_ body: String, to path: String, port: Int) async throws -> Int {
        var request = URLRequest(url: try #require(URL(string: "http://127.0.0.1:\(port)\(path)")))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data(body.utf8)
        let (_, response) = try await URLSession.shared.data(for: request)
        return try #require(response as? HTTPURLResponse).statusCode
    }

    @Test
    func aDeeplyNestedSchemaIsRejectedAndTheServerGoesOn() async throws {
        let engine = StubInferenceEngine(engineID: .mlxSwift)
        try await engine.load(
            LocalModel(
                id: Self.model, displayName: "Stub", directory: URL(filePath: "/tmp"), sizeBytes: 0,
                format: .mlx, quantization: nil, parameterCount: nil, architecture: nil))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 24_390)
        defer { Task { await server.stop() } }

        let depth = 500
        let schema = String(repeating: #"{"a":"#, count: depth) + "1" + String(repeating: "}", count: depth)
        let bodies = [
            "/v1/chat/completions":
                #"{"model":"stub-model","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":"#
                + schema + "}}]}",
            "/v1/messages":
                #"{"model":"stub-model","max_tokens":8,"messages":[{"role":"user","content":"hi"}],"tools":[{"name":"f","input_schema":"#
                + schema + "}]}",
            "/v1/responses":
                #"{"model":"stub-model","input":"hi","tools":[{"type":"function","name":"f","parameters":"#
                + schema + "}]}",
        ]
        for (path, body) in bodies {
            #expect(try await post(body, to: path, port: port) == 400, "\(path)")
        }

        let plain = #"{"model":"stub-model","messages":[{"role":"user","content":"hi"}]}"#
        #expect(try await post(plain, to: "/v1/chat/completions", port: port) == 200)
    }
}
