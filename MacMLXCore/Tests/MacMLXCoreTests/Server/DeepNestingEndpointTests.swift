import Foundation
import Testing

@testable import MacMLXCore

// MARK: - Deeply nested request bodies
//
// A real HummingbirdServer over HTTP, backed by the stub engine. No model, no
// Metal. A body nested a few hundred levels used to take the process down:
// `JSONValue` and Anthropic's `tool_result` blocks decoded by recursion, on a
// thread with a 512 KB stack. The server must answer such a body and go on
// serving.
//
// Port assignments (24_300 range, spaced by 30 for port-retry headroom):
//   aDeeplyNestedSchemaIsRejectedAndTheServerGoesOn : 24_390
//   nestedToolResultsAreNotDecodedByRecursion       : 24_420

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

    private func startServer(preferredPort: Int) async throws -> (server: HummingbirdServer, port: Int) {
        let engine = StubInferenceEngine(engineID: .mlxSwift)
        try await engine.load(
            LocalModel(
                id: Self.model, displayName: "Stub", directory: URL(filePath: "/tmp"), sizeBytes: 0,
                format: .mlx, quantization: nil, parameterCount: nil, architecture: nil))
        let server = HummingbirdServer(engine: engine)
        return (server, try await server.start(preferredPort: preferredPort))
    }

    @Test
    func aDeeplyNestedSchemaIsRejectedAndTheServerGoesOn() async throws {
        let (server, port) = try await startServer(preferredPort: 24_390)
        defer { Task { await server.stop() } }

        // Under the 512 levels Foundation refuses on its own, wrapping
        // included, so the 400 is ours.
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

    /// A tool result inside a tool result, 200 times over. The inner blocks of
    /// a tool result are read for their text only, so the body is served.
    @Test
    func nestedToolResultsAreNotDecodedByRecursion() async throws {
        let (server, port) = try await startServer(preferredPort: 24_420)
        defer { Task { await server.stop() } }

        // Two levels of JSON each, so under the 512 Foundation refuses.
        let depth = 200
        let body =
            #"{"model":"stub-model","max_tokens":8,"messages":[{"role":"user","content":"hi"},"#
            + #"{"role":"assistant","content":[{"type":"tool_use","id":"x","name":"f","input":{}}]},"#
            + #"{"role":"user","content":"#
            + String(repeating: #"[{"type":"tool_result","tool_use_id":"x","content":"#, count: depth)
            + #""hi""# + String(repeating: "}]", count: depth) + "}]}"
        #expect(try await post(body, to: "/v1/messages", port: port) == 200)
    }
}
