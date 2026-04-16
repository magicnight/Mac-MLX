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
}
