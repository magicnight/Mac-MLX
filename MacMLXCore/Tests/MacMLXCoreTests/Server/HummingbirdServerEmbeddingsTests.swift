import Foundation
import Testing
@testable import MacMLXCore

// MARK: - /v1/embeddings + /v1/rerank Tests
//
// These exercise route wiring, DTO decoding (string vs array input), and the
// model-not-found path. Actually producing embeddings needs a real embedder
// model, so that stays manual QA — but a 404 here proves the request body
// decoded far enough to reach cold-swap resolution (a decode failure would be
// a 400 instead).
//
// Port assignments (19_600 range, spaced by 10):
//   embeddingsStringInputUnknownModelReturns404 : 19_600
//   embeddingsArrayInputUnknownModelReturns404  : 19_610
//   embeddingsInvalidBodyReturns400             : 19_620
//   rerankUnknownModelReturns404                : 19_630
//   rerankInvalidBodyReturns400                 : 19_640
//   embeddingsNonEmbedderModelReturns400        : 19_650
//   rerankNonEmbedderModelReturns400            : 19_660

@Suite("HummingbirdServer embeddings/rerank")
struct HummingbirdServerEmbeddingsTests {

    // MARK: Helpers

    /// Server with the default nil resolver — every model id resolves to
    /// "not found", which is exactly what these decode/404 tests want.
    private func makeServer() -> HummingbirdServer {
        HummingbirdServer(engine: StubInferenceEngine(engineID: .mlxSwift))
    }

    /// Server whose resolver returns a single model of `format` for `id` (and
    /// nil otherwise), so the embedder kind-gate (P3-8) can be exercised
    /// without a real model on disk — a non-embedder is rejected at the gate,
    /// before any load is attempted.
    private func serverResolving(_ id: String, format: ModelFormat) -> HummingbirdServer {
        let model = LocalModel(
            id: id, displayName: id,
            directory: URL(filePath: "/tmp/\(id)"), sizeBytes: 0, format: format,
            quantization: nil, parameterCount: nil, architecture: nil
        )
        let resolver: HummingbirdServer.ModelResolver = { reqID in reqID == id ? model : nil }
        return HummingbirdServer(engine: StubInferenceEngine(engineID: .mlxSwift), modelResolver: resolver)
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

    private func errorCode(_ data: Data) -> String? {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let error = json["error"] as? [String: Any],
              let code = error["code"] as? String
        else { return nil }
        return code
    }

    // MARK: Tests

    @Test
    func embeddingsStringInputUnknownModelReturns404() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 19_600)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/embeddings")!

        let (data, response) = try await postRaw(url, jsonObject: [
            "model": "no-such-embedder-xyz",
            "input": "hello world",
        ])
        await server.stop()

        #expect(response.statusCode == 404)
        #expect(errorCode(data) == "model_not_found")
    }

    @Test
    func embeddingsArrayInputUnknownModelReturns404() async throws {
        // A 404 (not 400) proves the array-form `input` decoded successfully.
        let server = makeServer()
        let port = try await server.start(preferredPort: 19_610)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/embeddings")!

        let (data, response) = try await postRaw(url, jsonObject: [
            "model": "no-such-embedder-xyz",
            "input": ["first document", "second document"],
        ])
        await server.stop()

        #expect(response.statusCode == 404)
        #expect(errorCode(data) == "model_not_found")
    }

    @Test
    func embeddingsInvalidBodyReturns400() async throws {
        // Missing the required `input` field → decode fails → 400.
        let server = makeServer()
        let port = try await server.start(preferredPort: 19_620)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/embeddings")!

        let (data, response) = try await postRaw(url, jsonObject: [
            "model": "no-such-embedder-xyz",
        ])
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(errorCode(data) == "invalid_request_error")
    }

    /// P3-8: a request naming a real but NON-embedder model (e.g. a chat
    /// model) must be rejected with a 400 `model_not_embedder`, not silently
    /// embedded into meaningless vectors. Resolution succeeds (the model
    /// exists) but the kind gate rejects it before any load.
    @Test
    func embeddingsNonEmbedderModelReturns400() async throws {
        let server = serverResolving("chat-model-not-embedder", format: .mlx)
        let port = try await server.start(preferredPort: 19_650)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/embeddings")!

        let (data, response) = try await postRaw(url, jsonObject: [
            "model": "chat-model-not-embedder",
            "input": "hello world",
        ])
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(errorCode(data) == "model_not_embedder")
    }

    /// P3-8: same kind gate on the rerank path.
    @Test
    func rerankNonEmbedderModelReturns400() async throws {
        let server = serverResolving("chat-model-not-embedder", format: .mlx)
        let port = try await server.start(preferredPort: 19_660)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/rerank")!

        let (data, response) = try await postRaw(url, jsonObject: [
            "model": "chat-model-not-embedder",
            "query": "what is the capital of france",
            "documents": ["paris is the capital", "berlin is in germany"],
        ])
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(errorCode(data) == "model_not_embedder")
    }

    @Test
    func rerankUnknownModelReturns404() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 19_630)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/rerank")!

        let (data, response) = try await postRaw(url, jsonObject: [
            "model": "no-such-embedder-xyz",
            "query": "what is the capital of france",
            "documents": ["paris is the capital", "berlin is in germany"],
        ])
        await server.stop()

        #expect(response.statusCode == 404)
        #expect(errorCode(data) == "model_not_found")
    }

    @Test
    func rerankInvalidBodyReturns400() async throws {
        // Missing the required `documents` field → decode fails → 400.
        let server = makeServer()
        let port = try await server.start(preferredPort: 19_640)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/rerank")!

        let (data, response) = try await postRaw(url, jsonObject: [
            "model": "no-such-embedder-xyz",
            "query": "hello",
        ])
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(errorCode(data) == "invalid_request_error")
    }
}
