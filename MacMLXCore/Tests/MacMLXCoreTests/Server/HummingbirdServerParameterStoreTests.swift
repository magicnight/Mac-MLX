import Foundation
import Testing
@testable import MacMLXCore

// MARK: - The server's parameter store
//
// The server reads aliases and chat-template kwargs from the store it was
// given (`init(…parameters:)`) at every place it looks one up. Each test puts
// one override in a store of its own, in a temporary directory, and checks
// that the server acts on it. A server that built its own store would read
// `~/.mac-mlx/model-params`, where this alias does not exist, and fail here.
// The listing half (`/api/tags`) is `ollamaTagsReportsConfiguredAlias` in
// `HummingbirdServerTests`.
//
// Port assignments:
//   aChatRequestNamingAnAliasLoadsTheModelBehindIt     : 19_540
//   anEmbeddingsRequestNamingAnAliasReachesTheKindGate : 19_560

@Suite("HummingbirdServer parameter store")
struct HummingbirdServerParameterStoreTests {

    // MARK: Helpers

    private static let modelID = "some-model-dir"
    private static let alias = "macmlx-test-alias-in-the-given-store"
    private static let kwargs: [String: JSONValue] = ["enable_thinking": .bool(false)]

    private func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appending(path: "macmlx-test-params-\(UUID().uuidString)", directoryHint: .isDirectory)
    }

    /// A store under `directory` with one override: `modelID`, named `alias`,
    /// with `kwargs` as its chat-template kwargs.
    private func storeWithTheOverride(in directory: URL) async throws -> ModelParametersStore {
        let store = ModelParametersStore(directory: directory)
        var parameters = ModelParameters.default
        parameters.alias = Self.alias
        parameters.templateKwargs = Self.kwargs
        try await store.save(parameters, for: Self.modelID)
        return store
    }

    /// A server reading `store` whose resolver knows the model by its
    /// directory id only, so the alias can reach it through the store alone.
    private func makeServer(
        reading store: ModelParametersStore,
        engine: StubInferenceEngine = StubInferenceEngine(engineID: .mlxSwift)
    ) -> HummingbirdServer {
        let model = LocalModel(
            id: Self.modelID, displayName: Self.modelID,
            directory: URL(filePath: "/tmp/\(Self.modelID)"), sizeBytes: 0, format: .mlx,
            quantization: nil, parameterCount: nil, architecture: nil
        )
        let resolver: HummingbirdServer.ModelResolver = { id in id == Self.modelID ? model : nil }
        return HummingbirdServer(engineProvider: { engine }, modelResolver: resolver, parameters: store)
    }

    private func postRaw(_ url: URL, jsonObject: Any) async throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: jsonObject)
        let (data, response) = try await URLSession.shared.data(for: request)
        let http = try #require(response as? HTTPURLResponse)
        return (data, http)
    }

    private func errorCode(_ data: Data) -> String? {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let error = json["error"] as? [String: Any]
        else { return nil }
        return error["code"] as? String
    }

    // MARK: Tests

    @Test
    func templateKwargsAreReadFromTheGivenStoreByIDAndByAlias() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let server = makeServer(reading: try await storeWithTheOverride(in: directory))

        #expect(await server.templateKwargs(for: Self.modelID) == Self.kwargs)
        #expect(await server.templateKwargs(for: Self.alias) == Self.kwargs)
        #expect(await server.templateKwargs(for: "a-model-with-no-override") == nil)
    }

    @Test
    func anAliasResolvesOnlyThroughTheStoreThatHoldsIt() async throws {
        let directory = temporaryDirectory()
        let emptyDirectory = temporaryDirectory()
        defer {
            try? FileManager.default.removeItem(at: directory)
            try? FileManager.default.removeItem(at: emptyDirectory)
        }
        let withTheAlias = makeServer(reading: try await storeWithTheOverride(in: directory))
        let withoutIt = makeServer(reading: ModelParametersStore(directory: emptyDirectory))

        #expect(await withTheAlias.canResolveModel(Self.alias))
        #expect(await withTheAlias.canResolveModel(Self.modelID))
        #expect(await !withoutIt.canResolveModel(Self.alias))
        #expect(await withoutIt.canResolveModel(Self.modelID))
    }

    @Test
    func aChatRequestNamingAnAliasLoadsTheModelBehindIt() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let engine = StubInferenceEngine(engineID: .mlxSwift)
        let server = makeServer(reading: try await storeWithTheOverride(in: directory), engine: engine)
        let port = try await server.start(preferredPort: 19_540)

        let url = try #require(URL(string: "http://127.0.0.1:\(port)/v1/chat/completions"))
        let body: [String: Any] = [
            "model": Self.alias,
            "messages": [["role": "user", "content": "Hi"]],
            "stream": false,
        ]
        let (data, response) = try await postRaw(url, jsonObject: body)
        let loaded = await engine.loadedModel?.id

        await server.stop()

        #expect(response.statusCode == 200, "\(String(decoding: data, as: UTF8.self))")
        #expect(loaded == Self.modelID)
    }

    /// The embeddings and rerank lookup. The model behind the alias is a chat
    /// model, so a lookup that found it stops at the kind gate (400); one that
    /// did not would answer 404.
    @Test
    func anEmbeddingsRequestNamingAnAliasReachesTheKindGate() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let server = makeServer(reading: try await storeWithTheOverride(in: directory))
        let port = try await server.start(preferredPort: 19_560)

        let url = try #require(URL(string: "http://127.0.0.1:\(port)/v1/embeddings"))
        let (data, response) = try await postRaw(url, jsonObject: ["model": Self.alias, "input": "hello"])

        await server.stop()

        #expect(response.statusCode == 400)
        #expect(errorCode(data) == "model_not_embedder")
    }
}
