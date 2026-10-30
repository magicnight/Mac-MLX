import Foundation
import Testing
@testable import MacMLXCore

/// `GET /v1/models` and Ollama's `/api/tags` list every chat model the server
/// can serve, not only the loaded one: a request naming any of them is loaded
/// on demand. Xcode 27's locally hosted provider reads an empty list as a
/// failed connection, so with nothing loaded it could not be set up at all.
///
/// Port assignments (spaced by 10):
///   nothingLoadedListsTheLibrarysChatModels : 23_500
///   loadedModelComesFirstAndOnlyOnce        : 23_510
///   withoutAListerOnlyTheLoadedModelIsShown : 23_520
///   ollamaTagsListTheSameModels             : 23_530
///   aBrowserPageSeesOnlyTheLoadedModel      : 23_540
@Suite("ModelListing")
struct ModelListingTests {

    private actor StubEngine: InferenceEngine {
        nonisolated let engineID: EngineID = .mlxSwift
        private(set) var status: EngineStatus = .idle
        private(set) var loadedModel: LocalModel?
        let version = "model-listing-1"

        func load(_ model: LocalModel) async throws {
            loadedModel = model
            status = .ready(model: model.id)
        }

        func unload() async throws {
            loadedModel = nil
            status = .idle
        }

        nonisolated func generate(_ request: GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error> {
            AsyncThrowingStream { $0.finish() }
        }

        func healthCheck() async -> Bool { true }
    }

    private static func model(_ id: String, _ format: ModelFormat) -> LocalModel {
        LocalModel(
            id: id, displayName: id, directory: URL(filePath: "/tmp/\(id)"), sizeBytes: 0,
            format: format, quantization: nil, parameterCount: nil, architecture: nil)
    }

    /// Not in id order, as a directory listing need not be.
    private let library: [LocalModel] = [
        model("listing-text-c", .mlx),
        model("listing-vision-b", .mlxVLM),
        model("listing-embedder", .embedder),
        model("listing-text-a", .mlx),
    ]

    private func get(_ path: String, port: Int, origin: String? = nil) async throws -> [String: Any] {
        let url = try #require(URL(string: "http://127.0.0.1:\(port)\(path)"))
        var request = URLRequest(url: url)
        if let origin { request.setValue(origin, forHTTPHeaderField: "Origin") }
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        return try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private func modelIDs(port: Int, origin: String? = nil) async throws -> [String] {
        let body = try await get("/v1/models", port: port, origin: origin)
        #expect(body["object"] as? String == "list")
        return (body["data"] as? [[String: Any]] ?? []).compactMap { $0["id"] as? String }
    }

    private func server(_ engine: StubEngine, lister: HummingbirdServer.ModelLister?) -> HummingbirdServer {
        let library = library
        return HummingbirdServer(
            engineProvider: { engine },
            modelResolver: { id in library.first { $0.id == id } },
            modelLister: lister)
    }

    @Test
    func nothingLoadedListsTheLibrarysChatModels() async throws {
        let library = library
        let server = server(StubEngine(), lister: { library })
        let port = try await server.start(preferredPort: 23_500)
        let ids = try await modelIDs(port: port)
        await server.stop()

        #expect(ids == ["listing-text-a", "listing-text-c", "listing-vision-b"])
    }

    @Test
    func loadedModelComesFirstAndOnlyOnce() async throws {
        let engine = StubEngine()
        try await engine.load(library[0])
        let library = library
        let server = server(engine, lister: { library })
        let port = try await server.start(preferredPort: 23_510)
        let ids = try await modelIDs(port: port)
        await server.stop()

        #expect(ids == ["listing-text-c", "listing-text-a", "listing-vision-b"])
    }

    @Test
    func withoutAListerOnlyTheLoadedModelIsShown() async throws {
        let engine = StubEngine()
        let empty = server(engine, lister: nil)
        let port = try await empty.start(preferredPort: 23_520)
        let before = try await modelIDs(port: port)
        try await engine.load(library[3])
        let after = try await modelIDs(port: port)
        await empty.stop()

        #expect(before.isEmpty)
        #expect(after == ["listing-text-a"])
    }

    @Test
    func ollamaTagsListTheSameModels() async throws {
        let engine = StubEngine()
        try await engine.load(library[1])
        let library = library
        let server = server(engine, lister: { library })
        let port = try await server.start(preferredPort: 23_530)
        let tags = try await get("/api/tags", port: port)
        await server.stop()

        let names = (tags["models"] as? [[String: Any]] ?? []).compactMap { $0["name"] as? String }
        #expect(names == ["listing-vision-b", "listing-text-a", "listing-text-c"])
    }

    /// A browser always sends `Origin`, and the server answers it by default,
    /// so a web page does not learn which models are downloaded: it sees the
    /// loaded model, as before the library was listed. Native clients send
    /// no `Origin` and see the whole library.
    @Test
    func aBrowserPageSeesOnlyTheLoadedModel() async throws {
        let engine = StubEngine()
        let library = library
        let server = server(engine, lister: { library })
        let port = try await server.start(preferredPort: 23_540)
        let idle = try await modelIDs(port: port, origin: "https://example.com")
        try await engine.load(library[1])
        let loaded = try await modelIDs(port: port, origin: "https://example.com")
        let tags = try await get("/api/tags", port: port, origin: "https://example.com")
        let native = try await modelIDs(port: port)
        await server.stop()

        #expect(idle.isEmpty)
        #expect(loaded == ["listing-vision-b"])
        let names = (tags["models"] as? [[String: Any]] ?? []).compactMap { $0["name"] as? String }
        #expect(names == ["listing-vision-b"])
        #expect(native == ["listing-vision-b", "listing-text-a", "listing-text-c"])
    }
}
