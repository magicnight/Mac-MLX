import Foundation
import Testing

@testable import MacMLXCore

// MARK: - Auxiliary residency (#130)
//
// The server reports its reranker and embedder to the pool through
// `AuxiliaryResidency`. These tests run without a loadable model, so they
// pin down what the server tells the pool on the paths that need no weights:
// a cold request admits before it loads, and the load runs inside the
// admission (so the pool evicted and reserved first); the pool's answer, or
// the load's failure, comes back as a 500 with the lock released; the unload
// door names what it drops; `/x/status` shows the two slots; stopping the
// server forgets nothing it does not hold. The paths that need a resident
// model — a hit's use on a repeat, a use the pool refuses, forget on a swap,
// the pool's release emptying the slot, the use ended after the work — are
// the gated checkpoint smokes in `RerankEngineSmokeTests`.

/// Records what the server asks of the pool, in order, and whether the load
/// it was handed ran inside the admission.
actor RecordingResidency: AuxiliaryResidency {
    private(set) var events: [String] = []
    /// When set, `admit` throws this instead of running the load: the pool
    /// refusing admission.
    var refusal: (any Error)?

    func refuse(with error: any Error) { refusal = error }

    func admit(
        _ kind: AuxiliaryKind,
        _ model: LocalModel,
        load: @Sendable () async throws -> Void,
        release: @escaping @Sendable () async -> Void
    ) async throws {
        let key = ModelPool.auxiliaryKey(kind, model.id)
        events.append("admit \(key)")
        if let refusal { throw refusal }
        do {
            try await load()
            events.append("loaded \(key)")
        } catch {
            events.append("load threw \(key)")
            throw error
        }
    }

    func beginUse(_ kind: AuxiliaryKind, _ modelID: String) async -> Bool {
        events.append("use \(ModelPool.auxiliaryKey(kind, modelID))")
        return true
    }

    func endUse(_ kind: AuxiliaryKind, _ modelID: String) async {
        events.append("end \(ModelPool.auxiliaryKey(kind, modelID))")
    }

    func forget(_ kind: AuxiliaryKind, _ modelID: String) async {
        events.append("forget \(ModelPool.auxiliaryKey(kind, modelID))")
    }
}

@Suite("HummingbirdServer auxiliary residency (#130)")
struct HummingbirdServerAuxiliaryResidencyTests {

    private struct Refused: Error {}

    private func server(
        resolving id: String, format: ModelFormat, residency: RecordingResidency
    ) -> HummingbirdServer {
        let model = LocalModel(
            id: id, displayName: id,
            directory: URL(filePath: "/tmp/\(id)"), sizeBytes: 0, format: format,
            quantization: nil, parameterCount: nil, architecture: nil
        )
        let stub = StubInferenceEngine(engineID: .mlxSwift)
        return HummingbirdServer(
            engineProvider: { stub },
            modelResolver: { reqID in reqID == id ? model : nil },
            auxiliaryResidency: residency
        )
    }

    private func post(_ url: URL, body: Data?) async throws -> (Data, HTTPURLResponse) {
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = body
        let (data, response) = try await URLSession.shared.data(for: req)
        let http = try #require(response as? HTTPURLResponse)
        return (data, http)
    }

    private func post(_ url: URL, json: Any) async throws -> (Data, HTTPURLResponse) {
        try await post(url, body: try JSONSerialization.data(withJSONObject: json))
    }

    private func json(_ data: Data) -> [String: Any] {
        (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
    }

    private func errorCode(_ data: Data) -> String? {
        (json(data)["error"] as? [String: Any])?["code"] as? String
    }

    @Test("a cold rerank admits the reranker and loads it inside the admission")
    func coldRerankLoadsInsideTheAdmission() async throws {
        let residency = RecordingResidency()
        let server = server(resolving: "cross-encoder-model", format: .reranker, residency: residency)
        let port = try await server.start(preferredPort: 19_990)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/rerank")!

        let (data, response) = try await post(
            url, json: ["model": "cross-encoder-model", "query": "q", "documents": ["d"]])
        #expect(response.statusCode == 500)
        #expect(errorCode(data) == "load_failed")
        let events = await residency.events
        #expect(
            events == ["admit reranker:cross-encoder-model", "load threw reranker:cross-encoder-model"],
            "admitted once, the load ran inside the admission, nothing used, forgotten or ended")
        let lockHeld = await server.generationLockIsHeld
        #expect(!lockHeld)
        await server.stop()
        let afterStop = await residency.events
        #expect(afterStop.count == 2, "stopping a server that holds nothing forgets nothing")
    }

    @Test("a cold embeddings request admits the embedder the same way")
    func coldEmbeddingsLoadsInsideTheAdmission() async throws {
        let residency = RecordingResidency()
        let server = server(resolving: "embedder-model", format: .embedder, residency: residency)
        let port = try await server.start(preferredPort: 19_991)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/embeddings")!

        let (data, response) = try await post(url, json: ["model": "embedder-model", "input": "hello"])
        #expect(response.statusCode == 500)
        #expect(errorCode(data) == "load_failed")
        let events = await residency.events
        #expect(events == ["admit embedder:embedder-model", "load threw embedder:embedder-model"])
        await server.stop()
    }

    @Test("the pool refusing admission is a load failure with the lock released and no load run")
    func refusedAdmissionIsALoadFailure() async throws {
        let residency = RecordingResidency()
        await residency.refuse(with: Refused())
        let server = server(resolving: "cross-encoder-model", format: .reranker, residency: residency)
        let port = try await server.start(preferredPort: 19_992)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/rerank")!

        let (data, response) = try await post(
            url, json: ["model": "cross-encoder-model", "query": "q", "documents": ["d"]])
        #expect(response.statusCode == 500)
        #expect(errorCode(data) == "load_failed")
        let events = await residency.events
        #expect(events == ["admit reranker:cross-encoder-model"], "refused before the load ran")
        let lockHeld = await server.generationLockIsHeld
        #expect(!lockHeld, "a refusal leaves the lock free like any failed swap")
        await server.stop()
    }

    @Test("the unload door answers 404 for a model that is not resident, 400 for a bad body, and unloads the chat engine without one")
    func unloadDoorNamesWhatItDrops() async throws {
        let residency = RecordingResidency()
        let server = server(resolving: "cross-encoder-model", format: .reranker, residency: residency)
        let port = try await server.start(preferredPort: 19_993)
        let url = URL(string: "http://127.0.0.1:\(port)/x/models/unload")!

        let (missing, missingResponse) = try await post(url, json: ["model": "cross-encoder-model"])
        #expect(missingResponse.statusCode == 404)
        #expect(errorCode(missing) == "model_not_loaded")

        let (_, badResponse) = try await post(url, body: Data("{not json".utf8))
        #expect(badResponse.statusCode == 400)

        let (plain, plainResponse) = try await post(url, body: nil)
        #expect(plainResponse.statusCode == 200)
        #expect(json(plain)["status"] as? String == "unloaded")

        let (nullModel, nullResponse) = try await post(url, json: ["model": NSNull()])
        #expect(nullResponse.statusCode == 200)
        #expect(json(nullModel)["status"] as? String == "unloaded")

        let events = await residency.events
        #expect(events == [], "nothing was resident, so the pool heard nothing")
        let lockHeld = await server.generationLockIsHeld
        #expect(!lockHeld)
        await server.stop()
    }

    @Test("/x/status shows both auxiliary slots, empty")
    func statusShowsTheAuxiliarySlots() async throws {
        let residency = RecordingResidency()
        let server = server(resolving: "cross-encoder-model", format: .reranker, residency: residency)
        let port = try await server.start(preferredPort: 19_994)
        let url = URL(string: "http://127.0.0.1:\(port)/x/status")!

        let (data, response) = try await URLSession.shared.data(from: url)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        let body = json(data)
        #expect(body.keys.contains("reranker"))
        #expect(body.keys.contains("embedder"))
        #expect(body["reranker"] is NSNull)
        #expect(body["embedder"] is NSNull)
        await server.stop()
    }
}
