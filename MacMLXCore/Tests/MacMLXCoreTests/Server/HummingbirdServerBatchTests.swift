import Foundation
import Testing

@testable import MacMLXCore

// MARK: - HummingbirdServer A2d batched-path integration tests
//
// These verify the SERVER-side wiring of continuous batching against a stubbed
// `BatchGenerationServing` seam — no model, no Metal, mirroring how A2c covered
// the scheduler with a scripted `BatchInferenceCore`. The real engine→scheduler
// construction is a separate wave; here the seam is a deterministic echo.
//
// Port assignments (19_800 range, spaced by 10):
//   batchableRequestServedBySeamNotEngine          : 19_800
//   draftModelRequestBypassesBatch                 : 19_810
//   seamRefusalFallsBackToEngine                   : 19_820
//   noSeamAlwaysUsesEngine                         : 19_830
//   concurrentBatchRequestsEachGetOwnStream        : 19_840
//   streamingBatchRequestStreamsViaSeam             : 19_850
//   coldSwapDrainsSeamBeforeSwap                   : 19_860
//   markInFlightTrackedForBatchRequest              : 19_870
//   loadEndpointDrainsSeamBeforeLoad                : 19_880
//   unloadEndpointDrainsSeamBeforeUnload            : 19_890
//   concurrentSameModelDifferentPromptsNoCrossTalk  : 19_900

@Suite("HummingbirdServerBatch")
struct HummingbirdServerBatchTests {

    // MARK: Stubs

    /// Deterministic `BatchGenerationServing` seam. Accepts every request by
    /// default (echoing a `batch:<model>` stream), or REFUSES a named set by
    /// returning nil (forcing the legacy fallback). Records every submit + drain
    /// so tests can assert the seam was (or was not) consulted.
    private actor StubBatchServing: BatchGenerationServing {
        private let refuse: Set<String>
        /// When true, the echoed chunk includes the request's own prompt text
        /// (`"batch:<model>:<prompt>"`) instead of just the model
        /// (`"batch:<model>"`) — lets a test distinguish concurrent same-model
        /// requests by content. Default false preserves every existing test's
        /// expected `"batch:<model>"` echo.
        private let includePromptInEcho: Bool
        /// Optional probe invoked from `drainForModelChange`, BEFORE the drain
        /// count/bookkeeping below records anything else — lets a test capture
        /// "what model was still resident at the moment of the drain call",
        /// which proves (or disproves) that the drain ran before the swap/load/
        /// unload it is meant to precede.
        private let currentlyLoadedModelIDProbe: (@Sendable () async -> String?)?
        private(set) var submitted: [String] = []
        private(set) var drainCalls = 0
        private(set) var loadedModelIDsAtDrain: [String?] = []

        init(
            refuse: Set<String> = [],
            includePromptInEcho: Bool = false,
            currentlyLoadedModelIDProbe: (@Sendable () async -> String?)? = nil
        ) {
            self.refuse = refuse
            self.includePromptInEcho = includePromptInEcho
            self.currentlyLoadedModelIDProbe = currentlyLoadedModelIDProbe
        }

        func submit(_ request: GenerateRequest) async -> AsyncThrowingStream<GenerateChunk, Error>? {
            submitted.append(request.model)
            if refuse.contains(request.model) { return nil }
            let model = request.model
            let echoText: String
            if includePromptInEcho {
                let prompt = request.messages.last?.content ?? ""
                echoText = "batch:\(model):\(prompt)"
            } else {
                echoText = "batch:\(model)"
            }
            return AsyncThrowingStream { continuation in
                continuation.yield(GenerateChunk(text: echoText))
                continuation.yield(GenerateChunk(
                    text: "",
                    finishReason: .stop,
                    usage: TokenUsage(promptTokens: 3, completionTokens: 2)))
                continuation.finish()
            }
        }

        func drainForModelChange() async {
            if let currentlyLoadedModelIDProbe {
                loadedModelIDsAtDrain.append(await currentlyLoadedModelIDProbe())
            }
            drainCalls += 1
        }

        func submittedModels() -> [String] { submitted }
        func drainCount() -> Int { drainCalls }
        func loadedModelIDsAtDrainTime() -> [String?] { loadedModelIDsAtDrain }
    }

    /// Records the server's in-flight refcount hook (POOL-3) as `"<id>:<active>"`
    /// strings so a test can assert the batched path still marks a model busy.
    private actor InFlightRecorder {
        private(set) var log: [String] = []
        func record(_ id: String, _ active: Bool) { log.append("\(id):\(active)") }
        func snapshot() -> [String] { log }
        /// Exact occurrence count of one `"<id>:<active>"` entry — stronger than
        /// `contains`, which can't tell "happened once" from "happened twice".
        func count(_ entry: String) -> Int { log.filter { $0 == entry }.count }
    }

    // MARK: Helpers

    private func fixtureModel(id: String) -> LocalModel {
        LocalModel(
            id: id, displayName: id,
            directory: URL(filePath: "/tmp/\(id)"),
            sizeBytes: 0, format: .mlx,
            quantization: nil, parameterCount: nil, architecture: nil
        )
    }

    private func loadedEngine(id: String) async throws -> StubInferenceEngine {
        let engine = StubInferenceEngine(engineID: .mlxSwift)
        try await engine.load(fixtureModel(id: id))
        return engine
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

    private func chatContent(_ data: Data) throws -> String {
        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let choices = try #require(json["choices"] as? [[String: Any]])
        let message = try #require(choices.first?["message"] as? [String: Any])
        return try #require(message["content"] as? String)
    }

    /// Reassemble the streamed `choices[0].delta.content` from an SSE body.
    private func streamedContent(_ data: Data) -> String {
        let text = String(decoding: data, as: UTF8.self)
        var content = ""
        for rawLine in text.split(separator: "\n") {
            guard rawLine.hasPrefix("data: ") else { continue }
            let payload = rawLine.dropFirst("data: ".count)
            if payload == "[DONE]" { continue }
            guard let d = payload.data(using: .utf8),
                  let json = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
                  let choices = json["choices"] as? [[String: Any]],
                  let delta = choices.first?["delta"] as? [String: Any],
                  let piece = delta["content"] as? String
            else { continue }
            content += piece
        }
        return content
    }

    private func chatBody(
        _ model: String, stream: Bool = false, draftModel: String? = nil, content: String = "Hi"
    ) -> [String: Any] {
        var body: [String: Any] = [
            "model": model,
            "messages": [["role": "user", "content": content]],
            "stream": stream,
        ]
        if let draftModel { body["draft_model"] = draftModel }
        return body
    }

    // MARK: Tests

    /// An eligible request against an accepting seam is served by the SEAM, not
    /// the engine: the response content is the seam's `batch:<model>` echo, not
    /// the StubInferenceEngine's `stub-response`.
    @Test
    func batchableRequestServedBySeamNotEngine() async throws {
        let seam = StubBatchServing()
        let server = HummingbirdServer(
            engineProvider: { StubInferenceEngine(engineID: .mlxSwift) },
            batchServing: seam)
        let port = try await server.start(preferredPort: 19_800)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let (data, response) = try await postRaw(url, jsonObject: chatBody("m"))
        await server.stop()

        #expect(response.statusCode == 200)
        #expect(try chatContent(data) == "batch:m", "an accepted request must be served by the seam")
        let submitted = await seam.submittedModels()
        #expect(submitted == ["m"], "the seam must have been consulted exactly once")
    }

    /// A `draft_model` (speculative) request is mutually exclusive with batching:
    /// it must bypass the seam entirely and use the legacy single-stream engine.
    @Test
    func draftModelRequestBypassesBatch() async throws {
        let seam = StubBatchServing()
        let engine = try await loadedEngine(id: "m")
        let server = HummingbirdServer(engineProvider: { engine }, batchServing: seam)
        let port = try await server.start(preferredPort: 19_810)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let (data, response) = try await postRaw(url, jsonObject: chatBody("m", draftModel: "d"))
        await server.stop()

        #expect(response.statusCode == 200)
        #expect(try chatContent(data) == "stub-response", "a draft-model request must use the legacy engine path")
        let submitted = await seam.submittedModels()
        #expect(submitted.isEmpty, "the seam must NOT be consulted for a speculative request")
    }

    /// When the seam REFUSES (returns nil — e.g. a VLM / uncoverable resident
    /// model), the server falls back to the legacy engine path cleanly. The seam
    /// is consulted (submit recorded) but performs no work; the engine answers.
    @Test
    func seamRefusalFallsBackToEngine() async throws {
        let seam = StubBatchServing(refuse: ["m"])
        let engine = try await loadedEngine(id: "m")
        let server = HummingbirdServer(engineProvider: { engine }, batchServing: seam)
        let port = try await server.start(preferredPort: 19_820)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let (data, response) = try await postRaw(url, jsonObject: chatBody("m"))
        await server.stop()

        #expect(response.statusCode == 200)
        #expect(try chatContent(data) == "stub-response", "a refused request must fall back to the engine")
        let submitted = await seam.submittedModels()
        #expect(submitted == ["m"], "the seam must have been consulted, then declined")
    }

    /// With no seam installed (the default), behaviour is byte-for-byte the legacy
    /// path — the zero-regression guarantee.
    @Test
    func noSeamAlwaysUsesEngine() async throws {
        let engine = try await loadedEngine(id: "m")
        let server = HummingbirdServer(engine: engine)  // no seam
        let port = try await server.start(preferredPort: 19_830)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let (data, response) = try await postRaw(url, jsonObject: chatBody("m"))
        await server.stop()

        #expect(response.statusCode == 200)
        #expect(try chatContent(data) == "stub-response")
    }

    /// Four concurrent batched requests, each naming a different model, must each
    /// receive their OWN correct stream — no cross-talk at the fan-out boundary.
    @Test
    func concurrentBatchRequestsEachGetOwnStream() async throws {
        let seam = StubBatchServing()
        let server = HummingbirdServer(
            engineProvider: { StubInferenceEngine(engineID: .mlxSwift) },
            batchServing: seam)
        let port = try await server.start(preferredPort: 19_840)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!

        async let r0 = postRaw(url, jsonObject: chatBody("m0"))
        async let r1 = postRaw(url, jsonObject: chatBody("m1"))
        async let r2 = postRaw(url, jsonObject: chatBody("m2"))
        async let r3 = postRaw(url, jsonObject: chatBody("m3"))
        let (d0, _) = try await r0
        let (d1, _) = try await r1
        let (d2, _) = try await r2
        let (d3, _) = try await r3
        await server.stop()

        #expect(try chatContent(d0) == "batch:m0")
        #expect(try chatContent(d1) == "batch:m1")
        #expect(try chatContent(d2) == "batch:m2")
        #expect(try chatContent(d3) == "batch:m3")
        let submitted = Set(await seam.submittedModels())
        #expect(submitted == ["m0", "m1", "m2", "m3"], "all four requests must reach the seam")
    }

    /// Four concurrent batched requests naming the SAME model but with DIFFERENT
    /// prompts must each get back their OWN content — a content-level cross-talk
    /// guard complementing `concurrentBatchRequestsEachGetOwnStream` (which only
    /// varies the model, so it can't catch a fan-out bug that mixes up rows
    /// sharing one model).
    @Test
    func concurrentSameModelDifferentPromptsNoCrossTalk() async throws {
        let seam = StubBatchServing(includePromptInEcho: true)
        let server = HummingbirdServer(
            engineProvider: { StubInferenceEngine(engineID: .mlxSwift) },
            batchServing: seam)
        let port = try await server.start(preferredPort: 19_900)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!

        async let r0 = postRaw(url, jsonObject: chatBody("m", content: "prompt-0"))
        async let r1 = postRaw(url, jsonObject: chatBody("m", content: "prompt-1"))
        async let r2 = postRaw(url, jsonObject: chatBody("m", content: "prompt-2"))
        async let r3 = postRaw(url, jsonObject: chatBody("m", content: "prompt-3"))
        let (d0, _) = try await r0
        let (d1, _) = try await r1
        let (d2, _) = try await r2
        let (d3, _) = try await r3
        await server.stop()

        #expect(try chatContent(d0) == "batch:m:prompt-0", "each response must carry its OWN prompt's content")
        #expect(try chatContent(d1) == "batch:m:prompt-1", "each response must carry its OWN prompt's content")
        #expect(try chatContent(d2) == "batch:m:prompt-2", "each response must carry its OWN prompt's content")
        #expect(try chatContent(d3) == "batch:m:prompt-3", "each response must carry its OWN prompt's content")
    }

    /// A streaming batched request streams the seam's output as OpenAI SSE frames
    /// (through the same watchdog/splitter/`[DONE]` machinery as the single path).
    @Test
    func streamingBatchRequestStreamsViaSeam() async throws {
        let seam = StubBatchServing()
        let server = HummingbirdServer(
            engineProvider: { StubInferenceEngine(engineID: .mlxSwift) },
            batchServing: seam)
        let port = try await server.start(preferredPort: 19_850)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let (data, response) = try await postRaw(url, jsonObject: chatBody("m", stream: true))
        await server.stop()

        #expect(response.statusCode == 200)
        #expect(streamedContent(data) == "batch:m", "the streamed content must be the seam's echo")
    }

    /// A cold-swap (a legacy request forcing a model change) must DRAIN the seam
    /// first (SRV-2 generalized to drain-after-swap) so a resident batched cohort
    /// is never swapped out from under its live rows. A speculative request forces
    /// the legacy swap path; the drain must fire before the load. The probe also
    /// asserts ORDERING: at the moment of the drain call, the engine must still
    /// report the OLD model as loaded — proof the drain ran before the swap, not
    /// merely that it ran (matching the seam's submit/drain-epoch contract).
    @Test
    func coldSwapDrainsSeamBeforeSwap() async throws {
        let engine = try await loadedEngine(id: "model-a")
        let seam = StubBatchServing(currentlyLoadedModelIDProbe: { [engine] in
            await engine.loadedModel?.id
        })
        let modelB = fixtureModel(id: "model-b")
        let resolver: HummingbirdServer.ModelResolver = { id in
            id == "model-b" ? modelB : nil
        }
        let server = HummingbirdServer(
            engineProvider: { engine }, modelResolver: resolver, batchServing: seam)
        let port = try await server.start(preferredPort: 19_860)

        // draft_model forces the legacy path → beginGeneration → ensureModelLoaded
        // → swap model-a → model-b, which must drain the seam first.
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        let (_, response) = try await postRaw(url, jsonObject: chatBody("model-b", draftModel: "d"))
        let loaded = await engine.loadedModel?.id
        let drains = await seam.drainCount()
        let loadedAtDrain = await seam.loadedModelIDsAtDrainTime()
        await server.stop()

        #expect(response.statusCode == 200)
        #expect(loaded == "model-b", "the swap must have completed")
        #expect(drains == 1, "the seam must be drained exactly once, before the swap")
        #expect(loadedAtDrain == ["model-a"], "drain must observe the OLD model still loaded — i.e. run before the swap")
    }

    /// The manual `/x/models/load` door must also drain a resident batched cohort
    /// before the mutating load (the same drain-before-model-change invariant as
    /// the generation cold-swap, reached through a different door). The probe
    /// asserts ordering: nothing is resident yet at drain time, since this is a
    /// fresh load — the drain still fires (it's unconditional, cheap when idle)
    /// strictly before the load call that follows it.
    @Test
    func loadEndpointDrainsSeamBeforeLoad() async throws {
        let engine = StubInferenceEngine(engineID: .mlxSwift)
        let seam = StubBatchServing(currentlyLoadedModelIDProbe: { [engine] in
            await engine.loadedModel?.id
        })
        let server = HummingbirdServer(engineProvider: { engine }, batchServing: seam)
        let port = try await server.start(preferredPort: 19_880)

        let url = URL(string: "http://127.0.0.1:\(port)/x/models/load")!
        let (_, response) = try await postRaw(url, jsonObject: ["model_path": "/tmp/model-x"])
        let drains = await seam.drainCount()
        let loaded = await engine.loadedModel?.id
        let loadedAtDrain = await seam.loadedModelIDsAtDrainTime()
        await server.stop()

        #expect(response.statusCode == 200)
        #expect(loaded == "model-x", "the load must have completed")
        #expect(drains == 1, "the load door must drain the cohort before loading")
        #expect(loadedAtDrain == [nil], "drain must run before the load — nothing resident yet")
    }

    /// The manual `/x/models/unload` door must also drain a resident batched
    /// cohort before the mutating unload — mirrors `loadEndpointDrainsSeamBeforeLoad`,
    /// reached through the unload door instead. The probe asserts ordering: the
    /// model is STILL resident at drain time, proving the drain runs before the
    /// unload actually clears it.
    @Test
    func unloadEndpointDrainsSeamBeforeUnload() async throws {
        let engine = try await loadedEngine(id: "model-y")
        let seam = StubBatchServing(currentlyLoadedModelIDProbe: { [engine] in
            await engine.loadedModel?.id
        })
        let server = HummingbirdServer(engineProvider: { engine }, batchServing: seam)
        let port = try await server.start(preferredPort: 19_890)

        let url = URL(string: "http://127.0.0.1:\(port)/x/models/unload")!
        let (_, response) = try await postRaw(url, jsonObject: [String: Any]())
        let drains = await seam.drainCount()
        let loadedAtDrain = await seam.loadedModelIDsAtDrainTime()
        let loadedAfter = await engine.loadedModel?.id
        await server.stop()

        #expect(response.statusCode == 200)
        #expect(loadedAfter == nil, "the unload must have completed")
        #expect(drains == 1, "the unload door must drain the cohort before unloading")
        #expect(loadedAtDrain == ["model-y"], "drain must observe the model still resident — i.e. run before the unload")
    }

    /// The batched path still marks the model in-flight (POOL-3), so a concurrent
    /// load cannot LRU-evict it mid-generation. Asserts EXACT counts (not just
    /// `contains`) so a regression that double-marks (e.g. both
    /// `handleChatCompletions` and a batch responder marking `true`) is caught.
    @Test
    func markInFlightTrackedForBatchRequest() async throws {
        let seam = StubBatchServing()
        let recorder = InFlightRecorder()
        let hook: HummingbirdServer.InFlightHook = { id, active in
            await recorder.record(id, active)
        }
        let server = HummingbirdServer(
            engineProvider: { StubInferenceEngine(engineID: .mlxSwift) },
            inFlightHook: hook,
            batchServing: seam)
        let port = try await server.start(preferredPort: 19_870)

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        _ = try await postRaw(url, jsonObject: chatBody("m"))

        // The "active=true" mark happens synchronously in `handleChatCompletions`
        // BEFORE `submit` is even called (MEDIUM#2); the "false" mark is deferred
        // to a detached Task after the response returns, so poll briefly for it
        // rather than racing.
        var trueCount = 0
        var falseCount = 0
        for _ in 0..<40 {
            trueCount = await recorder.count("m:true")
            falseCount = await recorder.count("m:false")
            if trueCount > 0 && falseCount > 0 { break }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        await server.stop()

        #expect(trueCount == 1, "the batched path must mark the model in-flight EXACTLY once (POOL-3)")
        #expect(falseCount == 1, "the batched path must clear the in-flight mark EXACTLY once")
    }
}
