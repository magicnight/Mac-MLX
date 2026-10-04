import Foundation
import HTTPTypes
import NIOCore
import Testing
@testable import MacMLXCore

// MARK: - /v1/audio/transcriptions + /v1/audio/speech Tests
//
// These exercise route wiring, request decoding, every validation gate — all
// the paths that end in a 4xx before any model load is attempted — and, against
// a scripted `AudioBackend`, the handlers' lock contract and the shapes of
// their success and failure replies. Nothing here downloads weights, opens a
// network connection to the Hub, or touches Metal: the validation cases return
// before the backend is reached, and the scripted backend never runs a model.
// Only the real forward pass needs a checkpoint, and that stays deferred.
//
// Port assignments (20_600 range, spaced by 10):
//   transcriptionsRejectsNonMultipartBody            : 20_600
//   transcriptionsMissingFileReturns400              : 20_610
//   transcriptionsMissingModelReturns400             : 20_620
//   transcriptionsRejectsSubtitleResponseFormat      : 20_630
//   transcriptionsRejectsUnknownResponseFormat       : 20_640
//   transcriptionsRejectsPromptRatherThanIgnoringIt  : 20_650
//   transcriptionsRejectsOutOfRangeTemperature       : 20_660
//   transcriptionsRejectsEmptyBody                   : 20_670
//   transcriptionsRejectsMalformedMultipartBody      : 20_680
//   speechInvalidJSONReturns400                      : 20_690
//   speechMissingModelReturns400                     : 20_700
//   speechMissingInputReturns400                     : 20_710
//   speechMissingVoiceReturns400                     : 20_720
//   speechRejectsMP3AndEveryCompressedFormat         : 20_730
//   speechRejectsUnknownResponseFormat               : 20_740
//   speechRejectsOutOfRangeSpeed                     : 20_750
//   speechRejectsUnimplementedSpeed                  : 20_760
//   speechRejectsOverlongInput                       : 20_770
//   transcriptionsLoadWaitsForTheGenerationLockAndReleasesItOnFailure : 20_780
//   speechLoadWaitsForTheGenerationLockAndReleasesItOnFailure         : 20_790
//   aMalformedModelIDDoesNotWaitForTheLock                            : 20_810
//   transcriptionsServeTheBackendResultAndReleaseTheLock              : 20_820
//   speechServesTheBackendResultAndReleasesTheLock                    : 20_830
//   speechPCMAtTheWrongSampleRateIs400                                : 20_840
//   (20_800 belongs to HummingbirdServerBatchTests)

@Suite("HummingbirdServer audio endpoints")
struct HummingbirdServerAudioTests {

    // MARK: Helpers

    private func makeServer() -> HummingbirdServer {
        HummingbirdServer(engine: StubInferenceEngine(engineID: .mlxSwift))
    }

    private static let boundary = "----macmlxAudioTestBoundary"

    /// Build a `multipart/form-data` body from text fields plus an optional
    /// file part, matching what `curl -F` emits.
    private func multipartBody(
        fields: [(String, String)],
        file: (name: String, filename: String, bytes: Data)? = nil
    ) -> Data {
        var out = Data()
        for (name, value) in fields {
            out.append(Data("--\(Self.boundary)\r\n".utf8))
            out.append(Data("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n".utf8))
            out.append(Data(value.utf8))
            out.append(Data("\r\n".utf8))
        }
        if let file {
            out.append(Data("--\(Self.boundary)\r\n".utf8))
            let disposition = "Content-Disposition: form-data; name=\"\(file.name)\"; "
                + "filename=\"\(file.filename)\"\r\n"
            out.append(Data(disposition.utf8))
            out.append(Data("Content-Type: audio/wav\r\n\r\n".utf8))
            out.append(file.bytes)
            out.append(Data("\r\n".utf8))
        }
        out.append(Data("--\(Self.boundary)--\r\n".utf8))
        return out
    }

    /// A tiny but structurally valid WAV, so failures are attributable to the
    /// gate under test and never to "that wasn't audio".
    private var sampleWAV: Data {
        WAVEncoder.encode(samples: [0, 0.1, -0.1, 0], sampleRate: 16_000) ?? Data()
    }

    private func post(
        _ url: URL, body: Data, contentType: String
    ) async throws -> (Data, HTTPURLResponse) {
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue(contentType, forHTTPHeaderField: "Content-Type")
        req.httpBody = body
        let (data, response) = try await URLSession.shared.data(for: req)
        let http = try #require(response as? HTTPURLResponse)
        return (data, http)
    }

    private func postMultipart(
        _ url: URL,
        fields: [(String, String)],
        file: (name: String, filename: String, bytes: Data)? = nil
    ) async throws -> (Data, HTTPURLResponse) {
        try await post(
            url,
            body: multipartBody(fields: fields, file: file),
            contentType: "multipart/form-data; boundary=\(Self.boundary)")
    }

    private func postJSON(_ url: URL, object: Any) async throws -> (Data, HTTPURLResponse) {
        try await post(
            url,
            body: try JSONSerialization.data(withJSONObject: object),
            contentType: "application/json")
    }

    private func errorCode(_ data: Data) -> String? {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let error = json["error"] as? [String: Any],
              let code = error["code"] as? String
        else { return nil }
        return code
    }

    private func errorMessage(_ data: Data) -> String? {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let error = json["error"] as? [String: Any],
              let message = error["message"] as? String
        else { return nil }
        return message
    }

    private func transcriptionsURL(_ port: Int) -> URL {
        URL(string: "http://127.0.0.1:\(port)/v1/audio/transcriptions")!
    }

    private func speechURL(_ port: Int) -> URL {
        URL(string: "http://127.0.0.1:\(port)/v1/audio/speech")!
    }

    // MARK: /v1/audio/transcriptions

    @Test
    func transcriptionsRejectsNonMultipartBody() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_600)
        let (data, response) = try await postJSON(
            transcriptionsURL(port), object: ["model": "openai/whisper-tiny"])
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(errorCode(data) == "invalid_request_error")
        #expect(errorMessage(data)?.contains("multipart/form-data") == true)
    }

    @Test
    func transcriptionsMissingFileReturns400() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_610)
        let (data, response) = try await postMultipart(
            transcriptionsURL(port), fields: [("model", "openai/whisper-tiny")])
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(errorCode(data) == "invalid_request_error")
        #expect(errorMessage(data)?.contains("`file`") == true)
    }

    @Test
    func transcriptionsMissingModelReturns400() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_620)
        let (data, response) = try await postMultipart(
            transcriptionsURL(port),
            fields: [],
            file: (name: "file", filename: "a.wav", bytes: sampleWAV))
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(errorCode(data) == "invalid_request_error")
        #expect(errorMessage(data)?.contains("`model`") == true)
    }

    @Test
    func transcriptionsRejectsSubtitleResponseFormat() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_630)
        var results: [(Int, String?)] = []
        for format in ["srt", "vtt"] {
            let (data, response) = try await postMultipart(
                transcriptionsURL(port),
                fields: [("model", "openai/whisper-tiny"), ("response_format", format)],
                file: (name: "file", filename: "a.wav", bytes: sampleWAV))
            results.append((response.statusCode, errorCode(data)))
        }
        await server.stop()

        // Refused outright — never quietly served as JSON under a subtitle name.
        #expect(results.allSatisfy { $0.0 == 400 })
        #expect(results.allSatisfy { $0.1 == "unsupported_response_format" })
    }

    @Test
    func transcriptionsRejectsUnknownResponseFormat() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_640)
        let (data, response) = try await postMultipart(
            transcriptionsURL(port),
            fields: [("model", "openai/whisper-tiny"), ("response_format", "yaml")],
            file: (name: "file", filename: "a.wav", bytes: sampleWAV))
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(errorCode(data) == "unsupported_response_format")
    }

    @Test
    func transcriptionsRejectsPromptRatherThanIgnoringIt() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_650)
        let (data, response) = try await postMultipart(
            transcriptionsURL(port),
            fields: [("model", "openai/whisper-tiny"), ("prompt", "medical terminology")],
            file: (name: "file", filename: "a.wav", bytes: sampleWAV))
        await server.stop()

        // Accepting `prompt` and dropping it would be a silent lie about what
        // conditioned the transcript.
        #expect(response.statusCode == 400)
        #expect(errorCode(data) == "unsupported_parameter")
    }

    @Test
    func transcriptionsRejectsOutOfRangeTemperature() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_660)
        var statuses: [Int] = []
        for value in ["2.0", "-0.5", "hot"] {
            let (_, response) = try await postMultipart(
                transcriptionsURL(port),
                fields: [("model", "openai/whisper-tiny"), ("temperature", value)],
                file: (name: "file", filename: "a.wav", bytes: sampleWAV))
            statuses.append(response.statusCode)
        }
        await server.stop()
        #expect(statuses == [400, 400, 400])
    }

    @Test
    func transcriptionsRejectsEmptyBody() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_670)
        let (data, response) = try await post(
            transcriptionsURL(port),
            body: Data(),
            contentType: "multipart/form-data; boundary=\(Self.boundary)")
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(errorCode(data) == "invalid_request_error")
    }

    @Test
    func transcriptionsRejectsMalformedMultipartBody() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_680)
        // Truncated: the closing delimiter never arrives.
        var body = Data("--\(Self.boundary)\r\n".utf8)
        body.append(Data("Content-Disposition: form-data; name=\"model\"\r\n\r\nwhisper".utf8))
        let (data, response) = try await post(
            transcriptionsURL(port),
            body: body,
            contentType: "multipart/form-data; boundary=\(Self.boundary)")
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(errorCode(data) == "invalid_request_error")
        #expect(errorMessage(data)?.contains("Malformed") == true)
    }

    // MARK: /v1/audio/speech

    @Test
    func speechInvalidJSONReturns400() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_690)
        let (data, response) = try await post(
            speechURL(port), body: Data("{not json".utf8), contentType: "application/json")
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(errorCode(data) == "invalid_request_error")
        #expect(errorMessage(data)?.contains("Invalid JSON") == true)
    }

    @Test
    func speechMissingModelReturns400() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_700)
        let (data, response) = try await postJSON(
            speechURL(port), object: ["input": "hello", "voice": "af_heart"])
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(errorMessage(data)?.contains("`model`") == true)
    }

    @Test
    func speechMissingInputReturns400() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_710)
        let (data, response) = try await postJSON(
            speechURL(port), object: ["model": "mlx-community/Kokoro-82M-4bit", "voice": "af_heart"])
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(errorMessage(data)?.contains("`input`") == true)
    }

    @Test
    func speechMissingVoiceReturns400() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_720)
        let (data, response) = try await postJSON(
            speechURL(port), object: ["model": "mlx-community/Kokoro-82M-4bit", "input": "hello"])
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(errorMessage(data)?.contains("`voice`") == true)
    }

    @Test
    func speechRejectsMP3AndEveryCompressedFormat() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_730)
        var results: [(status: Int, code: String?, contentType: String?)] = []
        for format in ["mp3", "opus", "aac", "flac"] {
            let (data, response) = try await postJSON(speechURL(port), object: [
                "model": "mlx-community/Kokoro-82M-4bit",
                "input": "hello",
                "voice": "af_heart",
                "response_format": format,
            ])
            results.append((
                response.statusCode,
                errorCode(data),
                response.value(forHTTPHeaderField: "Content-Type")))
        }
        await server.stop()

        #expect(results.allSatisfy { $0.status == 400 })
        #expect(results.allSatisfy { $0.code == "unsupported_response_format" })
        // The whole point: no audio content type is ever returned for a format
        // macMLX cannot encode.
        #expect(results.allSatisfy { $0.contentType?.contains("audio/") != true })
    }

    @Test
    func speechRejectsUnknownResponseFormat() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_740)
        let (data, response) = try await postJSON(speechURL(port), object: [
            "model": "mlx-community/Kokoro-82M-4bit",
            "input": "hello",
            "voice": "af_heart",
            "response_format": "ogg",
        ])
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(errorCode(data) == "unsupported_response_format")
    }

    @Test
    func speechRejectsOutOfRangeSpeed() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_750)
        var statuses: [Int] = []
        for speed in [0.1, 5.0, -1.0] {
            let (_, response) = try await postJSON(speechURL(port), object: [
                "model": "mlx-community/Kokoro-82M-4bit",
                "input": "hello",
                "voice": "af_heart",
                "speed": speed,
            ])
            statuses.append(response.statusCode)
        }
        await server.stop()
        #expect(statuses == [400, 400, 400])
    }

    @Test
    func speechRejectsUnimplementedSpeed() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_760)
        let (data, response) = try await postJSON(speechURL(port), object: [
            "model": "mlx-community/Kokoro-82M-4bit",
            "input": "hello",
            "voice": "af_heart",
            "speed": 1.5,
        ])
        await server.stop()

        // In range per OpenAI, but macMLX has no rate control — refuse rather
        // than accept-and-discard.
        #expect(response.statusCode == 400)
        #expect(errorCode(data) == "unsupported_parameter")
    }

    @Test
    func speechRejectsOverlongInput() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_770)
        let (data, response) = try await postJSON(speechURL(port), object: [
            "model": "mlx-community/Kokoro-82M-4bit",
            "input": String(repeating: "a", count: 4097),
            "voice": "af_heart",
        ])
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(errorCode(data) == "invalid_request_error")
    }

    // MARK: Lock contract + reply shapes, against a scripted backend

    /// Answers every call with a fixed result or a fixed error and records the
    /// calls, so the handlers' lock discipline and reply shapes can be checked
    /// without weights, the Hub, or Metal. Once attached to its server it also
    /// records, per call, whether the generation lock was held while it ran —
    /// the property this whole seam exists to pin.
    private actor ScriptedAudioBackend: AudioBackend {
        struct TranscribeCall: Equatable {
            let model: String
            /// Size of the staged upload the handler pointed the backend at.
            let stagedBytes: Int
            let language: String?
            let temperature: Float?
        }
        struct SynthesizeCall: Equatable {
            let model: String
            let text: String
            let voice: String?
            let language: String?
        }

        static let transcript = "scripted transcript"
        static let samples: [Float] = [0, 0.5, -0.5, 0]

        private(set) var transcribeCalls: [TranscribeCall] = []
        private(set) var synthesizeCalls: [SynthesizeCall] = []
        /// One entry per backend call: was the server's generation lock held
        /// while the call ran? Recorded only after `attach(to:)`.
        private(set) var lockHeldDuringCalls: [Bool] = []
        private let failure: EngineError?
        private let speechSampleRate: Int
        private var server: HummingbirdServer?

        init(failing failure: EngineError? = nil, speechSampleRate: Int = 24_000) {
            self.failure = failure
            self.speechSampleRate = speechSampleRate
        }

        func attach(to server: HummingbirdServer) { self.server = server }

        /// Probe the lock from inside a call: try to take it and give it
        /// straight back. Getting it means the handler was NOT holding it;
        /// not getting it within 150 ms means it was. A probe that loses is
        /// cancelled, and the lock's cancellation path removes the parked
        /// waiter; if a release had already handed it ownership, the probe
        /// releases again, so it never leaves the lock owned by a dead task.
        private func recordLockState() async {
            guard let server else { return }
            let held = await withTaskGroup(of: Bool?.self) { group in
                group.addTask {
                    guard (try? await server.acquireGenerationLock()) != nil else { return nil }
                    await server.releaseGenerationLock()
                    return false
                }
                group.addTask {
                    try? await Task.sleep(nanoseconds: 150_000_000)
                    return true
                }
                let first = await group.next() ?? nil
                group.cancelAll()
                return first ?? true
            }
            lockHeldDuringCalls.append(held)
        }

        func transcribe(
            model: String, audioURL: URL, language: String?, temperature: Float?
        ) async throws -> AudioEngine.Transcription {
            await recordLockState()
            let staged = (try? Data(contentsOf: audioURL))?.count ?? -1
            transcribeCalls.append(
                .init(model: model, stagedBytes: staged, language: language, temperature: temperature))
            if let failure { throw failure }
            return AudioEngine.Transcription(
                text: Self.transcript, language: "en", duration: 1.25,
                segments: [.init(id: 0, start: 0, end: 1.25, text: Self.transcript)])
        }

        func synthesize(
            model: String, text: String, voice: String?, language: String?
        ) async throws -> AudioEngine.Speech {
            await recordLockState()
            synthesizeCalls.append(.init(model: model, text: text, voice: voice, language: language))
            if let failure { throw failure }
            return AudioEngine.Speech(samples: Self.samples, sampleRate: speechSampleRate)
        }
    }

    private func makeServer(backend: ScriptedAudioBackend) async -> HummingbirdServer {
        let server = makeServer()
        await server.useAudioBackend(backend)
        await backend.attach(to: server)
        return server
    }

    /// A completion flag for a request parked behind the lock. `Task.value`
    /// is not cancellable from a task group, so a timed wait needs a flag.
    private actor Done {
        private(set) var isSet = false
        func set() { isSet = true }
    }

    private func postMultipartInBackground(
        _ url: URL,
        fields: [(String, String)],
        file: (name: String, filename: String, bytes: Data)?,
        raising done: Done
    ) -> Task<(Data, HTTPURLResponse), any Error> {
        Task {
            let result = try await postMultipart(url, fields: fields, file: file)
            await done.set()
            return result
        }
    }

    private func postJSONInBackground(
        _ url: URL, object: [String: any Sendable], raising done: Done
    ) -> Task<(Data, HTTPURLResponse), any Error> {
        Task {
            let result = try await postJSON(url, object: object)
            await done.set()
            return result
        }
    }

    /// Whether every flag is set within `seconds`, polling.
    private func allSet(_ flags: [Done], within seconds: Double) async -> Bool {
        let deadline = ContinuousClock.now + .seconds(seconds)
        while ContinuousClock.now < deadline {
            var pending = false
            for flag in flags {
                let isSet = await flag.isSet
                if !isSet { pending = true }
            }
            if !pending { return true }
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        return false
    }

    /// A later acquire must succeed promptly, which proves every path that
    /// took the lock also released it.
    private func lockIsFree(_ server: HummingbirdServer) async -> Bool {
        await withTaskGroup(of: Bool.self) { group in
            group.addTask { (try? await server.acquireGenerationLock()) != nil }
            group.addTask {
                try? await Task.sleep(nanoseconds: 2_000_000_000)
                return false
            }
            let first = await group.next() ?? false
            group.cancelAll()
            return first
        }
    }

    /// SRV-2 for the audio engine: a transcription waits for the generation
    /// lock BEFORE it loads — the backend is not called while a "generation"
    /// holds the lock — and a load that fails releases the lock again. Before
    /// this, the handler loaded first and queued second, so a cold swap ran
    /// beside a generation, and a second request could replace the model, or
    /// empty the slot, while the first was parked at the lock.
    @Test
    func transcriptionsLoadWaitsForTheGenerationLockAndReleasesItOnFailure() async throws {
        let backend = ScriptedAudioBackend(failing: .modelLoadFailed(reason: "scripted"))
        let server = await makeServer(backend: backend)
        let port = try await server.start(preferredPort: 20_780)

        try await server.acquireGenerationLock()   // a generation holds the lock
        let done = Done()
        let pending = postMultipartInBackground(
            transcriptionsURL(port),
            fields: [("model", "openai/whisper-tiny")],
            file: (name: "file", filename: "a.wav", bytes: sampleWAV),
            raising: done)
        try await Task.sleep(nanoseconds: 300_000_000)
        let calledWhileLocked = await backend.transcribeCalls.count
        let answeredWhileLocked = await done.isSet
        #expect(calledWhileLocked == 0, "the load must queue behind the generation lock, not run beside a generation")
        #expect(!answeredWhileLocked)

        await server.releaseGenerationLock()
        let (data, response) = try await pending.value
        #expect(response.statusCode == 500)
        #expect(errorCode(data) == "load_failed")
        #expect(errorMessage(data)?.contains("openai/whisper-tiny") == true)
        let calls = await backend.transcribeCalls.count
        #expect(calls == 1)
        let lockHeld = await backend.lockHeldDuringCalls
        #expect(lockHeld == [true], "the load must run while this request holds the lock")

        let free = await lockIsFree(server)
        #expect(free, "the failed load must release the generation lock")
        if free { await server.releaseGenerationLock() }
        await server.stop()
    }

    /// Same contract on `/v1/audio/speech`.
    @Test
    func speechLoadWaitsForTheGenerationLockAndReleasesItOnFailure() async throws {
        let backend = ScriptedAudioBackend(failing: .modelLoadFailed(reason: "scripted"))
        let server = await makeServer(backend: backend)
        let port = try await server.start(preferredPort: 20_790)

        try await server.acquireGenerationLock()
        let done = Done()
        let pending = postJSONInBackground(
            speechURL(port),
            object: ["model": "mlx-community/Kokoro-82M-4bit", "input": "hello", "voice": "af_heart"],
            raising: done)
        try await Task.sleep(nanoseconds: 300_000_000)
        let calledWhileLocked = await backend.synthesizeCalls.count
        let answeredWhileLocked = await done.isSet
        #expect(calledWhileLocked == 0, "the load must queue behind the generation lock")
        #expect(!answeredWhileLocked)

        await server.releaseGenerationLock()
        let (data, response) = try await pending.value
        #expect(response.statusCode == 500)
        #expect(errorCode(data) == "load_failed")
        let calls = await backend.synthesizeCalls.count
        #expect(calls == 1)
        let lockHeld = await backend.lockHeldDuringCalls
        #expect(lockHeld == [true], "the load must run while this request holds the lock")

        let free = await lockIsFree(server)
        #expect(free, "the failed load must release the generation lock")
        if free { await server.releaseGenerationLock() }
        await server.stop()
    }

    /// A malformed model id is answered before the lock, like the lookup
    /// errors on `/v1/embeddings`: with the lock held, both routes still
    /// return their 400 promptly and never reach the backend.
    @Test
    func aMalformedModelIDDoesNotWaitForTheLock() async throws {
        let backend = ScriptedAudioBackend()
        let server = await makeServer(backend: backend)
        let port = try await server.start(preferredPort: 20_810)
        try await server.acquireGenerationLock()

        let transcriptionDone = Done()
        let pendingTranscription = postMultipartInBackground(
            transcriptionsURL(port),
            fields: [("model", "not-a-repo-id")],
            file: (name: "file", filename: "a.wav", bytes: sampleWAV),
            raising: transcriptionDone)
        let speechDone = Done()
        let pendingSpeech = postJSONInBackground(
            speechURL(port),
            object: ["model": "not-a-repo-id", "input": "hello", "voice": "af_heart"],
            raising: speechDone)
        let answered = await allSet([transcriptionDone, speechDone], within: 2)
        #expect(answered, "a malformed id must be answered without waiting for the lock")
        await server.releaseGenerationLock()

        let (transcriptionData, transcriptionResponse) = try await pendingTranscription.value
        #expect(transcriptionResponse.statusCode == 400)
        #expect(errorCode(transcriptionData) == "invalid_request_error")
        #expect(errorMessage(transcriptionData)?.contains("owner/name") == true)
        let (speechData, speechResponse) = try await pendingSpeech.value
        #expect(speechResponse.statusCode == 400)
        #expect(errorCode(speechData) == "invalid_request_error")
        let transcribeCalls = await backend.transcribeCalls.count
        let synthesizeCalls = await backend.synthesizeCalls.count
        #expect(transcribeCalls == 0 && synthesizeCalls == 0)
        await server.stop()
    }

    /// With a backend that answers, the route serves its result in every
    /// format and releases the lock: `text` as plain text, `verbose_json` with
    /// the measured duration and the segments. The backend sees the staged
    /// upload, the model id, and the knobs the client sent.
    @Test
    func transcriptionsServeTheBackendResultAndReleaseTheLock() async throws {
        let backend = ScriptedAudioBackend()
        let server = await makeServer(backend: backend)
        let port = try await server.start(preferredPort: 20_820)
        let wav = sampleWAV

        let (text, textResponse) = try await postMultipart(
            transcriptionsURL(port),
            fields: [
                ("model", "openai/whisper-tiny"), ("response_format", "text"),
                ("language", "en"), ("temperature", "0.2"),
            ],
            file: (name: "file", filename: "a.wav", bytes: wav))
        #expect(textResponse.statusCode == 200)
        #expect(textResponse.value(forHTTPHeaderField: "Content-Type")?.hasPrefix("text/plain") == true)
        #expect(String(decoding: text, as: UTF8.self) == ScriptedAudioBackend.transcript)

        let (verbose, verboseResponse) = try await postMultipart(
            transcriptionsURL(port),
            fields: [("model", "openai/whisper-tiny"), ("response_format", "verbose_json")],
            file: (name: "file", filename: "a.wav", bytes: wav))
        #expect(verboseResponse.statusCode == 200)
        let json = try #require(try JSONSerialization.jsonObject(with: verbose) as? [String: Any])
        #expect(json["task"] as? String == "transcribe")
        #expect(json["text"] as? String == ScriptedAudioBackend.transcript)
        #expect(json["duration"] as? Double == 1.25)
        #expect(json["language"] as? String == "en")
        let segments = try #require(json["segments"] as? [[String: Any]])
        #expect(segments.count == 1)
        #expect(segments.first?["text"] as? String == ScriptedAudioBackend.transcript)

        let calls = await backend.transcribeCalls
        #expect(calls == [
            .init(model: "openai/whisper-tiny", stagedBytes: wav.count, language: "en", temperature: 0.2),
            .init(model: "openai/whisper-tiny", stagedBytes: wav.count, language: nil, temperature: nil),
        ])
        let lockHeld = await backend.lockHeldDuringCalls
        #expect(lockHeld == [true, true], "load and inference run under the lock")
        let free = await lockIsFree(server)
        #expect(free, "a served request must release the generation lock")
        if free { await server.releaseGenerationLock() }
        await server.stop()
    }

    /// Same on `/v1/audio/speech`: the backend's samples come back as the
    /// requested container, and the lock is free afterwards.
    @Test
    func speechServesTheBackendResultAndReleasesTheLock() async throws {
        let backend = ScriptedAudioBackend()
        let server = await makeServer(backend: backend)
        let port = try await server.start(preferredPort: 20_830)
        let model = "mlx-community/Kokoro-82M-4bit"

        let (wav, wavResponse) = try await postJSON(
            speechURL(port), object: ["model": model, "input": "hello", "voice": "af_heart"])
        #expect(wavResponse.statusCode == 200)
        #expect(wavResponse.value(forHTTPHeaderField: "Content-Type") == "audio/wav")
        #expect(wav == WAVEncoder.encode(samples: ScriptedAudioBackend.samples, sampleRate: 24_000))

        let (pcm, pcmResponse) = try await postJSON(
            speechURL(port),
            object: ["model": model, "input": "hello", "voice": "af_heart", "response_format": "pcm"])
        #expect(pcmResponse.statusCode == 200)
        #expect(pcmResponse.value(forHTTPHeaderField: "Content-Type") == "audio/pcm")
        #expect(pcm == WAVEncoder.pcm16LittleEndian(samples: ScriptedAudioBackend.samples))

        let calls = await backend.synthesizeCalls
        #expect(calls == [
            .init(model: model, text: "hello", voice: "af_heart", language: nil),
            .init(model: model, text: "hello", voice: "af_heart", language: nil),
        ])
        let lockHeld = await backend.lockHeldDuringCalls
        #expect(lockHeld == [true, true], "load and inference run under the lock")
        let free = await lockIsFree(server)
        #expect(free, "a served request must release the generation lock")
        if free { await server.releaseGenerationLock() }
        await server.stop()
    }

    /// `pcm` is headerless 24 kHz by definition; a model that synthesizes at
    /// another rate gets a 400 that names both rates, not audio at the wrong
    /// pitch — and the lock is released on that path too.
    @Test
    func speechPCMAtTheWrongSampleRateIs400() async throws {
        let backend = ScriptedAudioBackend(speechSampleRate: 22_050)
        let server = await makeServer(backend: backend)
        let port = try await server.start(preferredPort: 20_840)

        let (data, response) = try await postJSON(
            speechURL(port),
            object: [
                "model": "mlx-community/Kokoro-82M-4bit", "input": "hello",
                "voice": "af_heart", "response_format": "pcm",
            ])
        #expect(response.statusCode == 400)
        #expect(errorCode(data) == "unsupported_response_format")
        #expect(errorMessage(data)?.contains("22050") == true)
        #expect(errorMessage(data)?.contains("24000") == true)

        let free = await lockIsFree(server)
        #expect(free)
        if free { await server.releaseGenerationLock() }
        await server.stop()
    }

}

// MARK: - Audio failure classification
//
// The two `/v1/audio/*` handlers answer with whatever these pure mappings
// decide, so the mappings are where the "is this the caller's fault or ours?"
// question is actually settled — and they are testable with no server, no
// port, no network, and no checkpoint. Both cases exist because reporting a
// client mistake as 5xx does real damage: it hides the mistake behind
// "internal error", and 5xx is the class client SDKs retry on by default, so
// an id that can never load gets replayed forever.

@Suite("HummingbirdServer audio failure classification")
struct HummingbirdServerAudioFailureClassificationTests {

    /// A stand-in for a failure with no relationship to `EngineError`.
    private struct UnrelatedFailure: Error {}

    // MARK: Model-load failures → 400 vs 500

    @Test
    func aLocallyRejectedModelIDBecomes400NotAServerError() {
        let hint = AudioEngine.repoIDHint("not-a-repo-id", kind: "STT")
        let failure = HummingbirdServer.audioFailure(
            EngineError.invalidAudioModelID(reason: hint), model: "not-a-repo-id",
            operation: "Transcription")

        #expect(failure.status == .badRequest)
        #expect(failure.code == "invalid_request_error")
        // The caller has to be able to see WHAT shape was expected.
        #expect(failure.message.contains("owner/name"))
        #expect(failure.message.contains("not-a-repo-id"))
    }

    @Test
    func aRealLoadFailureStays500() {
        let failure = HummingbirdServer.audioFailure(
            EngineError.modelLoadFailed(reason: "the Hub was unreachable"),
            model: "openai/whisper-tiny", operation: "Transcription")

        #expect(failure.status == .internalServerError)
        #expect(failure.code == "load_failed")
        #expect(failure.message.contains("openai/whisper-tiny"))
        #expect(failure.message.contains("the Hub was unreachable"))
    }

    /// The backend loads and runs in one call, so the mapping tells the phases
    /// apart by case: a forward pass that throws is `audio_failed`, named after
    /// the operation, never `load_failed`.
    @Test
    func anInferenceFailureIs500AudioFailed() {
        let failure = HummingbirdServer.audioFailure(
            EngineError.audioProcessingFailed(reason: "forward pass threw"),
            model: "mlx-community/Kokoro-82M-4bit", operation: "Speech synthesis")

        #expect(failure.status == .internalServerError)
        #expect(failure.code == "audio_failed")
        #expect(failure.message.hasPrefix("Speech synthesis failed: "))
        #expect(failure.message.contains("forward pass threw"))
    }

    @Test
    func anErrorFromOutsideEngineErrorStays500() {
        let failure = HummingbirdServer.audioFailure(
            UnrelatedFailure(), model: "openai/whisper-tiny", operation: "Transcription")

        #expect(failure.status == .internalServerError)
        #expect(failure.code == "audio_failed")
        #expect(failure.message.hasPrefix("Transcription failed: "))
    }

    @Test
    func everyOtherEngineErrorStays500() {
        // Only the LOCAL rejection is a client error; nothing else in the enum
        // gets downgraded to 400 by accident, and only a load failure is
        // reported as one.
        let others: [EngineError] = [
            .modelNotLoaded,
            .modelNotFound("openai/whisper-tiny"),
            .engineNotReady,
            .generationInProgress,
            .audioProcessingFailed(reason: "forward pass threw"),
            .unsupportedOperation("transcribe"),
        ]
        for error in others {
            let failure = HummingbirdServer.audioFailure(
                error, model: "openai/whisper-tiny", operation: "Transcription")
            #expect(failure.status == .internalServerError, "\(error) should stay 500")
            #expect(failure.code == "audio_failed", "\(error) is not a load failure")
        }
    }

    /// The engine and the mapping have to agree end to end: what `loadSTT` /
    /// `loadTTS` actually throw for a malformed id — and what the validators
    /// the server runs before the lock throw — must be what the mapping
    /// classifies as a 400. Asserting the halves separately would let them
    /// drift apart silently.
    @Test
    func whatTheEngineThrowsForAMalformedIDIsWhatTheMappingCalls400() async {
        let engine = AudioEngine()
        for badID in ["not-a-repo-id", "../etc", "owner/..", "a/b/c", "owner/na me"] {
            let rejections: [(String, () async throws -> Void)] = [
                ("validateSTTModelID", { try AudioEngine.validateSTTModelID(badID) }),
                ("loadSTT", { try await engine.loadSTT(badID) }),
                ("validateTTSModelID", { try AudioEngine.validateTTSModelID(badID) }),
                ("loadTTS", { try await engine.loadTTS(badID) }),
            ]
            for (name, attempt) in rejections {
                do {
                    try await attempt()
                    Issue.record("\(name)(\(badID)) should have been rejected locally")
                } catch {
                    let failure = HummingbirdServer.audioFailure(
                        error, model: badID, operation: "Transcription")
                    #expect(failure.status == .badRequest, "\(name)(\(badID)) should be a 400")
                    #expect(failure.code == "invalid_request_error")
                }
            }
        }
    }

    // MARK: Upload-collection failures → 413 vs everything else

    @Test
    func onlyTheSizeCeilingCountsAsTooLarge() {
        // The one error SwiftNIO's `collect(upTo:)` throws when the body runs
        // past the bound — see `HummingbirdServer.isUploadTooLarge`.
        #expect(HummingbirdServer.isUploadTooLarge(NIOTooManyBytesError(maxBytes: 25 * 1024 * 1024)))
    }

    @Test
    func cancellationAndStreamFailuresAreNotReportedAsTooLarge() {
        // Answering "your file is too large" to a cancelled request or a
        // client that hung up mid-upload sends the reader looking for a size
        // problem that does not exist.
        #expect(HummingbirdServer.isUploadTooLarge(CancellationError()) == false)
        #expect(HummingbirdServer.isUploadTooLarge(UnrelatedFailure()) == false)
        #expect(HummingbirdServer.isUploadTooLarge(
            EngineError.audioProcessingFailed(reason: "decode failed")) == false)
        #expect(HummingbirdServer.isUploadTooLarge(
            ChannelError.ioOnClosedChannel) == false)
    }
}
