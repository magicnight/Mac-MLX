import Foundation
import Testing
@testable import MacMLXCore

/// Pure unit tests for ``AudioEngine``'s helper surface — the parts that can
/// be exercised without downloading a checkpoint or touching Metal.
///
/// What is deliberately NOT here: anything that loads weights or runs a
/// forward pass. Segment normalization and repo-id validation are the two
/// pieces of real logic that sit between untrusted/undertyped input and the
/// wire, so those are what get tested.
@Suite("AudioEngine helpers")
struct AudioEngineHelpersTests {

    // MARK: Segment normalization

    @Test
    func normalizesUpstreamSegmentDictionaries() {
        let raw: [[String: Any]] = [
            ["text": "hello", "start": 0.0, "end": 1.5],
            ["text": " world", "start": 1.5, "end": 3.0],
        ]
        let segments = AudioEngine.segments(from: raw)
        #expect(segments.count == 2)
        #expect(segments[0] == AudioEngine.TranscriptionSegment(
            id: 0, start: 0.0, end: 1.5, text: "hello"))
        #expect(segments[1] == AudioEngine.TranscriptionSegment(
            id: 1, start: 1.5, end: 3.0, text: " world"))
    }

    @Test
    func assignsSequentialIdsEvenWhenEntriesAreDropped() {
        let raw: [[String: Any]] = [
            ["text": "a", "start": 0, "end": 1],
            ["start": 1, "end": 2],            // no text — dropped
            ["text": "c", "start": 2, "end": 3],
        ]
        let segments = AudioEngine.segments(from: raw)
        #expect(segments.map(\.id) == [0, 1])
        #expect(segments.map(\.text) == ["a", "c"])
    }

    @Test
    func acceptsTimestampsAsDoubleIntFloatOrNumericString() {
        let raw: [[String: Any]] = [
            ["text": "double", "start": 1.5 as Double, "end": 2.5 as Double],
            ["text": "int", "start": 3 as Int, "end": 4 as Int],
            ["text": "float", "start": 5.5 as Float, "end": 6.5 as Float],
            ["text": "string", "start": "7.5", "end": "8.5"],
        ]
        let segments = AudioEngine.segments(from: raw)
        #expect(segments.map(\.start) == [1.5, 3.0, 5.5, 7.5])
        #expect(segments.map(\.end) == [2.5, 4.0, 6.5, 8.5])
    }

    @Test
    func missingOrUnparseableTimestampsBecomeZero() {
        let raw: [[String: Any]] = [
            ["text": "no timestamps"],
            ["text": "junk", "start": "not a number", "end": ["nested"]],
        ]
        let segments = AudioEngine.segments(from: raw)
        #expect(segments.count == 2)
        #expect(segments.allSatisfy { $0.start == 0 && $0.end == 0 })
    }

    @Test
    func nilOrEmptySegmentsProduceAnEmptyArray() {
        #expect(AudioEngine.segments(from: nil).isEmpty)
        #expect(AudioEngine.segments(from: []).isEmpty)
    }

    @Test
    func entriesWithNonStringTextAreDropped() {
        let raw: [[String: Any]] = [["text": 42, "start": 0, "end": 1]]
        #expect(AudioEngine.segments(from: raw).isEmpty)
    }

    // MARK: Repo-id validation

    @Test
    func acceptsWellFormedHubRepoIDs() {
        #expect(AudioEngine.isHubRepoID("openai/whisper-tiny"))
        #expect(AudioEngine.isHubRepoID("mlx-community/Kokoro-82M-4bit"))
        #expect(AudioEngine.isHubRepoID("a/b"))
    }

    @Test
    func rejectsIDsThatAreNotOwnerSlashName() {
        #expect(AudioEngine.isHubRepoID("whisper-tiny") == false)      // no owner
        #expect(AudioEngine.isHubRepoID("a/b/c") == false)             // too many segments
        #expect(AudioEngine.isHubRepoID("/name") == false)             // empty owner
        #expect(AudioEngine.isHubRepoID("owner/") == false)            // empty name
        #expect(AudioEngine.isHubRepoID("") == false)
        #expect(AudioEngine.isHubRepoID("/") == false)
    }

    @Test
    func rejectsPathTraversalAndWhitespaceSoTheCacheCannotBeEscaped() {
        // A model id is client-controlled and ends up as a cache path
        // component, so `..` must never survive validation.
        #expect(AudioEngine.isHubRepoID("../etc") == false)
        #expect(AudioEngine.isHubRepoID("owner/..") == false)
        #expect(AudioEngine.isHubRepoID("./x") == false)
        #expect(AudioEngine.isHubRepoID("owner /name") == false)
        #expect(AudioEngine.isHubRepoID("owner/na me") == false)
        #expect(AudioEngine.isHubRepoID("owner/na\tme") == false)
        #expect(AudioEngine.isHubRepoID("owner/na\nme") == false)
    }

    @Test
    func localDirectoryDetectionRequiresAnExistingDirectoryWithAConfig() throws {
        #expect(AudioEngine.looksLikeLocalDirectory("openai/whisper-tiny") == false)
        #expect(AudioEngine.looksLikeLocalDirectory("relative/path") == false)
        #expect(AudioEngine.looksLikeLocalDirectory("/definitely/not/here-\(UUID())") == false)

        // A real directory WITHOUT config.json still does not qualify.
        let directory = FileManager.default.temporaryDirectory
            .appending(path: "macmlx-audio-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(AudioEngine.looksLikeLocalDirectory(directory.path) == false)

        try Data("{}".utf8).write(to: directory.appending(path: "config.json"))
        #expect(AudioEngine.looksLikeLocalDirectory(directory.path))
    }

    @Test
    func repoIDHintNamesTheExpectedShapeAndTheAudioCache() {
        let hint = AudioEngine.repoIDHint("whisper-tiny", kind: "STT")
        #expect(hint.contains("whisper-tiny"))
        #expect(hint.contains("owner/name"))
        #expect(hint.contains("STT"))
        #expect(hint.contains("audio-models"))
    }

    // MARK: Cache location

    @Test
    func audioModelsAreCachedUnderTheMacMLXDataRootNotTheSharedHFCache() {
        // Keeping downloads inside ~/.mac-mlx is what makes the GUI, the CLI,
        // and uninstall agree on one directory.
        let cache = AudioEngine.modelCacheDirectory
        #expect(cache.path.hasSuffix("/.mac-mlx/audio-models"))
        #expect(cache.path.hasPrefix(DataRoot.macMLX.path))
        #expect(AudioEngine.sttSampleRate == 16_000)
    }

    // MARK: Lifecycle before load

    @Test
    func aFreshEngineReportsNoResidentModels() async {
        let engine = AudioEngine()
        #expect(await engine.loadedSTTModelID == nil)
        #expect(await engine.loadedTTSModelID == nil)
    }

    /// The model-bound entry points — the ones the server and the app call —
    /// reject a malformed id exactly as the loaders do, before any network
    /// call, and leave nothing resident.
    @Test
    func theModelBoundEntryPointsRejectAMalformedIDLocally() async throws {
        let engine = AudioEngine()
        await #expect(throws: EngineError.invalidAudioModelID(
            reason: AudioEngine.repoIDHint("not-a-repo-id", kind: "STT"))) {
            _ = try await engine.transcribe(
                model: "not-a-repo-id", audioURL: URL(filePath: "/dev/null"))
        }
        await #expect(throws: EngineError.invalidAudioModelID(
            reason: AudioEngine.repoIDHint("not-a-repo-id", kind: "STT"))) {
            try await engine.prepareSTT(model: "not-a-repo-id")
        }
        await #expect(throws: EngineError.invalidAudioModelID(
            reason: AudioEngine.repoIDHint("../escape", kind: "TTS"))) {
            _ = try await engine.synthesize(model: "../escape", text: "hello")
        }
        await #expect(throws: EngineError.invalidAudioModelID(
            reason: AudioEngine.repoIDHint("../escape", kind: "TTS"))) {
            try await engine.prepareTTS(model: "../escape")
        }
        #expect(await engine.loadedSTTModelID == nil)
        #expect(await engine.loadedTTSModelID == nil)
    }

    // MARK: Fetching before the lock

    /// Two requests for one model that is not on disk share a single fetch:
    /// upstream clears a directory it finds incomplete, so two downloads of
    /// one repo would delete each other's files. A different repo fetches on
    /// its own, and once a fetch has finished the next prepare fetches again
    /// (nothing remembers "on disk"; upstream's own check is cheap). No timing
    /// is involved: the fake fetch is held at a gate until the test has seen
    /// the second caller join it.
    @Test
    func concurrentPreparesForOneRepoShareASingleFetch() async throws {
        let log = FetchLog()
        let gate = Gate()
        let fetcher = AudioSnapshotFetcher()
        let engine = AudioEngine(
            fetch: { modelID in
                await log.record(modelID)
                await gate.wait()
            },
            fetcher: fetcher)

        let first = Task { try await engine.prepareSTT(model: "openai/whisper-tiny") }
        try await poll { await log.entries == ["openai/whisper-tiny"] }
        let second = Task { try await engine.prepareSTT(model: "openai/whisper-tiny") }
        let other = Task { try await engine.prepareTTS(model: "mlx-community/Kokoro-82M-4bit") }
        try await poll { await fetcher.joins == 1 }
        try await poll { await log.entries.count == 2 }
        await gate.open()
        try await first.value
        try await second.value
        try await other.value
        let entries = await log.entries
        #expect(entries.sorted() == ["mlx-community/Kokoro-82M-4bit", "openai/whisper-tiny"])

        try await engine.prepareSTT(model: "openai/whisper-tiny")
        let after = await log.entries
        #expect(after.count == 3)
    }

    /// Poll `condition` until it holds, up to 60 s — generous for a starved CI
    /// runner; a correct engine satisfies every condition in milliseconds.
    private func poll(_ condition: @Sendable () async -> Bool) async throws {
        let deadline = ContinuousClock.now + .seconds(60)
        while ContinuousClock.now < deadline {
            if await condition() { return }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        throw PollTimeout()
    }

    private struct PollTimeout: Error {}

    /// Parks callers until opened. Built on continuations rather than a sleep
    /// loop so a cancelled task waits like any other instead of spinning.
    private actor Gate {
        private var isOpen = false
        private var waiters: [CheckedContinuation<Void, Never>] = []
        func open() {
            isOpen = true
            for waiter in waiters { waiter.resume() }
            waiters.removeAll()
        }
        func wait() async {
            if isOpen { return }
            await withCheckedContinuation { waiters.append($0) }
        }
    }

    // MARK: Cancellation

    /// A caller whose task was cancelled — the app's superseded request —
    /// stops before the load, so it neither downloads nor holds a second copy
    /// of a model. A local TTS directory stands in for a real model: the id
    /// passes validation, and had the load run it would have failed on the
    /// bogus `config.json` without touching the network.
    @Test
    func aCancelledCallerStopsBeforeTheLoad() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(path: "macmlx-cancel-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data("{}".utf8).write(to: directory.appending(path: "config.json"))

        let engine = AudioEngine()
        let gate = Gate()
        let call = Task {
            await gate.wait()
            return try await engine.synthesize(model: directory.path, text: "hello")
        }
        call.cancel()
        await gate.open()
        await #expect(throws: CancellationError.self) { _ = try await call.value }
        #expect(await engine.loadedTTSModelID == nil)
    }

    /// What the Hub throws becomes `modelLoadFailed` — the case the server
    /// reports as 500 `load_failed` — and nothing becomes resident.
    @Test
    func aFailedFetchIsALoadFailure() async throws {
        struct HubDown: Error {}
        let engine = AudioEngine(fetch: { _ in throw HubDown() })
        do {
            try await engine.prepareSTT(model: "openai/whisper-tiny")
            Issue.record("the fetch failure should have propagated")
        } catch let error as EngineError {
            guard case .modelLoadFailed = error else {
                Issue.record("expected modelLoadFailed, got \(error)")
                return
            }
        }
        #expect(await engine.loadedSTTModelID == nil)
    }

    private actor FetchLog {
        private(set) var entries: [String] = []
        func record(_ modelID: String) { entries.append(modelID) }
    }

    @Test
    func aMalformedModelIDIsRejectedLocallyWithoutAnyNetworkCall() async throws {
        // The guard runs before `STT.loadModel`, so this test never reaches
        // the Hub — it is safe to run offline and in CI.
        //
        // The EXACT case matters, not just "some EngineError": the server maps
        // `invalidAudioModelID` to 400 and everything else to 500, so throwing
        // `modelLoadFailed` here would silently turn a client typo back into a
        // retryable "internal error". See
        // `HummingbirdServer.audioFailure`.
        let engine = AudioEngine()
        await #expect(throws: EngineError.invalidAudioModelID(
            reason: AudioEngine.repoIDHint("not-a-repo-id", kind: "STT"))) {
            try await engine.loadSTT("not-a-repo-id")
        }
        #expect(await engine.loadedSTTModelID == nil)

        await #expect(throws: EngineError.invalidAudioModelID(
            reason: AudioEngine.repoIDHint("../escape", kind: "TTS"))) {
            try await engine.loadTTS("../escape")
        }
        #expect(await engine.loadedTTSModelID == nil)
    }

    @Test
    func aPathTraversalIDIsRejectedAsAClientErrorOnBothLoaders() async throws {
        // `..` in a client-controlled id would otherwise walk out of the audio
        // cache. It is refused locally, and as the caller's mistake — never as
        // a server fault, which a client SDK would retry.
        let engine = AudioEngine()
        for badID in ["../etc", "owner/..", "./x", "a/b/c"] {
            await #expect(throws: EngineError.invalidAudioModelID(
                reason: AudioEngine.repoIDHint(badID, kind: "STT"))) {
                try await engine.loadSTT(badID)
            }
            await #expect(throws: EngineError.invalidAudioModelID(
                reason: AudioEngine.repoIDHint(badID, kind: "TTS"))) {
                try await engine.loadTTS(badID)
            }
        }
        #expect(await engine.loadedSTTModelID == nil)
        #expect(await engine.loadedTTSModelID == nil)
    }
}
