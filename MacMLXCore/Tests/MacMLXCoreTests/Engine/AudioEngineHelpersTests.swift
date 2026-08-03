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

    @Test
    func transcribingBeforeLoadingThrowsModelNotLoaded() async throws {
        let engine = AudioEngine()
        await #expect(throws: EngineError.modelNotLoaded) {
            _ = try await engine.transcribe(audioURL: URL(filePath: "/dev/null"))
        }
    }

    @Test
    func synthesizingBeforeLoadingThrowsModelNotLoaded() async throws {
        let engine = AudioEngine()
        await #expect(throws: EngineError.modelNotLoaded) {
            _ = try await engine.synthesize(text: "hello")
        }
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
        // `HummingbirdServer.audioModelLoadFailure`.
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
