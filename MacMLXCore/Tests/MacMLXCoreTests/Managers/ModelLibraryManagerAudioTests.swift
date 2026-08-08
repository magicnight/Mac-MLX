import Testing
import Foundation
@testable import MacMLXCore

/// Discovery tests for `ModelLibraryManager.scanAudioModels(root:)`.
///
/// Pure filesystem work — a temp dir laid out the way
/// `ModelUtils.resolveOrDownloadModel` lays out its cache, with hand-written
/// `config.json` files. No Metal, no model download, no network.
///
/// Serialised for the same reason as the embedder/VLM/reranker suites
/// (parallel tmpdir thrash + actor scans trip a Swift-stdlib flake).
@Suite("ModelLibraryManager audio discovery", .serialized)
struct ModelLibraryManagerAudioTests {

    // MARK: - Happy path

    @Test
    func whisperCheckpointIsDiscoveredAsSTT() async throws {
        let temp = try AudioTempDir()
        try temp.writeAudioModel(folder: "openai_whisper-tiny", modelType: "whisper")

        let models = await ModelLibraryManager().scanAudioModels(root: temp.url)

        #expect(models.count == 1)
        let model = try #require(models.first)
        #expect(model.format == .audioSTT)
    }

    @Test
    func kokoroCheckpointIsDiscoveredAsTTS() async throws {
        let temp = try AudioTempDir()
        try temp.writeAudioModel(folder: "mlx-community_Kokoro-82M-4bit", modelType: "kokoro")

        let models = await ModelLibraryManager().scanAudioModels(root: temp.url)

        #expect(models.count == 1)
        let model = try #require(models.first)
        #expect(model.format == .audioTTS)
    }

    /// The id must be the REPO ID, not the folder name and not a path:
    /// `AudioEngine.loadSTT` accepts only `owner/name`, so any other shape
    /// produces a library entry the GUI cannot load.
    @Test
    func discoveredIDIsTheRepoIDNotTheFolderName() async throws {
        let temp = try AudioTempDir()
        try temp.writeAudioModel(folder: "openai_whisper-tiny", modelType: "whisper")

        let models = await ModelLibraryManager().scanAudioModels(root: temp.url)

        let model = try #require(models.first)
        #expect(model.id == "openai/whisper-tiny")
        #expect(AudioEngine.isHubRepoID(model.id))
    }

    /// These live in macMLX's own data root, so the app may delete them —
    /// unlike an HF-cache entry, which `delete(_:)` refuses.
    @Test
    func discoveredAudioModelsAreNotExternalReferences() async throws {
        let temp = try AudioTempDir()
        try temp.writeAudioModel(folder: "openai_whisper-tiny", modelType: "whisper")

        let models = await ModelLibraryManager().scanAudioModels(root: temp.url)

        #expect(try #require(models.first).isExternalReference == false)
    }

    @Test
    func directoryPointsAtTheCheckpointItself() async throws {
        let temp = try AudioTempDir()
        try temp.writeAudioModel(folder: "openai_whisper-tiny", modelType: "whisper")

        let models = await ModelLibraryManager().scanAudioModels(root: temp.url)

        #expect(try #require(models.first).directory.lastPathComponent == "openai_whisper-tiny")
    }

    @Test
    func resultsAreSortedByDisplayName() async throws {
        let temp = try AudioTempDir()
        try temp.writeAudioModel(folder: "zz-org_zebra-asr", modelType: "whisper")
        try temp.writeAudioModel(folder: "aa-org_alpha-tts", modelType: "kokoro")

        let models = await ModelLibraryManager().scanAudioModels(root: temp.url)

        #expect(models.map(\.id) == ["aa-org/alpha-tts", "zz-org/zebra-asr"])
    }

    // MARK: - The honest gate

    /// An audio checkpoint whose family neither upstream loader dispatches on
    /// must NOT be listed: clicking it could only fail. Same discipline as
    /// `LocalModel.isOCR`'s `.mlxVLM` gate.
    @Test
    func unrecognisedModelTypeIsNotListed() async throws {
        let temp = try AudioTempDir()
        try temp.writeAudioModel(folder: "someone_future-asr", modelType: "totally_new_asr")

        let models = await ModelLibraryManager().scanAudioModels(root: temp.url)

        #expect(models.isEmpty)
    }

    /// Upstream treats a zero-byte weight file as an interrupted download and
    /// clears the directory on next load. Listing it would offer a model that
    /// silently re-downloads the moment the user touches it.
    @Test
    func zeroByteWeightsAreNotListed() async throws {
        let temp = try AudioTempDir()
        try temp.writeAudioModel(
            folder: "openai_whisper-tiny", modelType: "whisper", weightBytes: 0)

        let models = await ModelLibraryManager().scanAudioModels(root: temp.url)

        #expect(models.isEmpty)
    }

    @Test
    func directoryWithNoWeightsIsNotListed() async throws {
        let temp = try AudioTempDir()
        try temp.writeAudioModel(
            folder: "openai_whisper-tiny", modelType: "whisper", includeWeights: false)

        let models = await ModelLibraryManager().scanAudioModels(root: temp.url)

        #expect(models.isEmpty)
    }

    @Test
    func directoryWithNoConfigIsNotListed() async throws {
        let temp = try AudioTempDir()
        try temp.writeAudioModel(
            folder: "openai_whisper-tiny", modelType: "whisper", includeConfig: false)

        let models = await ModelLibraryManager().scanAudioModels(root: temp.url)

        #expect(models.isEmpty)
    }

    /// Upstream re-downloads when `config.json` won't parse, so a half-written
    /// config is not a usable model.
    @Test
    func malformedConfigIsNotListed() async throws {
        let temp = try AudioTempDir()
        try temp.writeAudioModel(folder: "openai_whisper-tiny", modelType: "whisper")
        try Data("{ not json".utf8).write(
            to: temp.audioRoot
                .appendingPathComponent("openai_whisper-tiny")
                .appendingPathComponent("config.json"))

        let models = await ModelLibraryManager().scanAudioModels(root: temp.url)

        #expect(models.isEmpty)
    }

    /// A folder name with no `_` cannot yield a loadable repo id, so it is
    /// skipped rather than listed under a guessed id.
    @Test
    func folderNameWithoutSeparatorIsNotListed() async throws {
        let temp = try AudioTempDir()
        try temp.writeAudioModel(folder: "whisper-tiny", modelType: "whisper")

        let models = await ModelLibraryManager().scanAudioModels(root: temp.url)

        #expect(models.isEmpty)
    }

    // MARK: - Layout

    /// Downloads live one level deeper than the cache root, under
    /// `mlx-audio/`. Scanning the root directly would both miss real models
    /// and sweep in whatever else `HubCache` keeps beside them.
    @Test
    func checkpointsOutsideTheMLXAudioSubdirectoryAreIgnored() async throws {
        let temp = try AudioTempDir()
        // Correctly-shaped checkpoint, but placed at the cache ROOT rather
        // than inside `mlx-audio/`.
        try temp.writeModelDirectory(
            at: temp.url.appendingPathComponent("openai_whisper-tiny"), modelType: "whisper")

        let models = await ModelLibraryManager().scanAudioModels(root: temp.url)

        #expect(models.isEmpty)
    }

    @Test
    func missingCacheRootYieldsEmptyRatherThanThrowing() async throws {
        let absent = FileManager.default.temporaryDirectory
            .appendingPathComponent("macmlx-audio-absent-\(UUID().uuidString)")

        let models = await ModelLibraryManager().scanAudioModels(root: absent)

        #expect(models.isEmpty)
    }

    /// One unusable entry must not suppress a usable sibling — the same
    /// best-effort tolerance the other two scans show.
    @Test
    func oneBadEntryDoesNotSuppressTheRest() async throws {
        let temp = try AudioTempDir()
        try temp.writeAudioModel(folder: "openai_whisper-tiny", modelType: "whisper")
        try temp.writeAudioModel(folder: "someone_mystery", modelType: "totally_new_asr")
        try temp.writeAudioModel(
            folder: "broken_model", modelType: "whisper", includeConfig: false)

        let models = await ModelLibraryManager().scanAudioModels(root: temp.url)

        #expect(models.map(\.id) == ["openai/whisper-tiny"])
    }

    // MARK: - Config key fallbacks

    /// `ModelUtils.resolveModelType` falls back to `architecture` when
    /// `model_type` is absent, so discovery has to as well or the model would
    /// load fine yet never appear.
    @Test
    func architectureKeyIsHonouredWhenModelTypeIsAbsent() async throws {
        let temp = try AudioTempDir()
        try temp.writeModelDirectory(
            at: temp.audioRoot.appendingPathComponent("mlx-community_Kokoro-82M"),
            config: ["architecture": "kokoro"])

        let models = await ModelLibraryManager().scanAudioModels(root: temp.url)

        #expect(models.count == 1)
        #expect(try #require(models.first).format == .audioTTS)
    }
}

// MARK: - Fixture

/// Auto-created temp cache root shaped like `AudioEngine.modelCacheDirectory`,
/// i.e. with the `mlx-audio/` subdirectory `ModelUtils` writes into.
private struct AudioTempDir {
    let url: URL

    var audioRoot: URL { url.appendingPathComponent(AudioModelRegistry.cacheSubdirectory) }

    init() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("macmlx-audio-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        self.url = base
        try FileManager.default.createDirectory(at: audioRoot, withIntermediateDirectories: true)
    }

    /// Lay down one checkpoint inside `mlx-audio/`, the way a completed
    /// `resolveOrDownloadModel` leaves it: flat, `config.json` plus weights.
    /// Note there is deliberately no `tokenizer.json` — several speech
    /// families ship none, which is why this scan cannot reuse
    /// `ModelFormat.detect(in:)`.
    func writeAudioModel(
        folder: String,
        modelType: String,
        includeConfig: Bool = true,
        includeWeights: Bool = true,
        weightBytes: Int = 8
    ) throws {
        try writeModelDirectory(
            at: audioRoot.appendingPathComponent(folder),
            config: includeConfig ? ["model_type": modelType] : nil,
            includeWeights: includeWeights,
            weightBytes: weightBytes
        )
    }

    func writeModelDirectory(
        at directory: URL,
        modelType: String,
        includeWeights: Bool = true,
        weightBytes: Int = 8
    ) throws {
        try writeModelDirectory(
            at: directory,
            config: ["model_type": modelType],
            includeWeights: includeWeights,
            weightBytes: weightBytes
        )
    }

    func writeModelDirectory(
        at directory: URL,
        config: [String: Any]?,
        includeWeights: Bool = true,
        weightBytes: Int = 8
    ) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        if let config {
            try JSONSerialization.data(withJSONObject: config)
                .write(to: directory.appendingPathComponent("config.json"))
        }
        if includeWeights {
            try Data(repeating: 0, count: weightBytes)
                .write(to: directory.appendingPathComponent("model.safetensors"))
        }
    }
}
