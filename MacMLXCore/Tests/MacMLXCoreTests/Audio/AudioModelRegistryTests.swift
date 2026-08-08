import Testing
import Foundation
@testable import MacMLXCore

/// Pure classification tests for ``AudioModelRegistry`` — no filesystem, no
/// model, no network. Each one pins a property the audio badge's honesty
/// depends on.
@Suite("AudioModelRegistry")
struct AudioModelRegistryTests {

    // MARK: - The invariant that makes classification exact

    /// The whole design rests on this: because no `model_type` appears in both
    /// upstream dispatch tables, ``AudioModelRegistry/format(forModelType:)``
    /// is a lookup rather than a priority order, and no model can be both an
    /// STT and a TTS candidate. A dependency bump that introduced an overlap
    /// would silently make the STT-first check a coin toss for the overlapping
    /// family; this fails instead.
    @Test
    func sttAndTTSRegistriesAreDisjoint() {
        #expect(AudioModelRegistry.sttAndTTSTypesAreDisjoint)
    }

    // MARK: - format(forModelType:)

    @Test(arguments: ["whisper", "parakeet", "qwen3_asr", "moonshine", "granite_speech"])
    func knownSTTTypesClassifyAsSTT(_ modelType: String) {
        #expect(AudioModelRegistry.format(forModelType: modelType) == .audioSTT)
    }

    @Test(arguments: ["kokoro", "csm", "chatterbox", "kitten_tts", "orpheus"])
    func knownTTSTypesClassifyAsTTS(_ modelType: String) {
        #expect(AudioModelRegistry.format(forModelType: modelType) == .audioTTS)
    }

    /// The honest gate. An audio checkpoint whose family neither loader
    /// dispatches on cannot be loaded, so it must not be classified — the
    /// caller drops it rather than badging a model that would fail on click.
    @Test
    func unknownTypeClassifiesAsNil() {
        #expect(AudioModelRegistry.format(forModelType: "some_future_asr") == nil)
    }

    /// `qwen3_asr` (STT) and `qwen3` (TTS) are distinct keys that differ only
    /// by suffix. Exact-match lookup must not let one collapse into the other.
    @Test
    func qwen3ASRAndQwen3AreNotConfused() {
        #expect(AudioModelRegistry.format(forModelType: "qwen3_asr") == .audioSTT)
        #expect(AudioModelRegistry.format(forModelType: "qwen3") == .audioTTS)
    }

    /// Upstream lowercases before dispatching, so a `config.json` carrying
    /// `"Whisper"` still loads. Classification has to agree or the badge
    /// disappears for a model that works.
    @Test
    func classificationIsCaseAndWhitespaceInsensitive() {
        #expect(AudioModelRegistry.format(forModelType: "WHISPER") == .audioSTT)
        #expect(AudioModelRegistry.format(forModelType: "  Kokoro \n") == .audioTTS)
    }

    @Test
    func nilAndEmptyTypesClassifyAsNil() {
        #expect(AudioModelRegistry.format(forModelType: nil) == nil)
        #expect(AudioModelRegistry.format(forModelType: "") == nil)
        #expect(AudioModelRegistry.format(forModelType: "   ") == nil)
    }

    // MARK: - modelType(fromConfig:)

    /// Key order must match `ModelUtils.resolveModelType` exactly: disagreeing
    /// with the loader about which key wins is the one bug this path exists to
    /// prevent.
    @Test
    func modelTypeKeyWinsOverArchitectureAndVersion() {
        let config: [String: Any] = [
            "model_type": "whisper",
            "architecture": "kokoro",
            "model_version": "csm",
        ]
        #expect(AudioModelRegistry.modelType(fromConfig: config) == "whisper")
    }

    @Test
    func architectureIsUsedWhenModelTypeAbsent() {
        let config: [String: Any] = ["architecture": "kokoro", "model_version": "csm"]
        #expect(AudioModelRegistry.modelType(fromConfig: config) == "kokoro")
    }

    @Test
    func modelVersionIsTheLastResortKey() {
        let config: [String: Any] = ["model_version": "csm"]
        #expect(AudioModelRegistry.modelType(fromConfig: config) == "csm")
    }

    /// Upstream has a FOURTH fallback — guess the type from the repo name —
    /// that is deliberately not reproduced. A config carrying none of the
    /// three keys yields nothing, so the directory goes unbadged rather than
    /// being classified on the strength of its folder name.
    @Test
    func configWithNoRecognisedKeyYieldsNil() {
        #expect(AudioModelRegistry.modelType(fromConfig: ["hidden_size": 384]) == nil)
        #expect(AudioModelRegistry.modelType(fromConfig: [:]) == nil)
    }

    /// A non-string value under a recognised key is not a type. Bridging it to
    /// a description ("42") would invent a model family.
    @Test
    func nonStringModelTypeYieldsNil() {
        #expect(AudioModelRegistry.modelType(fromConfig: ["model_type": 42]) == nil)
    }

    // MARK: - repoID(fromCacheFolderName:)

    @Test
    func folderNameRoundTripsToRepoID() {
        #expect(
            AudioModelRegistry.repoID(fromCacheFolderName: "openai_whisper-tiny")
                == "openai/whisper-tiny")
        #expect(
            AudioModelRegistry.repoID(fromCacheFolderName: "mlx-community_Kokoro-82M-4bit")
                == "mlx-community/Kokoro-82M-4bit")
    }

    /// `resolveOrDownloadModel` replaces only the single `/`, so the FIRST `_`
    /// is the separator and every later one belongs to the repo name. Splitting
    /// on the last would produce an unloadable id.
    @Test
    func onlyTheFirstUnderscoreSeparatesOwnerFromRepo() {
        #expect(
            AudioModelRegistry.repoID(fromCacheFolderName: "openai_whisper_large_v3")
                == "openai/whisper_large_v3")
    }

    @Test
    func malformedFolderNamesYieldNil() {
        // No separator at all — cannot name a repo.
        #expect(AudioModelRegistry.repoID(fromCacheFolderName: "whisper") == nil)
        // Empty owner or empty repo half.
        #expect(AudioModelRegistry.repoID(fromCacheFolderName: "_whisper") == nil)
        #expect(AudioModelRegistry.repoID(fromCacheFolderName: "openai_") == nil)
        #expect(AudioModelRegistry.repoID(fromCacheFolderName: "") == nil)
    }

    /// A recovered repo id has to satisfy ``AudioEngine``'s own validator, or
    /// the GUI would surface a model whose load fails with
    /// `invalidAudioModelID`.
    @Test
    func recoveredRepoIDIsAcceptedByAudioEngine() throws {
        let repoID = try #require(
            AudioModelRegistry.repoID(fromCacheFolderName: "openai_whisper-tiny"))
        #expect(AudioEngine.isHubRepoID(repoID))
    }
}
