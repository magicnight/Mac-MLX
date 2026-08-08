import Testing
import Foundation
@testable import MacMLXCore

/// Pure tests for the audio-facing helpers on ``LocalModel``.
@Suite("LocalModel audio helpers")
struct LocalModelAudioTests {

    // MARK: - isAudio

    @Test
    func audioFormatsReportIsAudio() {
        #expect(model(id: "a/b", format: .audioSTT).isAudio)
        #expect(model(id: "a/b", format: .audioTTS).isAudio)
    }

    @Test
    func nonAudioFormatsDoNotReportIsAudio() {
        for format in [ModelFormat.mlx, .mlxVLM, .embedder, .reranker, .gguf, .unknown] {
            #expect(model(id: "a/b", format: format).isAudio == false)
        }
    }

    // MARK: - Draft-candidate exclusion

    /// Speculative decoding runs a draft model through the generation engine,
    /// which cannot load a speech checkpoint. The `.mlx`-only filter already
    /// excludes them; this pins that so a future relaxation of the filter
    /// cannot let one through.
    @Test
    func audioModelsAreNeverDraftCandidates() {
        let candidates = LocalModel.draftCandidates(
            from: [
                model(id: "openai/whisper-tiny", format: .audioSTT),
                model(id: "mlx-community/Kokoro-82M", format: .audioTTS),
                model(id: "Qwen3-8B-4bit", format: .mlx),
            ],
            excluding: nil)

        #expect(candidates.map(\.id) == ["Qwen3-8B-4bit"])
    }

    // MARK: - resolveAudioModelID

    @Test
    func preferredModelWinsWhenInstalled() {
        let installed = [
            model(id: "openai/whisper-tiny", format: .audioSTT),
            model(id: "openai/whisper-large-v3", format: .audioSTT),
        ]

        let resolved = LocalModel.resolveAudioModelID(
            preferred: "openai/whisper-large-v3", from: installed, format: .audioSTT)

        #expect(resolved == "openai/whisper-large-v3")
    }

    /// With no preference recorded and exactly one model of that kind
    /// installed, the choice is unambiguous, so the button works out of the box.
    @Test
    func singleInstalledModelResolvesWithoutAPreference() {
        let installed = [model(id: "openai/whisper-tiny", format: .audioSTT)]

        let resolved = LocalModel.resolveAudioModelID(
            preferred: nil, from: installed, format: .audioSTT)

        #expect(resolved == "openai/whisper-tiny")
    }

    /// The strictness that matters: with two installed and no preference,
    /// resolution must FAIL rather than pick one. Picking "the first" would
    /// transcribe with a model the user never chose — and would change which
    /// one as soon as an alphabetically-earlier model was installed.
    @Test
    func ambiguousChoiceResolvesToNil() {
        let installed = [
            model(id: "openai/whisper-tiny", format: .audioSTT),
            model(id: "nvidia/parakeet-tdt", format: .audioSTT),
        ]

        let resolved = LocalModel.resolveAudioModelID(
            preferred: nil, from: installed, format: .audioSTT)

        #expect(resolved == nil)
    }

    @Test
    func noInstalledModelOfThatKindResolvesToNil() {
        let installed = [model(id: "mlx-community/Kokoro-82M", format: .audioTTS)]

        let resolved = LocalModel.resolveAudioModelID(
            preferred: nil, from: installed, format: .audioSTT)

        #expect(resolved == nil)
    }

    /// A preference naming a model of the OTHER kind must not be honoured:
    /// handing a TTS repo id to `loadSTT` fails at load time.
    @Test
    func preferenceOfTheWrongKindIsIgnored() {
        let installed = [
            model(id: "mlx-community/Kokoro-82M", format: .audioTTS),
            model(id: "openai/whisper-tiny", format: .audioSTT),
        ]

        let resolved = LocalModel.resolveAudioModelID(
            preferred: "mlx-community/Kokoro-82M", from: installed, format: .audioSTT)

        #expect(resolved == "openai/whisper-tiny")
    }

    /// Uninstalling the preferred model must not brick the button while an
    /// unambiguous alternative remains.
    @Test
    func stalePreferenceFallsBackToTheSoleInstalledModel() {
        let installed = [model(id: "openai/whisper-tiny", format: .audioSTT)]

        let resolved = LocalModel.resolveAudioModelID(
            preferred: "openai/whisper-large-v3", from: installed, format: .audioSTT)

        #expect(resolved == "openai/whisper-tiny")
    }

    /// …but a stale preference with an ambiguous remainder still refuses to guess.
    @Test
    func stalePreferenceWithAmbiguousRemainderResolvesToNil() {
        let installed = [
            model(id: "openai/whisper-tiny", format: .audioSTT),
            model(id: "nvidia/parakeet-tdt", format: .audioSTT),
        ]

        let resolved = LocalModel.resolveAudioModelID(
            preferred: "openai/whisper-large-v3", from: installed, format: .audioSTT)

        #expect(resolved == nil)
    }

    // MARK: - Helper

    private func model(id: String, format: ModelFormat) -> LocalModel {
        LocalModel(
            id: id,
            displayName: id,
            directory: URL(filePath: "/tmp/\(id)"),
            sizeBytes: 1_000,
            format: format,
            quantization: nil,
            parameterCount: nil,
            architecture: nil
        )
    }
}
