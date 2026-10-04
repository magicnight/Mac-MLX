import Foundation

/// The audio work `HummingbirdServer` hands off from `/v1/audio/transcriptions`
/// and `/v1/audio/speech`: load the named model if it is not resident, run the
/// request on the model that load produced, and hand back a `Sendable` result.
///
/// `AudioEngine` is the only production conformer. The protocol exists so the
/// server's lock contract and wire shapes can be tested against a scripted
/// backend — the real engine needs weights, the Hub, and Metal, none of which
/// a unit test has.
///
/// A conformer throws `EngineError.invalidAudioModelID` for an id it will not
/// load, `EngineError.modelLoadFailed` when the load fails, and
/// `EngineError.audioProcessingFailed` when the inference does; the server
/// maps the three onto 400, 500 `load_failed`, and 500 `audio_failed`.
protocol AudioBackend: Sendable {
    func transcribe(
        model: String, audioURL: URL, language: String?, temperature: Float?
    ) async throws -> AudioEngine.Transcription

    func synthesize(
        model: String, text: String, voice: String?, language: String?
    ) async throws -> AudioEngine.Speech
}

extension AudioEngine: AudioBackend {}
