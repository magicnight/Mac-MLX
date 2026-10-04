import Foundation

/// The audio work `HummingbirdServer` hands off from `/v1/audio/transcriptions`
/// and `/v1/audio/speech`, in two steps. `prepareSTT` / `prepareTTS` make the
/// named model's files local — the download, which the server runs BEFORE the
/// generation lock so a first multi-minute fetch never has every other request
/// queued behind it. `transcribe` / `synthesize` then load the model if it is
/// not resident, run the request on the model that load produced, and hand
/// back a `Sendable` result; the server runs those under the lock.
///
/// `AudioEngine` is the only production conformer. The protocol exists so the
/// server's lock contract and wire shapes can be tested against a scripted
/// backend — the real engine needs weights, the Hub, and Metal, none of which
/// a unit test has.
///
/// A conformer throws `EngineError.invalidAudioModelID` for an id it will not
/// load, `EngineError.modelLoadFailed` when the fetch or the load fails, and
/// `EngineError.audioProcessingFailed` when the inference does; the server
/// maps the three onto 400, 500 `load_failed`, and 500 `audio_failed`.
protocol AudioBackend: Sendable {
    func prepareSTT(model: String) async throws
    func prepareTTS(model: String) async throws

    func transcribe(
        model: String, audioURL: URL, language: String?, temperature: Float?
    ) async throws -> AudioEngine.Transcription

    func synthesize(
        model: String, text: String, voice: String?, language: String?
    ) async throws -> AudioEngine.Speech
}

extension AudioEngine: AudioBackend {}
