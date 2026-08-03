import Foundation

/// The `response_format` values `POST /v1/audio/speech` actually produces.
///
/// **Format honesty is the point of this type.** macMLX ships no audio
/// encoder beyond ``WAVEncoder``, so the only bytes it can hand back are
/// 16-bit PCM — either wrapped in a RIFF/WAVE container (`wav`) or raw
/// (`pcm`). Every other name in OpenAI's spec is rejected with a 400 rather
/// than served as relabelled WAV: returning WAV bytes under
/// `Content-Type: audio/mpeg` would break every client that trusts the
/// header, which is worse than an explicit error.
///
/// - Note: ``default`` DIVERGES from OpenAI, whose default is `mp3`. Apple's
///   shipping audio frameworks decode mp3 but provide no mp3 *encoder*, so
///   there is no honest default-compatible path; `wav` is the closest lossless
///   substitute and, unlike `pcm`, self-describes its sample rate in the
///   header. The divergence is surfaced — never silent — because a request
///   that explicitly asks for `mp3` still gets a 400 explaining it.
public enum SpeechAudioFormat: String, Sendable, CaseIterable {
    /// RIFF/WAVE container, 16-bit signed little-endian PCM, mono, at the
    /// synthesis model's native sample rate (carried in the header).
    case wav
    /// Headerless 16-bit signed little-endian PCM, mono, 24 kHz — OpenAI's
    /// definition. Only served when the model's native rate is already 24 kHz;
    /// see ``pcmSampleRate``.
    case pcm

    /// Applied when a request omits `response_format`. See the type note for
    /// why this is `wav` and not OpenAI's `mp3`.
    public static let `default` = SpeechAudioFormat.wav

    /// The sample rate OpenAI's headerless `pcm` format is defined at. A
    /// headerless stream carries no rate of its own, so a model whose native
    /// rate differs cannot be served as `pcm` without misrepresenting it.
    public static let pcmSampleRate = 24_000

    /// OpenAI-spec speech formats macMLX recognizes but does not encode.
    /// Recognized separately from unknown values so the 400 can be specific.
    public static let recognizedButUnsupported: [String] = ["mp3", "opus", "aac", "flac"]

    /// The `Content-Type` for the bytes this format actually carries.
    public var contentType: String {
        switch self {
        case .wav: return "audio/wav"
        case .pcm: return "audio/pcm"
        }
    }

    /// Parse a request's `response_format`.
    ///
    /// - Returns: The format, or `nil` when the value is unsupported or
    ///   unrecognized. A `nil`/empty input yields ``default``. Matching is
    ///   case-insensitive and tolerates surrounding whitespace.
    public static func parse(_ raw: String?) -> SpeechAudioFormat? {
        guard let raw else { return `default` }
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if normalized.isEmpty { return `default` }
        return SpeechAudioFormat(rawValue: normalized)
    }

    /// Whether `raw` is an OpenAI format macMLX knowingly does not encode (as
    /// opposed to a value that isn't a speech format at all).
    public static func isRecognizedButUnsupported(_ raw: String?) -> Bool {
        guard let raw else { return false }
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return recognizedButUnsupported.contains(normalized)
    }

    /// OpenAI's accepted `speed` range. Values outside it are rejected.
    public static let speedRange: ClosedRange<Double> = 0.25...4.0

    /// The only `speed` macMLX can serve truthfully today: playback-rate
    /// control is not wired into any synthesis path yet, so anything other
    /// than 1.0 would be accepted and then ignored. Rejected instead.
    public static let supportedSpeed: Double = 1.0
}
