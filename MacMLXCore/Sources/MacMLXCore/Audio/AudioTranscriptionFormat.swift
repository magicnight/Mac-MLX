import Foundation

/// The `response_format` values `POST /v1/audio/transcriptions` actually
/// produces.
///
/// Deliberately does NOT include OpenAI's `srt` / `vtt`: macMLX does not emit
/// subtitle containers, and returning a JSON body under a subtitle format name
/// would be a lie. Those two are listed in ``recognizedButUnsupported`` purely
/// so the endpoint can 400 with an accurate reason instead of silently
/// substituting a different format.
public enum AudioTranscriptionFormat: String, Sendable, CaseIterable {
    /// `{"text": "…"}` — the OpenAI default, and ours.
    case json
    /// The transcript as a bare `text/plain` body.
    case text
    /// `{"task", "language", "duration", "text", "segments"}`.
    case verboseJSON = "verbose_json"

    /// Applied when a request omits `response_format`, matching OpenAI.
    public static let `default` = AudioTranscriptionFormat.json

    /// OpenAI-spec formats macMLX recognizes but cannot produce. Kept separate
    /// from "unknown garbage" so the 400 can say *why* rather than just
    /// "invalid".
    public static let recognizedButUnsupported: [String] = ["srt", "vtt"]

    /// Parse a request's `response_format`.
    ///
    /// - Returns: The format, or `nil` when the value is unsupported or
    ///   unrecognized. A `nil`/empty input is the documented default.
    ///   Matching is case-insensitive and tolerates surrounding whitespace.
    public static func parse(_ raw: String?) -> AudioTranscriptionFormat? {
        guard let raw else { return `default` }
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if normalized.isEmpty { return `default` }
        return AudioTranscriptionFormat(rawValue: normalized)
    }

    /// Whether `raw` is an OpenAI format macMLX knowingly does not implement
    /// (as opposed to a value that isn't a transcription format at all).
    public static func isRecognizedButUnsupported(_ raw: String?) -> Bool {
        guard let raw else { return false }
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return recognizedButUnsupported.contains(normalized)
    }
}
