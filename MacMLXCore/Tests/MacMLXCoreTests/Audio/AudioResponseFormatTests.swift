import Foundation
import Testing
@testable import MacMLXCore

/// Pure unit tests for the two `response_format` gatekeepers and the
/// transcription JSON shape.
///
/// The load-bearing property under test is FORMAT HONESTY: macMLX must never
/// return bytes under a format name it did not produce. Every "unsupported"
/// case here proves the request is refused rather than quietly served as
/// something else.
@Suite("Audio response formats")
struct AudioResponseFormatTests {

    // MARK: Transcription formats

    @Test
    func transcriptionDefaultsToJSONWhenUnspecified() {
        #expect(AudioTranscriptionFormat.parse(nil) == .json)
        #expect(AudioTranscriptionFormat.parse("") == .json)
        #expect(AudioTranscriptionFormat.parse("   ") == .json)
        #expect(AudioTranscriptionFormat.default == .json)
    }

    @Test
    func transcriptionParsesEverySupportedFormatCaseInsensitively() {
        #expect(AudioTranscriptionFormat.parse("json") == .json)
        #expect(AudioTranscriptionFormat.parse("text") == .text)
        #expect(AudioTranscriptionFormat.parse("verbose_json") == .verboseJSON)
        #expect(AudioTranscriptionFormat.parse("VERBOSE_JSON") == .verboseJSON)
        #expect(AudioTranscriptionFormat.parse("  Text  ") == .text)
    }

    @Test
    func transcriptionRejectsSubtitleFormatsRatherThanSubstitutingJSON() {
        #expect(AudioTranscriptionFormat.parse("srt") == nil)
        #expect(AudioTranscriptionFormat.parse("vtt") == nil)
        #expect(AudioTranscriptionFormat.isRecognizedButUnsupported("srt"))
        #expect(AudioTranscriptionFormat.isRecognizedButUnsupported("VTT"))
    }

    @Test
    func transcriptionRejectsUnknownFormats() {
        #expect(AudioTranscriptionFormat.parse("yaml") == nil)
        #expect(AudioTranscriptionFormat.isRecognizedButUnsupported("yaml") == false)
    }

    @Test
    func transcriptionErrorMessageDistinguishesUnimplementedFromUnknown() {
        let subtitle = HummingbirdServer.unsupportedTranscriptionFormatMessage("srt")
        #expect(subtitle.contains("srt"))
        #expect(subtitle.contains("subtitle"))
        #expect(subtitle.contains("verbose_json"))

        let unknown = HummingbirdServer.unsupportedTranscriptionFormatMessage("yaml")
        #expect(unknown.contains("not a known transcription format"))
    }

    // MARK: Speech formats

    @Test
    func speechDefaultsToWAVBecauseNoMP3EncoderExists() {
        // The divergence from OpenAI's mp3 default is intentional and must not
        // silently regress into "pretend mp3".
        #expect(SpeechAudioFormat.parse(nil) == .wav)
        #expect(SpeechAudioFormat.parse("") == .wav)
        #expect(SpeechAudioFormat.default == .wav)
        #expect(SpeechAudioFormat.allCases.contains(.wav))
        #expect(SpeechAudioFormat.allCases.contains(.pcm))
        #expect(SpeechAudioFormat.allCases.count == 2)
    }

    @Test
    func speechParsesSupportedFormatsCaseInsensitively() {
        #expect(SpeechAudioFormat.parse("wav") == .wav)
        #expect(SpeechAudioFormat.parse("PCM") == .pcm)
        #expect(SpeechAudioFormat.parse(" wav ") == .wav)
    }

    @Test
    func speechRejectsEveryCompressedFormat() {
        for name in ["mp3", "opus", "aac", "flac"] {
            #expect(SpeechAudioFormat.parse(name) == nil, "\(name) must not parse")
            #expect(
                SpeechAudioFormat.isRecognizedButUnsupported(name),
                "\(name) must be recognized as a known-but-unsupported format")
        }
    }

    @Test
    func speechRejectsUnknownFormats() {
        #expect(SpeechAudioFormat.parse("ogg") == nil)
        #expect(SpeechAudioFormat.isRecognizedButUnsupported("ogg") == false)
    }

    @Test
    func speechErrorMessageNamesTheMissingEncodersAndTheRealDefault() {
        let mp3 = HummingbirdServer.unsupportedSpeechFormatMessage("mp3")
        #expect(mp3.contains("mp3"))
        #expect(mp3.contains("encoder"))
        #expect(mp3.contains("wav"))
        #expect(mp3.contains("default: wav"))

        let unknown = HummingbirdServer.unsupportedSpeechFormatMessage("ogg")
        #expect(unknown.contains("not a known speech format"))
    }

    @Test
    func speechContentTypesMatchTheBytesActuallyReturned() {
        #expect(SpeechAudioFormat.wav.contentType == "audio/wav")
        #expect(SpeechAudioFormat.pcm.contentType == "audio/pcm")
        // Never audio/mpeg — that would be the lie this whole type prevents.
        #expect(SpeechAudioFormat.allCases.allSatisfy { $0.contentType != "audio/mpeg" })
    }

    @Test
    func speechSpeedBoundsMatchOpenAIAndOnlyUnityIsServable() {
        #expect(SpeechAudioFormat.speedRange.contains(0.25))
        #expect(SpeechAudioFormat.speedRange.contains(4.0))
        #expect(SpeechAudioFormat.speedRange.contains(0.24) == false)
        #expect(SpeechAudioFormat.speedRange.contains(4.01) == false)
        #expect(SpeechAudioFormat.supportedSpeed == 1.0)
    }

    // MARK: Transcription JSON shape

    private var sample: AudioEngine.Transcription {
        AudioEngine.Transcription(
            text: "hello world",
            language: "en",
            duration: 2.5,
            segments: [
                AudioEngine.TranscriptionSegment(id: 0, start: 0, end: 1.25, text: "hello"),
                AudioEngine.TranscriptionSegment(id: 1, start: 1.25, end: 2.5, text: " world"),
            ]
        )
    }

    @Test
    func jsonFormatCarriesOnlyTheText() throws {
        let body = HummingbirdServer.transcriptionJSON(sample, format: .json)
        #expect(body.count == 1)
        #expect(body["text"] as? String == "hello world")
    }

    @Test
    func verboseJSONCarriesTaskLanguageDurationAndSegments() throws {
        let body = HummingbirdServer.transcriptionJSON(sample, format: .verboseJSON)
        #expect(body["task"] as? String == "transcribe")
        #expect(body["language"] as? String == "en")
        #expect(body["duration"] as? Double == 2.5)
        #expect(body["text"] as? String == "hello world")

        let segments = try #require(body["segments"] as? [[String: Any]])
        #expect(segments.count == 2)
        #expect(segments[0]["id"] as? Int == 0)
        #expect(segments[0]["start"] as? Double == 0)
        #expect(segments[0]["end"] as? Double == 1.25)
        #expect(segments[0]["text"] as? String == "hello")
        #expect(segments[1]["id"] as? Int == 1)
    }

    @Test
    func verboseJSONOmitsLanguageWhenTheModelReportedNone() {
        let unknown = AudioEngine.Transcription(
            text: "…", language: nil, duration: 1, segments: [])
        let body = HummingbirdServer.transcriptionJSON(unknown, format: .verboseJSON)
        // Omitted, never guessed or stubbed with a placeholder.
        #expect(body["language"] == nil)
        #expect(body["task"] as? String == "transcribe")
        #expect((body["segments"] as? [[String: Any]])?.isEmpty == true)
    }

    @Test
    func everyTranscriptionBodyIsActuallySerializableToJSON() throws {
        for format in [AudioTranscriptionFormat.json, .verboseJSON] {
            let body = HummingbirdServer.transcriptionJSON(sample, format: format)
            #expect(JSONSerialization.isValidJSONObject(body), "\(format) body must be valid JSON")
            let data = try JSONSerialization.data(withJSONObject: body)
            #expect(data.isEmpty == false)
        }
    }

    // MARK: Upload filename sanitizing

    @Test
    func keepsAPlainAudioExtensionAsADecodeHint() {
        #expect(HummingbirdServer.sanitizedExtension(of: "speech.wav") == ".wav")
        #expect(HummingbirdServer.sanitizedExtension(of: "SPEECH.M4A") == ".m4a")
        #expect(HummingbirdServer.sanitizedExtension(of: "take2.mp3") == ".mp3")
    }

    @Test
    func dropsExtensionsThatCouldEscapeOrConfuseThePath() {
        #expect(HummingbirdServer.sanitizedExtension(of: nil) == "")
        #expect(HummingbirdServer.sanitizedExtension(of: "") == "")
        #expect(HummingbirdServer.sanitizedExtension(of: "noextension") == "")
        #expect(HummingbirdServer.sanitizedExtension(of: "../../etc/passwd") == "")
        #expect(HummingbirdServer.sanitizedExtension(of: "a.wav/../../x") == "")
        #expect(HummingbirdServer.sanitizedExtension(of: "a.w a v") == "")
        #expect(HummingbirdServer.sanitizedExtension(of: "a.verylongextension") == "")
        #expect(HummingbirdServer.sanitizedExtension(of: "a.录音") == "")
    }

    // MARK: Multipart failure classification

    @Test
    func onlyTheSizeCeilingMapsToA413() {
        #expect(HummingbirdServer.isBodyTooLarge(.bodyTooLarge(limit: 10)))
        #expect(HummingbirdServer.isBodyTooLarge(.missingBoundary) == false)
        #expect(HummingbirdServer.isBodyTooLarge(.malformed("truncated")) == false)
        #expect(HummingbirdServer.isBodyTooLarge(.notMultipart(contentType: nil)) == false)
        #expect(HummingbirdServer.isBodyTooLarge(.invalidBoundary("x")) == false)
    }

    @Test
    func parseAudioFormSurfacesFailuresAsResultsRatherThanThrowing() throws {
        let failure = HummingbirdServer.parseAudioForm(
            body: Data("nope".utf8), contentType: "application/json")
        switch failure {
        case .success: Issue.record("a non-multipart body must not parse")
        case .failure(let error): #expect(error == .notMultipart(contentType: "application/json"))
        }

        let boundary = "abc123"
        var body = Data("--\(boundary)\r\n".utf8)
        body.append(Data("Content-Disposition: form-data; name=\"model\"\r\n\r\nm\r\n".utf8))
        body.append(Data("--\(boundary)--\r\n".utf8))
        let success = HummingbirdServer.parseAudioForm(
            body: body, contentType: "multipart/form-data; boundary=\(boundary)")
        switch success {
        case .success(let form): #expect(form.value("model") == "m")
        case .failure(let error): Issue.record("unexpected parse failure: \(error)")
        }
    }
}
