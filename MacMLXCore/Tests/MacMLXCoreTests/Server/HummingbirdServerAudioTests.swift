import Foundation
import Testing
@testable import MacMLXCore

// MARK: - /v1/audio/transcriptions + /v1/audio/speech Tests
//
// These exercise route wiring, request decoding, and every validation gate —
// i.e. all the paths that end in a 4xx BEFORE any model load is attempted.
// Nothing here downloads weights, opens a network connection to the Hub, or
// touches Metal: each case is constructed so the handler returns before
// `AudioEngine.loadSTT` / `loadTTS` is reached. Actually producing audio needs
// a real checkpoint and stays deferred.
//
// Port assignments (20_600 range, spaced by 10):
//   transcriptionsRejectsNonMultipartBody            : 20_600
//   transcriptionsMissingFileReturns400              : 20_610
//   transcriptionsMissingModelReturns400             : 20_620
//   transcriptionsRejectsSubtitleResponseFormat      : 20_630
//   transcriptionsRejectsUnknownResponseFormat       : 20_640
//   transcriptionsRejectsPromptRatherThanIgnoringIt  : 20_650
//   transcriptionsRejectsOutOfRangeTemperature       : 20_660
//   transcriptionsRejectsEmptyBody                   : 20_670
//   transcriptionsRejectsMalformedMultipartBody      : 20_680
//   speechInvalidJSONReturns400                      : 20_690
//   speechMissingModelReturns400                     : 20_700
//   speechMissingInputReturns400                     : 20_710
//   speechMissingVoiceReturns400                     : 20_720
//   speechRejectsMP3AndEveryCompressedFormat         : 20_730
//   speechRejectsUnknownResponseFormat               : 20_740
//   speechRejectsOutOfRangeSpeed                     : 20_750
//   speechRejectsUnimplementedSpeed                  : 20_760
//   speechRejectsOverlongInput                       : 20_770

@Suite("HummingbirdServer audio endpoints")
struct HummingbirdServerAudioTests {

    // MARK: Helpers

    private func makeServer() -> HummingbirdServer {
        HummingbirdServer(engine: StubInferenceEngine(engineID: .mlxSwift))
    }

    private static let boundary = "----macmlxAudioTestBoundary"

    /// Build a `multipart/form-data` body from text fields plus an optional
    /// file part, matching what `curl -F` emits.
    private func multipartBody(
        fields: [(String, String)],
        file: (name: String, filename: String, bytes: Data)? = nil
    ) -> Data {
        var out = Data()
        for (name, value) in fields {
            out.append(Data("--\(Self.boundary)\r\n".utf8))
            out.append(Data("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n".utf8))
            out.append(Data(value.utf8))
            out.append(Data("\r\n".utf8))
        }
        if let file {
            out.append(Data("--\(Self.boundary)\r\n".utf8))
            let disposition = "Content-Disposition: form-data; name=\"\(file.name)\"; "
                + "filename=\"\(file.filename)\"\r\n"
            out.append(Data(disposition.utf8))
            out.append(Data("Content-Type: audio/wav\r\n\r\n".utf8))
            out.append(file.bytes)
            out.append(Data("\r\n".utf8))
        }
        out.append(Data("--\(Self.boundary)--\r\n".utf8))
        return out
    }

    /// A tiny but structurally valid WAV, so failures are attributable to the
    /// gate under test and never to "that wasn't audio".
    private var sampleWAV: Data {
        WAVEncoder.encode(samples: [0, 0.1, -0.1, 0], sampleRate: 16_000) ?? Data()
    }

    private func post(
        _ url: URL, body: Data, contentType: String
    ) async throws -> (Data, HTTPURLResponse) {
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue(contentType, forHTTPHeaderField: "Content-Type")
        req.httpBody = body
        let (data, response) = try await URLSession.shared.data(for: req)
        let http = try #require(response as? HTTPURLResponse)
        return (data, http)
    }

    private func postMultipart(
        _ url: URL,
        fields: [(String, String)],
        file: (name: String, filename: String, bytes: Data)? = nil
    ) async throws -> (Data, HTTPURLResponse) {
        try await post(
            url,
            body: multipartBody(fields: fields, file: file),
            contentType: "multipart/form-data; boundary=\(Self.boundary)")
    }

    private func postJSON(_ url: URL, object: Any) async throws -> (Data, HTTPURLResponse) {
        try await post(
            url,
            body: try JSONSerialization.data(withJSONObject: object),
            contentType: "application/json")
    }

    private func errorCode(_ data: Data) -> String? {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let error = json["error"] as? [String: Any],
              let code = error["code"] as? String
        else { return nil }
        return code
    }

    private func errorMessage(_ data: Data) -> String? {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let error = json["error"] as? [String: Any],
              let message = error["message"] as? String
        else { return nil }
        return message
    }

    private func transcriptionsURL(_ port: Int) -> URL {
        URL(string: "http://127.0.0.1:\(port)/v1/audio/transcriptions")!
    }

    private func speechURL(_ port: Int) -> URL {
        URL(string: "http://127.0.0.1:\(port)/v1/audio/speech")!
    }

    // MARK: /v1/audio/transcriptions

    @Test
    func transcriptionsRejectsNonMultipartBody() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_600)
        let (data, response) = try await postJSON(
            transcriptionsURL(port), object: ["model": "openai/whisper-tiny"])
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(errorCode(data) == "invalid_request_error")
        #expect(errorMessage(data)?.contains("multipart/form-data") == true)
    }

    @Test
    func transcriptionsMissingFileReturns400() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_610)
        let (data, response) = try await postMultipart(
            transcriptionsURL(port), fields: [("model", "openai/whisper-tiny")])
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(errorCode(data) == "invalid_request_error")
        #expect(errorMessage(data)?.contains("`file`") == true)
    }

    @Test
    func transcriptionsMissingModelReturns400() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_620)
        let (data, response) = try await postMultipart(
            transcriptionsURL(port),
            fields: [],
            file: (name: "file", filename: "a.wav", bytes: sampleWAV))
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(errorCode(data) == "invalid_request_error")
        #expect(errorMessage(data)?.contains("`model`") == true)
    }

    @Test
    func transcriptionsRejectsSubtitleResponseFormat() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_630)
        var results: [(Int, String?)] = []
        for format in ["srt", "vtt"] {
            let (data, response) = try await postMultipart(
                transcriptionsURL(port),
                fields: [("model", "openai/whisper-tiny"), ("response_format", format)],
                file: (name: "file", filename: "a.wav", bytes: sampleWAV))
            results.append((response.statusCode, errorCode(data)))
        }
        await server.stop()

        // Refused outright — never quietly served as JSON under a subtitle name.
        #expect(results.allSatisfy { $0.0 == 400 })
        #expect(results.allSatisfy { $0.1 == "unsupported_response_format" })
    }

    @Test
    func transcriptionsRejectsUnknownResponseFormat() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_640)
        let (data, response) = try await postMultipart(
            transcriptionsURL(port),
            fields: [("model", "openai/whisper-tiny"), ("response_format", "yaml")],
            file: (name: "file", filename: "a.wav", bytes: sampleWAV))
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(errorCode(data) == "unsupported_response_format")
    }

    @Test
    func transcriptionsRejectsPromptRatherThanIgnoringIt() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_650)
        let (data, response) = try await postMultipart(
            transcriptionsURL(port),
            fields: [("model", "openai/whisper-tiny"), ("prompt", "medical terminology")],
            file: (name: "file", filename: "a.wav", bytes: sampleWAV))
        await server.stop()

        // Accepting `prompt` and dropping it would be a silent lie about what
        // conditioned the transcript.
        #expect(response.statusCode == 400)
        #expect(errorCode(data) == "unsupported_parameter")
    }

    @Test
    func transcriptionsRejectsOutOfRangeTemperature() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_660)
        var statuses: [Int] = []
        for value in ["2.0", "-0.5", "hot"] {
            let (_, response) = try await postMultipart(
                transcriptionsURL(port),
                fields: [("model", "openai/whisper-tiny"), ("temperature", value)],
                file: (name: "file", filename: "a.wav", bytes: sampleWAV))
            statuses.append(response.statusCode)
        }
        await server.stop()
        #expect(statuses == [400, 400, 400])
    }

    @Test
    func transcriptionsRejectsEmptyBody() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_670)
        let (data, response) = try await post(
            transcriptionsURL(port),
            body: Data(),
            contentType: "multipart/form-data; boundary=\(Self.boundary)")
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(errorCode(data) == "invalid_request_error")
    }

    @Test
    func transcriptionsRejectsMalformedMultipartBody() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_680)
        // Truncated: the closing delimiter never arrives.
        var body = Data("--\(Self.boundary)\r\n".utf8)
        body.append(Data("Content-Disposition: form-data; name=\"model\"\r\n\r\nwhisper".utf8))
        let (data, response) = try await post(
            transcriptionsURL(port),
            body: body,
            contentType: "multipart/form-data; boundary=\(Self.boundary)")
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(errorCode(data) == "invalid_request_error")
        #expect(errorMessage(data)?.contains("Malformed") == true)
    }

    // MARK: /v1/audio/speech

    @Test
    func speechInvalidJSONReturns400() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_690)
        let (data, response) = try await post(
            speechURL(port), body: Data("{not json".utf8), contentType: "application/json")
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(errorCode(data) == "invalid_request_error")
        #expect(errorMessage(data)?.contains("Invalid JSON") == true)
    }

    @Test
    func speechMissingModelReturns400() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_700)
        let (data, response) = try await postJSON(
            speechURL(port), object: ["input": "hello", "voice": "af_heart"])
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(errorMessage(data)?.contains("`model`") == true)
    }

    @Test
    func speechMissingInputReturns400() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_710)
        let (data, response) = try await postJSON(
            speechURL(port), object: ["model": "mlx-community/Kokoro-82M-4bit", "voice": "af_heart"])
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(errorMessage(data)?.contains("`input`") == true)
    }

    @Test
    func speechMissingVoiceReturns400() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_720)
        let (data, response) = try await postJSON(
            speechURL(port), object: ["model": "mlx-community/Kokoro-82M-4bit", "input": "hello"])
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(errorMessage(data)?.contains("`voice`") == true)
    }

    @Test
    func speechRejectsMP3AndEveryCompressedFormat() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_730)
        var results: [(status: Int, code: String?, contentType: String?)] = []
        for format in ["mp3", "opus", "aac", "flac"] {
            let (data, response) = try await postJSON(speechURL(port), object: [
                "model": "mlx-community/Kokoro-82M-4bit",
                "input": "hello",
                "voice": "af_heart",
                "response_format": format,
            ])
            results.append((
                response.statusCode,
                errorCode(data),
                response.value(forHTTPHeaderField: "Content-Type")))
        }
        await server.stop()

        #expect(results.allSatisfy { $0.status == 400 })
        #expect(results.allSatisfy { $0.code == "unsupported_response_format" })
        // The whole point: no audio content type is ever returned for a format
        // macMLX cannot encode.
        #expect(results.allSatisfy { $0.contentType?.contains("audio/") != true })
    }

    @Test
    func speechRejectsUnknownResponseFormat() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_740)
        let (data, response) = try await postJSON(speechURL(port), object: [
            "model": "mlx-community/Kokoro-82M-4bit",
            "input": "hello",
            "voice": "af_heart",
            "response_format": "ogg",
        ])
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(errorCode(data) == "unsupported_response_format")
    }

    @Test
    func speechRejectsOutOfRangeSpeed() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_750)
        var statuses: [Int] = []
        for speed in [0.1, 5.0, -1.0] {
            let (_, response) = try await postJSON(speechURL(port), object: [
                "model": "mlx-community/Kokoro-82M-4bit",
                "input": "hello",
                "voice": "af_heart",
                "speed": speed,
            ])
            statuses.append(response.statusCode)
        }
        await server.stop()
        #expect(statuses == [400, 400, 400])
    }

    @Test
    func speechRejectsUnimplementedSpeed() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_760)
        let (data, response) = try await postJSON(speechURL(port), object: [
            "model": "mlx-community/Kokoro-82M-4bit",
            "input": "hello",
            "voice": "af_heart",
            "speed": 1.5,
        ])
        await server.stop()

        // In range per OpenAI, but macMLX has no rate control — refuse rather
        // than accept-and-discard.
        #expect(response.statusCode == 400)
        #expect(errorCode(data) == "unsupported_parameter")
    }

    @Test
    func speechRejectsOverlongInput() async throws {
        let server = makeServer()
        let port = try await server.start(preferredPort: 20_770)
        let (data, response) = try await postJSON(speechURL(port), object: [
            "model": "mlx-community/Kokoro-82M-4bit",
            "input": String(repeating: "a", count: 4097),
            "voice": "af_heart",
        ])
        await server.stop()

        #expect(response.statusCode == 400)
        #expect(errorCode(data) == "invalid_request_error")
    }
}
