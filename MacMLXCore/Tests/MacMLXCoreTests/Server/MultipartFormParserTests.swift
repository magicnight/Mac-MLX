import Foundation
import Testing
@testable import MacMLXCore

/// Pure unit tests for ``MultipartFormParser`` — the hand-written
/// `multipart/form-data` reader behind `POST /v1/audio/transcriptions`.
///
/// This is the thickest test surface in the audio wave on purpose: the parser
/// is the only place untrusted bytes are interpreted before they reach a
/// model, and it replaces a dependency we deliberately did not take. Every
/// case here is bytes-in / values-out — no server, no model, no Metal, no
/// network.
@Suite("MultipartFormParser")
struct MultipartFormParserTests {

    // MARK: Builders

    private static let boundary = "----macmlxTestBoundary7MA4YWxkTrZu0gW"

    /// Assemble a well-formed body from `(headers, payload)` pairs so each
    /// test states only what it is varying.
    private func body(
        boundary: String = MultipartFormParserTests.boundary,
        parts: [(headers: [String], payload: Data)],
        closed: Bool = true
    ) -> Data {
        var out = Data()
        for part in parts {
            out.append(Data("--\(boundary)\r\n".utf8))
            for header in part.headers {
                out.append(Data("\(header)\r\n".utf8))
            }
            out.append(Data("\r\n".utf8))
            out.append(part.payload)
            out.append(Data("\r\n".utf8))
        }
        out.append(Data((closed ? "--\(boundary)--\r\n" : "--\(boundary)\r\n").utf8))
        return out
    }

    private func textField(_ name: String, _ value: String) -> (headers: [String], payload: Data) {
        (["Content-Disposition: form-data; name=\"\(name)\""], Data(value.utf8))
    }

    private func filePart(
        name: String = "file",
        filename: String = "audio.wav",
        contentType: String? = "audio/wav",
        payload: Data
    ) -> (headers: [String], payload: Data) {
        var headers = ["Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(filename)\""]
        if let contentType { headers.append("Content-Type: \(contentType)") }
        return (headers, payload)
    }

    private var contentType: String { "multipart/form-data; boundary=\(Self.boundary)" }

    // MARK: Happy path

    @Test
    func parsesSingleFileWithMultipleTextFields() throws {
        let audio = Data([0x52, 0x49, 0x46, 0x46, 0x00, 0xFF, 0x10, 0x00])
        let data = body(parts: [
            textField("model", "openai/whisper-tiny"),
            textField("response_format", "verbose_json"),
            filePart(payload: audio),
            textField("language", "en"),
        ])

        let form = try MultipartFormParser.parse(
            body: data, contentType: contentType, maxBytes: 1 << 20)

        #expect(form.parts.count == 4)
        #expect(form.value("model") == "openai/whisper-tiny")
        #expect(form.value("response_format") == "verbose_json")
        #expect(form.value("language") == "en")

        let file = try #require(form.file("file"))
        #expect(file.filename == "audio.wav")
        #expect(file.contentType == "audio/wav")
        #expect(file.body == audio)
        // A file part is NOT a text field, even though it has a name.
        #expect(form.value("file") == nil)
    }

    @Test
    func preservesBinaryPayloadExactlyIncludingNulAndHighBytes() throws {
        let audio = Data([0x00, 0x01, 0xFF, 0x0D, 0x0A, 0x00, 0x2D, 0x2D, 0xFE])
        let data = body(parts: [textField("model", "m"), filePart(payload: audio)])
        let form = try MultipartFormParser.parse(
            body: data, contentType: contentType, maxBytes: 1 << 20)
        #expect(try #require(form.file("file")).body == audio)
    }

    @Test
    func emptyFilePayloadStillParsesAsAFilePart() throws {
        let data = body(parts: [textField("model", "m"), filePart(payload: Data())])
        let form = try MultipartFormParser.parse(
            body: data, contentType: contentType, maxBytes: 1 << 20)
        let file = try #require(form.file("file"))
        #expect(file.body.isEmpty)
    }

    @Test
    func ignoresPreambleBeforeTheFirstDelimiter() throws {
        var data = Data("This is a MIME preamble that clients may send.\r\n".utf8)
        data.append(body(parts: [textField("model", "m")]))
        let form = try MultipartFormParser.parse(
            body: data, contentType: contentType, maxBytes: 1 << 20)
        #expect(form.value("model") == "m")
    }

    // MARK: Boundary handling

    @Test
    func acceptsQuotedBoundaryContainingSpaces() throws {
        let quoted = "simple boundary"
        let data = body(boundary: quoted, parts: [
            textField("model", "m"),
            filePart(payload: Data([1, 2, 3])),
        ])
        let form = try MultipartFormParser.parse(
            body: data,
            contentType: "multipart/form-data; boundary=\"\(quoted)\"",
            maxBytes: 1 << 20)
        #expect(form.value("model") == "m")
        #expect(try #require(form.file("file")).body == Data([1, 2, 3]))
    }

    @Test
    func acceptsBoundaryParameterInAnyOrderAndAnyCase() throws {
        let data = body(parts: [textField("model", "m")])
        let form = try MultipartFormParser.parse(
            body: data,
            contentType: "multipart/form-data; charset=utf-8; BOUNDARY=\(Self.boundary)",
            maxBytes: 1 << 20)
        #expect(form.value("model") == "m")
    }

    @Test
    func rejectsNonMultipartContentType() {
        let data = body(parts: [textField("model", "m")])
        #expect(throws: MultipartFormParser.ParseError.notMultipart(contentType: "application/json")) {
            try MultipartFormParser.parse(
                body: data, contentType: "application/json", maxBytes: 1 << 20)
        }
    }

    @Test
    func rejectsAbsentContentType() {
        let data = body(parts: [textField("model", "m")])
        #expect(throws: MultipartFormParser.ParseError.notMultipart(contentType: nil)) {
            try MultipartFormParser.parse(body: data, contentType: nil, maxBytes: 1 << 20)
        }
    }

    @Test
    func rejectsAContentTypeThatMerelyContainsMultipartFormData() {
        // The media type is matched exactly, not searched for as a substring:
        // `application/not-multipart/form-data` is not multipart and must not
        // be parsed as one just because the literal appears inside it.
        let data = body(parts: [textField("model", "m")])
        let impostors = [
            "application/not-multipart/form-data",
            "application/not-multipart/form-data; boundary=\(Self.boundary)",
            "x-multipart/form-data; boundary=\(Self.boundary)",
            "multipart/form-data-ish; boundary=\(Self.boundary)",
            "text/plain; note=multipart/form-data; boundary=\(Self.boundary)",
        ]
        for contentType in impostors {
            #expect(throws: MultipartFormParser.ParseError.notMultipart(contentType: contentType)) {
                try MultipartFormParser.parse(
                    body: data, contentType: contentType, maxBytes: 1 << 20)
            }
        }
    }

    @Test
    func acceptsTheMediaTypeInAnyCaseAndWithSurroundingWhitespace() throws {
        // RFC 9110: the media type is case-insensitive, and optional whitespace
        // may sit around it and around the ';' that starts the parameters.
        let data = body(parts: [textField("model", "m")])
        let variants = [
            "multipart/form-data; boundary=\(Self.boundary)",
            "MULTIPART/FORM-DATA; boundary=\(Self.boundary)",
            "Multipart/Form-Data;boundary=\(Self.boundary)",
            "  multipart/form-data  ;  boundary=\(Self.boundary)",
        ]
        for contentType in variants {
            let form = try MultipartFormParser.parse(
                body: data, contentType: contentType, maxBytes: 1 << 20)
            #expect(form.value("model") == "m", "should have parsed '\(contentType)'")
        }
    }

    @Test
    func rejectsMissingBoundaryParameter() {
        let data = body(parts: [textField("model", "m")])
        #expect(throws: MultipartFormParser.ParseError.missingBoundary) {
            try MultipartFormParser.parse(
                body: data, contentType: "multipart/form-data", maxBytes: 1 << 20)
        }
    }

    @Test
    func aTrailingSemicolonWithNoBoundaryStillReportsTheMissingBoundary() {
        // Tightening the media-type match must not turn "multipart, but you
        // forgot the boundary" into "not multipart at all" — those are
        // different mistakes and the message has to name the right one.
        let data = body(parts: [textField("model", "m")])
        for contentType in ["multipart/form-data;", "multipart/form-data; charset=utf-8"] {
            #expect(throws: MultipartFormParser.ParseError.missingBoundary) {
                try MultipartFormParser.parse(
                    body: data, contentType: contentType, maxBytes: 1 << 20)
            }
        }
    }

    @Test
    func rejectsBoundaryLongerThanRFC2046Allows() {
        let tooLong = String(repeating: "a", count: 71)
        let data = body(boundary: tooLong, parts: [textField("model", "m")])
        #expect(throws: MultipartFormParser.ParseError.invalidBoundary(tooLong)) {
            try MultipartFormParser.parse(
                body: data,
                contentType: "multipart/form-data; boundary=\(tooLong)",
                maxBytes: 1 << 20)
        }
    }

    @Test
    func rejectsEmptyQuotedBoundary() {
        #expect(throws: MultipartFormParser.ParseError.invalidBoundary("")) {
            try MultipartFormParser.parse(
                body: Data("--\r\n".utf8),
                contentType: "multipart/form-data; boundary=\"\"",
                maxBytes: 1 << 20)
        }
    }

    // MARK: Missing fields (the endpoint's 400 inputs)

    @Test
    func missingFilePartIsReportedAsAbsentNotAsAnError() throws {
        let data = body(parts: [textField("model", "openai/whisper-tiny")])
        let form = try MultipartFormParser.parse(
            body: data, contentType: contentType, maxBytes: 1 << 20)
        #expect(form.file("file") == nil)
        #expect(form.value("model") == "openai/whisper-tiny")
    }

    @Test
    func missingModelFieldIsReportedAsAbsentNotAsAnError() throws {
        let data = body(parts: [filePart(payload: Data([1]))])
        let form = try MultipartFormParser.parse(
            body: data, contentType: contentType, maxBytes: 1 << 20)
        #expect(form.value("model") == nil)
        #expect(form.file("file") != nil)
    }

    // MARK: Malformed input

    @Test
    func rejectsBodyWithNoDelimiterAtAll() {
        #expect(throws: MultipartFormParser.ParseError.self) {
            try MultipartFormParser.parse(
                body: Data("just some bytes".utf8), contentType: contentType, maxBytes: 1 << 20)
        }
    }

    @Test
    func rejectsBodyTruncatedBeforeTheClosingDelimiter() {
        var data = body(parts: [textField("model", "m"), filePart(payload: Data([1, 2, 3]))])
        data = data.prefix(data.count - 20)  // chop the closing delimiter off
        #expect(throws: MultipartFormParser.ParseError.self) {
            try MultipartFormParser.parse(
                body: data, contentType: contentType, maxBytes: 1 << 20)
        }
    }

    @Test
    func rejectsBodyTruncatedImmediatelyAfterADelimiter() {
        #expect(throws: MultipartFormParser.ParseError.self) {
            try MultipartFormParser.parse(
                body: Data("--\(Self.boundary)".utf8),
                contentType: contentType,
                maxBytes: 1 << 20)
        }
    }

    @Test
    func rejectsPartHeadersThatAreNeverTerminatedByABlankLine() {
        var data = Data("--\(Self.boundary)\r\n".utf8)
        data.append(Data("Content-Disposition: form-data; name=\"model\"\r\n".utf8))
        data.append(Data("--\(Self.boundary)--\r\n".utf8))
        #expect(throws: MultipartFormParser.ParseError.self) {
            try MultipartFormParser.parse(
                body: data, contentType: contentType, maxBytes: 1 << 20)
        }
    }

    @Test
    func rejectsPartWithoutContentDisposition() {
        var data = Data("--\(Self.boundary)\r\n".utf8)
        data.append(Data("Content-Type: text/plain\r\n\r\nvalue\r\n".utf8))
        data.append(Data("--\(Self.boundary)--\r\n".utf8))
        #expect(throws: MultipartFormParser.ParseError.self) {
            try MultipartFormParser.parse(
                body: data, contentType: contentType, maxBytes: 1 << 20)
        }
    }

    @Test
    func rejectsContentDispositionWithoutAName() {
        var data = Data("--\(Self.boundary)\r\n".utf8)
        data.append(Data("Content-Disposition: form-data\r\n\r\nvalue\r\n".utf8))
        data.append(Data("--\(Self.boundary)--\r\n".utf8))
        #expect(throws: MultipartFormParser.ParseError.self) {
            try MultipartFormParser.parse(
                body: data, contentType: contentType, maxBytes: 1 << 20)
        }
    }

    @Test
    func rejectsHeaderLineWithoutAColon() {
        var data = Data("--\(Self.boundary)\r\n".utf8)
        data.append(Data("Content-Disposition: form-data; name=\"m\"\r\n".utf8))
        data.append(Data("this-is-not-a-header\r\n\r\nvalue\r\n".utf8))
        data.append(Data("--\(Self.boundary)--\r\n".utf8))
        #expect(throws: MultipartFormParser.ParseError.self) {
            try MultipartFormParser.parse(
                body: data, contentType: contentType, maxBytes: 1 << 20)
        }
    }

    @Test
    func rejectsNonIdentityContentTransferEncodingRatherThanMisdecodingIt() {
        var data = Data("--\(Self.boundary)\r\n".utf8)
        data.append(Data("Content-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n".utf8))
        data.append(Data("Content-Transfer-Encoding: base64\r\n\r\nUklGRg==\r\n".utf8))
        data.append(Data("--\(Self.boundary)--\r\n".utf8))
        #expect(throws: MultipartFormParser.ParseError.self) {
            try MultipartFormParser.parse(
                body: data, contentType: contentType, maxBytes: 1 << 20)
        }
    }

    @Test
    func acceptsIdentityContentTransferEncodings() throws {
        var data = Data("--\(Self.boundary)\r\n".utf8)
        data.append(Data("Content-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n".utf8))
        data.append(Data("Content-Transfer-Encoding: binary\r\n\r\nabc\r\n".utf8))
        data.append(Data("--\(Self.boundary)--\r\n".utf8))
        let form = try MultipartFormParser.parse(
            body: data, contentType: contentType, maxBytes: 1 << 20)
        #expect(try #require(form.file("file")).body == Data("abc".utf8))
    }

    // MARK: CRLF strictness

    @Test
    func rejectsBareLFLineEndingsInsteadOfSilentlyMisparsing() {
        // A client that wrote LF-only delimiters produces bytes that look
        // almost right; accepting them would silently fold headers into the
        // payload. Fail loudly instead.
        var data = Data("--\(Self.boundary)\n".utf8)
        data.append(Data("Content-Disposition: form-data; name=\"model\"\n\nm\n".utf8))
        data.append(Data("--\(Self.boundary)--\n".utf8))
        #expect(throws: MultipartFormParser.ParseError.self) {
            try MultipartFormParser.parse(
                body: data, contentType: contentType, maxBytes: 1 << 20)
        }
    }

    @Test
    func payloadKeepsBareLFAndBoundaryLikeTextThatIsNotCRLFPrefixed() throws {
        // Only "\r\n--boundary" terminates a part. Bare LF, and a
        // boundary-looking run that is not CRLF-prefixed, are ordinary bytes.
        let payload = Data("line one\nline two --\(Self.boundary) still payload\n".utf8)
        let data = body(parts: [filePart(payload: payload)])
        let form = try MultipartFormParser.parse(
            body: data, contentType: contentType, maxBytes: 1 << 20)
        #expect(try #require(form.file("file")).body == payload)
    }

    @Test
    func payloadEndingInCRIsNotTruncated() throws {
        let payload = Data([0x61, 0x0D])  // "a\r" — one byte short of a delimiter prefix
        let data = body(parts: [filePart(payload: payload)])
        let form = try MultipartFormParser.parse(
            body: data, contentType: contentType, maxBytes: 1 << 20)
        #expect(try #require(form.file("file")).body == payload)
    }

    // MARK: Filenames

    @Test
    func unescapesQuotedFilenameContainingQuotesSemicolonsAndSpaces() throws {
        var data = Data("--\(Self.boundary)\r\n".utf8)
        data.append(Data(
            #"Content-Disposition: form-data; name="file"; filename="my \"odd\"; take 2.wav""#.utf8))
        data.append(Data("\r\n\r\nxyz\r\n".utf8))
        data.append(Data("--\(Self.boundary)--\r\n".utf8))

        let form = try MultipartFormParser.parse(
            body: data, contentType: contentType, maxBytes: 1 << 20)
        #expect(try #require(form.file("file")).filename == #"my "odd"; take 2.wav"#)
    }

    @Test
    func keepsNonASCIIFilenameVerbatim() throws {
        let data = body(parts: [filePart(filename: "録音 テスト.m4a", payload: Data([1]))])
        let form = try MultipartFormParser.parse(
            body: data, contentType: contentType, maxBytes: 1 << 20)
        #expect(try #require(form.file("file")).filename == "録音 テスト.m4a")
    }

    @Test
    func doesNotSanitizePathTraversalItselfButSurfacesItToTheCaller() throws {
        // The parser is pure: it reports what the client sent. Sanitizing is
        // the endpoint's job (`sanitizedExtension`), and it never uses the
        // client's name as a path.
        let data = body(parts: [filePart(filename: "../../etc/passwd", payload: Data([1]))])
        let form = try MultipartFormParser.parse(
            body: data, contentType: contentType, maxBytes: 1 << 20)
        #expect(try #require(form.file("file")).filename == "../../etc/passwd")
    }

    @Test
    func emptyFilenameStillMarksThePartAsAFile() throws {
        let data = body(parts: [filePart(filename: "", payload: Data([1]))])
        let form = try MultipartFormParser.parse(
            body: data, contentType: contentType, maxBytes: 1 << 20)
        #expect(form.file("file") != nil)
        #expect(try #require(form.file("file")).filename == "")
    }

    @Test
    func headerFieldNamesAreCaseInsensitive() throws {
        var data = Data("--\(Self.boundary)\r\n".utf8)
        data.append(Data("CONTENT-DISPOSITION: form-data; NAME=\"file\"; FILENAME=\"a.wav\"\r\n".utf8))
        data.append(Data("content-type: audio/wav\r\n\r\nq\r\n".utf8))
        data.append(Data("--\(Self.boundary)--\r\n".utf8))

        let form = try MultipartFormParser.parse(
            body: data, contentType: contentType, maxBytes: 1 << 20)
        let file = try #require(form.file("file"))
        #expect(file.filename == "a.wav")
        #expect(file.contentType == "audio/wav")
    }

    @Test
    func unquotedParameterValuesAreAccepted() throws {
        var data = Data("--\(Self.boundary)\r\n".utf8)
        data.append(Data("Content-Disposition: form-data; name=model\r\n\r\nwhisper\r\n".utf8))
        data.append(Data("--\(Self.boundary)--\r\n".utf8))
        let form = try MultipartFormParser.parse(
            body: data, contentType: contentType, maxBytes: 1 << 20)
        #expect(form.value("model") == "whisper")
    }

    @Test
    func boundaryOverloadParsesTheBodyGrammarWithoutAContentType() throws {
        let data = body(parts: [textField("model", "m"), filePart(payload: Data([9]))])
        let form = try MultipartFormParser.parse(body: data, boundary: Self.boundary)
        #expect(form.value("model") == "m")
        #expect(try #require(form.file("file")).body == Data([9]))
    }

    // MARK: Size ceiling

    @Test
    func rejectsBodyOverTheLimitBeforeParsingAnything() {
        let data = body(parts: [filePart(payload: Data(repeating: 0xAB, count: 4096))])
        #expect(throws: MultipartFormParser.ParseError.bodyTooLarge(limit: 512)) {
            try MultipartFormParser.parse(body: data, contentType: contentType, maxBytes: 512)
        }
    }

    @Test
    func acceptsBodyExactlyAtTheLimit() throws {
        let data = body(parts: [textField("model", "m")])
        let form = try MultipartFormParser.parse(
            body: data, contentType: contentType, maxBytes: data.count)
        #expect(form.value("model") == "m")
    }

    // MARK: Lookup semantics

    @Test
    func lookupsReturnTheFirstMatchingPart() throws {
        let data = body(parts: [
            textField("model", "first"),
            textField("model", "second"),
        ])
        let form = try MultipartFormParser.parse(
            body: data, contentType: contentType, maxBytes: 1 << 20)
        #expect(form.value("model") == "first")
        #expect(form.parts.count == 2)
    }

    @Test
    func nonUTF8TextFieldDecodesToNilRatherThanCrashing() throws {
        let invalid = Data([0xFF, 0xFE, 0xFD])
        var data = Data("--\(Self.boundary)\r\n".utf8)
        data.append(Data("Content-Disposition: form-data; name=\"model\"\r\n\r\n".utf8))
        data.append(invalid)
        data.append(Data("\r\n--\(Self.boundary)--\r\n".utf8))

        let form = try MultipartFormParser.parse(
            body: data, contentType: contentType, maxBytes: 1 << 20)
        #expect(form.value("model") == nil)
        #expect(try #require(form.part("model")).body == invalid)
    }
}
