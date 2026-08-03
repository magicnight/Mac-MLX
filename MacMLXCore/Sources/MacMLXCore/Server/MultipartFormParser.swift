import Foundation

/// A minimal, dependency-free `multipart/form-data` reader for the one shape
/// macMLX needs: a handful of small text fields plus a single uploaded file.
///
/// Why hand-written: Hummingbird 2.25 ships no multipart product, and
/// `vapor/multipart-kit` 5.x is still pre-release — the pinning policy in
/// `Package.swift` keeps beta dependencies out of the graph. The subset here is
/// small enough to be fully unit-tested, which is the trade that makes it
/// acceptable.
///
/// Scope and deliberate limits (RFC 7578 / RFC 2046):
/// - Parses a flat part list. `multipart/mixed` sub-parts are NOT supported;
///   modern browsers and `curl -F` never emit them for form uploads.
/// - `Content-Transfer-Encoding` is NOT decoded. RFC 7578 §4.7 says form-data
///   parts should use the identity encoding, and no mainstream client
///   base64-encodes a file part. A part that declares a non-identity encoding
///   is rejected rather than silently mis-decoded.
/// - Part bodies are returned verbatim; nothing is written to disk here.
///
/// Everything is pure: bytes in, values out. No I/O, no Metal, no globals.
public enum MultipartFormParser {

    // MARK: Types

    /// One decoded part: its form field `name`, the original `filename` when
    /// the part was a file upload, its declared `contentType`, and the raw
    /// body bytes.
    public struct Part: Sendable, Equatable {
        public let name: String
        public let filename: String?
        public let contentType: String?
        public let body: Data

        public init(name: String, filename: String?, contentType: String?, body: Data) {
            self.name = name
            self.filename = filename
            self.contentType = contentType
            self.body = body
        }

        /// The body decoded as UTF-8, for text fields. `nil` when the bytes
        /// are not valid UTF-8.
        public var stringValue: String? { String(data: body, encoding: .utf8) }
    }

    /// A parsed form: the parts in wire order, plus lookups by field name.
    public struct Form: Sendable, Equatable {
        public let parts: [Part]

        public init(parts: [Part]) { self.parts = parts }

        /// First part named `name`, regardless of whether it is a file.
        public func part(_ name: String) -> Part? {
            parts.first { $0.name == name }
        }

        /// First **text** field named `name` (a part with no `filename`),
        /// decoded as UTF-8.
        public func value(_ name: String) -> String? {
            parts.first { $0.name == name && $0.filename == nil }?.stringValue
        }

        /// First **file** part named `name` (a part that carried a
        /// `filename` parameter, even an empty one).
        public func file(_ name: String) -> Part? {
            parts.first { $0.name == name && $0.filename != nil }
        }
    }

    /// Everything that can go wrong, each carrying enough detail for a caller
    /// to build a specific 400 rather than a generic one.
    public enum ParseError: Error, Equatable, LocalizedError {
        /// The request's `Content-Type` was absent or was not
        /// `multipart/form-data`.
        case notMultipart(contentType: String?)
        /// `multipart/form-data` with no usable `boundary` parameter.
        case missingBoundary
        /// A boundary that RFC 2046 forbids (empty, over 70 characters, or
        /// containing bytes that cannot appear in a delimiter).
        case invalidBoundary(String)
        /// The body was larger than the caller's ceiling.
        case bodyTooLarge(limit: Int)
        /// Structurally broken body — truncated, missing delimiters, bad
        /// headers. The payload names the specific defect.
        case malformed(String)

        public var errorDescription: String? {
            switch self {
            case .notMultipart(let contentType):
                return "Expected a multipart/form-data body but Content-Type was "
                    + "\(contentType.map { "'\($0)'" } ?? "absent")."
            case .missingBoundary:
                return "multipart/form-data Content-Type is missing its `boundary` parameter."
            case .invalidBoundary(let boundary):
                return "Invalid multipart boundary '\(boundary)'."
            case .bodyTooLarge(let limit):
                return "Request body exceeds the \(limit)-byte limit."
            case .malformed(let reason):
                return "Malformed multipart/form-data body: \(reason)"
            }
        }
    }

    // MARK: Boundary

    /// Extract the `boundary` parameter from a `multipart/form-data`
    /// `Content-Type`.
    ///
    /// Handles the quoted form (`boundary="…"`, which clients use whenever the
    /// delimiter contains characters that are not a bare token), parameters in
    /// any order, and case-insensitive attribute names. Per RFC 2046 the
    /// boundary must be 1–70 characters.
    public static func boundary(fromContentType contentType: String?) throws(ParseError) -> String {
        guard let contentType else { throw ParseError.notMultipart(contentType: nil) }
        let lowered = contentType.lowercased()
        guard lowered.contains("multipart/form-data") else {
            throw ParseError.notMultipart(contentType: contentType)
        }

        // Split on ';' at the top level. A quoted boundary may itself contain
        // ';', so track quoting while splitting.
        var parameters: [String] = []
        var current = ""
        var inQuotes = false
        var escaped = false
        for character in contentType {
            if escaped {
                current.append(character)
                escaped = false
                continue
            }
            switch character {
            case "\\" where inQuotes:
                current.append(character)
                escaped = true
            case "\"":
                inQuotes.toggle()
                current.append(character)
            case ";" where !inQuotes:
                parameters.append(current)
                current = ""
            default:
                current.append(character)
            }
        }
        parameters.append(current)

        for parameter in parameters.dropFirst() {
            let trimmed = parameter.trimmingCharacters(in: .whitespaces)
            guard let equals = trimmed.firstIndex(of: "="),
                  trimmed[trimmed.startIndex..<equals].lowercased() == "boundary"
            else { continue }
            let rawValue = String(trimmed[trimmed.index(after: equals)...])
                .trimmingCharacters(in: .whitespaces)
            let value = unquote(rawValue)
            guard !value.isEmpty, value.count <= 70 else {
                throw ParseError.invalidBoundary(value)
            }
            // The delimiter is written into the body as raw bytes; anything
            // non-ASCII or containing CR/LF could never match one.
            guard value.allSatisfy({ $0.isASCII && !$0.isNewline }) else {
                throw ParseError.invalidBoundary(value)
            }
            return value
        }
        throw ParseError.missingBoundary
    }

    // MARK: Parsing

    /// Parse a complete `multipart/form-data` body.
    ///
    /// - Parameters:
    ///   - body: The full request body. Callers MUST have bounded the read
    ///     that produced it; `maxBytes` here is a second line of defence, not
    ///     the first.
    ///   - contentType: The request's `Content-Type` header value.
    ///   - maxBytes: Reject bodies larger than this.
    /// - Throws: ``ParseError`` describing the specific defect.
    public static func parse(
        body: Data,
        contentType: String?,
        maxBytes: Int
    ) throws(ParseError) -> Form {
        guard body.count <= maxBytes else { throw ParseError.bodyTooLarge(limit: maxBytes) }
        let boundary = try boundary(fromContentType: contentType)
        return try parse(body: body, boundary: boundary)
    }

    /// Parse a body whose boundary is already known. Split out so tests can
    /// drive the body grammar without restating a `Content-Type`.
    public static func parse(body: Data, boundary: String) throws(ParseError) -> Form {
        let crlf = Data([0x0D, 0x0A])
        let dashes = Data([0x2D, 0x2D])
        let delimiter = dashes + Data(boundary.utf8)      // "--boundary"
        let crlfDelimiter = crlf + delimiter              // "\r\n--boundary"

        // RFC 2046 allows an ignorable preamble before the first delimiter, so
        // search rather than requiring offset 0. The first delimiter is not
        // CRLF-prefixed when there is no preamble.
        guard let first = body.range(of: delimiter, in: body.startIndex..<body.endIndex) else {
            throw ParseError.malformed("no `--\(boundary)` delimiter found")
        }

        var parts: [Part] = []
        var cursor = first.upperBound
        var sawClosingDelimiter = false

        while true {
            // A delimiter is followed either by "--" (this was the closing
            // delimiter) or by CRLF (another part starts).
            guard cursor + 2 <= body.endIndex else {
                throw ParseError.malformed("body truncated immediately after a boundary delimiter")
            }
            let marker = Data(body[cursor..<(cursor + 2)])
            if marker == dashes {
                sawClosingDelimiter = true
                break
            }
            guard marker == crlf else {
                // Most often a boundary that is a prefix of the real one, or a
                // client that used bare LF line endings.
                throw ParseError.malformed("expected CRLF or `--` after a boundary delimiter")
            }

            let headerStart = cursor + 2
            guard let headerTerminator = body.range(
                of: crlf + crlf, in: headerStart..<body.endIndex
            ) else {
                throw ParseError.malformed("part headers are not terminated by a blank line")
            }
            let headers = try parseHeaders(Data(body[headerStart..<headerTerminator.lowerBound]))

            let bodyStart = headerTerminator.upperBound
            guard let nextDelimiter = body.range(
                of: crlfDelimiter, in: bodyStart..<body.endIndex
            ) else {
                throw ParseError.malformed("a part body is not terminated by a closing boundary")
            }

            guard let disposition = headers.contentDisposition else {
                throw ParseError.malformed("a part is missing its Content-Disposition header")
            }
            guard let name = disposition.name else {
                throw ParseError.malformed("a part's Content-Disposition has no `name` parameter")
            }
            if let encoding = headers.contentTransferEncoding,
               !["7bit", "8bit", "binary"].contains(encoding.lowercased()) {
                throw ParseError.malformed(
                    "part '\(name)' declares Content-Transfer-Encoding '\(encoding)', which "
                        + "macMLX does not decode (RFC 7578 expects identity encoding)")
            }

            parts.append(Part(
                name: name,
                filename: disposition.filename,
                contentType: headers.contentType,
                body: Data(body[bodyStart..<nextDelimiter.lowerBound])
            ))
            cursor = nextDelimiter.upperBound
        }

        guard sawClosingDelimiter else {
            throw ParseError.malformed("missing closing `--\(boundary)--` delimiter")
        }
        return Form(parts: parts)
    }

    // MARK: Private

    /// The three part headers that matter here, pre-split.
    private struct PartHeaders {
        var contentDisposition: (name: String?, filename: String?)?
        var contentType: String?
        var contentTransferEncoding: String?
    }

    private static func parseHeaders(_ data: Data) throws(ParseError) -> PartHeaders {
        guard let text = String(data: data, encoding: .utf8) else {
            throw ParseError.malformed("part headers are not valid UTF-8")
        }
        var headers = PartHeaders()
        // Headers were captured between delimiters, so splitting on CRLF is
        // exact; `omittingEmptySubsequences` tolerates a stray blank line.
        for line in text.components(separatedBy: "\r\n") where !line.isEmpty {
            guard let colon = line.firstIndex(of: ":") else {
                throw ParseError.malformed("part header line '\(line)' has no colon")
            }
            let field = line[line.startIndex..<colon].trimmingCharacters(in: .whitespaces).lowercased()
            let value = String(line[line.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
            switch field {
            case "content-disposition":
                headers.contentDisposition = parseDisposition(value)
            case "content-type":
                headers.contentType = value
            case "content-transfer-encoding":
                headers.contentTransferEncoding = value
            default:
                continue  // Any other part header is irrelevant here.
            }
        }
        return headers
    }

    /// Pull `name` and `filename` out of a `Content-Disposition` value.
    ///
    /// Both may be quoted (`name="file"`), and a quoted value may contain
    /// `;`, `=`, escaped quotes, or non-ASCII characters — all of which real
    /// clients emit for filenames. An absent `filename` means "text field";
    /// a present-but-empty one still means "file part".
    private static func parseDisposition(_ value: String) -> (name: String?, filename: String?) {
        var name: String?
        var filename: String?

        var parameters: [String] = []
        var current = ""
        var inQuotes = false
        var escaped = false
        for character in value {
            if escaped {
                current.append(character)
                escaped = false
                continue
            }
            switch character {
            case "\\" where inQuotes:
                current.append(character)
                escaped = true
            case "\"":
                inQuotes.toggle()
                current.append(character)
            case ";" where !inQuotes:
                parameters.append(current)
                current = ""
            default:
                current.append(character)
            }
        }
        parameters.append(current)

        for parameter in parameters.dropFirst() {  // dropFirst = the disposition type
            let trimmed = parameter.trimmingCharacters(in: .whitespaces)
            guard let equals = trimmed.firstIndex(of: "=") else { continue }
            let key = trimmed[trimmed.startIndex..<equals].trimmingCharacters(in: .whitespaces)
                .lowercased()
            let raw = String(trimmed[trimmed.index(after: equals)...])
                .trimmingCharacters(in: .whitespaces)
            switch key {
            case "name" where name == nil: name = unquote(raw)
            case "filename" where filename == nil: filename = unquote(raw)
            default: continue
            }
        }
        return (name: name, filename: filename)
    }

    /// Strip surrounding double quotes and unescape `\"` / `\\` inside them.
    /// An unquoted token is returned as-is.
    private static func unquote(_ value: String) -> String {
        guard value.count >= 2, value.hasPrefix("\""), value.hasSuffix("\"") else { return value }
        let inner = value.dropFirst().dropLast()
        var out = ""
        var escaped = false
        for character in inner {
            if escaped {
                out.append(character)
                escaped = false
            } else if character == "\\" {
                escaped = true
            } else {
                out.append(character)
            }
        }
        if escaped { out.append("\\") }  // trailing lone backslash: keep it literal
        return out
    }
}
