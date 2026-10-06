// Copyright © 2026 macMLX. English comments only.

@testable import MacMLXCore

/// An independent reference for ``SchemaConstraintState``: a strict RFC 8259
/// parser plus a recursive validator over ``SchemaValueType``, written without
/// sharing any code with the automaton so the two can be compared document by
/// document.
///
/// It decides whole documents only, with the automaton's documented
/// semantics: keys and string-enum values are compared as the strings they
/// denote (escapes decoded, like any JSON parser; invalid UTF-8 matches
/// nothing), except that an ASCII scalar other than the quote, the backslash
/// and a control character must be spelled raw in a key or an enum value;
/// unpaired surrogate escapes are rejected (as `JSONSerialization` does),
/// duplicate keys are rejected, and an integer is a number lexeme with no
/// fraction or exponent.
enum ReferenceSchemaValidator {

    /// A parsed JSON value that keeps raw lexemes: strings as the bytes between
    /// the quotes (escapes validated, not decoded) and numbers as written.
    indirect enum Value {
        case object([(key: [UInt8], value: Value)])
        case array([Value])
        case string(raw: [UInt8])
        case number(lexeme: [UInt8])
        case bool(Bool)
        case null
    }

    /// Whether `document` is a single JSON object, optionally surrounded by
    /// whitespace, that conforms to `schema`.
    static func validate(_ document: [UInt8], _ schema: JSONSchemaObject) -> Bool {
        validate(document, root: .object(schema))
    }

    /// Whether `document` is a single JSON value, optionally surrounded by
    /// whitespace, that conforms to `root`.
    static func validate(_ document: [UInt8], root: SchemaValueType) -> Bool {
        guard let value = Parser.parse(document) else { return false }
        return conforms(value, to: root)
    }

    static func conforms(_ value: Value, to type: SchemaValueType) -> Bool {
        switch (type, value) {
        case (.string, .string):
            return true
        case (.stringEnum(let values), .string(let raw)):
            guard let scalars = decodeScalars(raw) else { return false }
            return values.contains { Array($0.unicodeScalars.map(\.value)) == scalars }
        case (.number, .number):
            return true
        case (.integer, .number(let lexeme)):
            return !lexeme.contains(0x2E) && !lexeme.contains(0x65) && !lexeme.contains(0x45)
        case (.boolean, .bool):
            return true
        case (.array(let items, let minItems, let maxItems), .array(let elements)):
            guard elements.count >= minItems else { return false }
            if let maxItems, elements.count > maxItems { return false }
            return elements.allSatisfy { conforms($0, to: items) }
        case (.object(let object), .object(let members)):
            var seen: [[UInt32]] = []
            for member in members {
                guard let key = decodeScalars(member.key) else { return false }
                if seen.contains(key) { return false }
                seen.append(key)
                guard let property = object.properties.first(where: { Array($0.name.unicodeScalars.map(\.value)) == key }),
                      conforms(member.value, to: property.type) else { return false }
            }
            return object.required.allSatisfy { seen.contains(Array($0.unicodeScalars.map(\.value))) }
        default:
            return false
        }
    }

    /// The scalars a validated string body denotes as a key or an enum value:
    /// escapes decoded (a surrogate pair into one scalar), raw runs decoded as
    /// strict UTF-8; `nil` when a raw run is not valid UTF-8, or when an
    /// escape spells an ASCII scalar the automaton only matches raw.
    static func decodeScalars(_ raw: [UInt8]) -> [UInt32]? {
        let shortEscapes: [UInt8: UInt32] = [
            0x22: 0x22, 0x5C: 0x5C, 0x2F: 0x2F, 0x62: 0x08, 0x66: 0x0C, 0x6E: 0x0A, 0x72: 0x0D, 0x74: 0x09,
        ]
        var scalars: [UInt32] = []
        var index = 0
        while index < raw.count {
            if raw[index] == 0x5C {
                // The parser validated every escape, so its bytes are present.
                let escape = raw[index + 1]
                if escape == 0x75 {
                    let parser = Parser(bytes: raw)
                    guard let unit = parser.hex4(at: index + 2) else { return nil }
                    if (0xD800...0xDBFF).contains(unit) {
                        guard let low = parser.hex4(at: index + 8) else { return nil }
                        scalars.append(UInt32(0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00)))
                        index += 12
                    } else {
                        guard mayBeEscaped(UInt32(unit)) else { return nil }
                        scalars.append(UInt32(unit))
                        index += 6
                    }
                } else {
                    guard let scalar = shortEscapes[escape], mayBeEscaped(scalar) else { return nil }
                    scalars.append(scalar)
                    index += 2
                }
                continue
            }
            var end = index
            while end < raw.count, raw[end] != 0x5C { end += 1 }
            var iterator = raw[index..<end].makeIterator()
            var decoder = Unicode.UTF8()
            decoding: while true {
                switch decoder.decode(&iterator) {
                case .scalarValue(let scalar): scalars.append(scalar.value)
                case .emptyInput: break decoding
                case .error: return nil
                }
            }
            index = end
        }
        return scalars
    }

    /// The automaton's rule, written again here: only a scalar outside ASCII,
    /// the quote, the backslash or a control character may be escaped.
    static func mayBeEscaped(_ scalar: UInt32) -> Bool {
        scalar > 0x7F || scalar == 0x22 || scalar == 0x5C || scalar < 0x20
    }

    /// A strict recursive-descent JSON parser over bytes.
    struct Parser {
        let bytes: [UInt8]
        var index = 0

        static func parse(_ bytes: [UInt8]) -> Value? {
            var parser = Parser(bytes: bytes)
            parser.skipWhitespace()
            guard let value = parser.value() else { return nil }
            parser.skipWhitespace()
            return parser.index == bytes.count ? value : nil
        }

        private var current: UInt8? { index < bytes.count ? bytes[index] : nil }

        mutating func skipWhitespace() {
            while let byte = current, byte == 0x20 || byte == 0x09 || byte == 0x0A || byte == 0x0D {
                index += 1
            }
        }

        mutating func value() -> Value? {
            switch current {
            case 0x7B?: return object()
            case 0x5B?: return array()
            case 0x22?: return string().map { .string(raw: $0) }
            case 0x74?: return literal("true", .bool(true))
            case 0x66?: return literal("false", .bool(false))
            case 0x6E?: return literal("null", .null)
            case .some: return number()
            case nil: return nil
            }
        }

        mutating func literal(_ text: String, _ value: Value) -> Value? {
            let expected = Array(text.utf8)
            guard index + expected.count <= bytes.count,
                  Array(bytes[index..<index + expected.count]) == expected else { return nil }
            index += expected.count
            return value
        }

        mutating func object() -> Value? {
            index += 1
            skipWhitespace()
            var members: [(key: [UInt8], value: Value)] = []
            if current == 0x7D {
                index += 1
                return .object(members)
            }
            while true {
                skipWhitespace()
                guard current == 0x22, let key = string() else { return nil }
                skipWhitespace()
                guard current == 0x3A else { return nil }
                index += 1
                skipWhitespace()
                guard let value = value() else { return nil }
                members.append((key, value))
                skipWhitespace()
                switch current {
                case 0x2C?: index += 1
                case 0x7D?:
                    index += 1
                    return .object(members)
                default: return nil
                }
            }
        }

        mutating func array() -> Value? {
            index += 1
            skipWhitespace()
            var items: [Value] = []
            if current == 0x5D {
                index += 1
                return .array(items)
            }
            while true {
                skipWhitespace()
                guard let item = value() else { return nil }
                items.append(item)
                skipWhitespace()
                switch current {
                case 0x2C?: index += 1
                case 0x5D?:
                    index += 1
                    return .array(items)
                default: return nil
                }
            }
        }

        /// The raw bytes between the quotes, with every escape validated.
        mutating func string() -> [UInt8]? {
            index += 1
            let start = index
            while let byte = current {
                if byte == 0x22 {
                    let raw = Array(bytes[start..<index])
                    index += 1
                    return raw
                }
                if byte < 0x20 { return nil }
                if byte == 0x5C {
                    guard index + 1 < bytes.count else { return nil }
                    let escape = bytes[index + 1]
                    if [0x22, 0x5C, 0x2F, 0x62, 0x66, 0x6E, 0x72, 0x74].contains(escape) {
                        index += 2
                        continue
                    }
                    guard escape == 0x75, let unit = hex4(at: index + 2) else { return nil }
                    if (0xDC00...0xDFFF).contains(unit) { return nil }
                    if (0xD800...0xDBFF).contains(unit) {
                        guard index + 7 < bytes.count, bytes[index + 6] == 0x5C, bytes[index + 7] == 0x75,
                              let low = hex4(at: index + 8), (0xDC00...0xDFFF).contains(low) else { return nil }
                        index += 12
                        continue
                    }
                    index += 6
                    continue
                }
                index += 1
            }
            return nil
        }

        func hex4(at position: Int) -> Int? {
            guard position + 4 <= bytes.count else { return nil }
            var unit = 0
            for byte in bytes[position..<position + 4] {
                let digit: Int
                switch byte {
                case 0x30...0x39: digit = Int(byte - 0x30)
                case 0x41...0x46: digit = Int(byte - 0x41) + 10
                case 0x61...0x66: digit = Int(byte - 0x61) + 10
                default: return nil
                }
                unit = unit * 16 + digit
            }
            return unit
        }

        mutating func number() -> Value? {
            let start = index
            if current == 0x2D { index += 1 }
            guard let first = current else { return nil }
            if first == 0x30 {
                index += 1
            } else if (0x31...0x39).contains(first) {
                skipDigits()
            } else {
                return nil
            }
            if current == 0x2E {
                index += 1
                guard let digit = current, (0x30...0x39).contains(digit) else { return nil }
                skipDigits()
            }
            if current == 0x65 || current == 0x45 {
                index += 1
                if current == 0x2B || current == 0x2D { index += 1 }
                guard let digit = current, (0x30...0x39).contains(digit) else { return nil }
                skipDigits()
            }
            return .number(lexeme: Array(bytes[start..<index]))
        }

        mutating func skipDigits() {
            while let byte = current, (0x30...0x39).contains(byte) { index += 1 }
        }
    }
}
