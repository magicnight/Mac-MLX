// Copyright © 2026 macMLX. English comments only.

/// A declared key, enum value or `const` of a compiled schema, in the form the
/// schema automaton matches it: scalar by scalar, each scalar spelled either
/// as its raw UTF-8 bytes or as a JSON escape (see ``LiteralMatch``).
@usableFromInline
struct SchemaLiteral: Hashable, Sendable {
    /// The literal's text, for diagnostics.
    @usableFromInline let text: String
    /// The literal's Unicode scalars — the unit of matching.
    @usableFromInline let scalars: [UInt32]

    @usableFromInline
    init(_ text: String) {
        self.text = text
        self.scalars = text.unicodeScalars.map(\.value)
    }

    /// The UTF-8 length (1–4) of scalar value `scalar`.
    @inlinable
    static func utf8Length(of scalar: UInt32) -> Int {
        if scalar < 0x80 { return 1 }
        if scalar < 0x800 { return 2 }
        if scalar < 0x10000 { return 3 }
        return 4
    }

    /// Byte `offset` (0-based, below ``utf8Length(of:)``) of the UTF-8
    /// encoding of `scalar`.
    @inlinable
    static func utf8Byte(of scalar: UInt32, at offset: Int) -> UInt8 {
        switch utf8Length(of: scalar) {
        case 1:
            return UInt8(scalar)
        case 2:
            return offset == 0 ? UInt8(0xC0 | (scalar >> 6)) : UInt8(0x80 | (scalar & 0x3F))
        case 3:
            switch offset {
            case 0: return UInt8(0xE0 | (scalar >> 12))
            case 1: return UInt8(0x80 | ((scalar >> 6) & 0x3F))
            default: return UInt8(0x80 | (scalar & 0x3F))
            }
        default:
            switch offset {
            case 0: return UInt8(0xF0 | (scalar >> 18))
            case 1: return UInt8(0x80 | ((scalar >> 12) & 0x3F))
            case 2: return UInt8(0x80 | ((scalar >> 6) & 0x3F))
            default: return UInt8(0x80 | (scalar & 0x3F))
            }
        }
    }

    /// The UTF-16 code unit a `\u` escape spells for `scalar`: the scalar
    /// itself in the BMP; above it, the high surrogate (`low == false`) or the
    /// low surrogate (`low == true`) of its pair.
    @inlinable
    static func escapeUnit(of scalar: UInt32, low: Bool) -> Int {
        guard scalar > 0xFFFF else { return Int(scalar) }
        let offset = scalar - 0x10000
        return low ? Int(0xDC00 + (offset & 0x3FF)) : Int(0xD800 + (offset >> 10))
    }

    /// The scalar a two-character escape `\X` denotes, or `nil` when `X` is
    /// not one of JSON's eight (`\"`, `\\`, `\/`, `\b`, `\f`, `\n`, `\r`, `\t`).
    @inlinable
    static func shortEscapeScalar(_ byte: UInt8) -> UInt32? {
        switch byte {
        case 0x22: return 0x22
        case 0x5C: return 0x5C
        case 0x2F: return 0x2F
        case 0x62: return 0x08
        case 0x66: return 0x0C
        case 0x6E: return 0x0A
        case 0x72: return 0x0D
        case 0x74: return 0x09
        default: return nil
        }
    }
}
