// Copyright © 2026 macMLX. English comments only.

/// Byte constants and byte classes shared by the schema automaton
/// (``SchemaConstraintState``) and its scalar sub-machine
/// (``SchemaScalarState``).
@usableFromInline
enum SchemaBytes {
    @usableFromInline static let quote: UInt8 = 0x22
    @usableFromInline static let backslash: UInt8 = 0x5C
    @usableFromInline static let slash: UInt8 = 0x2F
    @usableFromInline static let colon: UInt8 = 0x3A
    @usableFromInline static let comma: UInt8 = 0x2C
    @usableFromInline static let lBrace: UInt8 = 0x7B
    @usableFromInline static let rBrace: UInt8 = 0x7D
    @usableFromInline static let lBracket: UInt8 = 0x5B
    @usableFromInline static let rBracket: UInt8 = 0x5D
    @usableFromInline static let dot: UInt8 = 0x2E
    @usableFromInline static let plus: UInt8 = 0x2B
    @usableFromInline static let minus: UInt8 = 0x2D
    @usableFromInline static let zero: UInt8 = 0x30
    @usableFromInline static let lowerE: UInt8 = 0x65
    @usableFromInline static let upperE: UInt8 = 0x45
    @usableFromInline static let lowerB: UInt8 = 0x62
    @usableFromInline static let lowerF: UInt8 = 0x66
    @usableFromInline static let lowerN: UInt8 = 0x6E
    @usableFromInline static let lowerR: UInt8 = 0x72
    @usableFromInline static let lowerT: UInt8 = 0x74
    @usableFromInline static let lowerU: UInt8 = 0x75

    @inlinable static func isWhitespace(_ b: UInt8) -> Bool { b == 0x20 || b == 0x09 || b == 0x0A || b == 0x0D }
    @inlinable static func isDigit(_ b: UInt8) -> Bool { b >= 0x30 && b <= 0x39 }
    @inlinable static func isDigit1to9(_ b: UInt8) -> Bool { b >= 0x31 && b <= 0x39 }
    @inlinable static func isHexDigit(_ b: UInt8) -> Bool {
        isDigit(b) || (b >= 0x41 && b <= 0x46) || (b >= 0x61 && b <= 0x66)
    }

    /// Numeric value of a hex-digit byte (0–15); callers guard with
    /// ``isHexDigit(_:)`` first, so a non-hex byte yields 0 defensively.
    @inlinable static func hexValue(_ b: UInt8) -> Int {
        switch b {
        case 0x30...0x39: return Int(b - 0x30)
        case 0x41...0x46: return Int(b - 0x41) + 10
        case 0x61...0x66: return Int(b - 0x61) + 10
        default: return 0
        }
    }
}
