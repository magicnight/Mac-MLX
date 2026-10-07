// Copyright © 2026 macMLX. English comments only.

/// A decimal number as a schema bound or a value the automaton reads: exact,
/// with at most 19 significant digits and at most 19 decimals.
///
/// The value is `±mantissa / 10^scale`. The automaton compares values of this
/// form without rounding (two of them cross-multiplied fit in 128 bits), so a
/// bound means the decimal number the schema wrote — `0.3` is three tenths,
/// not the nearest double — and a generated number is judged by the digits
/// the model wrote. Zero has no sign and no trailing zeros are kept, so equal
/// values are equal structurally.
public struct SchemaDecimal: Hashable, Sendable, Codable, Comparable, CustomStringConvertible {

    /// The most significant digits a value may have, and the most decimals.
    public static let maximumDigits = 19

    /// `10^19`: every mantissa is below it.
    @usableFromInline static let limit: UInt64 = 10_000_000_000_000_000_000

    /// `10^0 … 10^19`.
    @usableFromInline static let powersOfTen: [UInt64] = {
        var powers: [UInt64] = [1]
        for _ in 0..<19 { powers.append(powers[powers.count - 1] * 10) }
        return powers
    }()

    @usableFromInline let negative: Bool
    @usableFromInline let mantissa: UInt64
    @usableFromInline let scale: UInt8

    /// The normalised form: `mantissa` below `limit` with no trailing zero
    /// past the point, `scale` within the limit, a zero never negative.
    private init(normalisedNegative negative: Bool, mantissa: UInt64, scale: UInt8) {
        self.negative = negative && mantissa != 0
        self.mantissa = mantissa
        self.scale = scale
    }

    /// `nil` when the magnitude needs more than 19 digits or decimals.
    @usableFromInline
    init?(negative: Bool, mantissa: UInt64, scale: Int) {
        var mantissa = mantissa
        var scale = scale
        while scale > 0, mantissa % 10 == 0 {
            mantissa /= 10
            scale -= 1
        }
        guard mantissa < Self.limit, scale >= 0, scale <= Self.maximumDigits else { return nil }
        self.init(normalisedNegative: negative, mantissa: mantissa, scale: UInt8(scale))
    }

    public init(_ value: Int) {
        // `Int.min`'s magnitude is below 10^19.
        self.init(normalisedNegative: value < 0, mantissa: UInt64(value.magnitude), scale: 0)
    }

    /// The decimal number `value` prints as (its shortest round-trip form), or
    /// `nil` when it is not finite or needs more than 19 digits or decimals.
    public init?(_ value: Double) {
        guard value.isFinite else { return nil }
        self.init(parsing: "\(value)")
    }

    /// A decimal literal — an optional sign, digits, an optional fraction, an
    /// optional exponent — or `nil` when malformed or beyond the limits.
    public init?<S: StringProtocol>(parsing text: S) {
        var bytes = Array(text.utf8)[...]
        var negative = false
        if bytes.first == UInt8(ascii: "-") {
            negative = true
            bytes = bytes.dropFirst()
        }
        var digits: [UInt8] = []
        var decimals = 0
        var sawDot = false
        var sawDigit = false
        while let byte = bytes.first, byte != UInt8(ascii: "e"), byte != UInt8(ascii: "E") {
            bytes = bytes.dropFirst()
            if byte == UInt8(ascii: ".") {
                guard !sawDot else { return nil }
                sawDot = true
                continue
            }
            guard byte >= UInt8(ascii: "0"), byte <= UInt8(ascii: "9") else { return nil }
            sawDigit = true
            if sawDot { decimals += 1 }
            if digits.isEmpty, byte == UInt8(ascii: "0") { continue }   // leading zeros carry nothing
            digits.append(byte)
        }
        guard sawDigit else { return nil }
        var exponent = 0
        if bytes.first == UInt8(ascii: "e") || bytes.first == UInt8(ascii: "E") {
            bytes = bytes.dropFirst()
            var exponentNegative = false
            if bytes.first == UInt8(ascii: "-") || bytes.first == UInt8(ascii: "+") {
                exponentNegative = bytes.first == UInt8(ascii: "-")
                bytes = bytes.dropFirst()
            }
            guard !bytes.isEmpty, bytes.allSatisfy({ $0 >= UInt8(ascii: "0") && $0 <= UInt8(ascii: "9") }) else { return nil }
            guard bytes.count <= 4 else { return nil }   // anything larger is out of range anyway
            for byte in bytes { exponent = exponent * 10 + Int(byte - UInt8(ascii: "0")) }
            if exponentNegative { exponent = -exponent }
        } else if !bytes.isEmpty {
            return nil
        }
        // value = digits × 10^(exponent − decimals); trailing zeros move into the exponent.
        var shift = exponent - decimals
        while let last = digits.last, last == UInt8(ascii: "0") {
            digits.removeLast()
            shift += 1
        }
        guard digits.count <= Self.maximumDigits else { return nil }
        var mantissa: UInt64 = 0
        for byte in digits { mantissa = mantissa * 10 + UInt64(byte - UInt8(ascii: "0")) }
        if mantissa == 0 { shift = 0 }
        if shift > 0 {
            guard shift <= Self.maximumDigits else { return nil }
            let (product, overflow) = mantissa.multipliedReportingOverflow(by: Self.powersOfTen[shift])
            guard !overflow else { return nil }
            mantissa = product
            shift = 0
        }
        self.init(negative: negative, mantissa: mantissa, scale: -shift)
    }

    /// Whether the value has no fractional part.
    public var isInteger: Bool { scale == 0 }

    /// The value as an `Int`, when it is an integer in range.
    public var integerValue: Int? {
        guard scale == 0 else { return nil }
        if negative {
            guard mantissa <= UInt64(Int.max) + 1 else { return nil }
            return mantissa == UInt64(Int.max) + 1 ? Int.min : -Int(mantissa)
        }
        guard mantissa <= UInt64(Int.max) else { return nil }
        return Int(mantissa)
    }

    /// The greatest integer not above the value, when it fits an `Int`.
    public var floor: Int? {
        let (quotient, remainder) = mantissa.quotientAndRemainder(dividingBy: Self.powersOfTen[Int(scale)])
        return negative ? Self.signed(-1, quotient + (remainder > 0 ? 1 : 0)) : Self.signed(1, quotient)
    }

    /// The least integer not below the value, when it fits an `Int`.
    public var ceiling: Int? {
        let (quotient, remainder) = mantissa.quotientAndRemainder(dividingBy: Self.powersOfTen[Int(scale)])
        return negative ? Self.signed(-1, quotient) : Self.signed(1, quotient + (remainder > 0 ? 1 : 0))
    }

    private static func signed(_ sign: Int, _ magnitude: UInt64) -> Int? {
        if sign < 0 {
            guard magnitude <= UInt64(Int.max) + 1 else { return nil }
            return magnitude == UInt64(Int.max) + 1 ? Int.min : -Int(magnitude)
        }
        guard magnitude <= UInt64(Int.max) else { return nil }
        return Int(magnitude)
    }

    /// The value with the opposite sign.
    @usableFromInline
    var negated: SchemaDecimal {
        SchemaDecimal(normalisedNegative: !negative, mantissa: mantissa, scale: scale)
    }

    /// The largest magnitude: 19 nines, the most a value can hold.
    @usableFromInline
    static func largest(scale: UInt8) -> SchemaDecimal {
        SchemaDecimal(normalisedNegative: false, mantissa: limit - 1, scale: scale)
    }

    public var description: String {
        var digits = String(mantissa)
        if scale > 0 {
            if digits.count <= Int(scale) {
                digits = String(repeating: "0", count: Int(scale) - digits.count + 1) + digits
            }
            digits.insert(".", at: digits.index(digits.endIndex, offsetBy: -Int(scale)))
        }
        return negative ? "-" + digits : digits
    }

    // MARK: Comparison

    /// `-1`, `0` or `1` as `a / 10^s` is below, equal to or above `b / 10^t`,
    /// exactly: the cross products fit in 128 bits.
    @usableFromInline
    static func compareMagnitudes(_ a: UInt64, scale s: UInt8, _ b: UInt64, scale t: UInt8) -> Int {
        let left = a.multipliedFullWidth(by: powersOfTen[Int(t)])
        let right = b.multipliedFullWidth(by: powersOfTen[Int(s)])
        if left.high != right.high { return left.high < right.high ? -1 : 1 }
        if left.low != right.low { return left.low < right.low ? -1 : 1 }
        return 0
    }

    @usableFromInline
    static func compare(_ lhs: SchemaDecimal, _ rhs: SchemaDecimal) -> Int {
        if lhs.negative != rhs.negative { return lhs.negative ? -1 : 1 }
        let magnitudes = compareMagnitudes(lhs.mantissa, scale: lhs.scale, rhs.mantissa, scale: rhs.scale)
        return lhs.negative ? -magnitudes : magnitudes
    }

    public static func < (lhs: SchemaDecimal, rhs: SchemaDecimal) -> Bool {
        compare(lhs, rhs) < 0
    }
}
