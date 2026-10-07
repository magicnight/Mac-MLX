// Copyright © 2026 macMLX. English comments only.

/// The typed scalar sub-machine of ``SchemaConstraintState``: one string,
/// string-enum, number, integer or boolean value, read byte by byte.
///
/// It knows nothing about where the value sits. ``step(_:)`` only reports how
/// a byte relates to the value — it continues it, ends it, or the value had
/// already ended before it — and the container automaton decides what follows.
@usableFromInline
enum SchemaScalarState: Hashable, Sendable {
    // String
    case stringBody
    case stringEscape
    /// Inside a `\u` escape — `digitsSeen` of 4 hex digits and the running
    /// code-unit `value`. `expectingLow` marks the SECOND `\u` of a surrogate
    /// pair, whose value must be a low surrogate (DC00–DFFF).
    case stringUnicode(digitsSeen: Int, value: Int, expectingLow: Bool)
    /// Read a high surrogate `\uD800–DBFF`; a `\u` low surrogate must follow
    /// (else `JSONSerialization` rejects the unpaired surrogate). Only `\` is
    /// legal next.
    case stringHighSurrogateBackslash
    /// Read the `\` after a high surrogate; only `u` is legal next.
    case stringHighSurrogateU
    /// String enum of scalar node `node`: the match against its values.
    case enumBody(node: Int32, match: LiteralMatch)
    // Number (integer or fractional)
    case numberAfterMinus
    case numberAfterLeadingZero
    case numberIntDigits
    case numberAfterDot
    case numberFracDigits
    case numberAfterExp
    case numberAfterExpSign
    case numberExpDigits
    // Integer only
    case intAfterMinus
    case intAfterZero
    case intDigits
    // Number or integer within a range (scalar node `node`): judged digit by
    // digit against the range, see ``NumberRange``.
    /// Read `-`; a digit must follow.
    case boundedMinus(node: Int32)
    /// The sign, the mantissa and decimals read so far, and where the prefix
    /// stands. The mantissa stays below 10^19 and `scale` at most 19.
    case bounded(node: Int32, negative: Bool, mantissa: UInt64, scale: UInt8, phase: NumberRange.Phase)
    /// `true` / `false`: the first `matched` bytes of the literal have been read.
    case literal(isTrue: Bool, matched: Int)

    /// Whether the value in progress is a string (or an enum literal), where
    /// whitespace is data rather than formatting.
    var isInsideString: Bool {
        switch self {
        case .stringBody, .stringEscape, .stringUnicode, .stringHighSurrogateBackslash,
            .stringHighSurrogateU, .enumBody:
            return true
        default:
            return false
        }
    }

    /// Whether the value in progress is a number that could end here: a root
    /// number has no terminator byte, so these states are accepting at the
    /// root. A bounded number ends only on a value in its range.
    @usableFromInline
    func isCompleteNumber(program: SchemaProgram) -> Bool {
        switch self {
        case .numberAfterLeadingZero, .numberIntDigits, .numberFracDigits, .numberExpDigits,
            .intAfterZero, .intDigits:
            return true
        case .bounded(let node, let negative, let mantissa, let scale, let phase):
            guard phase != .afterDot, case .boundedNumber(let range) = program.scalars[Int(node)] else { return false }
            return Self.inRange(range, negative: negative, mantissa: mantissa, scale: scale)
        default:
            return false
        }
    }

    /// Whether the value a bounded prefix spells lies in `range`.
    @usableFromInline
    static func inRange(_ range: NumberRange, negative: Bool, mantissa: UInt64, scale: UInt8) -> Bool {
        guard let value = SchemaDecimal(negative: negative, mantissa: mantissa, scale: Int(scale)) else { return false }
        return range.contains(value)
    }

    /// How one byte relates to the value in progress.
    @usableFromInline
    enum Step: Sendable {
        /// The byte belongs to the value, which continues in the given state.
        case consumed(SchemaScalarState)
        /// The byte belongs to the value and ends it (a closing `"`, the last
        /// byte of a literal).
        case completed
        /// The value — a number — ended before this byte; the container
        /// handles the byte itself.
        case endedBefore
        /// The byte is illegal here.
        case rejected
    }

    @usableFromInline static let trueBytes: [UInt8] = Array("true".utf8)
    @usableFromInline static let falseBytes: [UInt8] = Array("false".utf8)

    /// The state after the first byte of a value of scalar node `node`, of
    /// kind `kind`, or `nil` when the byte cannot start one.
    @usableFromInline
    static func start(_ byte: UInt8, node: Int32, kind: SchemaProgram.ScalarKind) -> SchemaScalarState? {
        switch kind {
        case .string:
            return byte == SchemaBytes.quote ? .stringBody : nil
        case .stringEnum(let values):
            guard byte == SchemaBytes.quote else { return nil }
            return .enumBody(node: node, match: LiteralMatch(candidates: .all(count: values.count)))
        case .number:
            if byte == SchemaBytes.minus { return .numberAfterMinus }
            if byte == SchemaBytes.zero { return .numberAfterLeadingZero }
            if SchemaBytes.isDigit1to9(byte) { return .numberIntDigits }
            return nil
        case .integer:
            if byte == SchemaBytes.minus { return .intAfterMinus }
            if byte == SchemaBytes.zero { return .intAfterZero }
            if SchemaBytes.isDigit1to9(byte) { return .intDigits }
            return nil
        case .boundedNumber(let range):
            if byte == SchemaBytes.minus { return range.admitsNegativeSign ? .boundedMinus(node: node) : nil }
            return boundedFirstDigit(byte, node: node, negative: false, range: range)
        case .boolean:
            if byte == SchemaBytes.lowerT { return .literal(isTrue: true, matched: 1) }
            if byte == SchemaBytes.lowerF { return .literal(isTrue: false, matched: 1) }
            return nil
        }
    }

    /// Advance over one byte of the value. `program` holds the enum values.
    @usableFromInline
    func step(_ byte: UInt8, program: SchemaProgram) -> Step {
        switch self {
        case .stringBody:
            if byte == SchemaBytes.quote { return .completed }
            if byte == SchemaBytes.backslash { return .consumed(.stringEscape) }
            return byte >= 0x20 ? .consumed(.stringBody) : .rejected

        case .stringEscape:
            switch byte {
            case SchemaBytes.quote, SchemaBytes.backslash, SchemaBytes.slash,
                 SchemaBytes.lowerB, SchemaBytes.lowerF, SchemaBytes.lowerN, SchemaBytes.lowerR, SchemaBytes.lowerT:
                return .consumed(.stringBody)
            case SchemaBytes.lowerU:
                return .consumed(.stringUnicode(digitsSeen: 0, value: 0, expectingLow: false))
            default:
                return .rejected
            }

        case .stringUnicode(let digitsSeen, let value, let expectingLow):
            return Self.unicodeDigit(byte, digitsSeen: digitsSeen, value: value, expectingLow: expectingLow)

        case .stringHighSurrogateBackslash:
            return byte == SchemaBytes.backslash ? .consumed(.stringHighSurrogateU) : .rejected

        case .stringHighSurrogateU:
            guard byte == SchemaBytes.lowerU else { return .rejected }
            return .consumed(.stringUnicode(digitsSeen: 0, value: 0, expectingLow: true))

        case .enumBody(let node, let match):
            guard case .stringEnum(let values) = program.scalars[Int(node)] else { return .rejected }
            switch match.step(byte, literals: values) {
            case .continued(let next): return .consumed(.enumBody(node: node, match: next))
            case .completed: return .completed
            case .rejected: return .rejected
            }

        case .numberAfterMinus:
            if byte == SchemaBytes.zero { return .consumed(.numberAfterLeadingZero) }
            if SchemaBytes.isDigit1to9(byte) { return .consumed(.numberIntDigits) }
            return .rejected

        case .numberAfterLeadingZero:
            return Self.numberTerminal(byte, allowMoreIntDigits: false)

        case .numberIntDigits:
            return Self.numberTerminal(byte, allowMoreIntDigits: true)

        case .numberAfterDot:
            return SchemaBytes.isDigit(byte) ? .consumed(.numberFracDigits) : .rejected

        case .numberFracDigits:
            if SchemaBytes.isDigit(byte) { return .consumed(.numberFracDigits) }
            if byte == SchemaBytes.lowerE || byte == SchemaBytes.upperE { return .consumed(.numberAfterExp) }
            return .endedBefore

        case .numberAfterExp:
            if byte == SchemaBytes.plus || byte == SchemaBytes.minus { return .consumed(.numberAfterExpSign) }
            return SchemaBytes.isDigit(byte) ? .consumed(.numberExpDigits) : .rejected

        case .numberAfterExpSign:
            return SchemaBytes.isDigit(byte) ? .consumed(.numberExpDigits) : .rejected

        case .numberExpDigits:
            return SchemaBytes.isDigit(byte) ? .consumed(.numberExpDigits) : .endedBefore

        case .intAfterMinus:
            if byte == SchemaBytes.zero { return .consumed(.intAfterZero) }
            if SchemaBytes.isDigit1to9(byte) { return .consumed(.intDigits) }
            return .rejected

        case .intAfterZero:
            return .endedBefore

        case .intDigits:
            return SchemaBytes.isDigit(byte) ? .consumed(.intDigits) : .endedBefore

        case .boundedMinus(let node):
            guard case .boundedNumber(let range) = program.scalars[Int(node)],
                  let next = Self.boundedFirstDigit(byte, node: node, negative: true, range: range)
            else { return .rejected }
            return .consumed(next)

        case .bounded(let node, let negative, let mantissa, let scale, let phase):
            guard case .boundedNumber(let range) = program.scalars[Int(node)] else { return .rejected }
            if SchemaBytes.isDigit(byte) {
                // The mantissa and the decimals stay within the limit.
                let digit = UInt64(byte - SchemaBytes.zero)
                // `grown + digit` could itself overflow (a mantissa of
                // 1844674407370955161, the first 19 digits of 2^64, times ten
                // fits; plus 6 does not), so the limit moves to the other side.
                let (grown, overflow) = mantissa.multipliedReportingOverflow(by: 10)
                guard !overflow, grown < SchemaDecimal.limit - digit else { return .rejected }
                let next = grown + digit
                switch phase {
                case .integerDigits:
                    guard range.admits(negative: negative, mantissa: next, scale: 0, phase: .integerDigits) else { return .rejected }
                    return .consumed(.bounded(node: node, negative: negative, mantissa: next, scale: 0, phase: .integerDigits))
                case .afterDot, .fraction:
                    guard scale < SchemaDecimal.maximumDigits,
                          range.admits(negative: negative, mantissa: next, scale: scale + 1, phase: .fraction)
                    else { return .rejected }
                    return .consumed(.bounded(node: node, negative: negative, mantissa: next, scale: scale + 1, phase: .fraction))
                case .loneZero:
                    // A lone zero takes no more integer digits: JSON forbids
                    // leading zeros.
                    return .rejected
                }
            }
            if byte == SchemaBytes.dot {
                guard !range.integersOnly, phase == .loneZero || phase == .integerDigits,
                      range.admits(negative: negative, mantissa: mantissa, scale: 0, phase: .afterDot)
                else { return .rejected }
                return .consumed(.bounded(node: node, negative: negative, mantissa: mantissa, scale: 0, phase: .afterDot))
            }
            // No exponent: a bounded number is spelled plain. Any other byte
            // ends the number, which must then be in range.
            guard phase != .afterDot, Self.inRange(range, negative: negative, mantissa: mantissa, scale: scale) else {
                return .rejected
            }
            return .endedBefore

        case .literal(let isTrue, let matched):
            let bytes = isTrue ? Self.trueBytes : Self.falseBytes
            guard matched < bytes.count, bytes[matched] == byte else { return .rejected }
            return matched + 1 == bytes.count ? .completed : .consumed(.literal(isTrue: isTrue, matched: matched + 1))
        }
    }

    /// The state after the first digit of a bounded number, or `nil` when no
    /// value starting with it lies in the range.
    @usableFromInline
    static func boundedFirstDigit(_ byte: UInt8, node: Int32, negative: Bool, range: NumberRange) -> SchemaScalarState? {
        if byte == SchemaBytes.zero {
            guard range.admits(negative: negative, mantissa: 0, scale: 0, phase: .loneZero) else { return nil }
            return .bounded(node: node, negative: negative, mantissa: 0, scale: 0, phase: .loneZero)
        }
        guard SchemaBytes.isDigit1to9(byte) else { return nil }
        let digit = UInt64(byte - SchemaBytes.zero)
        guard range.admits(negative: negative, mantissa: digit, scale: 0, phase: .integerDigits) else { return nil }
        return .bounded(node: node, negative: negative, mantissa: digit, scale: 0, phase: .integerDigits)
    }

    /// Number terminal sub-states (`numberAfterLeadingZero` / `numberIntDigits`):
    /// fraction, exponent, optional further integer digits, or the end of the
    /// number before this byte.
    @usableFromInline
    static func numberTerminal(_ byte: UInt8, allowMoreIntDigits: Bool) -> Step {
        if allowMoreIntDigits, SchemaBytes.isDigit(byte) { return .consumed(.numberIntDigits) }
        if byte == SchemaBytes.dot { return .consumed(.numberAfterDot) }
        if byte == SchemaBytes.lowerE || byte == SchemaBytes.upperE { return .consumed(.numberAfterExp) }
        return .endedBefore
    }

    /// Consume one hex digit of a `\uXXXX` escape, enforcing surrogate pairing
    /// so the output survives `JSONSerialization` (which, unlike RFC 8259,
    /// rejects unpaired surrogates): a high surrogate (D800–DBFF) must be
    /// followed by a `\u` low surrogate (DC00–DFFF); a lone low surrogate is
    /// rejected.
    ///
    /// The check is progressive: a digit is rejected as soon as no completion of
    /// the escape could be legal. Checking only at the fourth digit left dead
    /// prefixes (`\uDC`–`\uDF` outside a pair, `\uD83D\u00`) that the automaton
    /// accepted but could not continue, so generation hit the no-legal-token path
    /// and was cut off.
    @usableFromInline
    static func unicodeDigit(_ byte: UInt8, digitsSeen: Int, value: Int, expectingLow: Bool) -> Step {
        guard SchemaBytes.isHexDigit(byte) else { return .rejected }
        let newValue = value * 16 + SchemaBytes.hexValue(byte)
        let seen = digitsSeen + 1
        // The second half of a pair must start `D`, and its second digit must
        // make it DC–DF; any other escape must not reach DC–DF, which could only
        // end as a lone low surrogate.
        if seen == 1, expectingLow, newValue != 0xD { return .rejected }
        if seen == 2, expectingLow != (0xDC...0xDF).contains(newValue) { return .rejected }
        if seen < 4 {
            return .consumed(.stringUnicode(digitsSeen: seen, value: newValue, expectingLow: expectingLow))
        }
        if expectingLow {
            return (0xDC00...0xDFFF).contains(newValue) ? .consumed(.stringBody) : .rejected
        }
        if (0xD800...0xDBFF).contains(newValue) { return .consumed(.stringHighSurrogateBackslash) }
        if (0xDC00...0xDFFF).contains(newValue) { return .rejected }   // unpaired low surrogate
        return .consumed(.stringBody)
    }
}
