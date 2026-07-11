// Copyright © 2026 macMLX. English comments only.

/// A byte-level automaton that constrains generation to a specific
/// ``JSONSchemaObject`` (Track C — C2).
///
/// Where ``JSONGrammarState`` accepts *any* well-formed JSON, this accepts only
/// the flat object described by a compiled schema: keys drawn from the declared
/// set (each at most once, all required ones present, in any order) and each
/// value matching its declared ``SchemaValueType``. It is the runtime companion
/// to ``ResponseFormatDecoder`` and, like ``JSONGrammarState``, is a pure value
/// type — token classification is a non-mutating ``walk(_:)`` fold, MLX-free and
/// unit-testable.
public struct SchemaConstraintState: Equatable, Sendable {

    /// Typed value sub-automaton. Constructed by ``Phase/expectValue`` when the
    /// first value byte arrives, torn down when the value completes.
    @usableFromInline
    enum ValueState: Equatable, Sendable {
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
        // String enum: `candidates` are the enum values (as bytes) still
        // prefix-compatible with `accumulated`.
        case enumBody(accumulated: [UInt8], candidates: [[UInt8]])
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
        // Boolean literal: remaining bytes still to match.
        case literal(remaining: [UInt8])
    }

    /// The structural position within the object.
    @usableFromInline
    enum Phase: Equatable, Sendable {
        /// Before the object: whitespace then `{`.
        case beforeObject
        /// After `{` or after `,`. `afterComma` forbids the object close (no
        /// trailing comma).
        case expectKeyOrClose(afterComma: Bool)
        /// Inside a key string, matching declared names not yet emitted.
        case inKey(accumulated: [UInt8])
        /// A complete key was read; the `:` separator is required.
        case expectColon(key: String)
        /// After `:`; whitespace then the first byte of the typed value.
        case expectValue(key: String)
        /// Inside a typed value.
        case value(key: String, state: ValueState)
        /// A value completed; whitespace, `,`, or the object close `}`.
        case afterValue
        /// The object closed with all required keys present — the accept state.
        case done
    }

    @usableFromInline let schema: JSONSchemaObject
    @usableFromInline var emitted: Set<String>
    @usableFromInline var phase: Phase

    /// A fresh automaton positioned before the schema's object.
    public init(schema: JSONSchemaObject) {
        self.schema = schema
        self.emitted = []
        self.phase = .beforeObject
    }

    /// Whether the schema object has been fully and validly produced — the
    /// accept state, and the only state in which EOS is permitted.
    @inlinable
    public var isComplete: Bool { phase == .done }

    /// Advance over one byte, returning the resulting state or `nil` when the
    /// byte is illegal.
    @inlinable
    public func advanced(over byte: UInt8) -> SchemaConstraintState? {
        var next = self
        return next.applyInPlace(byte) ? next : nil
    }

    /// Fold ``advanced(over:)`` over a byte sequence; `nil` if any byte is
    /// rejected.
    @inlinable
    public func walk<S: Sequence>(_ bytes: S) -> SchemaConstraintState? where S.Element == UInt8 {
        var state = self
        for byte in bytes {
            guard state.applyInPlace(byte) else { return nil }
        }
        return state
    }

    /// A short description of the current structural position, for diagnostics
    /// (e.g. the constraint processor's "no legal token" log). Not a wire
    /// format — the reflected `phase`/`emitted` values are for humans.
    public var diagnosticDescription: String {
        "schema(phase: \(phase), emitted: \(emitted.sorted()), complete: \(isComplete))"
    }

    // MARK: - Transitions

    @usableFromInline
    mutating func applyInPlace(_ byte: UInt8) -> Bool {
        switch phase {
        case .beforeObject:
            if Self.isWhitespace(byte) { return true }
            if byte == Self.lBrace { phase = .expectKeyOrClose(afterComma: false); return true }
            return false

        case .expectKeyOrClose(let afterComma):
            return expectKeyOrClose(byte, afterComma: afterComma)

        case .inKey(let accumulated):
            return inKey(byte, accumulated: accumulated)

        case .expectColon(let key):
            if Self.isWhitespace(byte) { return true }
            if byte == Self.colon { phase = .expectValue(key: key); return true }
            return false

        case .expectValue(let key):
            if Self.isWhitespace(byte) { return true }
            return startValue(byte, key: key)

        case .value(let key, let state):
            return valueTransition(byte, key: key, state: state)

        case .afterValue:
            return afterValue(byte)

        case .done:
            return Self.isWhitespace(byte)
        }
    }

    @usableFromInline
    mutating func expectKeyOrClose(_ byte: UInt8, afterComma: Bool) -> Bool {
        if Self.isWhitespace(byte) { return true }
        if byte == Self.quote {
            guard !remainingKeys.isEmpty else { return false }
            phase = .inKey(accumulated: [])
            return true
        }
        if byte == Self.rBrace {
            guard !afterComma, requiredSatisfied else { return false }
            phase = .done
            return true
        }
        return false
    }

    @usableFromInline
    mutating func inKey(_ byte: UInt8, accumulated: [UInt8]) -> Bool {
        if byte == Self.quote {
            // Close the key only if it exactly equals a remaining declared name.
            guard let name = remainingKeys.first(where: { Array($0.utf8) == accumulated }) else {
                return false
            }
            phase = .expectColon(key: name)
            return true
        }
        // Otherwise the byte must extend the key toward some remaining name.
        let position = accumulated.count
        let stillViable = remainingKeys.contains { name in
            let bytes = Array(name.utf8)
            return bytes.count > position
                && bytes[position] == byte
                && Array(bytes[0..<position]) == accumulated
        }
        guard stillViable else { return false }
        phase = .inKey(accumulated: accumulated + [byte])
        return true
    }

    /// Enter the typed value machine for `key` from its first byte.
    @usableFromInline
    mutating func startValue(_ byte: UInt8, key: String) -> Bool {
        guard let type = schema.property(named: key)?.type else { return false }
        switch type {
        case .string:
            guard byte == Self.quote else { return false }
            phase = .value(key: key, state: .stringBody)
            return true
        case .stringEnum(let values):
            guard byte == Self.quote else { return false }
            phase = .value(key: key, state: .enumBody(accumulated: [], candidates: values.map { Array($0.utf8) }))
            return true
        case .number:
            guard let state = Self.numberStart(byte) else { return false }
            phase = .value(key: key, state: state)
            return true
        case .integer:
            guard let state = Self.integerStart(byte) else { return false }
            phase = .value(key: key, state: state)
            return true
        case .boolean:
            if byte == Self.lowerT { phase = .value(key: key, state: .literal(remaining: Array("rue".utf8))); return true }
            if byte == Self.lowerF { phase = .value(key: key, state: .literal(remaining: Array("alse".utf8))); return true }
            return false
        }
    }

    @usableFromInline
    mutating func valueTransition(_ byte: UInt8, key: String, state: ValueState) -> Bool {
        switch state {
        case .stringBody:
            if byte == Self.quote { return finishValue(key) }
            if byte == Self.backslash { phase = .value(key: key, state: .stringEscape); return true }
            guard byte >= 0x20 else { return false }
            phase = .value(key: key, state: .stringBody)
            return true

        case .stringEscape:
            switch byte {
            case Self.quote, Self.backslash, Self.slash,
                 Self.lowerB, Self.lowerF, Self.lowerN, Self.lowerR, Self.lowerT:
                phase = .value(key: key, state: .stringBody); return true
            case Self.lowerU:
                phase = .value(key: key, state: .stringUnicode(digitsSeen: 0, value: 0, expectingLow: false))
                return true
            default:
                return false
            }

        case .stringUnicode(let digitsSeen, let value, let expectingLow):
            return stringUnicodeValue(
                byte, key: key, digitsSeen: digitsSeen, value: value, expectingLow: expectingLow)

        case .stringHighSurrogateBackslash:
            guard byte == Self.backslash else { return false }
            phase = .value(key: key, state: .stringHighSurrogateU)
            return true

        case .stringHighSurrogateU:
            guard byte == Self.lowerU else { return false }
            phase = .value(key: key, state: .stringUnicode(digitsSeen: 0, value: 0, expectingLow: true))
            return true

        case .enumBody(let accumulated, let candidates):
            if byte == Self.quote {
                guard candidates.contains(accumulated) else { return false }
                return finishValue(key)
            }
            let position = accumulated.count
            let survivors = candidates.filter { $0.count > position && $0[position] == byte }
            guard !survivors.isEmpty else { return false }
            phase = .value(key: key, state: .enumBody(accumulated: accumulated + [byte], candidates: survivors))
            return true

        case .numberAfterMinus:
            if byte == Self.zero { phase = .value(key: key, state: .numberAfterLeadingZero); return true }
            if Self.isDigit1to9(byte) { phase = .value(key: key, state: .numberIntDigits); return true }
            return false

        case .numberAfterLeadingZero:
            return numberTerminal(byte, key: key, allowMoreIntDigits: false)

        case .numberIntDigits:
            return numberTerminal(byte, key: key, allowMoreIntDigits: true)

        case .numberAfterDot:
            if Self.isDigit(byte) { phase = .value(key: key, state: .numberFracDigits); return true }
            return false

        case .numberFracDigits:
            if Self.isDigit(byte) { phase = .value(key: key, state: .numberFracDigits); return true }
            if byte == Self.lowerE || byte == Self.upperE { phase = .value(key: key, state: .numberAfterExp); return true }
            return completeValueThenReprocess(byte, key: key)

        case .numberAfterExp:
            if byte == Self.plus || byte == Self.minus { phase = .value(key: key, state: .numberAfterExpSign); return true }
            if Self.isDigit(byte) { phase = .value(key: key, state: .numberExpDigits); return true }
            return false

        case .numberAfterExpSign:
            if Self.isDigit(byte) { phase = .value(key: key, state: .numberExpDigits); return true }
            return false

        case .numberExpDigits:
            if Self.isDigit(byte) { phase = .value(key: key, state: .numberExpDigits); return true }
            return completeValueThenReprocess(byte, key: key)

        case .intAfterMinus:
            if byte == Self.zero { phase = .value(key: key, state: .intAfterZero); return true }
            if Self.isDigit1to9(byte) { phase = .value(key: key, state: .intDigits); return true }
            return false

        case .intAfterZero:
            return completeValueThenReprocess(byte, key: key)

        case .intDigits:
            if Self.isDigit(byte) { phase = .value(key: key, state: .intDigits); return true }
            return completeValueThenReprocess(byte, key: key)

        case .literal(var remaining):
            guard let expected = remaining.first, expected == byte else { return false }
            remaining.removeFirst()
            if remaining.isEmpty { return finishValue(key) }
            phase = .value(key: key, state: .literal(remaining: remaining))
            return true
        }
    }

    /// Number terminal sub-states (`numberAfterLeadingZero` / `numberIntDigits`):
    /// fraction, exponent, optional further integer digits, or termination.
    @usableFromInline
    mutating func numberTerminal(_ byte: UInt8, key: String, allowMoreIntDigits: Bool) -> Bool {
        if allowMoreIntDigits, Self.isDigit(byte) { phase = .value(key: key, state: .numberIntDigits); return true }
        if byte == Self.dot { phase = .value(key: key, state: .numberAfterDot); return true }
        if byte == Self.lowerE || byte == Self.upperE { phase = .value(key: key, state: .numberAfterExp); return true }
        return completeValueThenReprocess(byte, key: key)
    }

    /// A number/integer value has ended: emit the key, move to `afterValue`, and
    /// re-dispatch the terminator byte there (one level only).
    @usableFromInline
    mutating func completeValueThenReprocess(_ byte: UInt8, key: String) -> Bool {
        guard finishValue(key) else { return false }
        return afterValue(byte)
    }

    /// Record `key` as emitted and move to `afterValue`. Always succeeds; typed
    /// to return `Bool` so it composes in the transition expressions.
    @usableFromInline
    mutating func finishValue(_ key: String) -> Bool {
        emitted.insert(key)
        phase = .afterValue
        return true
    }

    @usableFromInline
    mutating func afterValue(_ byte: UInt8) -> Bool {
        if Self.isWhitespace(byte) { return true }
        if byte == Self.comma { phase = .expectKeyOrClose(afterComma: true); return true }
        if byte == Self.rBrace {
            guard requiredSatisfied else { return false }
            phase = .done
            return true
        }
        return false
    }

    /// Consume one hex digit of a string value's `\uXXXX` escape, enforcing
    /// surrogate pairing so the output survives `JSONSerialization` (which, unlike
    /// RFC 8259, rejects unpaired surrogates): a high surrogate (D800–DBFF) must
    /// be followed by a `\u` low surrogate (DC00–DFFF); a lone low surrogate is
    /// rejected.
    @usableFromInline
    mutating func stringUnicodeValue(
        _ byte: UInt8, key: String, digitsSeen: Int, value: Int, expectingLow: Bool
    ) -> Bool {
        guard Self.isHexDigit(byte) else { return false }
        let newValue = value * 16 + Self.hexValue(byte)
        let seen = digitsSeen + 1
        if seen < 4 {
            phase = .value(key: key, state: .stringUnicode(digitsSeen: seen, value: newValue, expectingLow: expectingLow))
            return true
        }
        if expectingLow {
            guard (0xDC00...0xDFFF).contains(newValue) else { return false }
            phase = .value(key: key, state: .stringBody)
            return true
        }
        if (0xD800...0xDBFF).contains(newValue) {
            phase = .value(key: key, state: .stringHighSurrogateBackslash)
            return true
        }
        if (0xDC00...0xDFFF).contains(newValue) {
            return false   // unpaired low surrogate
        }
        phase = .value(key: key, state: .stringBody)
        return true
    }

    // MARK: - Schema helpers

    /// Declared property names not yet emitted — the only keys a new member may
    /// open, which also enforces "each key at most once".
    @usableFromInline
    var remainingKeys: [String] {
        schema.properties.map { $0.name }.filter { !emitted.contains($0) }
    }

    /// Whether every required name has been emitted (checked at the object close).
    @usableFromInline
    var requiredSatisfied: Bool {
        schema.required.allSatisfy { emitted.contains($0) }
    }

    @usableFromInline
    static func numberStart(_ byte: UInt8) -> ValueState? {
        if byte == minus { return .numberAfterMinus }
        if byte == zero { return .numberAfterLeadingZero }
        if isDigit1to9(byte) { return .numberIntDigits }
        return nil
    }

    @usableFromInline
    static func integerStart(_ byte: UInt8) -> ValueState? {
        if byte == minus { return .intAfterMinus }
        if byte == zero { return .intAfterZero }
        if isDigit1to9(byte) { return .intDigits }
        return nil
    }

    // MARK: - Byte constants / classes

    @usableFromInline static let quote: UInt8 = 0x22
    @usableFromInline static let backslash: UInt8 = 0x5C
    @usableFromInline static let slash: UInt8 = 0x2F
    @usableFromInline static let colon: UInt8 = 0x3A
    @usableFromInline static let comma: UInt8 = 0x2C
    @usableFromInline static let lBrace: UInt8 = 0x7B
    @usableFromInline static let rBrace: UInt8 = 0x7D
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
