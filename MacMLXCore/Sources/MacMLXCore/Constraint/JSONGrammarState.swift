// Copyright © 2026 macMLX. English comments only.

/// A byte-level pushdown automaton for RFC 8259 JSON (Track C, structured
/// output — the `response_format: {"type":"json_object"}` constraint).
///
/// The automaton is the correctness core of constrained decoding: at every
/// generation step the ``JSONConstraintProcessor`` masks any token whose bytes
/// this automaton would reject from the current state, so the model can only
/// ever emit a byte sequence that stays on a path to a complete, well-formed
/// JSON document.
///
/// ## Why byte-level
/// LLM tokenizers are byte-level (GPT-2 / Qwen BPE) — a single token is an
/// arbitrary byte string that can straddle JSON lexical boundaries (`":`,
/// `,{"`, `true`, a fragment of a UTF-8 scalar inside a string). JSON's grammar
/// is itself defined over bytes, so a byte-level automaton is the only shape
/// that classifies such tokens exactly and without a tokenizer round-trip.
///
/// ## Value semantics
/// The whole state (nesting stack + lexical mode) is a value type, so token
/// classification is a pure fold — ``walk(_:)`` returns the resulting state or
/// `nil` (rejected) without mutating the receiver. The one advancing edge (the
/// token actually sampled) is committed by the processor via ``walk(_:)`` +
/// assignment. No reference state, no hidden mutation: trivially unit-testable
/// with zero MLX / tokenizer dependencies.
///
/// ## Completion
/// ``isComplete`` is the accept predicate: a full top-level value has been
/// parsed and nothing but trailing whitespace could follow. Only in that state
/// is the end-of-sequence token permitted; until then EOS is masked, which is
/// what forces the model to close every brace and bracket before it may stop.
public struct JSONGrammarState: Equatable, Sendable {

    /// A currently-open structural container, tracked on the nesting stack so
    /// the automaton knows whether a `,` / `}` / `]` is legal and what must
    /// follow it.
    @usableFromInline
    enum Container: Equatable, Sendable {
        case object
        case array
    }

    /// The lexical expectation at the current byte position. Every transition
    /// in ``advance(_:)`` is a function of `(mode, byte)` plus, for the
    /// structural closers, the top of ``stack``.
    @usableFromInline
    enum Mode: Equatable, Sendable {
        /// A value is strictly required next (top-level start, after `:`, or
        /// after `,` inside an array). No container-close is legal here.
        case awaitingValue
        /// Right after `[` — a value OR the array close `]` (empty array).
        case arrayExpectValueOrClose
        /// A value has just completed; `,` / container-close / whitespace or,
        /// at the top level, end-of-input may follow. The legal set depends on
        /// the top of ``stack``.
        case afterValue
        /// Right after `{` — a key string OR the object close `}` (empty object).
        case objectExpectKeyOrClose
        /// After `,` inside an object — a key string is strictly required (no
        /// trailing comma).
        case objectExpectKey
        /// After a key string closed — the `:` separator is required.
        case objectExpectColon

        /// Inside a string body. `key` distinguishes an object key (which must
        /// be followed by `:`) from a value string (followed by `afterValue`).
        case string(key: Bool)
        /// Just consumed `\` inside a string — an escape char must follow.
        case stringEscape(key: Bool)
        /// Inside a `\u` escape — `digitsSeen` of the 4 hex digits consumed, with
        /// the running code-unit `value`. `expectingLow` marks the SECOND `\u` of
        /// a surrogate pair, whose value must be a low surrogate (DC00–DFFF).
        case stringUnicode(key: Bool, digitsSeen: Int, value: Int, expectingLow: Bool)
        /// Just consumed a high surrogate `\uD800–DBFF`; RFC 8259 is lenient but
        /// `JSONSerialization` rejects an unpaired surrogate, so a low surrogate
        /// MUST follow — the only legal next byte is `\`.
        case stringHighSurrogateBackslash(key: Bool)
        /// Consumed the `\` after a high surrogate; the only legal next byte is
        /// `u`, opening the low-surrogate `\u` escape.
        case stringHighSurrogateU(key: Bool)

        /// Consumed `-`; the first integer digit is required.
        case numberAfterMinus
        /// Consumed a leading `0`; no further integer digit is legal (only
        /// `.` / `e` / terminate).
        case numberAfterLeadingZero
        /// Consumed integer digits `[1-9][0-9]*`; more digits / frac / exp /
        /// terminate.
        case numberIntDigits
        /// Consumed `.`; the first fraction digit is required.
        case numberAfterDot
        /// Consumed fraction digits; more digits / exp / terminate.
        case numberFracDigits
        /// Consumed `e` / `E`; a sign or the first exponent digit is required.
        case numberAfterExp
        /// Consumed an exponent sign; the first exponent digit is required.
        case numberAfterExpSign
        /// Consumed exponent digits; more digits / terminate.
        case numberExpDigits

        /// Matching the remaining bytes of a `true` / `false` / `null` literal.
        /// Empty `remaining` never occurs as a stored mode (it collapses to
        /// `afterValue` the instant the last byte matches).
        case literal(remaining: [UInt8])
    }

    /// Open containers, outermost first. Empty at the top level.
    @usableFromInline var stack: [Container]
    /// The current lexical expectation.
    @usableFromInline var mode: Mode
    /// Hard cap on nesting depth. A `{` / `[` that would exceed it is rejected
    /// (masked), bounding both memory and the model's ability to run away into
    /// pathological nesting. `stack.count` never exceeds this.
    public let maxDepth: Int

    /// A fresh automaton positioned before the top-level value.
    ///
    /// - Parameter maxDepth: maximum structural nesting (default 64). Values
    ///   below 1 are clamped to 1 so at least a flat container is always
    ///   representable.
    public init(maxDepth: Int = 64) {
        self.stack = []
        self.mode = .awaitingValue
        self.maxDepth = Swift.max(1, maxDepth)
    }

    /// Whether a complete top-level JSON value has been parsed and only
    /// trailing whitespace could legally follow — the automaton's accept
    /// state. The end-of-sequence token is permitted only here.
    ///
    /// A bare top-level number is "complete" while still in a terminal number
    /// sub-state (no terminator byte has arrived to collapse it to
    /// ``Mode/afterValue``), so those sub-states count as accepting at the top
    /// level too.
    @inlinable
    public var isComplete: Bool {
        guard stack.isEmpty else { return false }
        switch mode {
        case .afterValue,
             .numberAfterLeadingZero, .numberIntDigits,
             .numberFracDigits, .numberExpDigits:
            return true
        default:
            return false
        }
    }

    /// A short description of the current lexical position, for diagnostics
    /// (e.g. the constraint processor's "no legal token" log). Not a wire
    /// format — the reflected `mode`/stack values are for humans reading logs.
    public var diagnosticDescription: String {
        "json(mode: \(mode), depth: \(stack.count), complete: \(isComplete))"
    }

    // MARK: - Transitions

    /// Advance over one byte, returning the resulting state or `nil` when the
    /// byte is illegal from the current state.
    ///
    /// Pure: the receiver is never mutated. This is the single primitive the
    /// whole constraint is built on.
    @inlinable
    public func advanced(over byte: UInt8) -> JSONGrammarState? {
        var next = self
        return next.applyInPlace(byte) ? next : nil
    }

    /// Fold ``advanced(over:)`` over a byte sequence. Returns the final state,
    /// or `nil` if any byte is rejected. An empty sequence returns the receiver
    /// unchanged.
    @inlinable
    public func walk<S: Sequence>(_ bytes: S) -> JSONGrammarState? where S.Element == UInt8 {
        var state = self
        for byte in bytes {
            guard state.applyInPlace(byte) else { return nil }
        }
        return state
    }

    /// Mutating core of a single transition. Returns `false` (leaving `self`
    /// in an unspecified state the caller discards) when the byte is illegal.
    @usableFromInline
    mutating func applyInPlace(_ byte: UInt8) -> Bool {
        switch mode {
        case .awaitingValue:
            if Self.isWhitespace(byte) { return true }
            return startValue(byte)

        case .arrayExpectValueOrClose:
            if Self.isWhitespace(byte) { return true }
            if byte == Self.rBracket { return closeContainer(.array) }
            return startValue(byte)

        case .afterValue:
            return afterValueTransition(byte)

        case .objectExpectKeyOrClose:
            if Self.isWhitespace(byte) { return true }
            if byte == Self.quote { mode = .string(key: true); return true }
            if byte == Self.rBrace { return closeContainer(.object) }
            return false

        case .objectExpectKey:
            if Self.isWhitespace(byte) { return true }
            if byte == Self.quote { mode = .string(key: true); return true }
            return false

        case .objectExpectColon:
            if Self.isWhitespace(byte) { return true }
            if byte == Self.colon { mode = .awaitingValue; return true }
            return false

        case .string(let key):
            return stringTransition(byte, key: key)

        case .stringEscape(let key):
            return stringEscapeTransition(byte, key: key)

        case .stringUnicode(let key, let digitsSeen, let value, let expectingLow):
            return stringUnicodeTransition(
                byte, key: key, digitsSeen: digitsSeen, value: value, expectingLow: expectingLow)

        case .stringHighSurrogateBackslash(let key):
            guard byte == Self.backslash else { return false }
            mode = .stringHighSurrogateU(key: key)
            return true

        case .stringHighSurrogateU(let key):
            guard byte == Self.lowerU else { return false }
            mode = .stringUnicode(key: key, digitsSeen: 0, value: 0, expectingLow: true)
            return true

        case .numberAfterMinus:
            if byte == Self.zero { mode = .numberAfterLeadingZero; return true }
            if Self.isDigit1to9(byte) { mode = .numberIntDigits; return true }
            return false

        case .numberAfterLeadingZero:
            return numberTerminalTransition(byte, allowMoreIntDigits: false)

        case .numberIntDigits:
            return numberTerminalTransition(byte, allowMoreIntDigits: true)

        case .numberAfterDot:
            if Self.isDigit(byte) { mode = .numberFracDigits; return true }
            return false

        case .numberFracDigits:
            if Self.isDigit(byte) { mode = .numberFracDigits; return true }
            if byte == Self.lowerE || byte == Self.upperE { mode = .numberAfterExp; return true }
            return completeNumberThenReprocess(byte)

        case .numberAfterExp:
            if byte == Self.plus || byte == Self.minus { mode = .numberAfterExpSign; return true }
            if Self.isDigit(byte) { mode = .numberExpDigits; return true }
            return false

        case .numberAfterExpSign:
            if Self.isDigit(byte) { mode = .numberExpDigits; return true }
            return false

        case .numberExpDigits:
            if Self.isDigit(byte) { mode = .numberExpDigits; return true }
            return completeNumberThenReprocess(byte)

        case .literal(var remaining):
            guard let expected = remaining.first, expected == byte else { return false }
            remaining.removeFirst()
            mode = remaining.isEmpty ? .afterValue : .literal(remaining: remaining)
            return true
        }
    }

    // MARK: - Transition helpers

    /// Begin a value from `byte` in a position where a value is legal. Sets the
    /// appropriate mode (pushing a container for `{` / `[`). Returns `false`
    /// for any byte that cannot start a JSON value or that would exceed
    /// ``maxDepth``.
    @usableFromInline
    mutating func startValue(_ byte: UInt8) -> Bool {
        switch byte {
        case Self.lBrace:
            guard stack.count < maxDepth else { return false }
            stack.append(.object)
            mode = .objectExpectKeyOrClose
            return true
        case Self.lBracket:
            guard stack.count < maxDepth else { return false }
            stack.append(.array)
            mode = .arrayExpectValueOrClose
            return true
        case Self.quote:
            mode = .string(key: false)
            return true
        case Self.minus:
            mode = .numberAfterMinus
            return true
        case Self.zero:
            mode = .numberAfterLeadingZero
            return true
        case Self.lowerT:
            mode = .literal(remaining: Array("rue".utf8))
            return true
        case Self.lowerF:
            mode = .literal(remaining: Array("alse".utf8))
            return true
        case Self.lowerN:
            mode = .literal(remaining: Array("ull".utf8))
            return true
        default:
            if Self.isDigit1to9(byte) {
                mode = .numberIntDigits
                return true
            }
            return false
        }
    }

    /// Transition out of ``Mode/afterValue``, where the legal continuations
    /// depend on the innermost open container.
    @usableFromInline
    mutating func afterValueTransition(_ byte: UInt8) -> Bool {
        if Self.isWhitespace(byte) { return true }
        guard let top = stack.last else {
            // Top level: a complete document. Only whitespace (handled above)
            // may follow; anything else is trailing garbage.
            return false
        }
        switch top {
        case .array:
            if byte == Self.comma { mode = .awaitingValue; return true }
            if byte == Self.rBracket { return closeContainer(.array) }
            return false
        case .object:
            if byte == Self.comma { mode = .objectExpectKey; return true }
            if byte == Self.rBrace { return closeContainer(.object) }
            return false
        }
    }

    /// Pop the expected container off the stack and enter ``Mode/afterValue``
    /// (the closed container is itself a completed value). Rejects a close that
    /// does not match the innermost container.
    @usableFromInline
    mutating func closeContainer(_ expected: Container) -> Bool {
        guard stack.last == expected else { return false }
        stack.removeLast()
        mode = .afterValue
        return true
    }

    @usableFromInline
    mutating func stringTransition(_ byte: UInt8, key: Bool) -> Bool {
        switch byte {
        case Self.quote:
            mode = key ? .objectExpectColon : .afterValue
            return true
        case Self.backslash:
            mode = .stringEscape(key: key)
            return true
        default:
            // Raw control characters (U+0000–U+001F) are illegal unescaped in a
            // JSON string. Every other byte — including 0x80–0xFF UTF-8 lead /
            // continuation bytes — is legal string content.
            return byte >= 0x20
        }
    }

    @usableFromInline
    mutating func stringEscapeTransition(_ byte: UInt8, key: Bool) -> Bool {
        switch byte {
        case Self.quote, Self.backslash, Self.slash,
             Self.lowerB, Self.lowerF, Self.lowerN, Self.lowerR, Self.lowerT:
            mode = .string(key: key)
            return true
        case Self.lowerU:
            mode = .stringUnicode(key: key, digitsSeen: 0, value: 0, expectingLow: false)
            return true
        default:
            return false
        }
    }

    /// Consume one hex digit of a `\uXXXX` escape, tracking the accumulated
    /// code-unit `value` so surrogate pairing can be enforced on the 4th digit.
    ///
    /// RFC 8259 permits unpaired surrogates, but `JSONSerialization` (the parser
    /// the output must satisfy) rejects them, so this closes that
    /// "automaton-valid / parser-invalid" gap: a high surrogate (D800–DBFF) must
    /// be followed by a `\u` low surrogate (DC00–DFFF), and a lone low surrogate
    /// is rejected.
    @usableFromInline
    mutating func stringUnicodeTransition(
        _ byte: UInt8, key: Bool, digitsSeen: Int, value: Int, expectingLow: Bool
    ) -> Bool {
        guard Self.isHexDigit(byte) else { return false }
        let newValue = value * 16 + Self.hexValue(byte)
        let seen = digitsSeen + 1
        if seen < 4 {
            mode = .stringUnicode(key: key, digitsSeen: seen, value: newValue, expectingLow: expectingLow)
            return true
        }
        // Fourth digit: the code unit is complete.
        if expectingLow {
            // This `\u` closes a surrogate pair: it must be a low surrogate.
            guard (0xDC00...0xDFFF).contains(newValue) else { return false }
            mode = .string(key: key)
            return true
        }
        if (0xD800...0xDBFF).contains(newValue) {
            // High surrogate — a `\u` low surrogate must follow.
            mode = .stringHighSurrogateBackslash(key: key)
            return true
        }
        if (0xDC00...0xDFFF).contains(newValue) {
            // Unpaired low surrogate — invalid UTF-16, rejected by the parser.
            return false
        }
        // Ordinary BMP scalar.
        mode = .string(key: key)
        return true
    }

    /// Shared handling for the two integer terminal sub-states
    /// (``Mode/numberAfterLeadingZero`` and ``Mode/numberIntDigits``): fraction,
    /// exponent, optional further integer digits, or termination.
    @usableFromInline
    mutating func numberTerminalTransition(_ byte: UInt8, allowMoreIntDigits: Bool) -> Bool {
        if allowMoreIntDigits, Self.isDigit(byte) {
            mode = .numberIntDigits
            return true
        }
        if byte == Self.dot { mode = .numberAfterDot; return true }
        if byte == Self.lowerE || byte == Self.upperE { mode = .numberAfterExp; return true }
        return completeNumberThenReprocess(byte)
    }

    /// The current (terminal) number has ended: collapse to ``Mode/afterValue``
    /// and re-dispatch `byte` there. Used when a number is followed directly by
    /// a structural byte or whitespace (`{"a":1,"b":2}`, `[1,2]`, `12 `).
    /// Exactly one level of re-dispatch — `afterValue` never routes back here.
    @usableFromInline
    mutating func completeNumberThenReprocess(_ byte: UInt8) -> Bool {
        mode = .afterValue
        return afterValueTransition(byte)
    }

    // MARK: - Byte class helpers

    @usableFromInline static let quote: UInt8 = 0x22       // "
    @usableFromInline static let backslash: UInt8 = 0x5C   // \
    @usableFromInline static let slash: UInt8 = 0x2F       // /
    @usableFromInline static let colon: UInt8 = 0x3A       // :
    @usableFromInline static let comma: UInt8 = 0x2C       // ,
    @usableFromInline static let lBrace: UInt8 = 0x7B      // {
    @usableFromInline static let rBrace: UInt8 = 0x7D      // }
    @usableFromInline static let lBracket: UInt8 = 0x5B    // [
    @usableFromInline static let rBracket: UInt8 = 0x5D    // ]
    @usableFromInline static let dot: UInt8 = 0x2E         // .
    @usableFromInline static let plus: UInt8 = 0x2B        // +
    @usableFromInline static let minus: UInt8 = 0x2D       // -
    @usableFromInline static let zero: UInt8 = 0x30        // 0
    @usableFromInline static let lowerE: UInt8 = 0x65      // e
    @usableFromInline static let upperE: UInt8 = 0x45      // E
    @usableFromInline static let lowerB: UInt8 = 0x62      // b
    @usableFromInline static let lowerF: UInt8 = 0x66      // f
    @usableFromInline static let lowerN: UInt8 = 0x6E      // n
    @usableFromInline static let lowerR: UInt8 = 0x72      // r
    @usableFromInline static let lowerT: UInt8 = 0x74      // t
    @usableFromInline static let lowerU: UInt8 = 0x75      // u

    @inlinable
    static func isWhitespace(_ byte: UInt8) -> Bool {
        byte == 0x20 || byte == 0x09 || byte == 0x0A || byte == 0x0D
    }

    @inlinable
    static func isDigit(_ byte: UInt8) -> Bool {
        byte >= 0x30 && byte <= 0x39
    }

    @inlinable
    static func isDigit1to9(_ byte: UInt8) -> Bool {
        byte >= 0x31 && byte <= 0x39
    }

    @inlinable
    static func isHexDigit(_ byte: UInt8) -> Bool {
        isDigit(byte)
            || (byte >= 0x41 && byte <= 0x46)   // A–F
            || (byte >= 0x61 && byte <= 0x66)   // a–f
    }

    /// Numeric value of a hex-digit byte (0–15). Callers guard with
    /// ``isHexDigit(_:)`` first; a non-hex byte yields 0 defensively.
    @inlinable
    static func hexValue(_ byte: UInt8) -> Int {
        switch byte {
        case 0x30...0x39: return Int(byte - 0x30)        // 0–9
        case 0x41...0x46: return Int(byte - 0x41) + 10   // A–F
        case 0x61...0x66: return Int(byte - 0x61) + 10   // a–f
        default: return 0
        }
    }
}
