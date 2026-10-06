// Copyright © 2026 macMLX. English comments only.

/// The automaton's position inside a quoted literal — an object key or a
/// string-enum value — matched scalar by scalar against a candidate set of
/// ``SchemaLiteral``s.
///
/// A scalar outside ASCII may arrive as its raw UTF-8 bytes or as a JSON
/// escape: `\uXXXX` in either hex case, or a surrogate pair above the BMP. The
/// quote, the backslash and control characters, which JSON cannot carry raw,
/// arrive as their short escape or as `\uXXXX`. Every other ASCII scalar
/// arrives raw (``SchemaLiteral/mayBeEscaped(_:)`` says why). Every candidate
/// still consistent with the bytes read stays in `candidates`; the bytes read
/// are the same for all of them, so they are all at the same scalar index and
/// the same point within its spelling. The closing quote completes the
/// literal when a candidate has exactly `unit` scalars.
///
/// Accepting escapes is what lets a schema declare any string. An escape is
/// spelled with ASCII bytes, for which a tokenizer can be expected to have
/// single-byte tokens, so a literal the model cannot spell raw — a token cut
/// inside a multi-byte scalar is unusable under the constraint — is still
/// reachable, and a required key need not deadlock the automaton.
@usableFromInline
struct LiteralMatch: Hashable, Sendable {

    @usableFromInline
    enum Progress: Hashable, Sendable {
        /// Between scalars: the next byte starts one (raw, or `\`), or `"`
        /// closes the literal.
        case boundary
        /// `offset` bytes into the raw UTF-8 of scalar `unit`.
        case raw(offset: Int)
        /// Read `\`; `u` or a two-character escape letter follows.
        case backslash
        /// Reading the hex digits of a `\u` escape: `digits` seen so far, each
        /// checked against the survivors' code unit as it arrived, so nothing
        /// accumulates. `low` marks the second escape of a surrogate pair.
        case hex(digits: Int, low: Bool)
        /// Read a high surrogate; `\` must follow.
        case lowBackslash
        /// Read the `\` after a high surrogate; `u` must follow.
        case lowU
    }

    @usableFromInline
    enum Outcome: Sendable {
        case continued(LiteralMatch)
        /// The closing quote matched candidate `member` exactly.
        case completed(member: Int)
        case rejected
    }

    /// Scalars matched so far.
    @usableFromInline var unit: Int
    /// The literals still consistent with the bytes read. Past the boundary
    /// every candidate has more than `unit` scalars.
    @usableFromInline var candidates: PropertyMask
    @usableFromInline var progress: Progress

    @inlinable
    init(candidates: PropertyMask) {
        self.unit = 0
        self.candidates = candidates
        self.progress = .boundary
    }

    /// Advance over `byte`; `literals` are the candidate set's literals.
    @usableFromInline
    func step(_ byte: UInt8, literals: [SchemaLiteral]) -> Outcome {
        switch progress {
        case .boundary:
            if byte == SchemaBytes.quote {
                guard let member = candidates.first(where: { literals[$0].scalars.count == unit }) else {
                    return .rejected
                }
                return .completed(member: member)
            }
            if byte == SchemaBytes.backslash {
                let escapable = candidates.filtered { member in
                    let scalars = literals[member].scalars
                    return scalars.count > unit && SchemaLiteral.mayBeEscaped(scalars[unit])
                }
                return narrowed(to: escapable, progress: .backslash)
            }
            // The quote and the backslash were handled above and JSON forbids
            // raw control characters, so this is the lead byte of a raw scalar.
            guard byte >= 0x20 else { return .rejected }
            let survivors = candidates.filtered { member in
                let scalars = literals[member].scalars
                return scalars.count > unit && SchemaLiteral.utf8Byte(of: scalars[unit], at: 0) == byte
            }
            guard let first = survivors.first else { return .rejected }
            // Survivors share the lead byte, so they share the length.
            let length = SchemaLiteral.utf8Length(of: literals[first].scalars[unit])
            return length == 1 ? advanced(survivors) : narrowed(to: survivors, progress: .raw(offset: 1))

        case .raw(let offset):
            let survivors = candidates.filtered { member in
                SchemaLiteral.utf8Byte(of: literals[member].scalars[unit], at: offset) == byte
            }
            guard let first = survivors.first else { return .rejected }
            let length = SchemaLiteral.utf8Length(of: literals[first].scalars[unit])
            return offset + 1 == length
                ? advanced(survivors)
                : narrowed(to: survivors, progress: .raw(offset: offset + 1))

        case .backslash:
            if byte == SchemaBytes.lowerU {
                return narrowed(to: candidates, progress: .hex(digits: 0, low: false))
            }
            guard let scalar = SchemaLiteral.shortEscapeScalar(byte) else { return .rejected }
            return advanced(candidates.filtered { literals[$0].scalars[unit] == scalar })

        case .hex(let digits, let low):
            guard SchemaBytes.isHexDigit(byte) else { return .rejected }
            let digit = SchemaBytes.hexValue(byte)
            let shift = 4 * (3 - digits)
            let survivors = candidates.filtered { member in
                (SchemaLiteral.escapeUnit(of: literals[member].scalars[unit], low: low) >> shift) & 0xF == digit
            }
            guard let first = survivors.first else { return .rejected }
            let seen = digits + 1
            if seen < 4 {
                return narrowed(to: survivors, progress: .hex(digits: seen, low: low))
            }
            // Four digits spell one code unit, the same for every survivor.
            // A high surrogate (a scalar above the BMP) needs its low half.
            if !low, literals[first].scalars[unit] > 0xFFFF {
                return narrowed(to: survivors, progress: .lowBackslash)
            }
            return advanced(survivors)

        case .lowBackslash:
            return byte == SchemaBytes.backslash ? narrowed(to: candidates, progress: .lowU) : .rejected

        case .lowU:
            return byte == SchemaBytes.lowerU
                ? narrowed(to: candidates, progress: .hex(digits: 0, low: true))
                : .rejected
        }
    }

    /// Continue with `survivors` within the same scalar, or reject when none.
    @usableFromInline
    func narrowed(to survivors: PropertyMask, progress: Progress) -> Outcome {
        guard !survivors.isEmpty else { return .rejected }
        var next = self
        next.candidates = survivors
        next.progress = progress
        return .continued(next)
    }

    /// Scalar `unit` is complete for `survivors`; continue at the next
    /// boundary, or reject when none survived.
    @usableFromInline
    func advanced(_ survivors: PropertyMask) -> Outcome {
        guard !survivors.isEmpty else { return .rejected }
        var next = self
        next.unit += 1
        next.candidates = survivors
        next.progress = .boundary
        return .continued(next)
    }
}
