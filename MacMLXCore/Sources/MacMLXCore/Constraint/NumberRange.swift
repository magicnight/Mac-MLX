// Copyright © 2026 macMLX. English comments only.

/// The compiled form of ``SchemaIntegerBounds`` or ``SchemaNumberBounds`` the
/// schema automaton reads a bounded number against, one decision per byte.
///
/// A number is read as a decimal prefix: a sign, then a mantissa `m` with `s`
/// decimals so far. The values the prefix can still become form a grid, since
/// a value holds at most 19 significant digits and 19 decimals: with `d`
/// digits in `m`, `r = min(19 − s, 19 − d)` more may follow. Once the
/// fraction has begun the completions are `m/10^s + j/10^(s+r)` for
/// `0 ≤ j < 10^r`, a closed cell from `m/10^s` to `(m+1)/10^s − 1/10^(s+r)`;
/// while integer digits may still follow there is one such cell per count
/// `k ≤ 19 − d` of further integer digits. A digit is legal exactly when one
/// of its cells holds a value in the range, the decimal point when a digit
/// can still follow it, and the number may end exactly when its value lies
/// in the range. Legal prefixes therefore always complete — the automaton
/// has no dead ends inside a bounded number — given that the range holds a
/// value of the grid at all, which ``SchemaNumberBounds`` guarantees.
@usableFromInline
struct NumberRange: Hashable, Sendable {

    /// Where a bounded number's prefix stands.
    @usableFromInline
    enum Phase: UInt8, Hashable, Sendable {
        /// A single `0` has been read: no more integer digits may follow.
        case loneZero
        /// One or more integer digits (not a lone zero) have been read.
        case integerDigits
        /// The decimal point has been read; a digit must follow.
        case afterDot
        /// One or more fraction digits have been read.
        case fraction
    }

    @usableFromInline let lower: SchemaDecimal?
    @usableFromInline let lowerOpen: Bool
    @usableFromInline let upper: SchemaDecimal?
    @usableFromInline let upperOpen: Bool
    /// Integers only: no decimal point, so a value is complete after its
    /// integer digits. The bounds are integers (the compiler folds them).
    @usableFromInline let integersOnly: Bool

    @usableFromInline
    init(lower: SchemaDecimal?, lowerOpen: Bool, upper: SchemaDecimal?, upperOpen: Bool, integersOnly: Bool) {
        self.lower = lower
        self.lowerOpen = lowerOpen
        self.upper = upper
        self.upperOpen = upperOpen
        self.integersOnly = integersOnly
    }

    @usableFromInline
    init(_ bounds: SchemaIntegerBounds) {
        self.init(
            lower: bounds.minimum.map(SchemaDecimal.init), lowerOpen: false,
            upper: bounds.maximum.map(SchemaDecimal.init), upperOpen: false, integersOnly: true)
    }

    @usableFromInline
    init(_ bounds: SchemaNumberBounds) {
        self.init(
            lower: bounds.minimum, lowerOpen: bounds.minimumIsExclusive,
            upper: bounds.maximum, upperOpen: bounds.maximumIsExclusive, integersOnly: false)
    }

    /// Whether `value` lies in the range.
    @usableFromInline
    func contains(_ value: SchemaDecimal) -> Bool {
        if let lower {
            let order = SchemaDecimal.compare(value, lower)
            if order < 0 || (order == 0 && lowerOpen) { return false }
        }
        if let upper {
            let order = SchemaDecimal.compare(value, upper)
            if order > 0 || (order == 0 && upperOpen) { return false }
        }
        return true
    }

    /// The digits a prefix with mantissa `mantissa` and `scale` decimals may
    /// still take: the digit limit less the digits it has, or the decimal
    /// limit less its decimals, whichever is smaller.
    @usableFromInline
    static func remainingDigits(mantissa: UInt64, scale: UInt8) -> Int {
        Swift.min(SchemaDecimal.maximumDigits - Int(scale), SchemaDecimal.maximumDigits - SchemaDecimal.digits(of: mantissa))
    }

    /// Whether a prefix with the given sign, mantissa and decimals, in the
    /// given phase, can still become a value in the range.
    @usableFromInline
    func admits(negative: Bool, mantissa: UInt64, scale: UInt8, phase: Phase) -> Bool {
        switch phase {
        case .loneZero:
            // The value is 0; for a number any fraction may follow.
            if integersOnly { return meets(from: 0, to: 0, scale: 0, negative: negative) }
            return meets(from: 0, to: SchemaDecimal.limit - 1, scale: UInt8(SchemaDecimal.maximumDigits), negative: negative)
        case .afterDot, .fraction:
            let remaining = Self.remainingDigits(mantissa: mantissa, scale: scale)
            // The point needs a digit after it.
            if phase == .afterDot, remaining < 1 { return false }
            let stretch = SchemaDecimal.powersOfTen[remaining]
            return meets(
                from: mantissa * stretch, to: (mantissa + 1) * stretch - 1,
                scale: UInt8(Int(scale) + remaining), negative: negative)
        case .integerDigits:
            // One cell per count of further integer digits. The cells are
            // disjoint and rise (fall, for a negative prefix), so the first one
            // past a bound ends the search.
            let digits = SchemaDecimal.digits(of: mantissa)
            let spare = SchemaDecimal.maximumDigits - digits
            for k in 0...spare {
                let low = mantissa * SchemaDecimal.powersOfTen[k]
                let met: Bool
                if integersOnly {
                    met = meets(from: low, to: (mantissa + 1) * SchemaDecimal.powersOfTen[k] - 1, scale: 0, negative: negative)
                } else {
                    let stretch = SchemaDecimal.powersOfTen[spare]
                    met = meets(
                        from: mantissa * stretch, to: (mantissa + 1) * stretch - 1,
                        scale: UInt8(spare - k), negative: negative)
                }
                if met { return true }
                if isPast(magnitude: low, negative: negative) { return false }
            }
            return false
        }
    }

    /// Whether `-` may start a value: some negative value, or `-0`, is in range.
    @usableFromInline
    var admitsNegativeSign: Bool {
        if admits(negative: true, mantissa: 0, scale: 0, phase: .loneZero) { return true }
        return (1...9).contains { admits(negative: true, mantissa: $0, scale: 0, phase: .integerDigits) }
    }

    /// Whether some value the automaton can spell lies in the range: a walk
    /// that takes the first legal byte at every step, ending as soon as it
    /// may. Every legal prefix completes once the range holds a value of the
    /// grid, so the walk ends in one when there is one and dies when there
    /// is none (the bounds then sit between two neighbouring grid points).
    @usableFromInline
    var holdsSomeValue: Bool {
        var negative = false
        var mantissa: UInt64 = 0
        var scale: UInt8 = 0
        var phase: Phase
        if admits(negative: false, mantissa: 0, scale: 0, phase: .loneZero) {
            phase = .loneZero
        } else if let digit = (1...9).first(where: { admits(negative: false, mantissa: $0, scale: 0, phase: .integerDigits) }) {
            mantissa = UInt64(digit)
            phase = .integerDigits
        } else if admits(negative: true, mantissa: 0, scale: 0, phase: .loneZero) {
            negative = true
            phase = .loneZero
        } else if let digit = (1...9).first(where: { admits(negative: true, mantissa: $0, scale: 0, phase: .integerDigits) }) {
            negative = true
            mantissa = UInt64(digit)
            phase = .integerDigits
        } else {
            return false
        }
        for _ in 0..<(2 * SchemaDecimal.maximumDigits + 2) {
            if phase != .afterDot, let value = SchemaDecimal(negative: negative, mantissa: mantissa, scale: Int(scale)),
               contains(value) {
                return true
            }
            var stepped = false
            if phase != .loneZero {
                for digit in UInt64(0)...9 {
                    let next = mantissa * 10 + digit
                    guard next < SchemaDecimal.limit else { break }
                    let nextScale = phase == .integerDigits ? scale : scale + 1
                    let nextPhase: Phase = phase == .integerDigits ? .integerDigits : .fraction
                    if admits(negative: negative, mantissa: next, scale: nextScale, phase: nextPhase) {
                        mantissa = next; scale = nextScale; phase = nextPhase
                        stepped = true
                        break
                    }
                }
            }
            if !stepped, !integersOnly, phase == .loneZero || phase == .integerDigits,
               admits(negative: negative, mantissa: mantissa, scale: 0, phase: .afterDot) {
                phase = .afterDot
                stepped = true
            }
            if !stepped { return false }
        }
        return false
    }

    /// Whether the magnitude, with the sign, is already beyond the range on
    /// its own side (above the upper bound for a positive prefix, below the
    /// lower bound for a negative one): no larger magnitude can come back.
    private func isPast(magnitude: UInt64, negative: Bool) -> Bool {
        guard let value = SchemaDecimal(negative: negative, mantissa: magnitude, scale: 0) else { return true }
        if negative {
            guard let lower else { return false }
            return SchemaDecimal.compare(value, lower) < 0
        }
        guard let upper else { return false }
        return SchemaDecimal.compare(value, upper) > 0
    }

    /// Whether the closed cell from `from / 10^scale` to `to / 10^scale` —
    /// mirrored to the negative side when `negative` — holds a value in the
    /// range. The cell's ends are values of the grid, so a range end that
    /// ties with one is in the cell exactly when it is closed.
    private func meets(from: UInt64, to: UInt64, scale: UInt8, negative: Bool) -> Bool {
        guard let near = SchemaDecimal(negative: false, mantissa: from, scale: Int(scale)),
              let far = SchemaDecimal(negative: false, mantissa: to, scale: Int(scale))
        else { return false }
        var low: SchemaDecimal, lowOpen = false, high: SchemaDecimal, highOpen = false
        if negative {
            low = far.negated
            high = near.negated
        } else {
            low = near
            high = far
        }
        // Intersect with the range: the greater lower end, the lesser upper end,
        // an end open when the end that wins is open (either, when they tie).
        if let lower {
            let order = SchemaDecimal.compare(lower, low)
            if order > 0 { low = lower; lowOpen = lowerOpen }
            else if order == 0 { lowOpen = lowOpen || lowerOpen }
        }
        if let upper {
            let order = SchemaDecimal.compare(upper, high)
            if order < 0 { high = upper; highOpen = upperOpen }
            else if order == 0 { highOpen = highOpen || upperOpen }
        }
        let order = SchemaDecimal.compare(low, high)
        return order < 0 || (order == 0 && !lowOpen && !highOpen)
    }
}
