// Copyright © 2026 macMLX. English comments only.

/// The compiled form of ``SchemaIntegerBounds`` or ``SchemaNumberBounds`` the
/// schema automaton reads a bounded number against, one decision per byte.
///
/// A number is read as a decimal prefix: a sign, then a mantissa `m` with `s`
/// decimals so far. Every digit the model may append keeps the value inside a
/// half-open interval that the prefix determines — `[m/10^s, (m+1)/10^s)` once
/// the fraction has begun, and `[m·10^k, (m+1)·10^k)` for every `k ≥ 0` while
/// integer digits may still follow — so a digit is legal exactly when one of
/// those intervals meets the range, and the number may end exactly when its
/// value lies in the range. Legal prefixes therefore always complete: the
/// automaton has no dead ends inside a bounded number.
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
    init(_ bounds: SchemaIntegerBounds) {
        lower = bounds.minimum.map(SchemaDecimal.init)
        lowerOpen = false
        upper = bounds.maximum.map(SchemaDecimal.init)
        upperOpen = false
        integersOnly = true
    }

    @usableFromInline
    init(_ bounds: SchemaNumberBounds) {
        lower = bounds.minimum
        lowerOpen = bounds.minimumIsExclusive
        upper = bounds.maximum
        upperOpen = bounds.maximumIsExclusive
        integersOnly = false
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

    /// Whether a prefix with the given sign, mantissa and decimals, in the
    /// given phase, can still become a value in the range.
    @usableFromInline
    func admits(negative: Bool, mantissa: UInt64, scale: UInt8, phase: Phase) -> Bool {
        switch phase {
        case .loneZero:
            // The value is 0; a fraction may follow (not for integers).
            return meets(from: 0, to: integersOnly ? 0 : 1, upperClosed: integersOnly, scale: 0, negative: negative)
        case .afterDot, .fraction:
            return meets(from: mantissa, to: mantissa + 1, upperClosed: false, scale: scale, negative: negative)
        case .integerDigits:
            // [m·10^k, (m+1)·10^k) for k = 0, 1, … until the interval passes the
            // range or the 19-digit limit. The intervals are disjoint and rise
            // (or fall, for a negative prefix), so the first one past a bound
            // ends the search.
            var power: UInt64 = 1
            while true {
                let (low, lowOverflow) = mantissa.multipliedReportingOverflow(by: power)
                guard !lowOverflow, low < SchemaDecimal.limit else { return false }
                let (high, highOverflow) = (mantissa + 1).multipliedReportingOverflow(by: power)
                let highOrNil: UInt64? = highOverflow || high >= SchemaDecimal.limit ? nil : high
                if meets(from: low, to: highOrNil, upperClosed: false, scale: 0, negative: negative) { return true }
                if isPast(magnitude: low, negative: negative) { return false }
                power *= 10
                if power > SchemaDecimal.limit / 10 { return false }   // 10^19: no further digit fits
            }
        }
    }

    /// Whether `-` may start a value: some value at or below zero is in range.
    @usableFromInline
    var admitsNegativeSign: Bool {
        meets(from: 0, to: nil, upperClosed: false, scale: 0, negative: true)
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

    /// Whether the interval from `from / 10^scale` (closed) to `to / 10^scale`
    /// (open unless `upperClosed`) — mirrored to the negative side when
    /// `negative` — meets the range. A `to` of `nil`, or one past the 19-digit
    /// limit, means every magnitude up to the limit: the interval then ends,
    /// closed, at the largest value a mantissa can spell.
    private func meets(from: UInt64, to: UInt64?, upperClosed: Bool, scale: UInt8, negative: Bool) -> Bool {
        guard let near = SchemaDecimal(negative: false, mantissa: from, scale: Int(scale)) else { return false }
        var far = SchemaDecimal.largest(scale: scale)
        var farOpen = false
        if let to, let bounded = SchemaDecimal(negative: false, mantissa: to, scale: Int(scale)) {
            far = bounded
            farOpen = !upperClosed
        }
        var low: SchemaDecimal?, lowOpen: Bool, high: SchemaDecimal?, highOpen: Bool
        if negative {
            low = far.negated; lowOpen = farOpen
            high = near.negated; highOpen = false
        } else {
            low = near; lowOpen = false
            high = far; highOpen = farOpen
        }
        // Intersect with the range: the greater lower end, the lesser upper end,
        // an end open when the end that wins is open (both, when they tie).
        if let lower {
            if let current = low {
                let order = SchemaDecimal.compare(lower, current)
                if order > 0 { low = lower; lowOpen = lowerOpen }
                else if order == 0 { lowOpen = lowOpen || lowerOpen }
            } else {
                low = lower; lowOpen = lowerOpen
            }
        }
        if let upper {
            if let current = high {
                let order = SchemaDecimal.compare(upper, current)
                if order < 0 { high = upper; highOpen = upperOpen }
                else if order == 0 { highOpen = highOpen || upperOpen }
            } else {
                high = upper; highOpen = upperOpen
            }
        }
        guard let low, let high else { return true }
        let order = SchemaDecimal.compare(low, high)
        return order < 0 || (order == 0 && !lowOpen && !highOpen)
    }
}
