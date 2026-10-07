// Copyright © 2026 macMLX. English comments only.

/// The range a number value must lie in: `minimum` / `exclusiveMinimum` and
/// `maximum` / `exclusiveMaximum`, each side closed or open. A missing side is
/// unbounded.
///
/// A bounded number is spelled as a plain decimal on the wire — digits, an
/// optional fraction, no exponent — of at most 19 significant digits and 19
/// decimals, so that the automaton can judge every digit exactly (see
/// ``SchemaDecimal``). That restricts spellings, never values: every number in
/// the range has such a spelling.
public struct SchemaNumberBounds: Equatable, Hashable, Sendable, Codable {
    public let minimum: SchemaDecimal?
    public let minimumIsExclusive: Bool
    public let maximum: SchemaDecimal?
    public let maximumIsExclusive: Bool

    /// `nil` when no number lies between the bounds.
    public init?(
        minimum: SchemaDecimal?, minimumIsExclusive: Bool = false,
        maximum: SchemaDecimal?, maximumIsExclusive: Bool = false
    ) {
        if let minimum, let maximum {
            if minimum > maximum { return nil }
            if minimum == maximum, minimumIsExclusive || maximumIsExclusive { return nil }
        }
        self.minimum = minimum
        self.minimumIsExclusive = minimum == nil ? false : minimumIsExclusive
        self.maximum = maximum
        self.maximumIsExclusive = maximum == nil ? false : maximumIsExclusive
    }
}
