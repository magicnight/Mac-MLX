// Copyright © 2026 macMLX. English comments only.

/// The range a number value must lie in: `minimum` / `exclusiveMinimum` and
/// `maximum` / `exclusiveMaximum`, each side closed or open. A missing side is
/// unbounded.
///
/// A bounded number is spelled as a plain decimal on the wire — digits, an
/// optional fraction, no exponent — of at most 19 significant digits and 19
/// decimals, so that the automaton can judge every digit exactly (see
/// ``SchemaDecimal``). A value that needs more digits than that cannot be
/// produced, so a range holds at least one value it can spell, or it is not
/// a range: bounds that sit between two neighbouring values of that grid
/// (`exclusiveMinimum: 0, exclusiveMaximum: 1e-19`) are refused, as empty
/// ones are.
public struct SchemaNumberBounds: Equatable, Hashable, Sendable, Codable {
    public let minimum: SchemaDecimal?
    public let minimumIsExclusive: Bool
    public let maximum: SchemaDecimal?
    public let maximumIsExclusive: Bool

    /// `nil` when no number the automaton can spell lies between the bounds.
    public init?(
        minimum: SchemaDecimal?, minimumIsExclusive: Bool = false,
        maximum: SchemaDecimal?, maximumIsExclusive: Bool = false
    ) {
        let lowerOpen = minimum == nil ? false : minimumIsExclusive
        let upperOpen = maximum == nil ? false : maximumIsExclusive
        let range = NumberRange(lower: minimum, lowerOpen: lowerOpen, upper: maximum, upperOpen: upperOpen, integersOnly: false)
        guard range.holdsSomeValue else { return nil }
        self.minimum = minimum
        self.minimumIsExclusive = lowerOpen
        self.maximum = maximum
        self.maximumIsExclusive = upperOpen
    }

    private enum CodingKeys: String, CodingKey {
        case minimum, minimumIsExclusive, maximum, maximumIsExclusive
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        guard let bounds = SchemaNumberBounds(
            minimum: try container.decodeIfPresent(SchemaDecimal.self, forKey: .minimum),
            minimumIsExclusive: try container.decode(Bool.self, forKey: .minimumIsExclusive),
            maximum: try container.decodeIfPresent(SchemaDecimal.self, forKey: .maximum),
            maximumIsExclusive: try container.decode(Bool.self, forKey: .maximumIsExclusive))
        else {
            throw DecodingError.dataCorruptedError(forKey: .minimum, in: container, debugDescription: "no number lies between the bounds")
        }
        self = bounds
    }
}
