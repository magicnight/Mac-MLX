// Copyright © 2026 macMLX. English comments only.

/// The closed range an integer value must lie in: `minimum`, `maximum`,
/// `exclusiveMinimum` and `exclusiveMaximum` folded to integers (an exclusive
/// or fractional bound moved to the nearest integer inside it). A missing side
/// is unbounded.
public struct SchemaIntegerBounds: Equatable, Hashable, Sendable, Codable {
    public let minimum: Int?
    public let maximum: Int?

    /// `nil` when no integer lies between the bounds.
    public init?(minimum: Int?, maximum: Int?) {
        if let minimum, let maximum, minimum > maximum { return nil }
        self.minimum = minimum
        self.maximum = maximum
    }

    private enum CodingKeys: String, CodingKey {
        case minimum, maximum
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        guard let bounds = SchemaIntegerBounds(
            minimum: try container.decodeIfPresent(Int.self, forKey: .minimum),
            maximum: try container.decodeIfPresent(Int.self, forKey: .maximum))
        else {
            throw DecodingError.dataCorruptedError(forKey: .minimum, in: container, debugDescription: "no integer lies between the bounds")
        }
        self = bounds
    }
}
