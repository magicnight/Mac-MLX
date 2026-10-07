// Copyright © 2026 macMLX. English comments only.

/// The value constraint at one position of the supported JSON-schema subset
/// (Track C — C2): the root, an object member or an array item, at any depth.
///
/// The subset is deliberately small and enforced exactly: anything outside it
/// is rejected at compile time with a 400 rather than silently downgraded (see
/// ``ResponseFormatDecoder``).
public enum SchemaValueType: Equatable, Hashable, Sendable, Codable {
    /// `{"type":"string"}` — any JSON string.
    case string
    /// `{"type":"number"}` — any JSON number (integer or fractional).
    case number
    /// `{"type":"integer"}` — a JSON integer: optional sign then digits, with
    /// no fraction or exponent.
    case integer
    /// `{"type":"integer","minimum":…,"maximum":…}` — an integer within the
    /// bounds (`exclusiveMinimum` / `exclusiveMaximum` folded in).
    case boundedInteger(SchemaIntegerBounds)
    /// `{"type":"number","minimum":…,"maximum":…}` — a number within the
    /// bounds, spelled as a plain decimal (no exponent) of at most 19
    /// significant digits and 19 decimals, which every number in the range
    /// has (see ``SchemaNumberBounds``).
    case boundedNumber(SchemaNumberBounds)
    /// `{"type":"boolean"}` — `true` or `false`.
    case boolean
    /// `{"type":"string","enum":[…]}` — exactly one of the given string
    /// literals, any Unicode (a scalar outside ASCII spelled raw or escaped on
    /// the wire, see `LiteralMatch`). The list is
    /// non-empty (guaranteed by the compiler). A string `const` compiles to a
    /// one-value enum.
    case stringEnum([String])
    /// A nested object: inline `properties`, or a resolved `$ref`.
    case object(JSONSchemaObject)
    /// An array whose every element is `items`, with between `minItems` and
    /// `maxItems` elements; `maxItems == nil` means unbounded. The compiler
    /// guarantees `0 <= minItems <= (maxItems ?? .max)`.
    indirect case array(items: SchemaValueType, minItems: Int, maxItems: Int?)
}
