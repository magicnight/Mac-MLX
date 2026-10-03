// Copyright © 2026 macMLX. English comments only.

/// The value constraint at one position of the supported JSON-schema subset
/// (Track C — C2): an object member or an array item, at any depth.
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
    /// `{"type":"boolean"}` — `true` or `false`.
    case boolean
    /// `{"type":"string","enum":[…]}` — exactly one of the given string
    /// literals. The list is non-empty (guaranteed by the compiler). A string
    /// `const` compiles to a one-value enum.
    case stringEnum([String])
    /// A nested object: inline `properties`, or a resolved `$ref`.
    case object(JSONSchemaObject)
    /// An array whose every element is `items`, with between `minItems` and
    /// `maxItems` elements; `maxItems == nil` means unbounded. The compiler
    /// guarantees `0 <= minItems <= (maxItems ?? .max)`.
    indirect case array(items: SchemaValueType, minItems: Int, maxItems: Int?)
}
