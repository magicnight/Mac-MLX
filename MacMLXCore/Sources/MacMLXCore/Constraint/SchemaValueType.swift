// Copyright © 2026 macMLX. English comments only.

/// The value constraint for a single property in the supported JSON-schema
/// subset (Track C — C2).
///
/// The subset is deliberately small and enforced exactly: anything outside it
/// is rejected at compile time with a 400 rather than silently downgraded (see
/// ``ResponseFormatDecoder``). Nested objects and arrays are intentionally NOT
/// members — a request that asks for them is an explicit
/// `unsupported schema feature` error.
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
    /// literals. The list is non-empty (guaranteed by the compiler).
    case stringEnum([String])
}
