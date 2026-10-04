// Copyright © 2026 macMLX. English comments only.

/// The structured-output constraint attached to a ``GenerateRequest`` (Track C).
///
/// Mirrors the two OpenAI `response_format` shapes macMLX supports:
///
///  - ``jsonObject`` ⇄ `{"type":"json_object"}` — output must be some
///    well-formed JSON document (C1).
///  - ``jsonSchema(_:)`` ⇄ `{"type":"json_schema", …}` — output must conform to
///    the compiled schema subset (C2): a root value of any supported type.
///
/// Decoding from the wire (and rejecting unsupported schema features with a 400)
/// is ``ResponseFormatDecoder``'s job; this type is the already-validated,
/// engine-facing result. It is `Codable`/`Hashable`/`Sendable` so it rides
/// inside `GenerateRequest` without disturbing that struct's conformances.
public enum ResponseFormat: Equatable, Hashable, Sendable, Codable {
    /// Constrain output to any well-formed JSON value.
    case jsonObject
    /// Constrain output to the given compiled schema: the root value's type,
    /// an object, an array or a scalar.
    case jsonSchema(SchemaValueType)
}
