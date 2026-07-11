// Copyright © 2026 macMLX. English comments only.

/// A compiled top-level object schema — the whole of the supported JSON-schema
/// subset (Track C — C2).
///
/// Produced by ``ResponseFormatDecoder`` from an OpenAI
/// `response_format: {"type":"json_schema", …}` body, and consumed at decode
/// time by ``SchemaConstraintState`` to constrain generation to an object that:
///
///  - opens with `{` and closes with `}`;
///  - contains only keys drawn from ``properties`` (no additional keys);
///  - contains each key at most once;
///  - contains every name in ``required``; and
///  - gives each present key a value of its declared ``SchemaValueType``.
///
/// Keys may appear in any order (JSON objects are unordered), so the runtime
/// automaton tracks the set of already-emitted keys rather than a fixed
/// sequence.
public struct JSONSchemaObject: Equatable, Hashable, Sendable, Codable {

    /// One declared property: its wire name and value constraint.
    public struct Property: Equatable, Hashable, Sendable, Codable {
        public let name: String
        public let type: SchemaValueType

        public init(name: String, type: SchemaValueType) {
            self.name = name
            self.type = type
        }
    }

    /// The declared properties, in schema declaration order. Order is retained
    /// only for stable diagnostics; it does NOT constrain key order on the wire.
    public let properties: [Property]

    /// The names that MUST be present. Guaranteed by the compiler to be a subset
    /// of ``properties`` names.
    public let required: [String]

    public init(properties: [Property], required: [String]) {
        self.properties = properties
        self.required = required
    }

    /// The property declared under `name`, if any.
    public func property(named name: String) -> Property? {
        properties.first { $0.name == name }
    }
}
