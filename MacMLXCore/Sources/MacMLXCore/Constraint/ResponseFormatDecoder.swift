// Copyright © 2026 macMLX. English comments only.

/// Compiles an OpenAI `response_format` value into a validated
/// ``ResponseFormat``, rejecting anything outside the supported subset with a
/// ``ResponseFormatError`` (which the server turns into a 400).
///
/// This is the single gate between untrusted request JSON and the decode-time
/// constraint. It is pure and MLX-free — it takes a ``JSONValue`` (already
/// parsed by the server) and returns a value type — so the full matrix of
/// accept / unsupported / invalid cases is unit-testable without a running
/// server or model.
///
/// ## Supported subset
///  - `{"type":"text"}` and an absent/`null` field → no constraint (`nil`).
///  - `{"type":"json_object"}` → ``ResponseFormat/jsonObject`` (C1).
///  - `{"type":"json_schema","json_schema":{"schema":{…}}}` where the inner
///    schema is a flat object of `string` / `number` / `integer` / `boolean` /
///    string-`enum` properties with an optional `required` list → C2.
///
/// Everything else (nested objects/arrays, combinators, `$ref`, non-object
/// roots, `additionalProperties: true`, …) is an explicit
/// ``ResponseFormatError/unsupportedFeature(_:)``.
public enum ResponseFormatDecoder {

    /// The ONLY property-schema keys we honor. This is a strict allow-list, not a
    /// blocklist: any other keyword (`pattern`, `minLength`, `maximum`, `format`,
    /// `properties`, `$ref`, combinators, …) is rejected with a 400 rather than
    /// silently ignored — a value constraint we cannot enforce must never be
    /// silently downgraded (see ``ResponseFormatError``). `type` and `enum` are
    /// enforced; `description` / `title` / `default` are purely annotative and
    /// accepted-but-ignored.
    private static let allowedPropertyKeys: Set<String> = [
        "type", "enum",
        "description", "title", "default",
    ]

    /// Decode the raw `response_format` field.
    ///
    /// - Parameter raw: the field value, or `nil` when the request omitted it.
    /// - Returns: the validated constraint, or `nil` when no constraint applies
    ///   (absent, `null`, or `{"type":"text"}`).
    /// - Throws: ``ResponseFormatError`` on any unsupported or malformed input.
    public static func decode(_ raw: JSONValue?) throws -> ResponseFormat? {
        guard let raw, raw != .null else { return nil }
        guard case .object(let root) = raw else {
            throw ResponseFormatError.invalidFormat("response_format must be an object")
        }
        guard let typeValue = root["type"] else {
            throw ResponseFormatError.invalidFormat("response_format.type is required")
        }
        guard case .string(let type) = typeValue else {
            throw ResponseFormatError.invalidFormat("response_format.type must be a string")
        }

        switch type {
        case "text":
            return nil
        case "json_object":
            return .jsonObject
        case "json_schema":
            return .jsonSchema(try compileJSONSchemaEnvelope(root))
        default:
            throw ResponseFormatError.unsupportedFeature("response_format type '\(type)'")
        }
    }

    /// Pull the inner schema out of the `{"json_schema":{"schema":{…}}}`
    /// envelope and compile it.
    private static func compileJSONSchemaEnvelope(
        _ root: [String: JSONValue]
    ) throws -> JSONSchemaObject {
        guard let envelopeValue = root["json_schema"] else {
            throw ResponseFormatError.invalidFormat("json_schema object is required")
        }
        guard case .object(let envelope) = envelopeValue else {
            throw ResponseFormatError.invalidFormat("json_schema must be an object")
        }
        guard let schemaValue = envelope["schema"] else {
            throw ResponseFormatError.invalidFormat("json_schema.schema object is required")
        }
        guard case .object(let schema) = schemaValue else {
            throw ResponseFormatError.invalidFormat("json_schema.schema must be an object")
        }
        return try compileObjectSchema(schema)
    }

    /// Compile the top-level object schema (the whole supported C2 surface).
    static func compileObjectSchema(_ schema: [String: JSONValue]) throws -> JSONSchemaObject {
        if let typeValue = schema["type"] {
            guard case .string(let type) = typeValue else {
                throw ResponseFormatError.invalidFormat("schema.type must be a string")
            }
            guard type == "object" else {
                throw ResponseFormatError.unsupportedFeature(
                    "top-level type '\(type)' (only 'object' is supported)")
            }
        }

        // We forbid additional properties structurally; honoring an explicit
        // `additionalProperties: true` would contradict that, so reject it.
        if let additional = schema["additionalProperties"], additional != .bool(false) {
            throw ResponseFormatError.unsupportedFeature("additionalProperties (only false is supported)")
        }

        guard let propertiesValue = schema["properties"] else {
            throw ResponseFormatError.invalidFormat("schema.properties object is required")
        }
        guard case .object(let properties) = propertiesValue else {
            throw ResponseFormatError.invalidFormat("schema.properties must be an object")
        }
        guard !properties.isEmpty else {
            throw ResponseFormatError.invalidFormat("schema.properties must declare at least one property")
        }

        // Sorted for a deterministic declaration order (diagnostics only — the
        // runtime automaton accepts keys in any order).
        var compiled: [JSONSchemaObject.Property] = []
        for name in properties.keys.sorted() {
            // The runtime key matcher compares literal UTF-8 bytes, so a declared
            // key the model could never spell would deadlock a `required` object
            // into the no-legal-token path — reject it up front (M2).
            try requireLiteralMatchable(name, role: "property key '\(name)'")
            guard let propertyValue = properties[name] else { continue }
            guard case .object(let property) = propertyValue else {
                throw ResponseFormatError.invalidFormat("property '\(name)' must be an object")
            }
            let type = try compilePropertyType(name: name, schema: property)
            compiled.append(JSONSchemaObject.Property(name: name, type: type))
        }

        var required: [String] = []
        if let requiredValue = schema["required"] {
            guard case .array(let entries) = requiredValue else {
                throw ResponseFormatError.invalidFormat("schema.required must be an array")
            }
            for entry in entries {
                guard case .string(let name) = entry else {
                    throw ResponseFormatError.invalidFormat("schema.required entries must be strings")
                }
                guard compiled.contains(where: { $0.name == name }) else {
                    throw ResponseFormatError.invalidFormat(
                        "required property '\(name)' is not declared in properties")
                }
                required.append(name)
            }
        }

        return JSONSchemaObject(properties: compiled, required: required)
    }

    /// Compile one property's value constraint.
    static func compilePropertyType(
        name: String,
        schema property: [String: JSONValue]
    ) throws -> SchemaValueType {
        // Allow-list gate (M1): reject any keyword we do not model — a value
        // constraint (`pattern`, `minLength`, `maximum`, `format`, …) or a
        // structural one (`properties`, `$ref`, combinators) must 400, never be
        // silently dropped.
        for key in property.keys where !allowedPropertyKeys.contains(key) {
            throw ResponseFormatError.unsupportedFeature(
                "unsupported schema keyword '\(key)' on property '\(name)'")
        }

        guard let typeValue = property["type"] else {
            throw ResponseFormatError.invalidFormat("property '\(name)' is missing 'type'")
        }
        guard case .string(let type) = typeValue else {
            throw ResponseFormatError.invalidFormat("property '\(name)' type must be a string")
        }

        if let enumValue = property["enum"] {
            guard case .array(let entries) = enumValue, !entries.isEmpty else {
                throw ResponseFormatError.invalidFormat(
                    "property '\(name)' enum must be a non-empty array")
            }
            guard type == "string" else {
                throw ResponseFormatError.unsupportedFeature(
                    "enum on non-string property '\(name)'")
            }
            var values: [String] = []
            for entry in entries {
                guard case .string(let value) = entry else {
                    throw ResponseFormatError.unsupportedFeature(
                        "non-string enum value on property '\(name)'")
                }
                // Enum values are matched as literal bytes at runtime, so one the
                // model could never spell would narrow (or, if the only choice,
                // deadlock) the value — reject it up front (M2).
                try requireLiteralMatchable(value, role: "enum value on property '\(name)'")
                values.append(value)
            }
            return .stringEnum(values)
        }

        switch type {
        case "string": return .string
        case "number": return .number
        case "integer": return .integer
        case "boolean": return .boolean
        case "object", "array":
            throw ResponseFormatError.unsupportedFeature("nested \(type) on property '\(name)'")
        default:
            throw ResponseFormatError.unsupportedFeature("property type '\(type)' on property '\(name)'")
        }
    }

    /// Reject a declared key or enum literal the byte-level runtime matcher could
    /// never match (M2). That matcher compares literal UTF-8 bytes with no
    /// JSON-unescaping and can only use whole-scalar tokens, so:
    ///
    ///  - a `"`, `\`, or control character (< 0x20) would require the string
    ///    escaping we do not model — the literal bytes can never appear raw; and
    ///  - a non-ASCII scalar may be unspellable by the tokenizer's complete-scalar
    ///    tokens (a conservative v1 restriction; it can be relaxed later by
    ///    modeling `\uXXXX` / multi-byte escapes).
    ///
    /// A `required` property whose key hits either case would deadlock the
    /// automaton into the no-legal-token path, so both are 400s at compile time
    /// rather than a silent narrowing.
    private static func requireLiteralMatchable(_ value: String, role: String) throws {
        for scalar in value.unicodeScalars
        where scalar == "\"" || scalar == "\\" || scalar.value < 0x20 {
            throw ResponseFormatError.unsupportedFeature(
                "\(role) requires JSON escaping we do not model (contains '\"', '\\', or a control character)")
        }
        for scalar in value.unicodeScalars where scalar.value > 0x7F {
            throw ResponseFormatError.unsupportedFeature(
                "\(role) contains non-ASCII characters (unsupported in v1)")
        }
    }
}
