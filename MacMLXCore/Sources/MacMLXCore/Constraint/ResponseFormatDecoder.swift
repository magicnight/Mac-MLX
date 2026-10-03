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
///  - `{"type":"json_schema","json_schema":{"schema":{…}}}` → C2, where the
///    schema is an object whose properties are, at any depth:
///     - `string`, `number`, `integer` or `boolean`;
///     - a string `enum`, or a string `const` (a one-value enum);
///     - a nested object: `properties`, an optional `required` list, and
///       `additionalProperties: false` when present;
///     - an array: an `items` schema and optional `minItems` / `maxItems`;
///     - a `$ref` to `#/$defs/<name>` or `#/definitions/<name>` of the root.
///
///    `description`, `title`, `default`, `examples`, `$comment`, `deprecated`,
///    `readOnly` and `writeOnly` are accepted and ignored anywhere; so is
///    `x-order` on an object, and
///    `$schema` / `$id` at the root. Property order never constrains key order
///    on the wire. A schema may hold at most ``maxSchemaDepth`` containers
///    open at once; after `$ref` expansion, at most ``maxSchemaNodes`` nodes,
///    ``maxSchemaLiterals`` enum and `const` values and ``maxSchemaBytes``
///    bytes of property names (declared and `required`) and values. A
///    recursive schema is rejected because no bound on its documents exists,
///    and a `required` list may not repeat a name.
///
/// Everything else — combinators, `null`, type arrays, numeric and string
/// bounds (`minimum`, `pattern`, …), non-object roots,
/// `additionalProperties: true`, free-form objects and arrays, any unknown
/// keyword — is an explicit ``ResponseFormatError/unsupportedFeature(_:)``.
/// An enforceable-looking constraint is never silently dropped.
public enum ResponseFormatDecoder {

    /// The most containers a compiled schema may hold open at once, the root
    /// object counting as 1 — the same counting as ``JSONGrammarState/maxDepth``.
    /// Staying below that automaton's default of 64 keeps every schema document
    /// acceptable to the generic JSON automaton too.
    static let maxSchemaDepth = 32

    /// The most schema nodes one compile may visit, every `$ref` hop included.
    /// Bounds compile time and the compiled tree against exponential `$ref`
    /// fan-out (a definition used four times by one used four times, …).
    static let maxSchemaNodes = 4096

    /// The most enum and `const` values one compile may produce after `$ref`
    /// expansion; each enum value and each `const` counts 1. The node budget
    /// counts a 60,000-value enum as one node, but every reference to it
    /// compiles the values again and the automaton encodes them again, so a
    /// small request could otherwise expand into gigabytes.
    static let maxSchemaLiterals = 65_536

    /// The most UTF-8 bytes of property names, `required` entries, enum values
    /// and `const` values one compile may produce after `$ref` expansion. A
    /// value count alone does not bound size: one long name or value
    /// referenced many times is copied once per reference.
    static let maxSchemaBytes = 4 * 1_024 * 1_024

    /// The largest `minItems` accepted, a server limit. Every document needs at
    /// least `minItems` items, so a minimum in the billions would compile and
    /// then cut every generation off at `max_tokens`. `maxItems` needs no cap:
    /// a large maximum forces nothing and is enforced exactly.
    static let maxMinItems = 65_536

    /// Purely annotative keywords, accepted and ignored on every kind of schema:
    /// the JSON Schema annotation and metadata vocabulary, none of which
    /// constrains a value.
    private static let annotationKeys: Set<String> = [
        "description", "title", "default", "examples", "$comment",
        "deprecated", "readOnly", "writeOnly",
    ]

    /// The keywords a scalar property schema may carry besides annotations.
    /// This is a strict allow-list, not a blocklist: any other keyword
    /// (`pattern`, `minLength`, `maximum`, `format`, combinators, …) is
    /// rejected with a 400 rather than silently ignored — a value constraint we
    /// cannot enforce must never be silently downgraded (see
    /// ``ResponseFormatError``).
    private static let scalarKeys: Set<String> = ["type", "enum"]
    /// The keywords of an object schema besides annotations. `x-order` (Apple's
    /// declaration order) is accepted and ignored.
    private static let objectKeys: Set<String> = [
        "type", "properties", "required", "additionalProperties", "x-order",
    ]
    /// The keywords of an array schema besides annotations.
    private static let arrayKeys: Set<String> = ["type", "items", "minItems", "maxItems"]
    /// The keywords of a `const` schema besides annotations.
    private static let constKeys: Set<String> = ["type", "const"]
    /// Keywords honoured only on the root object. A nested `$id` would rebase
    /// `$ref` resolution and nested `$defs` would need scoped lookup, so both
    /// stay a 400 below the root.
    private static let rootOnlyKeys: Set<String> = ["$defs", "definitions", "$schema", "$id"]

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
        return try compileRootSchema(schema)
    }

    /// Per-compile state: the root's definition tables, the `$ref`s being
    /// expanded on the current path (cycle detection), and the size budgets.
    struct Context {
        let defs: [String: JSONValue]
        let definitions: [String: JSONValue]
        /// The refs expanded on the path to the current position. Scoped to the
        /// path, so one definition used at two sibling positions is fine.
        var expanding: [String] = []
        var nodes = 0
        var literals = 0
        var bytes = 0

        /// Count one schema node against ``maxSchemaNodes``.
        mutating func spend(at path: String) throws {
            nodes += 1
            guard nodes <= ResponseFormatDecoder.maxSchemaNodes else {
                throw ResponseFormatError.unsupportedFeature(
                    "schema too large (more than \(ResponseFormatDecoder.maxSchemaNodes) nodes "
                        + "after '$ref' expansion, at property '\(path)')")
            }
        }

        /// Count `count` enum or `const` values against ``maxSchemaLiterals``.
        mutating func spendLiterals(_ count: Int, at path: String) throws {
            literals += count
            guard literals <= ResponseFormatDecoder.maxSchemaLiterals else {
                throw ResponseFormatError.unsupportedFeature(
                    "schema too large (more than \(ResponseFormatDecoder.maxSchemaLiterals) enum and const values "
                        + "after '$ref' expansion, at property '\(path)')")
            }
        }

        /// Count the UTF-8 bytes of a property name, `required` entry or value
        /// against ``maxSchemaBytes``.
        mutating func spendBytes(of text: String, at path: String) throws {
            bytes += text.utf8.count
            guard bytes <= ResponseFormatDecoder.maxSchemaBytes else {
                throw ResponseFormatError.unsupportedFeature(
                    "schema too large (more than \(ResponseFormatDecoder.maxSchemaBytes) bytes of property names "
                        + "and enum and const values after '$ref' expansion, at property '\(path)')")
            }
        }
    }

    /// Compile the top-level schema, which must be an object.
    static func compileRootSchema(_ schema: [String: JSONValue]) throws -> JSONSchemaObject {
        if let typeValue = schema["type"] {
            guard case .string(let type) = typeValue else {
                throw ResponseFormatError.invalidFormat("schema.type must be a string")
            }
            guard type == "object" else {
                throw ResponseFormatError.unsupportedFeature(
                    "top-level type '\(type)' (only 'object' is supported)")
            }
        }
        var context = Context(
            defs: try definitionTable(schema["$defs"], keyword: "$defs"),
            definitions: try definitionTable(schema["definitions"], keyword: "definitions"))
        return try compileObject(schema, path: "", depth: 1, isRoot: true, context: &context)
    }

    private static func definitionTable(_ value: JSONValue?, keyword: String) throws -> [String: JSONValue] {
        guard let value else { return [:] }
        guard case .object(let table) = value else {
            throw ResponseFormatError.invalidFormat("schema.\(keyword) must be an object")
        }
        return table
    }

    /// Compile an object schema: the root (`isRoot`) or a nested object at
    /// `path`. `depth` counts this object's own container.
    static func compileObject(
        _ schema: [String: JSONValue],
        path: String,
        depth: Int,
        isRoot: Bool,
        context: inout Context
    ) throws -> JSONSchemaObject {
        let location = isRoot ? "at the schema root" : "on property '\(path)'"
        // Allow-list gate. At the root this is the C3 fix: keywords such as
        // `allOf` or `minProperties` used to be accepted there and enforced by
        // nothing.
        let allowed = objectKeys.union(annotationKeys).union(isRoot ? rootOnlyKeys : [])
        for key in schema.keys.sorted() where !allowed.contains(key) {
            throw ResponseFormatError.unsupportedFeature("unsupported schema keyword '\(key)' \(location)")
        }
        guard depth <= maxSchemaDepth else {
            throw ResponseFormatError.unsupportedFeature(
                "schema nesting deeper than \(maxSchemaDepth) levels \(location)")
        }

        // We forbid additional properties structurally; honoring an explicit
        // `additionalProperties: true` would contradict that, so reject it.
        if let additional = schema["additionalProperties"], additional != .bool(false) {
            throw ResponseFormatError.unsupportedFeature(
                isRoot
                    ? "additionalProperties (only false is supported)"
                    : "additionalProperties (only false is supported) on property '\(path)'")
        }

        let owner = isRoot ? "schema" : "property '\(path)'"
        guard let propertiesValue = schema["properties"] else {
            if isRoot {
                throw ResponseFormatError.invalidFormat("schema.properties object is required")
            }
            throw ResponseFormatError.unsupportedFeature(
                "nested object without 'properties' on property '\(path)' (free-form objects are not supported)")
        }
        guard case .object(let properties) = propertiesValue else {
            throw ResponseFormatError.invalidFormat("\(owner).properties must be an object")
        }
        guard !properties.isEmpty else {
            throw ResponseFormatError.invalidFormat("\(owner).properties must declare at least one property")
        }

        // Sorted for a deterministic declaration order (diagnostics only — the
        // runtime automaton accepts keys in any order).
        var compiled: [JSONSchemaObject.Property] = []
        for name in properties.keys.sorted() {
            let childPath = path.isEmpty ? name : "\(path).\(name)"
            try context.spendBytes(of: name, at: childPath)
            // The runtime key matcher compares literal UTF-8 bytes, so a declared
            // key the model could never spell would deadlock a `required` object
            // into the no-legal-token path — reject it up front (M2).
            try requireLiteralMatchable(name, role: "property key '\(childPath)'")
            guard let propertyValue = properties[name] else { continue }
            guard case .object(let property) = propertyValue else {
                throw ResponseFormatError.invalidFormat("property '\(childPath)' must be an object")
            }
            let type = try compileValue(property, path: childPath, depth: depth, context: &context)
            compiled.append(JSONSchemaObject.Property(name: name, type: type))
        }

        var required: [String] = []
        if let requiredValue = schema["required"] {
            guard case .array(let entries) = requiredValue else {
                throw ResponseFormatError.invalidFormat("\(owner).required must be an array")
            }
            let declared = Set(compiled.map(\.name))
            var listed = Set<String>()
            for entry in entries {
                guard case .string(let name) = entry else {
                    throw ResponseFormatError.invalidFormat("\(owner).required entries must be strings")
                }
                // A `$ref` repeats this list once per reference, like the names.
                try context.spendBytes(of: name, at: path.isEmpty ? name : "\(path).\(name)")
                // JSON Schema requires unique entries. Refusing the first repeat,
                // with every entry declared, keeps this loop to one pass over the
                // declared properties however long the list is or however many
                // times a `$ref` compiles it.
                guard listed.insert(name).inserted else {
                    throw ResponseFormatError.invalidFormat(
                        isRoot
                            ? "required property '\(name)' is listed more than once"
                            : "required property '\(name)' is listed more than once in '\(path)'")
                }
                guard declared.contains(name) else {
                    throw ResponseFormatError.invalidFormat(
                        isRoot
                            ? "required property '\(name)' is not declared in properties"
                            : "required property '\(name)' is not declared in properties of '\(path)'")
                }
                required.append(name)
            }
        }

        return JSONSchemaObject(properties: compiled, required: required)
    }

    /// Compile the schema of one value position at `path`. `depth` is the
    /// number of containers open around it; an object or array value adds its
    /// own. Dispatch precedence: `$ref`, then `const`, then `type`.
    static func compileValue(
        _ schema: [String: JSONValue],
        path: String,
        depth: Int,
        context: inout Context
    ) throws -> SchemaValueType {
        try context.spend(at: path)

        if let refValue = schema["$ref"] {
            for key in schema.keys.sorted() where key != "$ref" && !annotationKeys.contains(key) {
                throw ResponseFormatError.unsupportedFeature(
                    "schema keyword '\(key)' alongside '$ref' on property '\(path)' "
                        + "(only annotations may accompany '$ref')")
            }
            guard case .string(let ref) = refValue else {
                throw ResponseFormatError.invalidFormat("'$ref' on property '\(path)' must be a string")
            }
            let target = try resolve(ref, path: path, context: context)
            guard !context.expanding.contains(ref) else {
                throw ResponseFormatError.unsupportedFeature(
                    "recursive schema: '$ref' '\(ref)' on property '\(path)' refers back to itself "
                        + "(recursive schemas cannot be bounded)")
            }
            guard context.expanding.count < maxSchemaDepth else {
                throw ResponseFormatError.unsupportedFeature(
                    "'$ref' chain deeper than \(maxSchemaDepth) on property '\(path)'")
            }
            context.expanding.append(ref)
            defer { context.expanding.removeLast() }
            return try compileValue(target, path: path, depth: depth, context: &context)
        }

        // Apple's `@Guide(.constant(…))` emits `const` without a `type`.
        if let constValue = schema["const"] {
            for key in schema.keys.sorted() where !constKeys.contains(key) && !annotationKeys.contains(key) {
                throw ResponseFormatError.unsupportedFeature(
                    "unsupported schema keyword '\(key)' on property '\(path)'")
            }
            if let typeValue = schema["type"], typeValue != .string("string") {
                throw ResponseFormatError.unsupportedFeature("'const' on non-string property '\(path)'")
            }
            guard case .string(let value) = constValue else {
                throw ResponseFormatError.unsupportedFeature("non-string 'const' on property '\(path)'")
            }
            try context.spendLiterals(1, at: path)
            try context.spendBytes(of: value, at: path)
            // Matched as a literal at runtime, like an enum value (M2).
            try requireLiteralMatchable(value, role: "const value on property '\(path)'")
            return .stringEnum([value])
        }

        // A type array (a nullable union, say) is refused before the kind
        // dispatch. Otherwise `["object","null"]` with `properties` would fall
        // through to the scalar rules and be reported as an unknown
        // `properties` keyword instead of what it is.
        if case .array? = schema["type"] {
            throw ResponseFormatError.unsupportedFeature("type arrays (e.g. nullable unions) on property '\(path)'")
        }

        switch schema["type"] {
        case .string("object")?:
            return .object(try compileObject(schema, path: path, depth: depth + 1, isRoot: false, context: &context))
        case .string("array")?:
            return try compileArray(schema, path: path, depth: depth + 1, context: &context)
        default:
            return try compileScalar(schema, path: path, context: &context)
        }
    }

    /// Compile an array schema at `path`. `depth` counts the array's own
    /// container; its items are compiled at path `path[]`.
    static func compileArray(
        _ schema: [String: JSONValue],
        path: String,
        depth: Int,
        context: inout Context
    ) throws -> SchemaValueType {
        for key in schema.keys.sorted() where !arrayKeys.contains(key) && !annotationKeys.contains(key) {
            throw ResponseFormatError.unsupportedFeature("unsupported schema keyword '\(key)' on property '\(path)'")
        }
        guard depth <= maxSchemaDepth else {
            throw ResponseFormatError.unsupportedFeature(
                "schema nesting deeper than \(maxSchemaDepth) levels on property '\(path)'")
        }
        let items: [String: JSONValue]
        switch schema["items"] {
        case nil:
            throw ResponseFormatError.unsupportedFeature(
                "nested array without 'items' on property '\(path)' (free-form arrays are not supported)")
        case .object(let object)?:
            items = object
        case .array?:
            throw ResponseFormatError.unsupportedFeature("tuple-form 'items' on property '\(path)'")
        case .bool?:
            throw ResponseFormatError.unsupportedFeature("boolean 'items' schema on property '\(path)'")
        default:
            throw ResponseFormatError.invalidFormat("'items' on property '\(path)' must be a schema object")
        }
        let minItems = try itemCount(schema["minItems"], keyword: "minItems", path: path) ?? 0
        guard minItems <= maxMinItems else {
            throw ResponseFormatError.unsupportedFeature(
                "schema too large (minItems \(minItems) on property '\(path)' is above the limit of \(maxMinItems))")
        }
        let maxItems = try itemCount(schema["maxItems"], keyword: "maxItems", path: path)
        // `minItems > maxItems` admits no document: `]` could never close the
        // array, so the automaton would be stuck at its last item.
        if let maxItems, minItems > maxItems {
            throw ResponseFormatError.invalidFormat(
                "minItems (\(minItems)) exceeds maxItems (\(maxItems)) on property '\(path)'")
        }
        let item = try compileValue(items, path: "\(path)[]", depth: depth, context: &context)
        return .array(items: item, minItems: minItems, maxItems: maxItems)
    }

    /// A `minItems` / `maxItems` value: a non-negative integer, or `nil` when
    /// absent. (`JSONValue` already decodes `3.0` and `1e2` as integers.)
    private static func itemCount(_ value: JSONValue?, keyword: String, path: String) throws -> Int? {
        guard let value else { return nil }
        guard case .int(let count) = value, count >= 0 else {
            throw ResponseFormatError.invalidFormat(
                "\(keyword) on property '\(path)' must be a non-negative integer")
        }
        return count
    }

    /// Compile one scalar property's value constraint — the flat subset's
    /// original rules, check for check.
    static func compileScalar(
        _ property: [String: JSONValue],
        path name: String,
        context: inout Context
    ) throws -> SchemaValueType {
        // Allow-list gate (M1): reject any keyword we do not model — a value
        // constraint (`pattern`, `minLength`, `maximum`, `format`, …) or a
        // structural one (`properties`, combinators) must 400, never be
        // silently dropped. Sorted so the reported keyword is deterministic.
        for key in property.keys.sorted() where !scalarKeys.contains(key) && !annotationKeys.contains(key) {
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
            try context.spendLiterals(entries.count, at: name)
            var values: [String] = []
            for entry in entries {
                guard case .string(let value) = entry else {
                    throw ResponseFormatError.unsupportedFeature(
                        "non-string enum value on property '\(name)'")
                }
                try context.spendBytes(of: value, at: name)
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
        default:
            throw ResponseFormatError.unsupportedFeature("property type '\(type)' on property '\(name)'")
        }
    }

    /// Resolve `#/$defs/<name>` or `#/definitions/<name>` against the root's
    /// tables: exactly one non-empty JSON-pointer segment (RFC 6901 `~0` / `~1`
    /// unescaping, no percent-encoding). Anything else — a remote ref, `#`,
    /// `#/properties/…`, a deeper pointer — is unsupported.
    private static func resolve(_ ref: String, path: String, context: Context) throws -> [String: JSONValue] {
        let unsupported = ResponseFormatError.unsupportedFeature(
            "'$ref' '\(ref)' on property '\(path)' "
                + "(only '#/$defs/<name>' and '#/definitions/<name>' are supported)")
        let table: [String: JSONValue]
        let segment: Substring
        if ref.hasPrefix("#/$defs/") {
            table = context.defs
            segment = ref.dropFirst("#/$defs/".count)
        } else if ref.hasPrefix("#/definitions/") {
            table = context.definitions
            segment = ref.dropFirst("#/definitions/".count)
        } else {
            throw unsupported
        }
        guard !segment.isEmpty, !segment.contains("/"), !segment.contains("%"),
              let name = decodePointerSegment(segment) else {
            throw unsupported
        }
        guard let value = table[name] else {
            throw ResponseFormatError.invalidFormat(
                "'$ref' '\(ref)' on property '\(path)' does not resolve to a definition")
        }
        guard case .object(let target) = value else {
            throw ResponseFormatError.invalidFormat("'$ref' target '\(ref)' must be a schema object")
        }
        return target
    }

    /// RFC 6901: `~1` → `/`, `~0` → `~`; any other `~` sequence is malformed.
    private static func decodePointerSegment(_ segment: Substring) -> String? {
        var decoded = ""
        var iterator = segment.makeIterator()
        while let character = iterator.next() {
            guard character == "~" else {
                decoded.append(character)
                continue
            }
            switch iterator.next() {
            case "0"?: decoded.append("~")
            case "1"?: decoded.append("/")
            default: return nil
            }
        }
        return decoded
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
