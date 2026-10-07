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
///    schema's root, and every property and item below it, is one of:
///     - `string`, `number`, `integer` or `boolean`;
///     - a string `enum`, or a string `const` (a one-value enum);
///     - an object: `properties`, an optional `required` list, and
///       `additionalProperties: false` when present;
///     - an array: an `items` schema and optional `minItems` / `maxItems`;
///     - a `$ref` to `#/$defs/<name>` or `#/definitions/<name>` of the root.
///
///    A root without `type` (and without `$ref`, `const` or `enum`) is an
///    object.
///    Property names, enum values and `const` values may be any string: the
///    automaton matches them scalar by scalar — a scalar outside ASCII raw or
///    as a JSON escape, the quote, the backslash and control characters as
///    an escape, every other ASCII scalar raw — so nothing a schema declares
///    is unspellable.
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
///    A `number` or `integer` may carry `minimum`, `maximum`,
///    `exclusiveMinimum` and `exclusiveMaximum` (numbers of at most 19
///    significant digits and decimals): an integer's bounds fold to the
///    nearest integers inside them; a bounded number is spelled as a plain
///    decimal, without an exponent.
///
/// Everything else — combinators, `null`, type arrays, string bounds
/// (`pattern`, `minLength`, …), `multipleOf`, `additionalProperties: true`,
/// free-form objects and arrays, any unknown keyword — is an explicit
/// ``ResponseFormatError/unsupportedFeature(_:)``. An enforceable-looking
/// constraint is never silently dropped.
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
    private static let scalarKeys: Set<String> = [
        "type", "enum", "minimum", "exclusiveMinimum", "maximum", "exclusiveMaximum",
    ]
    /// The keywords that bound a number, in the order they are reported.
    private static let boundKeys = ["minimum", "exclusiveMinimum", "maximum", "exclusiveMaximum"]
    /// The keywords of an object schema besides annotations. `x-order` (Apple's
    /// declaration order) is accepted and ignored.
    private static let objectKeys: Set<String> = [
        "type", "properties", "required", "additionalProperties", "x-order",
    ]
    /// The keywords of an array schema besides annotations.
    private static let arrayKeys: Set<String> = ["type", "items", "minItems", "maxItems"]
    /// The keywords of a `const` schema besides annotations.
    private static let constKeys: Set<String> = ["type", "const"]
    /// Keywords honoured only at the root, whatever its type. A nested `$id`
    /// would rebase `$ref` resolution and nested `$defs` would need scoped
    /// lookup, so both stay a 400 below the root.
    private static let rootOnlyKeys: Set<String> = ["$defs", "definitions", "$schema", "$id"]

    /// Where a diagnostic points: the root, or a property path.
    private static func location(_ path: String) -> String {
        path.isEmpty ? "at the schema root" : "on property '\(path)'"
    }

    /// What a diagnostic names as the owner of a keyword: the schema itself at
    /// the root, or a property.
    private static func owner(_ path: String) -> String {
        path.isEmpty ? "schema" : "property '\(path)'"
    }

    /// The path of property `name` under `path`. A property named "" is
    /// written `""`, so that only the root has the empty path.
    private static func childPath(_ path: String, _ name: String) -> String {
        let segment = name.isEmpty ? "\"\"" : name
        return path.isEmpty ? segment : "\(path).\(segment)"
    }

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
    ) throws -> SchemaValueType {
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
                        + "after '$ref' expansion, \(ResponseFormatDecoder.location(path)))")
            }
        }

        /// Count `count` enum or `const` values against ``maxSchemaLiterals``.
        mutating func spendLiterals(_ count: Int, at path: String) throws {
            literals += count
            guard literals <= ResponseFormatDecoder.maxSchemaLiterals else {
                throw ResponseFormatError.unsupportedFeature(
                    "schema too large (more than \(ResponseFormatDecoder.maxSchemaLiterals) enum and const values "
                        + "after '$ref' expansion, \(ResponseFormatDecoder.location(path)))")
            }
        }

        /// Count the UTF-8 bytes of a property name, `required` entry or value
        /// against ``maxSchemaBytes``.
        mutating func spendBytes(of text: String, at path: String) throws {
            bytes += text.utf8.count
            guard bytes <= ResponseFormatDecoder.maxSchemaBytes else {
                throw ResponseFormatError.unsupportedFeature(
                    "schema too large (more than \(ResponseFormatDecoder.maxSchemaBytes) bytes of property names "
                        + "and enum and const values after '$ref' expansion, \(ResponseFormatDecoder.location(path)))")
            }
        }
    }

    /// Compile the top-level schema: a value of any supported type. The root's
    /// definition tables are read here and the root-only keywords stripped, so
    /// the value compiler sees the root like any other position.
    static func compileRootSchema(_ schema: [String: JSONValue]) throws -> SchemaValueType {
        var context = Context(
            defs: try definitionTable(schema["$defs"], keyword: "$defs"),
            definitions: try definitionTable(schema["definitions"], keyword: "definitions"))
        var root = schema
        for key in rootOnlyKeys { root[key] = nil }
        // A root without a type is an object, as it always was — unless it is
        // plainly something else: a `$ref`, a `const`, or an `enum`, which
        // the value rules then report as it is (an enum needs its `type`).
        if root["type"] == nil, root["$ref"] == nil, root["const"] == nil, root["enum"] == nil {
            return .object(try compileObject(root, path: "", depth: 1, context: &context))
        }
        return try compileValue(root, path: "", depth: 0, context: &context)
    }

    private static func definitionTable(_ value: JSONValue?, keyword: String) throws -> [String: JSONValue] {
        guard let value else { return [:] }
        guard case .object(let table) = value else {
            throw ResponseFormatError.invalidFormat("schema.\(keyword) must be an object")
        }
        return table
    }

    /// Compile an object schema: the root, or a nested object at `path`.
    /// `depth` counts this object's own container.
    static func compileObject(
        _ schema: [String: JSONValue],
        path: String,
        depth: Int,
        context: inout Context
    ) throws -> JSONSchemaObject {
        // Only the root position has the empty path (a property named "" is
        // written `""`, see `childPath`); a root `$ref` target is compiled
        // there too. The root is worded as "the schema", the rest as a
        // property.
        let isRoot = path.isEmpty
        let location = isRoot ? "at the schema root" : "on property '\(path)'"
        // Allow-list gate. At the root this is the C3 fix: keywords such as
        // `allOf` or `minProperties` used to be accepted there and enforced by
        // nothing. The root-only keywords are not on the list: the root's were
        // read and stripped by `compileRootSchema`, so one that reaches here
        // sits below the root — in a `$ref` target or under a property named
        // "", which share the root's empty path — and is a 400 like any other.
        let allowed = objectKeys.union(annotationKeys)
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
            let child = childPath(path, name)
            try context.spendBytes(of: name, at: child)
            guard let propertyValue = properties[name] else { continue }
            guard case .object(let property) = propertyValue else {
                throw ResponseFormatError.invalidFormat("property '\(child)' must be an object")
            }
            let type = try compileValue(property, path: child, depth: depth, context: &context)
            compiled.append(JSONSchemaObject.Property(name: name, type: type))
        }

        var required: [String] = []
        if let requiredValue = schema["required"] {
            guard case .array(let entries) = requiredValue else {
                throw ResponseFormatError.invalidFormat("\(owner).required must be an array")
            }
            // Compared scalar by scalar, as the automaton matches keys: `String`
            // equality is canonical, so a decomposed "é" in `required` would
            // otherwise pass for a precomposed one it can never match.
            let declared = Set(compiled.map { Array($0.name.unicodeScalars) })
            var listed = Set<[Unicode.Scalar]>()
            for entry in entries {
                guard case .string(let name) = entry else {
                    throw ResponseFormatError.invalidFormat("\(owner).required entries must be strings")
                }
                // A `$ref` repeats this list once per reference, like the names.
                try context.spendBytes(of: name, at: childPath(path, name))
                // JSON Schema requires unique entries. Refusing the first repeat,
                // with every entry declared, keeps this loop to one pass over the
                // declared properties however long the list is or however many
                // times a `$ref` compiles it.
                guard listed.insert(Array(name.unicodeScalars)).inserted else {
                    throw ResponseFormatError.invalidFormat(
                        isRoot
                            ? "required property '\(name)' is listed more than once"
                            : "required property '\(name)' is listed more than once in '\(path)'")
                }
                guard declared.contains(Array(name.unicodeScalars)) else {
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
                    "schema keyword '\(key)' alongside '$ref' \(location(path)) "
                        + "(only annotations may accompany '$ref')")
            }
            guard case .string(let ref) = refValue else {
                throw ResponseFormatError.invalidFormat("'$ref' \(location(path)) must be a string")
            }
            let target = try resolve(ref, path: path, context: context)
            guard !context.expanding.contains(ref) else {
                throw ResponseFormatError.unsupportedFeature(
                    "recursive schema: '$ref' '\(ref)' \(location(path)) refers back to itself "
                        + "(recursive schemas cannot be bounded)")
            }
            guard context.expanding.count < maxSchemaDepth else {
                throw ResponseFormatError.unsupportedFeature(
                    "'$ref' chain deeper than \(maxSchemaDepth) \(location(path))")
            }
            context.expanding.append(ref)
            defer { context.expanding.removeLast() }
            // The path names the position, not the definition, so a problem
            // inside the target also says which reference led there.
            do {
                return try compileValue(target, path: path, depth: depth, context: &context)
            } catch ResponseFormatError.unsupportedFeature(let message) {
                throw ResponseFormatError.unsupportedFeature("\(message) (via '$ref' '\(ref)')")
            } catch ResponseFormatError.invalidFormat(let message) {
                throw ResponseFormatError.invalidFormat("\(message) (via '$ref' '\(ref)')")
            }
        }

        // Apple's `@Guide(.constant(…))` emits `const` without a `type`.
        if let constValue = schema["const"] {
            for key in schema.keys.sorted() where !constKeys.contains(key) && !annotationKeys.contains(key) {
                throw ResponseFormatError.unsupportedFeature(
                    "unsupported schema keyword '\(key)' \(location(path))")
            }
            if let typeValue = schema["type"], typeValue != .string("string") {
                throw ResponseFormatError.unsupportedFeature("'const' on non-string \(owner(path))")
            }
            guard case .string(let value) = constValue else {
                throw ResponseFormatError.unsupportedFeature("non-string 'const' \(location(path))")
            }
            try context.spendLiterals(1, at: path)
            try context.spendBytes(of: value, at: path)
            return .stringEnum([value])
        }

        // A type array (a nullable union, say) is refused before the kind
        // dispatch. Otherwise `["object","null"]` with `properties` would fall
        // through to the scalar rules and be reported as an unknown
        // `properties` keyword instead of what it is.
        if case .array? = schema["type"] {
            throw ResponseFormatError.unsupportedFeature("type arrays (e.g. nullable unions) \(location(path))")
        }

        switch schema["type"] {
        case .string("object")?:
            return .object(
                try compileObject(schema, path: path, depth: depth + 1, context: &context))
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
            throw ResponseFormatError.unsupportedFeature("unsupported schema keyword '\(key)' \(location(path))")
        }
        guard depth <= maxSchemaDepth else {
            throw ResponseFormatError.unsupportedFeature(
                "schema nesting deeper than \(maxSchemaDepth) levels \(location(path))")
        }
        let items: [String: JSONValue]
        switch schema["items"] {
        case nil:
            throw ResponseFormatError.unsupportedFeature(
                "\(path.isEmpty ? "array" : "nested array") without 'items' \(location(path)) "
                    + "(free-form arrays are not supported)")
        case .object(let object)?:
            items = object
        case .array?:
            throw ResponseFormatError.unsupportedFeature("tuple-form 'items' \(location(path))")
        case .bool?:
            throw ResponseFormatError.unsupportedFeature("boolean 'items' schema \(location(path))")
        default:
            throw ResponseFormatError.invalidFormat("'items' \(location(path)) must be a schema object")
        }
        let minItems = try itemCount(schema["minItems"], keyword: "minItems", path: path) ?? 0
        guard minItems <= maxMinItems else {
            throw ResponseFormatError.unsupportedFeature(
                "schema too large (minItems \(minItems) \(location(path)) is above the limit of \(maxMinItems))")
        }
        let maxItems = try itemCount(schema["maxItems"], keyword: "maxItems", path: path)
        // `minItems > maxItems` admits no document: `]` could never close the
        // array, so the automaton would be stuck at its last item.
        if let maxItems, minItems > maxItems {
            throw ResponseFormatError.invalidFormat(
                "minItems (\(minItems)) exceeds maxItems (\(maxItems)) \(location(path))")
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
                "\(keyword) \(location(path)) must be a non-negative integer")
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
                "unsupported schema keyword '\(key)' \(location(name))")
        }

        guard let typeValue = property["type"] else {
            throw ResponseFormatError.invalidFormat("\(owner(name)) is missing 'type'")
        }
        guard case .string(let type) = typeValue else {
            throw ResponseFormatError.invalidFormat("\(owner(name)) type must be a string")
        }

        let boundKeyword = boundKeys.first { property[$0] != nil }

        if let enumValue = property["enum"] {
            if let boundKeyword {
                throw ResponseFormatError.unsupportedFeature("'\(boundKeyword)' alongside 'enum' \(location(name))")
            }
            guard case .array(let entries) = enumValue, !entries.isEmpty else {
                throw ResponseFormatError.invalidFormat(
                    "\(owner(name)) enum must be a non-empty array")
            }
            guard type == "string" else {
                throw ResponseFormatError.unsupportedFeature(
                    "enum on non-string \(owner(name))")
            }
            try context.spendLiterals(entries.count, at: name)
            var values: [String] = []
            for entry in entries {
                guard case .string(let value) = entry else {
                    throw ResponseFormatError.unsupportedFeature(
                        "non-string enum value \(location(name))")
                }
                try context.spendBytes(of: value, at: name)
                values.append(value)
            }
            return .stringEnum(values)
        }

        switch type {
        case "number":
            guard boundKeyword != nil else { return .number }
            return .boundedNumber(try numberBounds(try rawBounds(of: property, path: name), path: name))
        case "integer":
            guard boundKeyword != nil else { return .integer }
            return .boundedInteger(try integerBounds(try rawBounds(of: property, path: name), path: name))
        case "string", "boolean":
            if let boundKeyword {
                throw ResponseFormatError.unsupportedFeature(
                    "'\(boundKeyword)' on non-numeric \(owner(name)) (type '\(type)')")
            }
            return type == "string" ? .string : .boolean
        default:
            throw ResponseFormatError.unsupportedFeature("property type '\(type)' \(location(name))")
        }
    }

    /// The bounds a scalar schema declares, as the decimals they were written.
    private struct RawBounds {
        var minimum: SchemaDecimal?
        var exclusiveMinimum: SchemaDecimal?
        var maximum: SchemaDecimal?
        var exclusiveMaximum: SchemaDecimal?
    }

    /// The numeric bounds of `property`. A bound is a JSON number of at most
    /// 19 significant digits and decimals (what the automaton can hold
    /// exactly); the draft-4 boolean form of `exclusiveMinimum` /
    /// `exclusiveMaximum` is refused by name.
    private static func rawBounds(of property: [String: JSONValue], path: String) throws -> RawBounds {
        var bounds = RawBounds()
        for keyword in boundKeys {
            guard let value = property[keyword] else { continue }
            let decimal: SchemaDecimal?
            switch value {
            case .int(let integer):
                decimal = SchemaDecimal(integer)
            case .double(let double):
                decimal = SchemaDecimal(double)
            case .bool:
                throw ResponseFormatError.unsupportedFeature(
                    "boolean '\(keyword)' \(location(path)) (JSON Schema draft 4; give the bound as a number)")
            default:
                throw ResponseFormatError.invalidFormat("'\(keyword)' \(location(path)) must be a number")
            }
            guard let decimal else {
                throw ResponseFormatError.unsupportedFeature(
                    "'\(keyword)' \(location(path)) has more than \(SchemaDecimal.maximumDigits) significant digits or decimals")
            }
            switch keyword {
            case "minimum": bounds.minimum = decimal
            case "exclusiveMinimum": bounds.exclusiveMinimum = decimal
            case "maximum": bounds.maximum = decimal
            default: bounds.exclusiveMaximum = decimal
            }
        }
        return bounds
    }

    /// Integer bounds: each bound moves to the nearest integer inside it (a
    /// fractional `minimum` rounds up, an `exclusiveMinimum` steps past
    /// itself), and when both forms of a side are given the stricter wins.
    private static func integerBounds(_ raw: RawBounds, path: String) throws -> SchemaIntegerBounds {
        func integer(_ value: Int?, _ keyword: String) throws -> Int {
            guard let value else {
                throw ResponseFormatError.unsupportedFeature("'\(keyword)' \(location(path)) is beyond the integer range")
            }
            return value
        }
        var minimum: Int?
        if let bound = raw.minimum { minimum = try integer(bound.ceiling, "minimum") }
        if let bound = raw.exclusiveMinimum {
            let (stepped, overflow) = try integer(bound.floor, "exclusiveMinimum").addingReportingOverflow(1)
            guard !overflow else {
                throw ResponseFormatError.unsupportedFeature("'exclusiveMinimum' \(location(path)) is beyond the integer range")
            }
            minimum = Swift.max(minimum ?? stepped, stepped)
        }
        var maximum: Int?
        if let bound = raw.maximum { maximum = try integer(bound.floor, "maximum") }
        if let bound = raw.exclusiveMaximum {
            let (stepped, overflow) = try integer(bound.ceiling, "exclusiveMaximum").subtractingReportingOverflow(1)
            guard !overflow else {
                throw ResponseFormatError.unsupportedFeature("'exclusiveMaximum' \(location(path)) is beyond the integer range")
            }
            maximum = Swift.min(maximum ?? stepped, stepped)
        }
        guard let bounds = SchemaIntegerBounds(minimum: minimum, maximum: maximum) else {
            throw ResponseFormatError.invalidFormat(
                "the bounds \(location(path)) admit no integer (minimum \(minimum.map(String.init) ?? "-"), "
                    + "maximum \(maximum.map(String.init) ?? "-") once folded)")
        }
        return bounds
    }

    /// Number bounds: when both forms of a side are given the stricter wins
    /// (an exclusive bound at the same value, or inside the closed one).
    private static func numberBounds(_ raw: RawBounds, path: String) throws -> SchemaNumberBounds {
        var minimum = raw.minimum
        var minimumIsExclusive = false
        if let exclusive = raw.exclusiveMinimum, minimum.map({ exclusive >= $0 }) ?? true {
            minimum = exclusive
            minimumIsExclusive = true
        }
        var maximum = raw.maximum
        var maximumIsExclusive = false
        if let exclusive = raw.exclusiveMaximum, maximum.map({ exclusive <= $0 }) ?? true {
            maximum = exclusive
            maximumIsExclusive = true
        }
        guard let bounds = SchemaNumberBounds(
            minimum: minimum, minimumIsExclusive: minimumIsExclusive,
            maximum: maximum, maximumIsExclusive: maximumIsExclusive)
        else {
            throw ResponseFormatError.invalidFormat(
                "the bounds \(location(path)) admit no number (\(minimumIsExclusive ? "exclusiveMinimum" : "minimum") "
                    + "\(minimum.map(\.description) ?? "-"), \(maximumIsExclusive ? "exclusiveMaximum" : "maximum") "
                    + "\(maximum.map(\.description) ?? "-"))")
        }
        return bounds
    }

    /// Resolve `#/$defs/<name>` or `#/definitions/<name>` against the root's
    /// tables: exactly one non-empty JSON-pointer segment (RFC 6901 `~0` / `~1`
    /// unescaping, no percent-encoding). Anything else — a remote ref, `#`,
    /// `#/properties/…`, a deeper pointer — is unsupported.
    private static func resolve(_ ref: String, path: String, context: Context) throws -> [String: JSONValue] {
        let unsupported = ResponseFormatError.unsupportedFeature(
            "'$ref' '\(ref)' \(location(path)) "
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
                "'$ref' '\(ref)' \(location(path)) does not resolve to a definition")
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
}
