import Testing

@testable import MacMLXCore

// MARK: - ResponseFormatDecoder Tests (Track C — C1 + C2)
//
// The 400 gate: every accept / unsupported / invalid branch, driven by the same
// `JSONValue` the server hands over. MLX-free.

@Suite("ResponseFormatDecoder")
struct ResponseFormatDecoderTests {

    private func obj(_ pairs: [String: JSONValue]) -> JSONValue { .object(pairs) }

    // MARK: Absent / text / json_object

    @Test
    func absentOrNullOrTextYieldsNoConstraint() throws {
        #expect(try ResponseFormatDecoder.decode(nil) == nil)
        #expect(try ResponseFormatDecoder.decode(.null) == nil)
        #expect(try ResponseFormatDecoder.decode(obj(["type": .string("text")])) == nil)
    }

    @Test
    func jsonObjectDecodes() throws {
        #expect(try ResponseFormatDecoder.decode(obj(["type": .string("json_object")])) == .jsonObject)
    }

    // MARK: json_schema — supported subset

    @Test
    func compilesFlatSchema() throws {
        let schema = obj([
            "type": .string("object"),
            "properties": obj([
                "name": obj(["type": .string("string")]),
                "age": obj(["type": .string("integer")]),
                "score": obj(["type": .string("number")]),
                "active": obj(["type": .string("boolean")]),
                "role": obj(["type": .string("string"), "enum": .array([.string("admin"), .string("user")])]),
            ]),
            "required": .array([.string("name"), .string("age")]),
        ])
        let format = obj([
            "type": .string("json_schema"),
            "json_schema": obj(["name": .string("Person"), "schema": schema]),
        ])
        let decoded = try ResponseFormatDecoder.decode(format)
        guard case .jsonSchema(let object) = decoded else {
            Issue.record("expected .jsonSchema, got \(String(describing: decoded))")
            return
        }
        #expect(object.properties.count == 5)
        #expect(object.required == ["name", "age"])
        #expect(object.property(named: "role")?.type == .stringEnum(["admin", "user"]))
        #expect(object.property(named: "age")?.type == .integer)
    }

    // MARK: json_schema — unsupported features → 400

    @Test
    func rejectsNestedObjectProperty() {
        let schema = obj([
            "type": .string("object"),
            "properties": obj([
                "address": obj(["type": .string("object")]),
            ]),
        ])
        expectUnsupported(schema: schema, containing: "nested object")
    }

    @Test
    func rejectsArrayProperty() {
        let schema = obj([
            "type": .string("object"),
            "properties": obj(["tags": obj(["type": .string("array")])]),
        ])
        expectUnsupported(schema: schema, containing: "nested array")
    }

    @Test
    func rejectsNonObjectRoot() {
        let schema = obj([
            "type": .string("array"),
            "properties": obj(["x": obj(["type": .string("string")])]),
        ])
        expectUnsupported(schema: schema, containing: "top-level type 'array'")
    }

    @Test
    func rejectsCombinatorsAndRefs() {
        for key in ["properties", "items", "$ref", "anyOf", "allOf", "oneOf"] {
            let property: [String: JSONValue] = ["type": .string("string"), key: .string("y")]
            let schema = obj([
                "type": .string("object"),
                "properties": obj(["x": .object(property)]),
            ])
            expectUnsupported(schema: schema, containing: "'\(key)'")
        }
    }

    @Test
    func rejectsEnumOnNonString() {
        let schema = obj([
            "type": .string("object"),
            "properties": obj(["n": obj(["type": .string("integer"), "enum": .array([.int(1)])])]),
        ])
        expectUnsupported(schema: schema, containing: "enum on non-string")
    }

    @Test
    func rejectsAdditionalPropertiesTrue() {
        let schema = obj([
            "type": .string("object"),
            "properties": obj(["x": obj(["type": .string("string")])]),
            "additionalProperties": .bool(true),
        ])
        expectUnsupported(schema: schema, containing: "additionalProperties")
    }

    // MARK: json_schema — property keyword allow-list (M1)

    @Test
    func rejectsUnsupportedValueConstraintKeywords() {
        // Value-constraint keywords we cannot enforce must 400, never be silently
        // dropped ("never silently downgraded").
        for key in ["pattern", "minimum", "format", "maximum", "minLength", "multipleOf"] {
            let property: [String: JSONValue] = ["type": .string("string"), key: .string("x")]
            let schema = obj([
                "type": .string("object"),
                "properties": obj(["field": .object(property)]),
            ])
            expectUnsupported(schema: schema, containing: "'\(key)'")
        }
    }

    // MARK: json_schema — unmatchable keys / enum values (M2)

    @Test
    func rejectsKeyRequiringJSONEscaping() {
        // A `"` in a declared key can never be matched by the literal-byte key
        // matcher, so a required object with it would deadlock — reject up front.
        let schema = obj([
            "type": .string("object"),
            "properties": obj(["na\"me": obj(["type": .string("string")])]),
        ])
        expectUnsupported(schema: schema, containing: "requires JSON escaping")
    }

    @Test
    func rejectsKeyWithControlCharacter() {
        let schema = obj([
            "type": .string("object"),
            "properties": obj(["a\u{01}b": obj(["type": .string("string")])]),
        ])
        expectUnsupported(schema: schema, containing: "requires JSON escaping")
    }

    @Test
    func rejectsNonASCIIKey() {
        let schema = obj([
            "type": .string("object"),
            "properties": obj(["café": obj(["type": .string("string")])]),
        ])
        expectUnsupported(schema: schema, containing: "non-ASCII")
    }

    @Test
    func rejectsEnumValueWithBackslash() {
        let schema = obj([
            "type": .string("object"),
            "properties": obj(["p": obj([
                "type": .string("string"),
                "enum": .array([.string("a\\b"), .string("ok")]),
            ])]),
        ])
        expectUnsupported(schema: schema, containing: "requires JSON escaping")
    }

    @Test
    func rejectsNonASCIIEnumValue() {
        let schema = obj([
            "type": .string("object"),
            "properties": obj(["p": obj([
                "type": .string("string"),
                "enum": .array([.string("naïve")]),
            ])]),
        ])
        expectUnsupported(schema: schema, containing: "non-ASCII")
    }

    // MARK: json_schema — malformed → 400 invalid

    @Test
    func rejectsMissingProperties() {
        let schema = obj(["type": .string("object")])
        expectInvalid(schema: schema, containing: "properties")
    }

    @Test
    func rejectsRequiredNamingUndeclaredProperty() {
        let schema = obj([
            "type": .string("object"),
            "properties": obj(["x": obj(["type": .string("string")])]),
            "required": .array([.string("y")]),
        ])
        expectInvalid(schema: schema, containing: "not declared")
    }

    @Test
    func rejectsEmptyEnum() {
        let schema = obj([
            "type": .string("object"),
            "properties": obj(["r": obj(["type": .string("string"), "enum": .array([])])]),
        ])
        expectInvalid(schema: schema, containing: "enum must be a non-empty array")
    }

    @Test
    func rejectsUnknownTopLevelType() {
        let format = obj(["type": .string("xml")])
        #expect(throws: ResponseFormatError.self) {
            try ResponseFormatDecoder.decode(format)
        }
    }

    // MARK: json_schema — nested objects, arrays, $ref, const

    private let string: JSONValue = .object(["type": .string("string")])
    private let integer: JSONValue = .object(["type": .string("integer")])

    private func compile(_ schema: JSONValue) throws -> JSONSchemaObject {
        try StructuredOutputFixtures.compile(schema)
    }

    /// An object schema with `properties`, an optional `required` list and any
    /// extra keywords.
    private func root(
        _ properties: [String: JSONValue],
        required: [String]? = nil,
        extra: [String: JSONValue] = [:]
    ) -> JSONValue {
        var members: [String: JSONValue] = ["type": .string("object"), "properties": .object(properties)]
        if let required { members["required"] = .array(required.map(JSONValue.string)) }
        members.merge(extra) { _, new in new }
        return .object(members)
    }

    private func array(_ items: JSONValue, _ bounds: [String: JSONValue] = [:]) -> JSONValue {
        var members: [String: JSONValue] = ["type": .string("array"), "items": items]
        members.merge(bounds) { _, new in new }
        return .object(members)
    }

    private func ref(_ target: String) -> JSONValue { obj(["$ref": .string(target)]) }

    private var address: SchemaValueType {
        .object(JSONSchemaObject(
            properties: [.init(name: "street", type: .string), .init(name: "zip", type: .integer)],
            required: ["street"]))
    }

    @Test
    func compilesNestedObject() throws {
        let home = obj([
            "type": .string("object"),
            "properties": obj(["street": string, "zip": integer]),
            "required": .array([.string("street")]),
            "additionalProperties": .bool(false),
        ])
        let object = try compile(root(["home": home], required: ["home"]))
        #expect(object.property(named: "home")?.type == address)
        #expect(object.required == ["home"])
    }

    @Test
    func compilesArraysWithAndWithoutBounds() throws {
        let object = try compile(root([
            "tags": array(string, ["minItems": .int(1), "maxItems": .int(3)]),
            "free": array(string),
            "exact": array(string, ["minItems": .int(2), "maxItems": .int(2)]),
            "floor": array(string, ["minItems": .int(1)]),
        ]))
        #expect(object.property(named: "tags")?.type == .array(items: .string, minItems: 1, maxItems: 3))
        #expect(object.property(named: "free")?.type == .array(items: .string, minItems: 0, maxItems: nil))
        #expect(object.property(named: "exact")?.type == .array(items: .string, minItems: 2, maxItems: 2))
        #expect(object.property(named: "floor")?.type == .array(items: .string, minItems: 1, maxItems: nil))
    }

    @Test
    func compilesArraysOfArraysAndOfObjects() throws {
        let object = try compile(root([
            "rows": array(array(integer)),
            "people": array(root(["n": string])),
        ]))
        let row = SchemaValueType.array(items: .integer, minItems: 0, maxItems: nil)
        #expect(object.property(named: "rows")?.type == .array(items: row, minItems: 0, maxItems: nil))
        let person = SchemaValueType.object(JSONSchemaObject(properties: [.init(name: "n", type: .string)], required: []))
        #expect(object.property(named: "people")?.type == .array(items: person, minItems: 0, maxItems: nil))
    }

    /// `$defs` and `definitions` are separate tables; an annotation may sit
    /// next to `$ref`; an unreferenced definition is never compiled.
    @Test
    func resolvesRefsToDefsAndDefinitions() throws {
        let schema = root(
            [
                "home": obj(["$ref": .string("#/$defs/Addr"), "description": .string("d")]),
                "tags": array(ref("#/definitions/Tag")),
            ],
            extra: [
                "$defs": obj([
                    "Addr": root(["street": string, "zip": integer], required: ["street"]),
                    "Unused": obj(["type": .string("object"), "patternProperties": obj([:])]),
                ]),
                "definitions": obj(["Tag": obj(["type": .string("string"), "enum": .array([.string("a"), .string("b")])])]),
            ])
        let object = try compile(schema)
        #expect(object.property(named: "home")?.type == address)
        #expect(object.property(named: "tags")?.type == .array(items: .stringEnum(["a", "b"]), minItems: 0, maxItems: nil))
    }

    /// Cycle detection is scoped to the current path: one definition used at
    /// two sibling positions is not a cycle.
    @Test
    func compilesOneDefinitionUsedTwice() throws {
        let schema = root(
            ["x": ref("#/$defs/Addr"), "y": array(ref("#/$defs/Addr"))],
            extra: ["$defs": obj(["Addr": root(["street": string, "zip": integer], required: ["street"])])])
        let object = try compile(schema)
        #expect(object.property(named: "x")?.type == address)
        #expect(object.property(named: "y")?.type == .array(items: address, minItems: 0, maxItems: nil))
    }

    @Test
    func resolvesEscapedPointerSegments() throws {
        let schema = root(
            ["a": ref("#/$defs/a~1b"), "t": ref("#/$defs/t~0")],
            extra: ["$defs": obj(["a/b": string, "t~": integer])])
        let object = try compile(schema)
        #expect(object.property(named: "a")?.type == .string)
        #expect(object.property(named: "t")?.type == .integer)
    }

    /// Annotations change nothing: `examples` and `$comment` anywhere,
    /// `x-order` on objects, `$schema` and `$id` at the root.
    @Test
    func ignoresAnnotations() throws {
        let bare = root(["name": string, "home": root(["street": string])], required: ["name"])
        let annotated = root(
            [
                "name": obj([
                    "type": .string("string"), "examples": .array([.string("Ada")]),
                    "default": .string("x"), "$comment": .string("c"),
                ]),
                "home": obj([
                    "type": .string("object"), "properties": obj(["street": string]),
                    "x-order": .array([.string("street")]), "title": .string("Address"),
                ]),
            ],
            required: ["name"],
            extra: [
                "title": .string("Person"),
                "description": .string("A person"),
                "$schema": .string("https://json-schema.org/draft/2020-12/schema"),
                "$id": .string("https://example.com/person"),
                "x-order": .array([.string("name"), .string("home")]),
                "$comment": .string("root"),
            ])
        #expect(try compile(annotated) == compile(bare))
    }

    /// The metadata keywords `deprecated`, `readOnly` and `writeOnly` change
    /// nothing either, at the root or on a property.
    @Test
    func ignoresMetadataKeywords() throws {
        let metadata: [String: JSONValue] = [
            "deprecated": .bool(true), "readOnly": .bool(false), "writeOnly": .bool(true),
        ]
        func annotated(_ schema: [String: JSONValue]) -> JSONValue {
            .object(schema.merging(metadata) { current, _ in current })
        }
        let bare = root(["name": string, "tags": array(string)], required: ["name"])
        let withMetadata = root(
            ["name": annotated(["type": .string("string")]), "tags": annotated(["type": .string("array"), "items": string])],
            required: ["name"],
            extra: metadata)
        #expect(try compile(withMetadata) == compile(bare))
    }

    /// Apple's `@Guide(.constant(…))` emits `const` without a `type`.
    @Test
    func compilesStringConstAsAOneValueEnum() throws {
        let object = try compile(root([
            "a": obj(["const": .string("fixed")]),
            "b": obj(["type": .string("string"), "const": .string("fixed"), "description": .string("d")]),
        ]))
        #expect(object.property(named: "a")?.type == .stringEnum(["fixed"]))
        #expect(object.property(named: "b")?.type == .stringEnum(["fixed"]))
    }

    /// The schemas Apple's framework emits for real `@Generable` types.
    @Test
    func compilesFoundationModelsGenerableSchemas() throws {
        let person = try compile(StructuredOutputFixtures.generable("PersonNoRange"))
        #expect(person.property(named: "tags")?.type == .array(items: .string, minItems: 2, maxItems: 2))
        #expect(person.property(named: "scores")?.type == .array(items: .number, minItems: 1, maxItems: 3))
        #expect(person.property(named: "home")?.type == address)
        #expect(person.property(named: "previous")?.type == address)
        #expect(person.property(named: "addresses")?.type == .array(items: address, minItems: 0, maxItems: nil))
        let matrix = SchemaValueType.array(
            items: .array(items: .integer, minItems: 0, maxItems: nil), minItems: 0, maxItems: nil)
        #expect(person.property(named: "matrix")?.type == matrix)
        #expect(person.property(named: "konst")?.type == .stringEnum(["fixed"]))
        #expect(person.property(named: "mood")?.type == .stringEnum(["happy", "sad"]))
        #expect(person.properties.count == 13)
        #expect(person.required.count == 10)

        let explicit = try compile(StructuredOutputFixtures.generable("Explicit"))
        #expect(explicit.required.isEmpty)
        #expect(explicit.property(named: "maybe")?.type == .string)
        #expect(explicit.property(named: "maybeObj")?.type == address)
        #expect(explicit.property(named: "maybeList")?.type == .array(items: .integer, minItems: 0, maxItems: nil))
    }

    /// Property names are names, even when they spell a schema keyword.
    @Test
    func acceptsPropertiesNamedLikeKeywords() throws {
        let object = try compile(root(["type": string, "items": string, "properties": string, "$ref": string, "description": string]))
        #expect(object.properties.map(\.name) == ["$ref", "description", "items", "properties", "type"])
        #expect(object.properties.allSatisfy { $0.type == .string })
    }

    /// The root counts as one open container; at most 32 may be open.
    @Test
    func capsNestingAtThirtyTwoContainers() throws {
        func objects(_ depth: Int) -> JSONValue {
            var value = root(["leaf": string])
            for _ in 1..<depth { value = root(["c": value]) }
            return value
        }
        _ = try compile(objects(32))
        expectUnsupported(schema: objects(33), containing: "nesting deeper than 32")

        func arrays(_ count: Int) -> JSONValue {
            var item = string
            for _ in 0..<count { item = array(item) }
            return root(["a": item])
        }
        _ = try compile(arrays(31))
        expectUnsupported(schema: arrays(32), containing: "nesting deeper than 32")
    }

    @Test
    func reportsNestedPaths() {
        let pattern: JSONValue = obj(["type": .string("string"), "pattern": .string("x")])
        expectUnsupported(schema: root(["o": root(["e": pattern])]), containing: "on property 'o.e'")
        expectUnsupported(schema: root(["l": array(pattern)]), containing: "on property 'l[]'")
        expectUnsupported(schema: root(["l": array(root(["f": pattern]))]), containing: "on property 'l[].f'")
    }

    @Test
    func rejectsFreeFormObjectsAndArrays() {
        expectUnsupported(schema: root(["o": obj(["type": .string("object")])]), containing: "free-form objects")
        expectUnsupported(schema: root(["a": obj(["type": .string("array")])]), containing: "free-form arrays")
        expectInvalid(schema: root(["o": obj(["type": .string("object"), "properties": obj([:])])]), containing: "at least one property")
        expectInvalid(
            schema: root(["o": obj(["type": .string("object"), "properties": obj(["x": string]), "required": .array([.string("y")])])]),
            containing: "not declared")
    }

    // MARK: json_schema — $ref, bounds, root and literal rejections

    @Test
    func rejectsRecursiveSchemas() {
        let selfReferencing = root(
            ["n": ref("#/$defs/N")],
            extra: ["$defs": obj(["N": root(["next": ref("#/$defs/N")])])])
        expectUnsupported(schema: selfReferencing, containing: "recursive schema")

        let cycle = root(
            ["n": ref("#/$defs/A")],
            extra: ["$defs": obj(["A": root(["b": ref("#/$defs/B")]), "B": root(["a": ref("#/$defs/A")])])])
        expectUnsupported(schema: cycle, containing: "recursive schema")

        let bareCycle = root(
            ["n": ref("#/$defs/A")],
            extra: ["$defs": obj(["A": ref("#/$defs/B"), "B": ref("#/$defs/A")])])
        expectUnsupported(schema: bareCycle, containing: "recursive schema")
    }

    @Test
    func rejectsUnresolvedRef() {
        expectInvalid(schema: root(["n": ref("#/$defs/Missing")]), containing: "does not resolve")
        // Present only in the other table.
        expectInvalid(
            schema: root(["n": ref("#/$defs/T")], extra: ["definitions": obj(["T": string])]),
            containing: "does not resolve")
    }

    @Test
    func rejectsUnsupportedRefForms() {
        let defs: JSONValue = obj(["a": string, "A": string])
        for target in ["https://x/y.json", "#/properties/m", "#", "#/$defs/a/b", "#/$defs/", "#/$defs/%41", "#/$defs/a~2", "a"] {
            expectUnsupported(schema: root(["n": ref(target)], extra: ["$defs": defs]), containing: "'$ref'")
        }
        expectInvalid(schema: root(["n": obj(["$ref": .int(1)])]), containing: "must be a string")
    }

    @Test
    func rejectsKeywordsAlongsideRef() {
        let defs: JSONValue = obj(["A": root(["x": string])])
        expectUnsupported(
            schema: root(["n": obj(["$ref": .string("#/$defs/A"), "type": .string("object")])], extra: ["$defs": defs]),
            containing: "alongside '$ref'")
        expectUnsupported(
            schema: root(["n": obj(["$ref": .string("#/$defs/A"), "properties": obj(["y": string])])], extra: ["$defs": defs]),
            containing: "alongside '$ref'")
    }

    @Test
    func rejectsMalformedItemBounds() {
        expectInvalid(schema: root(["t": array(string, ["minItems": .int(3), "maxItems": .int(2)])]), containing: "exceeds maxItems")
        expectInvalid(schema: root(["t": array(string, ["minItems": .int(-1)])]), containing: "non-negative integer")
        expectInvalid(schema: root(["t": array(string, ["maxItems": .int(-1)])]), containing: "non-negative integer")
        expectInvalid(schema: root(["t": array(string, ["maxItems": .double(2.5)])]), containing: "non-negative integer")
        expectInvalid(schema: root(["t": array(string, ["minItems": .string("3")])]), containing: "non-negative integer")
    }

    /// `minItems` and `maxItems` are capped at 65,536: a minimum in the
    /// billions would compile and then cut every generation off at
    /// `max_tokens`.
    @Test
    func capsItemBounds() throws {
        let cap = ResponseFormatDecoder.maxItemCount
        let object = try compile(root([
            "a": array(string, ["minItems": .int(cap)]),
            "b": array(string, ["maxItems": .int(cap)]),
        ]))
        #expect(object.property(named: "a")?.type == .array(items: .string, minItems: cap, maxItems: nil))
        #expect(object.property(named: "b")?.type == .array(items: .string, minItems: 0, maxItems: cap))
        expectInvalid(schema: root(["a": array(string, ["minItems": .int(cap + 1)])]), containing: "minItems on property 'a' must be at most 65536")
        expectInvalid(schema: root(["b": array(string, ["maxItems": .int(cap + 1)])]), containing: "maxItems on property 'b' must be at most 65536")
        expectInvalid(schema: root(["c": array(string, ["minItems": .int(1_000_000_000)])]), containing: "must be at most 65536")
    }

    @Test
    func rejectsUnsupportedArrayForms() {
        expectUnsupported(schema: root(["t": array(.array([string]))]), containing: "tuple-form")
        expectUnsupported(schema: root(["t": array(.bool(true))]), containing: "boolean 'items'")
        for key in ["uniqueItems", "contains", "prefixItems", "minContains"] {
            expectUnsupported(schema: root(["t": array(string, [key: .bool(true)])]), containing: "'\(key)'")
        }
        expectInvalid(schema: root(["t": array(.string("x"))]), containing: "'items'")
    }

    /// Keywords that only make sense at the root, or not at all, stay a 400
    /// below it.
    @Test
    func rejectsRootOnlyKeywordsBelowTheRoot() {
        func nested(_ key: String, _ value: JSONValue) -> JSONValue {
            root(["o": obj(["type": .string("object"), "properties": obj(["x": string]), key: value])])
        }
        expectUnsupported(schema: nested("additionalProperties", .bool(true)), containing: "additionalProperties")
        for key in ["$defs", "definitions", "$id", "$schema"] {
            expectUnsupported(schema: nested(key, obj([:])), containing: "'\(key)'")
        }
    }

    /// Root keywords nothing enforces are a 400 naming the keyword. They used
    /// to be accepted and silently ignored (C3).
    @Test
    func rejectsUnenforceableRootKeywords() {
        for key in ["minProperties", "allOf", "anyOf", "patternProperties", "dependentRequired", "propertyNames"] {
            let schema = root(["x": string], extra: [key: .array([])])
            expectUnsupported(schema: schema, containing: "'\(key)'")
            expectUnsupported(schema: schema, containing: "at the schema root")
        }
    }

    @Test
    func rejectsNullAndUnions() {
        expectUnsupported(schema: root(["x": obj(["type": .array([.string("string"), .string("null")])])]), containing: "type arrays")
        // On an object or array schema the type array is named too, not the
        // first object or array keyword next to it.
        expectUnsupported(
            schema: root(["o": obj(["type": .array([.string("object"), .string("null")]), "properties": obj(["a": string])])]),
            containing: "type arrays")
        expectUnsupported(
            schema: root(["l": obj(["type": .array([.string("array"), .string("null")]), "items": string])]),
            containing: "type arrays")
        expectUnsupported(schema: root(["x": obj(["anyOf": .array([string, obj(["type": .string("null")])])])]), containing: "'anyOf'")
        expectUnsupported(schema: root(["x": obj(["type": .string("null")])]), containing: "property type 'null'")
    }

    /// The literal-byte rules (M2) apply at every depth and to `const`.
    @Test
    func appliesLiteralRulesAtEveryDepth() {
        expectUnsupported(schema: root(["o": root(["café": string])]), containing: "non-ASCII")
        expectUnsupported(
            schema: root(["o": root(["e": obj(["type": .string("string"), "enum": .array([.string("a\\b")])])])]),
            containing: "requires JSON escaping")
        expectUnsupported(
            schema: root(["l": array(obj(["type": .string("string"), "enum": .array([.string("naïve")])]))]),
            containing: "non-ASCII")
        expectUnsupported(schema: root(["k": obj(["const": .string("say \"hi\"")])]), containing: "requires JSON escaping")
        expectUnsupported(schema: root(["k": obj(["const": .string("Lençóis")])]), containing: "non-ASCII")
    }

    /// Exponential `$ref` fan-out (20 levels, 4 uses each) hits the node
    /// budget instead of hanging.
    @Test
    func boundsRefFanOut() {
        var defs: [String: JSONValue] = ["D0": string]
        for level in 1...20 {
            var properties: [String: JSONValue] = [:]
            for branch in 0..<4 { properties["p\(branch)"] = ref("#/$defs/D\(level - 1)") }
            defs["D\(level)"] = root(properties)
        }
        let schema = root(["x": ref("#/$defs/D20")], extra: ["$defs": .object(defs)])
        let clock = ContinuousClock()
        let elapsed = clock.measure {
            expectUnsupported(schema: schema, containing: "schema too large")
        }
        #expect(elapsed < .seconds(2))
    }

    /// A schema small on the wire can expand to a huge one: every `$ref` to a
    /// large enum compiles the whole enum again, and the automaton later
    /// encodes it again. 2,000 references to a 20,000-value enum are 4,001
    /// nodes, under the node budget; the value budget turns them into a 400
    /// after a few references. Mutation: without the value budget the byte
    /// budget still refuses this, later and with its own message; without
    /// both budgets it compiles (no 400) and takes seconds.
    @Test
    func boundsEnumValuesAfterRefExpansion() {
        let values = (0..<20_000).map { JSONValue.string("v\($0)") }
        var properties: [String: JSONValue] = [:]
        for index in 0..<2_000 { properties["p\(index)"] = ref("#/$defs/E") }
        let schema = root(
            ["o": root(properties)],
            extra: ["$defs": obj(["E": obj(["type": .string("string"), "enum": .array(values)])])])
        let elapsed = ContinuousClock().measure {
            expectUnsupported(schema: schema, containing: "schema too large (more than 65536 enum and const values")
        }
        #expect(elapsed < .seconds(1), "took \(elapsed)")
    }

    /// The value cap itself: one enum of 65,536 values compiles, one more value
    /// does not.
    @Test
    func valueBudgetAdmitsExactlyItsCap() throws {
        func schema(values count: Int) -> JSONValue {
            root(["e": obj(["type": .string("string"), "enum": .array((0..<count).map { .string("v\($0)") })])])
        }
        _ = try compile(schema(values: ResponseFormatDecoder.maxSchemaLiterals))
        expectUnsupported(schema: schema(values: ResponseFormatDecoder.maxSchemaLiterals + 1), containing: "enum and const values")
    }

    /// The byte cap itself: a property name of exactly the cap compiles, one
    /// more byte does not.
    @Test
    func byteBudgetAdmitsExactlyItsCap() throws {
        func schema(nameBytes count: Int) -> JSONValue {
            root([String(repeating: "k", count: count): string])
        }
        _ = try compile(schema(nameBytes: ResponseFormatDecoder.maxSchemaBytes))
        expectUnsupported(schema: schema(nameBytes: ResponseFormatDecoder.maxSchemaBytes + 1), containing: "bytes of property names")
    }

    /// The same expansion with long strings instead of many: 1,000 references
    /// to a 64 KiB `const`, or to an object whose one key is 64 KiB, stay under
    /// the node and value budgets, yet the automaton's program would copy the
    /// string once per reference — 64 MB. The byte budget refuses both.
    /// Mutation: without it both compile (no 400).
    @Test
    func boundsNameAndValueBytesAfterRefExpansion() {
        let long = String(repeating: "a", count: 65_536)
        var properties: [String: JSONValue] = [:]
        for index in 0..<1_000 { properties["p\(index)"] = ref("#/$defs/E") }
        let viaValue = root(["o": root(properties)], extra: ["$defs": obj(["E": obj(["const": .string(long)])])])
        let viaKey = root(["o": root(properties)], extra: ["$defs": obj(["E": root([long: string])])])
        for schema in [viaValue, viaKey] {
            let elapsed = ContinuousClock().measure {
                expectUnsupported(schema: schema, containing: "schema too large")
            }
            #expect(elapsed < .seconds(1), "took \(elapsed)")
        }
    }

    @Test
    func rejectsNonStringConst() {
        expectUnsupported(schema: root(["k": obj(["const": .int(3)])]), containing: "non-string 'const'")
        expectUnsupported(schema: root(["k": obj(["type": .string("integer"), "const": .int(3)])]), containing: "'const' on non-string")
        expectUnsupported(
            schema: root(["k": obj(["const": .string("x"), "enum": .array([.string("x")])])]),
            containing: "'enum'")
    }

    /// `@Guide(.range(…))` becomes `minimum`/`maximum`, which nothing here can
    /// enforce: a 400, not a silently unbounded integer.
    @Test
    func rejectsTheRangeGuide() throws {
        expectUnsupported(schema: try StructuredOutputFixtures.generable("Person"), containing: "'maximum'")
    }

    /// Apple's TripPlanner sample still needs non-ASCII enum values
    /// ("Lençóis Maranhenses"); without that one value it compiles.
    @Test
    func tripPlannerNeedsOnlyNonASCIIEnumValues() throws {
        expectUnsupported(schema: try StructuredOutputFixtures.itinerary(), containing: "non-ASCII")
        let trip = try compile(StructuredOutputFixtures.asciiItinerary())
        #expect(trip.required == ["title", "destinationName", "description", "rationale", "days"])
    }

    // MARK: Helpers

    private func wrap(_ schema: JSONValue) -> JSONValue {
        obj([
            "type": .string("json_schema"),
            "json_schema": obj(["name": .string("T"), "schema": schema]),
        ])
    }

    private func expectUnsupported(schema: JSONValue, containing needle: String) {
        do {
            _ = try ResponseFormatDecoder.decode(wrap(schema))
            Issue.record("expected unsupportedFeature containing '\(needle)'")
        } catch let error as ResponseFormatError {
            guard case .unsupportedFeature = error else {
                Issue.record("expected .unsupportedFeature, got \(error)")
                return
            }
            #expect(error.description.contains(needle), "‘\(error.description)’ lacked ‘\(needle)’")
        } catch {
            Issue.record("unexpected error \(error)")
        }
    }

    private func expectInvalid(schema: JSONValue, containing needle: String) {
        do {
            _ = try ResponseFormatDecoder.decode(wrap(schema))
            Issue.record("expected invalidFormat containing '\(needle)'")
        } catch let error as ResponseFormatError {
            guard case .invalidFormat = error else {
                Issue.record("expected .invalidFormat, got \(error)")
                return
            }
            #expect(error.description.contains(needle), "‘\(error.description)’ lacked ‘\(needle)’")
        } catch {
            Issue.record("unexpected error \(error)")
        }
    }
}
