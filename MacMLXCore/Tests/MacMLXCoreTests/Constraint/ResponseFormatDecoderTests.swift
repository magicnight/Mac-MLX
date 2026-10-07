import Foundation
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
        guard case .jsonSchema(.object(let object)) = decoded else {
            Issue.record("expected .jsonSchema(.object), got \(String(describing: decoded))")
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

    /// A root of another type is compiled by that type's rules, so an object
    /// keyword on an array root is the unsupported keyword it is.
    @Test
    func rejectsObjectKeywordsOnAnArrayRoot() {
        let schema = obj([
            "type": .string("array"),
            "properties": obj(["x": obj(["type": .string("string")])]),
        ])
        expectUnsupported(schema: schema, containing: "'properties' at the schema root")
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

    // MARK: json_schema — keys and enum values may be any string

    /// Keys and enum values outside ASCII, or containing a quote, a backslash
    /// or a control character, compile: the automaton matches a scalar outside
    /// ASCII raw or as a JSON escape and those three as an escape, so every
    /// literal has an all-ASCII spelling.
    @Test
    func compilesAnyStringAsAKeyOrEnumValue() throws {
        let schema = obj([
            "type": .string("object"),
            "properties": obj([
                "na\"me": obj(["type": .string("string")]),
                "a\u{01}b": obj(["type": .string("string")]),
                "café": obj(["type": .string("string")]),
                "p": obj(["type": .string("string"), "enum": .array([.string("a\\b"), .string("naïve"), .string("😀")])]),
            ]),
            "required": .array([.string("café"), .string("na\"me")]),
        ])
        let object = try compile(schema)
        #expect(object.properties.map(\.name) == ["a\u{01}b", "café", "na\"me", "p"])
        #expect(object.property(named: "p")?.type == .stringEnum(["a\\b", "naïve", "😀"]))
        #expect(object.required == ["café", "na\"me"])
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

    private func compileRoot(_ schema: JSONValue) throws -> SchemaValueType {
        try StructuredOutputFixtures.compileRoot(schema)
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

    /// `minItems` is capped at 65,536, a server limit: every document needs
    /// that many items, so a minimum in the billions would compile and then cut
    /// every generation off at `max_tokens`. `maxItems` is not capped: a large
    /// maximum forces nothing and is enforced exactly.
    @Test
    func capsItemBounds() throws {
        let cap = ResponseFormatDecoder.maxMinItems
        let object = try compile(root([
            "a": array(string, ["minItems": .int(cap)]),
            "b": array(string, ["maxItems": .int(100_000)]),
            "c": array(string, ["maxItems": .int(1_000_000_000)]),
        ]))
        #expect(object.property(named: "a")?.type == .array(items: .string, minItems: cap, maxItems: nil))
        #expect(object.property(named: "b")?.type == .array(items: .string, minItems: 0, maxItems: 100_000))
        #expect(object.property(named: "c")?.type == .array(items: .string, minItems: 0, maxItems: 1_000_000_000))
        expectUnsupported(
            schema: root(["a": array(string, ["minItems": .int(cap + 1)])]),
            containing: "schema too large (minItems 65537 on property 'a' is above the limit of 65536)")
        expectUnsupported(schema: root(["d": array(string, ["minItems": .int(1_000_000_000)])]), containing: "schema too large")
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
        // A `$ref` target is compiled at the root position and a property
        // named "" has the root's empty path; neither is the root, whose
        // root-only keywords were read and stripped before compilation.
        for key in ["$defs", "$id"] {
            let target = obj(["type": .string("object"), "properties": obj(["x": string]), key: obj([:])])
            expectUnsupported(schema: obj(["$ref": .string("#/$defs/T"), "$defs": obj(["T": target])]), containing: "'\(key)'")
            expectUnsupported(schema: root(["": target]), containing: "'\(key)'")
        }
        // Nor does a property named "" take the root's wording or the root's
        // empty path: it is written `""`.
        expectUnsupported(schema: root(["": obj(["type": .string("object")])]), containing: "nested object without 'properties' on property '\"\"'")
        expectUnsupported(schema: root(["": obj(["type": .string("string"), "pattern": .string("x")])]), containing: "'pattern' on property '\"\"'")
        expectUnsupported(
            schema: root(["": obj(["type": .string("object"), "properties": obj(["x": obj(["type": .string("null")])])])]),
            containing: "'null' on property '\"\".x'")
        // A problem inside a `$ref` target names the reference that led there.
        expectUnsupported(
            schema: obj(["$ref": .string("#/$defs/T"), "$defs": obj(["T": obj(["type": .string("object"), "properties": obj(["x": string]), "$id": obj([:])])])]),
            containing: "'$id' at the schema root (via '$ref' '#/$defs/T')")
    }

    /// `required` names are compared scalar by scalar, as the automaton
    /// matches keys: a decomposed "é" does not name a precomposed one, although
    /// Swift's `String` says the two are equal.
    @Test
    func comparesRequiredNamesByScalar() throws {
        let precomposed = "\u{E9}", decomposed = "e\u{301}"
        #expect(precomposed == decomposed)
        let compiled = try compile(root([decomposed: string], required: [decomposed]))
        #expect(compiled.properties.map { Array($0.name.unicodeScalars) } == [Array(decomposed.unicodeScalars)])
        #expect(compiled.required.map { Array($0.unicodeScalars) } == [Array(decomposed.unicodeScalars)])
        expectInvalid(schema: root([precomposed: string], required: [decomposed]), containing: "is not declared")
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

    /// Any string is a literal at every depth and in a `const` too.
    @Test
    func compilesAnyStringLiteralAtEveryDepth() throws {
        let object = try compile(root([
            "o": root(["café": string, "e": obj(["type": .string("string"), "enum": .array([.string("a\\b")])])]),
            "l": array(obj(["type": .string("string"), "enum": .array([.string("naïve")])])),
            "k": obj(["const": .string("say \"hi\"")]),
            "m": obj(["const": .string("Lençóis")]),
        ]))
        let inner = JSONSchemaObject(
            properties: [.init(name: "café", type: .string), .init(name: "e", type: .stringEnum(["a\\b"]))], required: [])
        #expect(object.property(named: "o")?.type == .object(inner))
        #expect(object.property(named: "l")?.type == .array(items: .stringEnum(["naïve"]), minItems: 0, maxItems: nil))
        #expect(object.property(named: "k")?.type == .stringEnum(["say \"hi\""]))
        #expect(object.property(named: "m")?.type == .stringEnum(["Lençóis"]))
    }

    // MARK: json_schema — roots of any type

    /// A root may be an array, a scalar, an enum, a `const` or a `$ref`; the
    /// root-only keywords and annotations apply to it as to an object root.
    @Test
    func compilesNonObjectRoots() throws {
        let items = root(["id": integer], required: ["id"])
        let item = SchemaValueType.object(JSONSchemaObject(properties: [.init(name: "id", type: .integer)], required: ["id"]))
        #expect(try compileRoot(array(items, ["minItems": .int(1)])) == .array(items: item, minItems: 1, maxItems: nil))
        #expect(try compileRoot(string) == .string)
        #expect(try compileRoot(integer) == .integer)
        #expect(try compileRoot(obj(["type": .string("number"), "description": .string("d")])) == .number)
        #expect(try compileRoot(obj(["type": .string("boolean")])) == .boolean)
        #expect(try compileRoot(obj(["type": .string("string"), "enum": .array([.string("a"), .string("é")])])) == .stringEnum(["a", "é"]))
        #expect(try compileRoot(obj(["const": .string("fixed")])) == .stringEnum(["fixed"]))

        let defs: JSONValue = obj(["Addr": root(["street": string, "zip": integer], required: ["street"])])
        let viaRef = obj([
            "$ref": .string("#/$defs/Addr"), "$defs": defs,
            "$schema": .string("https://json-schema.org/draft/2020-12/schema"), "title": .string("T"),
        ])
        #expect(try compileRoot(viaRef) == address)
        let listOfRefs = obj([
            "type": .string("array"), "items": ref("#/$defs/Addr"), "maxItems": .int(3),
            "$defs": defs, "$id": .string("https://example.com/list"),
        ])
        #expect(try compileRoot(listOfRefs) == .array(items: address, minItems: 0, maxItems: 3))
    }

    /// Problems at a non-object root are reported at the root, and a root
    /// with no `type` is still an object.
    @Test
    func reportsRootProblemsAtTheRoot() {
        expectUnsupported(schema: obj(["type": .string("array")]), containing: "array without 'items' at the schema root")
        expectUnsupported(schema: obj(["type": .string("null")]), containing: "property type 'null' at the schema root")
        expectUnsupported(
            schema: obj(["type": .array([.string("string"), .string("null")])]),
            containing: "type arrays (e.g. nullable unions) at the schema root")
        expectUnsupported(schema: obj(["type": .string("string"), "pattern": .string("x")]), containing: "'pattern' at the schema root")
        expectUnsupported(schema: obj(["type": .string("integer"), "enum": .array([.int(1)])]), containing: "enum on non-string schema")
        expectUnsupported(schema: obj(["type": .string("array"), "items": string, "$defs": obj([:]), "x": .int(1)]), containing: "'x' at the schema root")
        expectInvalid(schema: obj(["type": .string("array"), "items": string, "minItems": .int(-1)]), containing: "minItems at the schema root")
        expectInvalid(schema: obj(["enum": .array([.string("a")])]), containing: "schema is missing 'type'")
        expectInvalid(schema: obj(["title": .string("T")]), containing: "schema.properties object is required")
        expectInvalid(schema: obj(["$ref": .string("#/$defs/Missing")]), containing: "at the schema root does not resolve")
        let tooMany = JSONValue.array((0...ResponseFormatDecoder.maxSchemaLiterals).map { .string("v\($0)") })
        expectUnsupported(
            schema: obj(["type": .string("string"), "enum": tooMany]),
            containing: "enum and const values after '$ref' expansion, at the schema root")
        expectUnsupported(
            schema: root(["x": obj(["type": .string("string"), "enum": tooMany])]),
            containing: "enum and const values after '$ref' expansion, on property 'x'")
    }

    /// The constraint rides inside `GenerateRequest` as `Codable`; a
    /// non-object root survives the round trip.
    @Test
    func responseFormatRoundTripsThroughCodable() throws {
        let formats: [ResponseFormat] = [
            .jsonObject,
            .jsonSchema(.array(items: .stringEnum(["é", "x"]), minItems: 1, maxItems: nil)),
            .jsonSchema(.integer),
            .jsonSchema(address),
        ]
        for format in formats {
            let data = try JSONEncoder().encode(format)
            #expect(try JSONDecoder().decode(ResponseFormat.self, from: data) == format)
        }
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

    /// `required` entries are multiplied by `$ref` too: 1,300 references to an
    /// object whose `required` repeats its one key 100,000 times are about
    /// 3,900 nodes, no enum values and a few kilobytes of names, so every
    /// other budget passes, yet the entries would be compiled 130 million
    /// times. JSON Schema requires them to be unique, and the first repeat is
    /// refused. The byte budget, which also charges `required` entries, would
    /// refuse this request too, later and with its own message; both are kept.
    /// Mutation: without the duplicate check this test goes red.
    @Test
    func refusesRepeatedRequiredEntriesUnderRefExpansion() {
        let repeated = JSONValue.array(Array(repeating: .string("a"), count: 100_000))
        var properties: [String: JSONValue] = [:]
        for index in 0..<1_300 { properties["p\(index)"] = ref("#/$defs/D") }
        let schema = root(
            ["o": root(properties)],
            extra: ["$defs": obj(["D": obj(["type": .string("object"), "properties": obj(["a": string]), "required": repeated])])])
        let elapsed = ContinuousClock().measure {
            expectInvalid(schema: schema, containing: "required property 'a' is listed more than once")
        }
        #expect(elapsed < .seconds(1), "took \(elapsed)")
    }

    @Test
    func rejectsDuplicateRequiredEntry() {
        let schema = root(["a": string, "b": string], required: ["a", "b", "a"])
        expectInvalid(schema: schema, containing: "required property 'a' is listed more than once")
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
    /// to a 64 KiB `const`, to an object whose one key is 64 KiB, or to an enum
    /// of 30 values of 60 KiB stay under the node and value budgets (the enum
    /// is 30,000 values in all), yet the automaton's program would copy the
    /// strings once per reference — 64 MB, 64 MB and 1.8 GB. The byte budget
    /// refuses all three. Mutation: without the charge on a name, a `const` or
    /// an enum value, that variant compiles (no 400).
    @Test
    func boundsNameAndValueBytesAfterRefExpansion() {
        let long = String(repeating: "a", count: 65_536)
        var properties: [String: JSONValue] = [:]
        for index in 0..<1_000 { properties["p\(index)"] = ref("#/$defs/E") }
        let enumValues = (0..<30).map { JSONValue.string(String(repeating: "e", count: 61_440) + "\($0)") }
        let variants: [(String, JSONValue)] = [
            ("const", root(["o": root(properties)], extra: ["$defs": obj(["E": obj(["const": .string(long)])])])),
            ("key", root(["o": root(properties)], extra: ["$defs": obj(["E": root([long: string])])])),
            ("enum", root(
                ["o": root(properties)],
                extra: ["$defs": obj(["E": obj(["type": .string("string"), "enum": .array(enumValues)])])])),
        ]
        for (name, schema) in variants {
            let elapsed = ContinuousClock().measure {
                expectUnsupported(schema: schema, containing: "bytes of property names")
            }
            #expect(elapsed < .seconds(1), "\(name): took \(elapsed)")
        }
    }

    /// `required` entries are charged too: a property name just over half the
    /// byte cap compiles while it is optional and is refused once it is also
    /// required, because the name is then counted twice. Mutation: without the
    /// charge on `required` entries the second schema compiles.
    @Test
    func requiredEntriesCountAgainstTheByteBudget() throws {
        let name = String(repeating: "k", count: ResponseFormatDecoder.maxSchemaBytes / 2 + 1)
        _ = try compile(root([name: string]))
        expectUnsupported(schema: root([name: string], required: [name]), containing: "bytes of property names")
    }

    @Test
    func rejectsNonStringConst() {
        expectUnsupported(schema: root(["k": obj(["const": .int(3)])]), containing: "non-string 'const'")
        expectUnsupported(schema: root(["k": obj(["type": .string("integer"), "const": .int(3)])]), containing: "'const' on non-string")
        expectUnsupported(
            schema: root(["k": obj(["const": .string("x"), "enum": .array([.string("x")])])]),
            containing: "'enum'")
    }

    /// `@Guide(.range(1...10))` becomes `minimum`/`maximum`: the Person
    /// fixture compiles as the framework emits it, its rating bounded.
    @Test
    func compilesTheRangeGuide() throws {
        let person = try compile(StructuredOutputFixtures.generable("Person"))
        #expect(person.property(named: "rating")?.type == .boundedInteger(try #require(SchemaIntegerBounds(minimum: 1, maximum: 10))))
        #expect(person.properties.count == 14)
        #expect(person.required.count == 11)
    }

    // MARK: json_schema — numeric bounds

    private func bounded(_ type: String, _ extra: [String: JSONValue]) throws -> SchemaValueType? {
        var property: [String: JSONValue] = ["type": .string(type)]
        property.merge(extra) { _, new in new }
        return try compile(root(["n": obj(property)])).property(named: "n")?.type
    }

    // The helpers record an issue instead of throwing: `#expect` evaluates
    // each operand in its own autoclosure, where a `try` does not reach.

    private func integers(_ minimum: Int?, _ maximum: Int?) -> SchemaValueType {
        guard let bounds = SchemaIntegerBounds(minimum: minimum, maximum: maximum) else {
            Issue.record("no integer between \(String(describing: minimum)) and \(String(describing: maximum))")
            return .integer
        }
        return .boundedInteger(bounds)
    }

    private func numbers(_ minimum: String?, _ maximum: String?, openBelow: Bool = false, openAbove: Bool = false) -> SchemaValueType {
        func decimal(_ text: String) -> SchemaDecimal {
            guard let value = SchemaDecimal(parsing: text) else {
                Issue.record("not a decimal: \(text)")
                return SchemaDecimal(0)
            }
            return value
        }
        guard let bounds = SchemaNumberBounds(
            minimum: minimum.map(decimal), minimumIsExclusive: openBelow,
            maximum: maximum.map(decimal), maximumIsExclusive: openAbove)
        else {
            Issue.record("no number between \(String(describing: minimum)) and \(String(describing: maximum))")
            return .number
        }
        return .boundedNumber(bounds)
    }

    /// Integer bounds fold to the nearest integer inside them: a fractional
    /// `minimum` rounds up, an exclusive bound steps past itself, and the
    /// stricter of the two forms of a side wins.
    @Test
    func compilesIntegerBounds() throws {
        #expect(try bounded("integer", ["minimum": .int(1), "maximum": .int(10)]) == integers(1, 10))
        #expect(try bounded("integer", ["minimum": .int(7)]) == integers(7, nil))
        #expect(try bounded("integer", ["maximum": .int(-3)]) == integers(nil, -3))
        #expect(try bounded("integer", ["minimum": .double(1.5)]) == integers(2, nil))
        #expect(try bounded("integer", ["minimum": .double(-1.5)]) == integers(-1, nil))
        #expect(try bounded("integer", ["exclusiveMinimum": .int(2)]) == integers(3, nil))
        #expect(try bounded("integer", ["exclusiveMinimum": .double(1.5)]) == integers(2, nil))
        #expect(try bounded("integer", ["maximum": .double(2.5)]) == integers(nil, 2))
        #expect(try bounded("integer", ["maximum": .double(-2.5)]) == integers(nil, -3))
        #expect(try bounded("integer", ["exclusiveMaximum": .int(3)]) == integers(nil, 2))
        #expect(try bounded("integer", ["exclusiveMaximum": .double(2.5)]) == integers(nil, 2))
        #expect(try bounded("integer", ["minimum": .int(1), "exclusiveMinimum": .int(1)]) == integers(2, nil))
        #expect(try bounded("integer", ["minimum": .int(5), "exclusiveMinimum": .int(1)]) == integers(5, nil))
        #expect(try bounded("integer", ["maximum": .int(3), "exclusiveMaximum": .int(9)]) == integers(nil, 3))
        #expect(try bounded("integer", ["minimum": .double(-0.0)]) == integers(0, nil))
        #expect(try bounded("integer", ["exclusiveMinimum": .int(2), "exclusiveMaximum": .int(4)]) == integers(3, 3))
        #expect(try bounded("integer", ["minimum": .int(Int.min), "maximum": .int(Int.max)]) == integers(Int.min, Int.max))
    }

    @Test
    func compilesNumberBounds() throws {
        #expect(try bounded("number", ["minimum": .double(0.5), "maximum": .double(2.75)]) == numbers("0.5", "2.75"))
        #expect(try bounded("number", ["minimum": .int(1), "maximum": .int(10)]) == numbers("1", "10"))
        #expect(try bounded("number", ["exclusiveMinimum": .int(0), "exclusiveMaximum": .int(1)]) == numbers("0", "1", openBelow: true, openAbove: true))
        #expect(try bounded("number", ["minimum": .int(0), "exclusiveMinimum": .int(0)]) == numbers("0", nil, openBelow: true))
        #expect(try bounded("number", ["minimum": .int(1), "exclusiveMinimum": .double(0.5)]) == numbers("1", nil))
        #expect(try bounded("number", ["minimum": .double(0.5), "exclusiveMinimum": .int(1)]) == numbers("1", nil, openBelow: true))
        #expect(try bounded("number", ["maximum": .int(1), "exclusiveMaximum": .int(1)]) == numbers(nil, "1", openAbove: true))
        #expect(try bounded("number", ["maximum": .int(1), "exclusiveMaximum": .int(5)]) == numbers(nil, "1"))
        #expect(try bounded("number", ["maximum": .double(-1.25)]) == numbers(nil, "-1.25"))
        #expect(try bounded("number", ["minimum": .double(0.1), "maximum": .double(0.1)]) == numbers("0.1", "0.1"))
        #expect(try bounded("number", ["minimum": .double(-0.0)]) == numbers("0", nil))
    }

    @Test
    func rejectsUnusableBounds() {
        expectUnsupported(schema: root(["s": obj(["type": .string("string"), "minimum": .int(1)])]), containing: "'minimum' on non-numeric property 's'")
        expectUnsupported(schema: root(["b": obj(["type": .string("boolean"), "exclusiveMaximum": .int(1)])]), containing: "'exclusiveMaximum' on non-numeric property 'b'")
        expectUnsupported(
            schema: root(["e": obj(["type": .string("string"), "enum": .array([.string("a")]), "maximum": .int(1)])]),
            containing: "'maximum' alongside 'enum'")
        expectUnsupported(schema: root(["n": obj(["type": .string("number"), "exclusiveMinimum": .bool(true), "minimum": .int(1)])]), containing: "draft 4")
        expectInvalid(schema: root(["n": obj(["type": .string("integer"), "minimum": .string("5")])]), containing: "'minimum' on property 'n' must be a number")
        expectInvalid(schema: root(["n": obj(["type": .string("integer"), "minimum": .int(5), "maximum": .int(3)])]), containing: "admit no integer")
        expectInvalid(schema: root(["n": obj(["type": .string("integer"), "exclusiveMinimum": .int(2), "exclusiveMaximum": .int(3)])]), containing: "admit no integer")
        expectInvalid(schema: root(["n": obj(["type": .string("integer"), "minimum": .double(2.5), "maximum": .double(2.9)])]), containing: "admit no integer")
        expectInvalid(schema: root(["n": obj(["type": .string("number"), "minimum": .int(5), "maximum": .int(3)])]), containing: "admit no number")
        expectInvalid(schema: root(["n": obj(["type": .string("number"), "minimum": .int(1), "exclusiveMaximum": .int(1)])]), containing: "admit no number")
        expectUnsupported(schema: root(["n": obj(["type": .string("number"), "minimum": .double(1e25)])]), containing: "more than 19 significant digits")
        expectUnsupported(schema: root(["n": obj(["type": .string("number"), "maximum": .double(1e-20)])]), containing: "more than 19 significant digits")
        expectUnsupported(schema: root(["n": obj(["type": .string("integer"), "maximum": .double(9.3e18)])]), containing: "'maximum' on property 'n' is beyond the integer range")
        expectUnsupported(schema: root(["n": obj(["type": .string("integer"), "exclusiveMinimum": .int(Int.max)])]), containing: "beyond the integer range")
        expectUnsupported(schema: root(["n": obj(["type": .string("number"), "multipleOf": .int(2)])]), containing: "'multipleOf'")
        expectInvalid(schema: obj(["type": .string("integer"), "minimum": .int(3), "maximum": .int(1)]), containing: "the bounds at the schema root admit no integer")
    }

    /// Apple's TripPlanner sample compiles as the framework emits it, its
    /// enum value "Lençóis Maranhenses" included.
    @Test
    func compilesTheTripPlannerSchemaInFull() throws {
        let trip = try compile(StructuredOutputFixtures.itinerary())
        #expect(trip.required == ["title", "destinationName", "description", "rationale", "days"])
        guard case .stringEnum(let destinations)? = trip.property(named: "destinationName")?.type else {
            Issue.record("destinationName should be an enum")
            return
        }
        #expect(destinations.contains("Lençóis Maranhenses"))
        #expect(try compile(StructuredOutputFixtures.asciiItinerary()) != trip)
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
