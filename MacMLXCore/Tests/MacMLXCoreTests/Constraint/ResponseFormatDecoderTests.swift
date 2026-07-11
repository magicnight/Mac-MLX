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
