import Testing

@testable import MacMLXCore

// MARK: - SchemaConstraintState Tests (Track C — C2)
//
// Pure, MLX-free tests for the schema-specific automaton: only the declared
// object shape is accepted, keys are unique and any-order, required keys are
// enforced, and each value must match its declared type.

@Suite("SchemaConstraintState")
struct SchemaConstraintStateTests {

    private func schema(
        _ properties: [(String, SchemaValueType)],
        required: [String] = []
    ) -> JSONSchemaObject {
        JSONSchemaObject(
            properties: properties.map { .init(name: $0.0, type: $0.1) },
            required: required
        )
    }

    private func accepts(_ text: String, _ object: JSONSchemaObject) -> Bool {
        guard let end = SchemaConstraintState(schema: object).walk(Array(text.utf8)) else { return false }
        return end.isComplete
    }

    // MARK: Types

    @Test
    func acceptsTypedValues() {
        let s = schema([
            ("name", .string), ("age", .integer), ("score", .number), ("active", .boolean),
        ])
        #expect(accepts("{\"name\":\"Ada\",\"age\":36,\"score\":9.5,\"active\":true}", s))
        #expect(accepts("{ \"name\" : \"Ada\" , \"age\" : -1 }", s))   // ws + subset of props
        #expect(accepts("{}", s))                                     // nothing required
    }

    @Test
    func enforcesIntegerVsNumber() {
        let s = schema([("age", .integer)])
        #expect(accepts("{\"age\":36}", s))
        #expect(accepts("{\"age\":-36}", s))
        #expect(!accepts("{\"age\":3.6}", s))     // fraction not allowed for integer
        #expect(!accepts("{\"age\":1e3}", s))     // exponent not allowed for integer
        #expect(!accepts("{\"age\":01}", s))      // leading zero
    }

    @Test
    func acceptsNumberFractionsAndExponents() {
        let s = schema([("x", .number)])
        for v in ["0", "-0", "3.14", "1e10", "-2.5e-3", "42"] {
            #expect(accepts("{\"x\":\(v)}", s), "expected \(v)")
        }
        #expect(!accepts("{\"x\":.5}", s))
        #expect(!accepts("{\"x\":1.}", s))
    }

    @Test
    func enforcesBooleanLiterals() {
        let s = schema([("b", .boolean)])
        #expect(accepts("{\"b\":true}", s))
        #expect(accepts("{\"b\":false}", s))
        #expect(!accepts("{\"b\":True}", s))
        #expect(!accepts("{\"b\":1}", s))
        #expect(!accepts("{\"b\":null}", s))
    }

    @Test
    func enforcesStringEnum() {
        let s = schema([("role", .stringEnum(["admin", "user", "guest"]))])
        #expect(accepts("{\"role\":\"admin\"}", s))
        #expect(accepts("{\"role\":\"guest\"}", s))
        #expect(!accepts("{\"role\":\"root\"}", s))       // not in enum
        #expect(!accepts("{\"role\":\"admi\"}", s))       // prefix, not complete
        #expect(!accepts("{\"role\":\"adminx\"}", s))     // superset
        #expect(!accepts("{\"role\":admin}", s))          // missing quotes
    }

    @Test
    func acceptsStringWithEscapes() {
        let s = schema([("msg", .string)])
        #expect(accepts("{\"msg\":\"hi\\nthere\"}", s))
        #expect(accepts("{\"msg\":\"q\\\"q\"}", s))
        #expect(accepts("{\"msg\":\"u\\u00e9\"}", s))
        #expect(!accepts("{\"msg\":\"bad\\x\"}", s))
    }

    @Test
    func enforcesSurrogatePairingInStringValues() {
        let s = schema([("msg", .string)])
        // A complete surrogate pair is accepted; unpaired surrogates (which
        // JSONSerialization rejects) are not.
        #expect(accepts("{\"msg\":\"\\uD83D\\uDE00\"}", s))
        #expect(!accepts("{\"msg\":\"\\uD83D\"}", s))          // lone high
        #expect(!accepts("{\"msg\":\"\\uDE00\"}", s))          // lone low
        #expect(!accepts("{\"msg\":\"\\uD83D\\u0041\"}", s))   // high + non-low
    }

    // MARK: Keys

    @Test
    func rejectsUndeclaredKeys() {
        let s = schema([("a", .string)])
        #expect(!accepts("{\"b\":\"x\"}", s))
        #expect(!accepts("{\"a\":\"x\",\"b\":\"y\"}", s))
    }

    @Test
    func rejectsDuplicateKeys() {
        let s = schema([("a", .string), ("b", .string)])
        #expect(!accepts("{\"a\":\"x\",\"a\":\"y\"}", s))
        #expect(accepts("{\"a\":\"x\",\"b\":\"y\"}", s))
    }

    @Test
    func acceptsKeysInAnyOrder() {
        let s = schema([("a", .string), ("b", .integer)], required: ["a", "b"])
        #expect(accepts("{\"a\":\"x\",\"b\":1}", s))
        #expect(accepts("{\"b\":1,\"a\":\"x\"}", s))
    }

    // MARK: Required

    @Test
    func enforcesRequiredPresence() {
        let s = schema([("a", .string), ("b", .integer)], required: ["a"])
        #expect(accepts("{\"a\":\"x\"}", s))
        #expect(accepts("{\"a\":\"x\",\"b\":2}", s))
        #expect(!accepts("{}", s))                 // missing required 'a'
        #expect(!accepts("{\"b\":2}", s))          // missing required 'a'
    }

    @Test
    func requiredNotSatisfiedIsNotComplete() {
        let s = schema([("a", .string)], required: ["a"])
        // A prefix that opened the brace but hasn't supplied 'a' is not complete,
        // and the close brace is illegal there.
        let state = SchemaConstraintState(schema: s)
        #expect(state.walk(Array("{".utf8))?.isComplete == false)
        #expect(state.walk(Array("{}".utf8)) == nil)
    }

    // MARK: Structure

    @Test
    func rejectsNonObjectRoot() {
        let s = schema([("a", .string)])
        #expect(!accepts("[]", s))
        #expect(!accepts("\"x\"", s))
        #expect(!accepts("123", s))
    }

    @Test
    func rejectsTrailingCommaAndGarbage() {
        let s = schema([("a", .string), ("b", .string)])
        #expect(!accepts("{\"a\":\"x\",}", s))
        #expect(!accepts("{\"a\":\"x\"}x", s))
        #expect(accepts("{\"a\":\"x\"}  ", s))    // trailing whitespace ok
    }
}
