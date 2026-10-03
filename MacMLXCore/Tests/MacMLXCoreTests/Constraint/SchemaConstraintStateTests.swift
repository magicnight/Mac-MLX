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

    /// Seen on a real checkpoint: with every declared key emitted, a comma was
    /// still accepted, after which only whitespace was legal — the model could
    /// never close the object and ran to max_tokens emitting blanks.
    @Test
    func rejectsCommaOnceEveryKeyIsEmitted() {
        let s = schema([("a", .string), ("b", .string)], required: ["a"])
        #expect(accepts("{\"a\":\"x\",\"b\":\"y\"}", s))
        #expect(!accepts("{\"a\":\"x\",\"b\":\"y\",}", s))

        let afterLast = SchemaConstraintState(schema: s).walk(Array("{\"a\":\"x\",\"b\":\"y\"".utf8))
        #expect(afterLast?.walk(Array(",".utf8)) == nil, "no key left to promise")
        #expect(afterLast?.walk(Array(" ,".utf8)) == nil)
        #expect(afterLast?.walk(Array(" }".utf8))?.isComplete == true)

        // With a key still available the comma stays legal.
        let afterFirst = SchemaConstraintState(schema: s).walk(Array("{\"a\":\"x\"".utf8))
        #expect(afterFirst?.walk(Array(",".utf8)) != nil)
        // ...and so does closing early, since only `a` is required.
        #expect(afterFirst?.walk(Array("}".utf8))?.isComplete == true)
    }

    // MARK: Surrogate escapes (C2)

    /// A `\u` escape is cut off as soon as no completion of it could be legal.
    /// The surrogate range used to be checked only at the fourth digit, so
    /// `\uDC`–`\uDF` outside a pair and `\uD83D\u00` were accepted and then no
    /// byte could follow: the no-legal-token path, and a truncated document.
    @Test
    func prunesDeadSurrogateEscapePrefixes() {
        let s = schema([("msg", .string)])
        let start = SchemaConstraintState(schema: s)
        func walks(_ escape: String) -> Bool {
            start.walk(Array("{\"msg\":\"\(escape)".utf8)) != nil
        }
        // Outside a pair, the second digit decides a lone low surrogate.
        for escape in ["\\uDC", "\\uDD", "\\uDE", "\\uDF", "\\udc"] {
            #expect(!walks(escape), "\(escape)")
        }
        // The second half of a pair must be DC–DF, decided by its first two digits.
        #expect(!walks("\\uD83D\\u00"))
        #expect(!walks("\\uD83D\\u0"))
        #expect(!walks("\\uD83D\\uD8"))
        // Not over-pruned: prefixes of legal escapes still walk.
        #expect(walks("\\uD83D\\uD"))
        #expect(walks("\\uD83D\\uDC"))
        for escape in ["\\uD8", "\\uDB", "\\uD7", "\\uE0", "\\u00"] {
            #expect(walks(escape), "\(escape)")
        }
        #expect(accepts("{\"msg\":\"\\uD83D\\uDE00\\uD7FF\\uE000\"}", s))
    }

    // MARK: No trap states

    /// Every state reachable over a small alphabet can still reach a complete
    /// document. A state that cannot is a trap: the processor finds no legal
    /// token there and forces EOS on a truncated output. Breadth-first over at
    /// most 40k states per schema, then backward co-reachability from the
    /// complete states (see ``SchemaTrapSearch``). Fixed schemas cover the
    /// shapes of known traps (a comma after the last key, a dead surrogate
    /// escape); seeded schemas cover the rest.
    @Test
    func noReachableStateIsATrap() {
        var schemas: [JSONSchemaObject] = [
            schema([("a", .string)]),
            schema([("a", .string)], required: ["a"]),
            schema([("a", .string), ("ab", .integer), ("b", .number)], required: ["ab"]),
            schema([("e", .stringEnum(["x", "xy"])), ("f", .boolean)], required: ["e", "f"]),
            schema([("", .stringEnum([""])), ("n", .number)]),
        ]
        var generator = RandomSchemaGenerator(seed: 42)
        for _ in 0..<12 {
            schemas.append(generator.object())
        }
        for object in schemas {
            let result = SchemaTrapSearch.run(
                from: SchemaConstraintState(schema: object),
                alphabet: SchemaTrapSearch.alphabet(for: object),
                limit: 40_000)
            #expect(
                result.traps.isEmpty,
                "\(result.traps.count) trap(s) in \(result.explored) states, first: \(result.traps.first?.diagnosticDescription ?? "-") for \(object)")
        }
    }
}
