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
            // Nested: the same shapes one level down, and the array bounds.
            schema([("o", nested([("k", .boolean)]))]),
            schema([("o", nested([("a", .integer), ("ab", .integer)], required: ["ab"]))], required: ["o"]),
            schema([("t", .array(items: .string, minItems: 2, maxItems: 3))], required: ["t"]),
            schema([("z", .array(items: .boolean, minItems: 0, maxItems: 0))]),
            schema([("m", .array(items: .array(items: .integer, minItems: 1, maxItems: 2), minItems: 0, maxItems: nil))]),
            schema([("i", .array(items: nested([("id", .integer), ("t", .string)], required: ["id"]), minItems: 1, maxItems: 2))]),
        ]
        var generator = RandomSchemaGenerator(seed: 42)
        for _ in 0..<12 {
            schemas.append(generator.object())
        }
        for _ in 0..<12 {
            schemas.append(generator.object(nested: true))
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

    // MARK: Nested objects

    private func nested(_ properties: [(String, SchemaValueType)], required: [String] = []) -> SchemaValueType {
        .object(schema(properties, required: required))
    }

    private func walk(_ text: String, _ object: JSONSchemaObject) -> SchemaConstraintState? {
        SchemaConstraintState(schema: object).walk(Array(text.utf8))
    }

    private var address: SchemaValueType {
        nested([("street", .string), ("zip", .integer)], required: ["street"])
    }

    /// Each object has its own emitted set and its own required keys.
    @Test
    func nestedObjectKeysAreScopedToTheirObject() {
        let s = schema([("name", .string), ("home", address)], required: ["home"])
        #expect(accepts("{\"home\":{\"street\":\"Main\"}}", s))
        #expect(accepts("{\"name\":\"x\",\"home\":{\"zip\":1,\"street\":\"M\"}}", s))
        #expect(accepts("{ \"home\" : { \"street\" : \"M\" } , \"name\" : \"x\" }", s))
        #expect(!accepts("{\"home\":{}}", s))                               // nested required
        #expect(!accepts("{\"home\":{\"zip\":1}}", s))
        #expect(!accepts("{\"home\":{\"street\":\"M\",\"street\":\"N\"}}", s))  // nested duplicate
        #expect(!accepts("{\"home\":{\"street\":\"M\",\"name\":\"N\"}}", s))    // parent's key, nested
        #expect(!accepts("{\"home\":\"x\"}", s))                             // wrong type
        #expect(!accepts("{\"home\":{\"street\":\"M\"},\"home\":{\"street\":\"M\"}}", s))  // parent duplicate
        #expect(!accepts("{\"name\":\"x\"}", s))                             // parent required
    }

    @Test
    func emptyNestedObjectWhenNothingIsRequired() {
        let s = schema([("home", nested([("zip", .integer)]))])
        #expect(accepts("{\"home\":{}}", s))
        #expect(accepts("{\"home\":{ }}", s))
        #expect(!accepts("{\"home\":{,}}", s))
    }

    // MARK: Arrays

    @Test
    func arrayItemBoundsAreEnforcedByteByByte() {
        let s = schema([("tags", .array(items: .string, minItems: 2, maxItems: 3))], required: ["tags"])
        #expect(!accepts("{\"tags\":[]}", s))
        #expect(!accepts("{\"tags\":[\"a\"]}", s))
        #expect(accepts("{\"tags\":[\"a\",\"b\"]}", s))
        #expect(accepts("{\"tags\":[ \"a\" , \"b\" , \"c\" ]}", s))
        #expect(walk("{\"tags\":[\"a\",\"b\",\"c\",", s) == nil, "a comma promises an item the array cannot hold")
        #expect(walk("{\"tags\":[\"a\",\"b\",\"c\" ,", s) == nil)
        #expect(walk("{\"tags\":[\"a\"]", s) == nil, "closing below minItems")
        #expect(walk("{\"tags\":[\"a\",\"b\",\"c\"", s)?.walk(Array("]}".utf8))?.isComplete == true)
        #expect(!accepts("{\"tags\":[\"a\",\"b\",]}", s))   // trailing comma
        #expect(!accepts("{\"tags\":[,\"a\",\"b\"]}", s))   // leading comma
        #expect(!accepts("{\"tags\":[\"a\" \"b\"]}", s))    // missing comma
    }

    @Test
    func unboundedIntegerItems() {
        let s = schema([("n", .array(items: .integer, minItems: 0, maxItems: nil))])
        #expect(accepts("{\"n\":[]}", s))
        #expect(accepts("{\"n\":[ ]}", s))
        #expect(accepts("{\"n\":[1,-2,30]}", s))
        #expect(accepts("{\"n\":[0 ,0]}", s))
        #expect(accepts("{\"n\":[\(Array(repeating: "7", count: 50).joined(separator: ","))]}", s))
        #expect(!accepts("{\"n\":[1.5]}", s))
        #expect(!accepts("{\"n\":[01]}", s))
        #expect(!accepts("{\"n\":[1}", s))     // a number's terminator goes to the array, not the object
        #expect(!accepts("{\"n\":[\"1\"]}", s))
    }

    @Test
    func maxItemsZeroAdmitsOnlyTheEmptyArray() {
        let s = schema([("z", .array(items: .boolean, minItems: 0, maxItems: 0))])
        #expect(accepts("{\"z\":[]}", s))
        #expect(walk("{\"z\":[t", s) == nil)
        #expect(walk("{\"z\":[ f", s) == nil)
    }

    @Test
    func enumItems() {
        let s = schema([("e", .array(items: .stringEnum(["x", "xy"]), minItems: 1, maxItems: nil))])
        #expect(accepts("{\"e\":[\"x\",\"xy\",\"x\"]}", s))
        #expect(!accepts("{\"e\":[\"y\"]}", s))
        #expect(!accepts("{\"e\":[\"xyz\"]}", s))
    }

    @Test
    func numberItemsEndOnTheContainersBytes() {
        let s = schema([("f", .array(items: .number, minItems: 1, maxItems: 2))])
        #expect(accepts("{\"f\":[1e5,-0.5]}", s))
        #expect(accepts("{\"f\":[3]}", s))
        #expect(accepts("{\"f\":[0 ]}", s))
        #expect(!accepts("{\"f\":[1,2,3]}", s))
        #expect(!accepts("{\"f\":[1e]}", s))
    }

    /// Each item of an array of objects is its own object: a fresh emitted set,
    /// its own required keys, and one item counted per `{`.
    @Test
    func arrayOfObjects() {
        let item = nested([("id", .integer), ("tag", .string)], required: ["id"])
        let s = schema([("items", .array(items: item, minItems: 1, maxItems: 2))], required: ["items"])
        #expect(accepts("{\"items\":[{\"id\":1},{\"tag\":\"t\",\"id\":2}]}", s))
        #expect(accepts("{\"items\":[{\"id\":1,\"tag\":\"a\"}]}", s))
        #expect(accepts("{\"items\":[{\"id\":1},{\"id\":1}]}", s))          // same keys in each item
        #expect(!accepts("{\"items\":[{\"tag\":\"t\"}]}", s))              // required per item
        #expect(!accepts("{\"items\":[{\"id\":1},{\"tag\":\"t\"}]}", s))
        #expect(!accepts("{\"items\":[{\"id\":1},{\"id\":2},{\"id\":3}]}", s))  // third item
        #expect(walk("{\"items\":[{\"id\":1},{\"id\":2},", s) == nil)      // comma when full
        #expect(!accepts("{\"items\":[]}", s))
    }

    @Test
    func arrayOfArrays() {
        let row = SchemaValueType.array(items: .integer, minItems: 1, maxItems: 2)
        let s = schema([("m", .array(items: row, minItems: 0, maxItems: nil))])
        #expect(accepts("{\"m\":[[1],[2,3],[4]]}", s))
        #expect(accepts("{\"m\":[]}", s))
        #expect(accepts("{\"m\":[ [ 1 ] , [2 ,3] ]}", s))
        #expect(!accepts("{\"m\":[[]]}", s))           // inner minItems
        #expect(!accepts("{\"m\":[[1,2,3]]}", s))      // inner maxItems
        #expect(!accepts("{\"m\":[1]}", s))            // item must be an array
        #expect(!accepts("{\"m\":[[1],]}", s))
    }

    /// One token can close and open several containers; every frame it crosses
    /// keeps its own bounds.
    @Test
    func oneTokenSpanningSeveralFrames() throws {
        let inner = nested([("y", .array(items: .string, minItems: 0, maxItems: 1))], required: ["y"])
        let s = schema([("x", .array(items: inner, minItems: 1, maxItems: 2))], required: ["x"])
        let mid = try #require(walk("{\"x\":[{\"y\":[\"a", s))
        #expect(mid.walk(Array("\"]},{\"y\":[]}]}".utf8))?.isComplete == true)
        #expect(mid.walk(Array("\",\"b".utf8)) == nil, "the inner array holds at most one item")
        #expect(mid.walk(Array("\"]}]}".utf8))?.isComplete == true)
        #expect(mid.walk(Array("\"]}]".utf8))?.isComplete == false)
        #expect(mid.walk(Array("\"]},{\"y\":[]},{".utf8)) == nil, "the outer array holds at most two")
    }

    /// Key candidates are narrowed byte by byte and exclude emitted keys, in a
    /// nested object as at the root.
    @Test
    func nestedKeysSharingAPrefix() {
        let s = schema([("o", nested([("a", .integer), ("ab", .integer)], required: ["ab"]))])
        #expect(accepts("{\"o\":{\"a\":1,\"ab\":2}}", s))
        #expect(accepts("{\"o\":{\"ab\":2,\"a\":1}}", s))
        #expect(accepts("{\"o\":{\"ab\":2}}", s))
        #expect(!accepts("{\"o\":{\"a\":1}}", s))
        #expect(walk("{\"o\":{\"a\":1,\"a\"", s) == nil)
        #expect(walk("{\"o\":{\"a\":1,\"a", s) != nil, "still a prefix of 'ab'")
        #expect(walk("{\"o\":{\"ab\":1,\"ab", s) == nil)
    }

    /// The comma guard (C1) holds in every object: no `,` once every declared
    /// key is emitted, at the root or nested.
    @Test
    func noCommaAfterTheLastKeyAtAnyDepth() {
        #expect(walk("{\"a\":\"x\",", schema([("a", .string)])) == nil)
        let s = schema([("o", nested([("k", .boolean)]))])
        #expect(walk("{\"o\":{\"k\":true,", s) == nil)
        #expect(walk("{\"o\":{\"k\":true", s) != nil)
        #expect(accepts("{\"o\":{\"k\":true}}", s))
        #expect(walk("{\"o\":{\"k\":true},", s) == nil)
    }

    // MARK: Wide objects and enums

    /// More than 64 members: emitted, required and candidate masks spill into
    /// a second word.
    @Test
    func objectsWiderThanSixtyFourMembers() throws {
        let names = (0..<70).map { "p\($0)" }
        let wide = schema(names.map { ($0, SchemaValueType.integer) }, required: ["p65", "p3"])
        #expect(accepts("{\"p65\":1,\"p3\":2}", wide))
        #expect(accepts("{\"p3\":2,\"p69\":0,\"p65\":1}", wide))
        #expect(!accepts("{\"p3\":2}", wide))           // high-word required missing
        #expect(!accepts("{\"p65\":1}", wide))          // low-word required missing
        #expect(walk("{\"p66\":1,\"p66\"", wide) == nil)  // duplicate high-index key
        #expect(walk("{\"p66\":1,\"p6\"", wide) != nil)   // its low-word prefix name is still free
        let all = names.map { "\"\($0)\":1" }.joined(separator: ",")
        #expect(accepts("{" + all + "}", wide))
        #expect(walk("{" + all + ",", wide) == nil, "no key left after all 70")
        let allButLast = try #require(walk("{" + names.dropLast().map { "\"\($0)\":1" }.joined(separator: ","), wide))
        #expect(allButLast.walk(Array(",\"p69\":1}".utf8))?.isComplete == true)

        let values = (0..<100).map { "v\($0)" }
        let choice = schema([("e", .stringEnum(values))], required: ["e"])
        for value in ["v0", "v1", "v10", "v63", "v64", "v99"] {
            #expect(accepts("{\"e\":\"\(value)\"}", choice), "\(value)")
        }
        for value in ["v100", "v", "w1", "v999", "v640"] {
            #expect(!accepts("{\"e\":\"\(value)\"}", choice), "\(value)")
        }
    }

    // MARK: Equality

    /// States are equal at the same position of equal schemas, even when the
    /// schemas were compiled separately; the hash agrees.
    @Test
    func equalityIsByPositionAndSchema() throws {
        let object = schema([("o", nested([("k", .boolean)])), ("t", .array(items: .integer, minItems: 0, maxItems: nil))])
        let a = SchemaConstraintState(schema: object)
        let b = SchemaConstraintState(schema: object)
        #expect(a == b)
        #expect(a.hashValue == b.hashValue)

        let prefix = Array("{\"o\":{\"k\":true},\"t\":[1,".utf8)
        let a1 = try #require(a.walk(prefix))
        let b1 = try #require(b.walk(prefix))
        #expect(a1 == b1)
        #expect(a1.hashValue == b1.hashValue)
        #expect(Set([a1, b1]).count == 1)

        #expect(a1 != a)
        #expect(a.walk(Array("{\"t\":[1".utf8)) != a.walk(Array("{\"t\":[1,2".utf8)))
        // Different paths to the same position are the same state.
        #expect(a.walk(Array("{\"o\":{\"k\":true}".utf8)) == a.walk(Array("{\"o\":{\"k\":false}".utf8)))

        // Same node layout, different schema: not equal.
        let other = schema([("o", nested([("k", .boolean)])), ("t", .array(items: .number, minItems: 0, maxItems: nil))])
        #expect(SchemaConstraintState(schema: other) != a)
    }

    // MARK: Differential test against a reference validator

    /// Seeded schemas (flat and nested) × valid and mutated documents: the
    /// automaton accepts exactly what ``ReferenceSchemaValidator`` accepts,
    /// and everything it accepts is well-formed JSON to ``JSONGrammarState``.
    @Test
    func agreesWithTheReferenceValidator() {
        var generator = RandomSchemaGenerator(seed: 0xC0FFEE)
        var accepted = 0
        var mismatches: [String] = []
        for round in 0..<500 {
            let object = generator.object(nested: round % 2 == 1)
            let start = SchemaConstraintState(schema: object)
            for k in 0..<12 {
                let valid = generator.document(for: .object(object))
                let document = k < 4 ? Array(valid.utf8) : generator.mutate(valid)
                let automaton = start.walk(document)?.isComplete ?? false
                let reference = ReferenceSchemaValidator.validate(document, object)
                if automaton != reference {
                    mismatches.append("automaton=\(automaton) reference=\(reference) \(String(decoding: document, as: UTF8.self)) for \(object)")
                    continue
                }
                if automaton {
                    accepted += 1
                    #expect(JSONGrammarState().walk(document)?.isComplete == true, "\(String(decoding: document, as: UTF8.self))")
                }
            }
        }
        #expect(mismatches.isEmpty, "\(mismatches.count) mismatches, first: \(mismatches.first ?? "-")")
        #expect(accepted > 2_000, "too few accepted documents (\(accepted)) to mean anything")
    }
}
