import Testing

@testable import MacMLXCore

// MARK: - SchemaConstraintState Tests (Track C — C2)
//
// Pure, MLX-free tests for the schema-specific automaton: only the declared
// shape is accepted (an object, an array or a scalar at the root), keys are
// unique and any-order, required keys are enforced, each value must match its
// declared type, and keys and enum values match scalar by scalar, raw or
// escaped.

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
        acceptsRoot(text, .object(object))
    }

    private func acceptsRoot(_ text: String, _ root: SchemaValueType) -> Bool {
        guard let end = SchemaConstraintState(root: root).walk(Array(text.utf8)) else { return false }
        return end.isComplete
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

    // MARK: Literals — any Unicode, raw or escaped

    /// A key or enum value outside ASCII matches its raw UTF-8, its `\u`
    /// escapes in either hex case, or a mix, and a multi-byte scalar may
    /// arrive one byte at a time (a token can end inside it).
    @Test
    func matchesNonASCIILiteralsRawOrEscaped() {
        let s = schema([("ciudad", .stringEnum(["Lençóis Maranhenses", "Bogotá"])), ("日本", .string)], required: ["ciudad"])
        #expect(accepts("{\"ciudad\":\"Lençóis Maranhenses\"}", s))
        #expect(accepts("{\"ciudad\":\"Len\\u00e7\\u00f3is Maranhenses\"}", s))
        #expect(accepts("{\"ciudad\":\"Len\\u00E7óis Maranhenses\"}", s))
        #expect(accepts("{\"ciudad\":\"Bogot\\u00e1\",\"日本\":\"x\"}", s))
        #expect(accepts("{\"\\u65e5\\u672c\":\"x\",\"ciudad\":\"Bogotá\"}", s))
        #expect(accepts("{\"\\u65e5本\":\"x\",\"ciudad\":\"Bogotá\"}", s))
        #expect(!accepts("{\"ciudad\":\"Lencois Maranhenses\"}", s))
        #expect(!accepts("{\"ciudad\":\"Len\\u00e8óis Maranhenses\"}", s))
        #expect(!accepts("{\"ciudad\":\"Len\\u00e7\"}", s))
        #expect(!accepts("{\"\\u65e5\":\"x\"}", s))
        #expect(!accepts("{\"ciudad\":\"Len\\u00e7\\u00f3is Maranhenses\",\"ciudad\":\"Bogotá\"}", s))

        let start = SchemaConstraintState(schema: s)
        let mid = start.walk(Array("{\"ciudad\":\"Len".utf8) + [0xC3])
        #expect(mid != nil)
        #expect(mid?.walk([0xA9]) == nil, "é is not ç")
        #expect(mid?.walk([0xA7] + Array("óis Maranhenses\"}".utf8))?.isComplete == true)
        #expect(start.walk(Array("{\"ciudad\":\"Len".utf8) + [0xE7]) == nil, "a lead byte of the wrong length")
        #expect(start.walk(Array("{\"ciudad\":\"Len\\u00".utf8))?.isInsideString == true)
    }

    /// A scalar above the BMP is one raw four-byte sequence or one surrogate
    /// pair; half a pair, or a pair whose second half is not a low surrogate,
    /// matches nothing.
    @Test
    func matchesSupplementaryScalarsAsSurrogatePairs() {
        let s = schema([("mood", .stringEnum(["😀", "😁"])), ("x", .string)])
        #expect(accepts("{\"mood\":\"😀\"}", s))
        #expect(accepts("{\"mood\":\"\\ud83d\\ude00\"}", s))
        #expect(accepts("{\"mood\":\"\\uD83D\\uDE01\"}", s))
        #expect(!accepts("{\"mood\":\"\\ud83d\\ude02\"}", s))
        #expect(!accepts("{\"mood\":\"\\ud83d\"}", s))
        #expect(!accepts("{\"mood\":\"\\ud83d\\u0041\"}", s))
        #expect(walk("{\"mood\":\"\\ud83d\\u", s) != nil)
        #expect(walk("{\"mood\":\"\\ud83d\\ud", s) != nil, "a low surrogate starts with D")
        #expect(walk("{\"mood\":\"\\ud83d\\ud8", s) == nil, "the second half must be a low surrogate")
        #expect(walk("{\"mood\":\"\\ud83d\\u0", s) == nil)
        #expect(walk("{\"mood\":\"\\ud83dx", s) == nil)
        #expect(walk("{\"mood\":\"\\ude00", s) == nil, "a lone low surrogate")
    }

    /// The quote, the backslash and control characters can only be matched
    /// escaped — the short escape or `\u` — while every other ASCII scalar is
    /// matched raw only: `\u0041` is not an `A` here, so a model that is off
    /// its preferred path cannot drift into spelling a whole literal as
    /// escapes (seen on a real checkpoint before this rule).
    @Test
    func matchesEscapedASCIIAndControlCharactersInLiterals() {
        let s = schema([("q\"q", .stringEnum(["a\\b", "line\nbreak", "A", "a/b"]))], required: ["q\"q"])
        #expect(accepts("{\"q\\\"q\":\"a\\\\b\"}", s))
        #expect(accepts("{\"q\\u0022q\":\"a\\u005cb\"}", s))
        #expect(accepts("{\"q\\\"q\":\"line\\nbreak\"}", s))
        #expect(accepts("{\"q\\\"q\":\"line\\u000Abreak\"}", s))
        #expect(accepts("{\"q\\\"q\":\"A\"}", s))
        #expect(accepts("{\"q\\\"q\":\"a/b\"}", s))
        #expect(!accepts("{\"q\\\"q\":\"\\u0041\"}", s))
        #expect(!accepts("{\"q\\\"q\":\"a\\/b\"}", s))
        #expect(!accepts("{\"\\u0071\\\"q\":\"A\"}", s))
        #expect(!accepts("{\"q\"q\":\"A\"}", s))
        #expect(!accepts("{\"q\\\"q\":\"line\nbreak\"}", s))
        #expect(!accepts("{\"q\\\"q\":\"a\\b\"}", s))
        #expect(!accepts("{\"q\\\"q\":\"a\\x\"}", s))
        #expect(walk("{\"q\\\"q\":\"A\\", s) == nil, "nothing after A may be escaped")
    }

    // MARK: Structure

    @Test
    func rejectsNonObjectDocumentUnderAnObjectRoot() {
        let s = schema([("a", .string)])
        #expect(!accepts("[]", s))
        #expect(!accepts("\"x\"", s))
        #expect(!accepts("123", s))
    }

    // MARK: Roots of any type

    @Test
    func acceptsNonObjectRoots() {
        let item = nested([("id", .integer)], required: ["id"])
        let list = SchemaValueType.array(items: item, minItems: 1, maxItems: 2)
        #expect(acceptsRoot("[{\"id\":1}]", list))
        #expect(acceptsRoot(" [ {\"id\":1} , {\"id\":2} ] ", list))
        #expect(!acceptsRoot("[]", list))
        #expect(!acceptsRoot("[{\"id\":1},{\"id\":2},{\"id\":3}]", list))
        #expect(!acceptsRoot("{\"id\":1}", list))
        #expect(!acceptsRoot("[{\"id\":1}]]", list))
        #expect(acceptsRoot("\"hi\"", .string))
        #expect(acceptsRoot("\"\\u00e9\"", .string))
        #expect(!acceptsRoot("hi", .string))
        #expect(!acceptsRoot("\"hi\" \"\"", .string))
        #expect(acceptsRoot("\"admin\"", .stringEnum(["admin", "user"])))
        #expect(!acceptsRoot("\"\\u0061dmin\"", .stringEnum(["admin", "user"])))
        #expect(acceptsRoot("\"caf\\u00e9\"", .stringEnum(["café", "user"])))
        #expect(!acceptsRoot("\"root\"", .stringEnum(["admin", "user"])))
        #expect(acceptsRoot("true", .boolean))
        #expect(acceptsRoot(" false ", .boolean))
        #expect(!acceptsRoot("tru", .boolean))
        #expect(!acceptsRoot("{}", .boolean))
    }

    /// A root number has no terminator: it is complete while more digits
    /// could still follow, whitespace ends it, and nothing else may follow.
    @Test
    func rootNumbersAreCompleteWithoutATerminator() {
        for text in ["42", "-0", "3.14", "1e5", "-2.5E-3", "0"] {
            #expect(acceptsRoot(text, .number), "\(text)")
            #expect(acceptsRoot(text + " ", .number), "\(text)")
        }
        for text in ["-", "1.", "1e", "1e+", ".5", "01", "+1", "1 2", "1,"] {
            #expect(!acceptsRoot(text, .number), "\(text)")
        }
        for text in ["42", "-7", "0"] { #expect(acceptsRoot(text, .integer), "\(text)") }
        for text in ["1.5", "1e3", "01", "-", "7]"] { #expect(!acceptsRoot(text, .integer), "\(text)") }
        let afterDigits = SchemaConstraintState(root: .integer).walk(Array("12".utf8))
        #expect(afterDigits?.isComplete == true)
        #expect(afterDigits?.walk(Array("3".utf8))?.isComplete == true)
        #expect(afterDigits?.walk(Array("}".utf8)) == nil)
        // Inside a container a number still needs its terminator.
        #expect(walk("{\"n\":12", schema([("n", .integer)]))?.isComplete == false)
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

    // MARK: Numeric bounds

    /// A bounded integer: a digit is legal only while some completion fits
    /// the range, and the number ends only on a value in it.
    @Test
    func boundedIntegersEndOnlyInRange() throws {
        let r = schema([("r", integers(-12, 35))], required: ["r"])
        for v in ["-12", "-1", "0", "7", "35", "-0", "30"] { #expect(accepts("{\"r\":\(v)}", r), "\(v)") }
        for v in ["-13", "36", "40", "100", "01", "3.5", "1e1", "-", "350", "+1", "-20"] { #expect(!accepts("{\"r\":\(v)}", r), "\(v)") }
        let open = try #require(walk("{\"r\":", r))
        #expect(open.walk(Array("4".utf8)) != nil, "4 is in range on its own")
        #expect(open.walk(Array("4}".utf8))?.isComplete == true)
        #expect(open.walk(Array("40".utf8)) == nil, "but no value in [-12, 35] starts with 40")
        #expect(open.walk(Array("3".utf8)) != nil)
        #expect(open.walk(Array("36".utf8)) == nil)
        #expect(open.walk(Array("35}".utf8))?.isComplete == true)
        #expect(open.walk(Array("3}".utf8))?.isComplete == true)
        #expect(open.walk(Array("-1".utf8)) != nil)
        #expect(open.walk(Array("-13".utf8)) == nil)
        #expect(open.walk(Array("-12}".utf8))?.isComplete == true)
        #expect(open.walk(Array("-2".utf8)) != nil)
        #expect(open.walk(Array("-20".utf8)) == nil, "-2x is below -12")
        // A range that starts past the single digits refuses them outright.
        let tens = try #require(walk("{\"r\":", schema([("r", integers(10, 35))])))
        #expect(tens.walk(Array("4".utf8)) == nil, "no value in [10, 35] starts with 4")
        #expect(tens.walk(Array("1".utf8)) != nil)
        #expect(tens.walk(Array("1}".utf8)) == nil)
    }

    @Test
    func boundedIntegerSingletonsAndOneSidedRanges() throws {
        let zero = schema([("z", integers(0, 0))])
        #expect(accepts("{\"z\":0}", zero))
        #expect(accepts("{\"z\":-0}", zero))
        #expect(!accepts("{\"z\":1}", zero))
        #expect(!accepts("{\"z\":00}", zero))
        #expect(walk("{\"z\":-1", zero) == nil)

        let hundred = schema([("c", integers(100, 100))])
        #expect(accepts("{\"c\":100}", hundred))
        #expect(!accepts("{\"c\":10}", hundred))
        #expect(!accepts("{\"c\":1000}", hundred))
        #expect(walk("{\"c\":1", hundred) != nil)
        #expect(walk("{\"c\":2", hundred) == nil)
        #expect(walk("{\"c\":101", hundred) == nil)

        let atLeast = schema([("n", integers(7, nil))])
        #expect(accepts("{\"n\":7}", atLeast))
        #expect(accepts("{\"n\":700000000000000000}", atLeast))
        #expect(accepts("{\"n\":\(String(repeating: "9", count: 19))}", atLeast), "19 digits is the most a value may have")
        #expect(walk("{\"n\":\(String(repeating: "9", count: 20))", atLeast) == nil)
        #expect(!accepts("{\"n\":6}", atLeast))
        #expect(!accepts("{\"n\":-7}", atLeast))
        #expect(walk("{\"n\":-", atLeast) == nil, "nothing at or below zero is in [7, ...)")

        let atMost = schema([("n", integers(nil, -3))])
        #expect(accepts("{\"n\":-3}", atMost))
        #expect(accepts("{\"n\":-4000}", atMost))
        #expect(!accepts("{\"n\":-2}", atMost))
        #expect(!accepts("{\"n\":0}", atMost))
        #expect(walk("{\"n\":1", atMost) == nil)
        #expect(walk("{\"n\":0", atMost) == nil)

        // A range above zero refuses the sign outright; one that touches zero takes -0.
        #expect(walk("{\"r\":-", schema([("r", integers(1, 10))])) == nil)
        #expect(accepts("{\"r\":-0}", schema([("r", integers(0, 10))])))
    }

    /// A bounded number is spelled plain — no exponent — and judged digit by
    /// digit: 0 may still become 0.5, but 0 itself is below [0.5, 2.75].
    @Test
    func boundedNumbersEndOnlyInRangeAndTakeNoExponent() throws {
        let r = schema([("x", numbers("0.5", "2.75"))], required: ["x"])
        for v in ["0.5", "0.50", "1", "1.0", "2.75", "2.7", "0.999", "2.749999", "2.750000000000000000"] {
            #expect(accepts("{\"x\":\(v)}", r), "\(v)")
        }
        for v in ["0.4", "0.49", "2.76", "2.8", "3", "0", "-0.5", "-1", "1e0", "1E0", "0.5e1", "1.", ".5", "00.5", "2.75000000000000000001"] {
            #expect(!accepts("{\"x\":\(v)}", r), "\(v)")
        }
        let open = try #require(walk("{\"x\":", r))
        #expect(open.walk(Array("0".utf8)) != nil, "0 can still become 0.5")
        #expect(open.walk(Array("0}".utf8)) == nil, "but 0 itself is below the range")
        #expect(open.walk(Array("0.4".utf8)) == nil)
        #expect(open.walk(Array("2.7".utf8)) != nil)
        #expect(open.walk(Array("2.76".utf8)) == nil)
        #expect(open.walk(Array("2.75}".utf8))?.isComplete == true)
        #expect(open.walk(Array("3".utf8)) == nil)
        #expect(open.walk(Array("-".utf8)) == nil)
        #expect(open.walk(Array("1e".utf8)) == nil)
        #expect(open.walk(Array("1.}".utf8)) == nil, "a dot needs a digit")
    }

    @Test
    func openNegativeAndOneSidedNumberRanges() throws {
        let unit = schema([("p", numbers("0", "1", openBelow: true, openAbove: true))])
        for v in ["0.01", "0.5", "0.999", "0.0000000000000000001"] { #expect(accepts("{\"p\":\(v)}", unit), "\(v)") }
        for v in ["0", "0.0", "-0", "1", "1.0", "0.00000000000000000001", "-0.5"] { #expect(!accepts("{\"p\":\(v)}", unit), "\(v)") }
        #expect(walk("{\"p\":-", unit) == nil)
        #expect(walk("{\"p\":0.0", unit) != nil, "0.0 can still become 0.01")

        let negative = schema([("t", numbers("-1.5", "-0.25"))])
        for v in ["-1.5", "-0.25", "-1", "-0.3", "-1.50", "-0.250"] { #expect(accepts("{\"t\":\(v)}", negative), "\(v)") }
        for v in ["-0.2", "-1.6", "0", "-0", "0.5", "-2", "1"] { #expect(!accepts("{\"t\":\(v)}", negative), "\(v)") }
        #expect(walk("{\"t\":-0", negative) != nil, "-0 can still become -0.25")
        #expect(walk("{\"t\":-0.2", negative) != nil, "-0.2 can still become -0.25")
        #expect(walk("{\"t\":-0.24", negative) == nil, "-0.24x is above -0.25")

        let below = schema([("t", numbers(nil, "-1.25"))])
        for v in ["-1.25", "-100", "-1.250", "-999999999999999999.9"] { #expect(accepts("{\"t\":\(v)}", below), "\(v)") }
        for v in ["-1.24", "-1.2", "0", "-0", "-1", "5"] { #expect(!accepts("{\"t\":\(v)}", below), "\(v)") }

        let fixed = schema([("f", numbers("10", "10"))])
        for v in ["10", "10.0", "10.000"] { #expect(accepts("{\"f\":\(v)}", fixed), "\(v)") }
        for v in ["1", "100", "10.1", "9.99", "-10"] { #expect(!accepts("{\"f\":\(v)}", fixed), "\(v)") }
        #expect(walk("{\"f\":1", fixed) != nil)
        #expect(walk("{\"f\":1}", fixed) == nil)

        let above = schema([("a", numbers("0.3", nil, openBelow: true))])
        for v in ["0.31", "1", "999", "0.3000000000000000001"] { #expect(accepts("{\"a\":\(v)}", above), "\(v)") }
        for v in ["0.3", "0.30", "0.29", "-5", "0"] { #expect(!accepts("{\"a\":\(v)}", above), "\(v)") }
    }

    @Test
    func rootBoundedNumbersAreCompleteWithoutATerminator() throws {
        let digits = integers(0, 9)
        for t in ["0", "5", "9", "-0", "7 "] { #expect(acceptsRoot(t, digits), "\(t)") }
        for t in ["10", "-1", "1.5", "5 1", "", "-"] { #expect(!acceptsRoot(t, digits), "\(t)") }
        let n = numbers("0.5", "2.75")
        for t in ["0.5", "2.75", "1", "1.0 "] { #expect(acceptsRoot(t, n), "\(t)") }
        for t in ["0", "0.", "3", "1e0"] { #expect(!acceptsRoot(t, n), "\(t)") }
        // At the root a value in range is complete as soon as its digits are,
        // and stays complete as digits that keep it in range follow.
        let afterOne = SchemaConstraintState(root: n).walk(Array("1".utf8))
        #expect(afterOne?.isComplete == true)
        #expect(afterOne?.walk(Array(".".utf8))?.isComplete == false)
        #expect(afterOne?.walk(Array(".5".utf8))?.isComplete == true)
        #expect(SchemaConstraintState(root: n).walk(Array("0".utf8))?.isComplete == false)
        #expect(SchemaConstraintState(root: n).walk(Array("2.75".utf8))?.walk(Array("1".utf8)) == nil)
    }

    /// Every legal prefix of a bounded value completes: from every state the
    /// automaton reaches within five bytes of a bounded value, a complete
    /// document is reached within six more. The trap search proves this for
    /// bounded integers with two-sided bounds, whose states are finite; a
    /// bounded number has a state per prefix (every fraction digit opens
    /// ten more), so this walks them to a fixed depth instead. The search
    /// for a completion has a budget of walks: a correct automaton completes
    /// every prefix within a few, and a dead end would otherwise be explored
    /// to the full depth, thousands of times over.
    @Test
    func boundedPrefixesAlwaysComplete() throws {
        let roots: [SchemaValueType] = [
            integers(-12, 35), integers(0, 0), integers(7, nil), integers(nil, -3),
            numbers("0.5", "2.75"), numbers("0", "1", openBelow: true, openAbove: true),
            numbers("-1.5", "-0.25"), numbers(nil, "-1.25"), numbers("10", "10"),
            numbers("0.3", nil, openBelow: true), numbers("-0.5", "0.5"), numbers("0.125", "0.125"),
            numbers("99.5", "100.25"), numbers("-0.001", "0.001", openBelow: true, openAbove: true),
        ]
        let digits = Array("-0123456789.".utf8)
        for root in roots {
            // The root itself, and the same value inside an object (where `}` must still follow).
            let object = JSONSchemaObject(properties: [.init(name: "v", type: root)], required: ["v"])
            let starts: [(SchemaConstraintState, [UInt8])] = [
                (SchemaConstraintState(root: root), digits),
                (try #require(walk("{\"v\":", object)), digits + [SchemaBytes.rBrace]),
            ]
            for (start, alphabet) in starts {
                var frontier = [start]
                var seen: Set<SchemaConstraintState> = [start]
                for _ in 0..<5 {
                    var next: [SchemaConstraintState] = []
                    for state in frontier {
                        for byte in alphabet {
                            if let advanced = state.advanced(over: byte), seen.insert(advanced).inserted { next.append(advanced) }
                        }
                    }
                    frontier = next
                }
                var deadEnds = 0
                for state in seen {
                    var budget = 10_000
                    if !completes(state, within: 6, alphabet, budget: &budget) {
                        Issue.record("\(budget == 0 ? "no completion within the budget" : "dead end") at \(state.diagnosticDescription) for \(root)")
                        deadEnds += 1
                        if deadEnds == 3 { break }   // three say enough
                    }
                }
            }
        }
    }

    /// Whether a value of `type` has unboundedly many legal prefixes: a bounded
    /// number (every fraction digit opens ten more) or an integer bounded on
    /// one side only. The trap search cannot finish on those.
    private static func hasUnboundedPrefixes(_ type: SchemaValueType) -> Bool {
        switch type {
        case .boundedNumber: return true
        case .boundedInteger(let bounds): return bounds.minimum == nil || bounds.maximum == nil
        case .object(let object): return object.properties.contains { hasUnboundedPrefixes($0.type) }
        case .array(let items, _, _): return hasUnboundedPrefixes(items)
        case .string, .number, .integer, .boolean, .stringEnum: return false
        }
    }

    /// Whether a complete document is reachable from `state` within `depth`
    /// bytes, trying at most `budget` walks in all.
    private func completes(_ state: SchemaConstraintState, within depth: Int, _ alphabet: [UInt8], budget: inout Int) -> Bool {
        if state.isComplete { return true }
        guard depth > 0 else { return false }
        for byte in alphabet {
            guard budget > 0 else { return false }
            budget -= 1
            if let next = state.advanced(over: byte), completes(next, within: depth - 1, alphabet, budget: &budget) {
                return true
            }
        }
        return false
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
    func noReachableStateIsATrap() throws {
        let schemas: [JSONSchemaObject] = [
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
            // Literals outside ASCII and ones that need escapes: the escape
            // paths (surrogate pairs included) must not strand the matcher.
            schema([("c", .stringEnum(["Lençóis", "Bogotá", "😀"]))], required: ["c"]),
            schema([("日本", .string), ("q\"q", .stringEnum(["a\\b"]))], required: ["日本", "q\"q"]),
            // Bounded integers with two-sided bounds: finitely many prefixes.
            schema([("r", integers(-12, 35)), ("z", integers(0, 0))], required: ["r", "z"]),
            schema([("c", integers(100, 100)), ("one", integers(1, 10))], required: ["one"]),
            schema([("i", .array(items: integers(-5, 5), minItems: 1, maxItems: 2))]),
        ]
        var roots: [SchemaValueType] = schemas.map { .object($0) }
        roots += [
            .array(items: nested([("id", .integer)], required: ["id"]), minItems: 1, maxItems: 2),
            .array(items: .stringEnum(["é", "e"]), minItems: 0, maxItems: nil),
            .string, .number, .integer, .boolean, .stringEnum(["😀", "x"]),
            integers(-12, 35), integers(0, 0),
        ]
        // A search the cap cuts off counts the frontier as live and could hide
        // a trap, so the hand-written schemas must be explored whole (an
        // unbounded array's count holds at `minItems`, so its states are
        // finite). A seeded nested object can be too wide for that — every
        // level multiplies the states by its emitted-key sets — so those are
        // checked as far as the search reaches, and counted: with seed 42 two
        // of the 32 exceed 40k states (one of them 400k). A jump in that
        // number would mean the seeded coverage had mostly vanished.
        var generator = RandomSchemaGenerator(seed: 42)
        var seeded: [SchemaValueType] = []
        for _ in 0..<12 {
            seeded.append(.object(generator.object()))
        }
        for _ in 0..<12 {
            seeded.append(.object(generator.object(nested: true)))
        }
        for _ in 0..<8 {
            seeded.append(generator.root())
        }
        // A bounded number, or an integer bounded on one side only, has a
        // state per prefix and always reaches the cap: checked as far as the
        // search goes (`boundedPrefixesAlwaysComplete` walks them to depth).
        let unbounded: [SchemaValueType] = [
            numbers("0.5", "2.75"), numbers("0", "1", openBelow: true, openAbove: true), integers(7, nil),
            .object(schema([("p", numbers("-1.5", "-0.25")), ("n", integers(nil, -3))], required: ["p"])),
        ]
        let limit = 40_000
        var capped = 0
        let runs = roots.map { ($0, "fixed") } + seeded.map { ($0, Self.hasUnboundedPrefixes($0) ? "unbounded" : "seeded") }
            + unbounded.map { ($0, "unbounded") }
        for (root, group) in runs {
            let result = SchemaTrapSearch.run(
                from: SchemaConstraintState(root: root),
                alphabet: SchemaTrapSearch.alphabet(for: root),
                limit: limit)
            #expect(
                result.traps.isEmpty,
                "\(result.traps.count) trap(s) in \(result.explored) states, first: \(result.traps.first?.diagnosticDescription ?? "-") for \(root)")
            switch group {
            case "fixed": #expect(result.explored < limit, "the search hit the state cap at \(result.explored) for \(root)")
            case "seeded": if result.explored >= limit { capped += 1 }
            default: break
            }
        }
        #expect(capped <= 4, "\(capped) seeded roots hit the state cap")
    }

    /// Each of JSON's short escapes matches the scalar it denotes, as does the
    /// `\u00XX` spelling; the wrong letter does not, nor does the scalar raw
    /// where JSON forbids it. `\/` is the exception: `/` is plain ASCII, which
    /// the matcher takes raw only.
    @Test
    func matchesEveryShortEscape() {
        let escapes: [(letter: String, scalar: UInt32)] = [
            ("\"", 0x22), ("\\", 0x5C), ("b", 0x08), ("f", 0x0C), ("n", 0x0A), ("r", 0x0D), ("t", 0x09),
        ]
        for (letter, scalar) in escapes {
            let value = String(UnicodeScalar(scalar).map(Character.init) ?? "?")
            let object = schema([("k", .stringEnum([value]))], required: ["k"])
            let hex = String(scalar, radix: 16)
            let padded = String(repeating: "0", count: 4 - hex.count) + hex
            #expect(accepts("{\"k\":\"\\\(letter)\"}", object), "\\\(letter)")
            #expect(accepts("{\"k\":\"\\u\(padded)\"}", object), "\\u\(padded)")
            #expect(!accepts("{\"k\":\"\\\(letter == "t" ? "n" : "t")\"}", object), "the wrong letter for \\\(letter)")
            if scalar < 0x20 {
                #expect(!accepts("{\"k\":\"\(value)\"}", object), "raw U+\(padded) in a string")
            }
        }
        let slash = schema([("k", .stringEnum(["/"]))], required: ["k"])
        #expect(accepts("{\"k\":\"/\"}", slash))
        #expect(!accepts("{\"k\":\"\\/\"}", slash))
        #expect(!accepts("{\"k\":\"\\u002f\"}", slash))
    }

    /// `required` names are matched scalar by scalar, as keys are: a name that
    /// Swift's `String` calls equal to a declared key (a decomposed "é"
    /// against a precomposed one) is not that key. The decoder refuses such a
    /// schema; built directly, it names a member that is never emitted, so
    /// the object can never close — it never falls back to the look-alike.
    @Test
    func requiredNamesAreMatchedByScalar() {
        let precomposed = "\u{E9}", decomposed = "e\u{301}"
        #expect(precomposed == decomposed, "String equality is canonical")
        #expect(Array(precomposed.unicodeScalars) != Array(decomposed.unicodeScalars))
        let declared = schema([(precomposed, .string)], required: [precomposed])
        #expect(accepts("{\"\(precomposed)\":\"x\"}", declared))
        let lookalike = schema([(precomposed, .string)], required: [decomposed])
        #expect(!accepts("{\"\(precomposed)\":\"x\"}", lookalike))
        #expect(!accepts("{\"\(decomposed)\":\"x\"}", lookalike))
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

    // MARK: Real schemas

    /// Upstream mlx-swift-lm's constrained-decoding goldens (tiers 1–4, up to
    /// five nested containers): each golden document is accepted, by the
    /// reference validator and the generic JSON automaton too.
    @Test
    func acceptsUpstreamGoldenDocuments() throws {
        for tier in 1...4 {
            let golden = try StructuredOutputFixtures.golden(tier: tier)
            let object = try StructuredOutputFixtures.compile(golden.schema)
            let document = Array(golden.document.utf8)
            #expect(accepts(golden.document, object), "tier \(tier)")
            #expect(ReferenceSchemaValidator.validate(document, object), "tier \(tier)")
            #expect(JSONGrammarState().walk(document)?.isComplete == true, "tier \(tier)")
        }
        // Tier 4's activities hold exactly three items.
        let tier4 = try StructuredOutputFixtures.golden(tier: 4)
        let object = try StructuredOutputFixtures.compile(tier4.schema)
        let short = tier4.document.replacingOccurrences(
            of: ",{\"type\":\"X\",\"title\":\"T\",\"description\":\"D\"}]", with: "]")
        #expect(short != tier4.document)
        #expect(!accepts(short, object))
    }

    /// Apple's TripPlanner `Itinerary` schema as the framework emits it:
    /// `$defs`, `$ref` as `items`, exact array counts, an enum nested two
    /// arrays deep, and the destination enum value "Lençóis Maranhenses",
    /// which may arrive raw or escaped.
    @Test
    func acceptsATripPlannerItinerary() throws {
        let trip = try StructuredOutputFixtures.compile(StructuredOutputFixtures.itinerary())
        let day = { (last: String) in
            "{\"title\":\"T\",\"subtitle\":\"S\",\"destination\":\"D\",\"activities\":["
                + "{\"type\":\"sightseeing\",\"title\":\"T\",\"description\":\"D\"},"
                + "{\"type\":\"shopping\",\"title\":\"T\",\"description\":\"D\"},"
                + "{\"type\":\"\(last)\",\"title\":\"T\",\"description\":\"D\"}]}"
        }
        func itinerary(_ destination: String) -> String {
            "{\"title\":\"T\",\"destinationName\":\"\(destination)\",\"description\":\"E\",\"rationale\":\"R\","
                + "\"days\":[\(day("foodAndDining")),\(day("foodAndDining")),\(day("hotelAndLodging"))]}"
        }
        let document = itinerary("Mount Fuji")
        #expect(accepts(document, trip))
        #expect(ReferenceSchemaValidator.validate(Array(document.utf8), trip))
        for spelling in ["Lençóis Maranhenses", "Len\\u00e7\\u00f3is Maranhenses", "Len\\u00E7óis Maranhenses"] {
            #expect(accepts(itinerary(spelling), trip), "\(spelling)")
            #expect(ReferenceSchemaValidator.validate(Array(itinerary(spelling).utf8), trip), "\(spelling)")
        }
        #expect(!accepts(itinerary("Lencois Maranhenses"), trip))
        #expect(!accepts(document.replacingOccurrences(of: "\"shopping\"", with: "\"golf\""), trip))
        #expect(!accepts(itinerary("Mount Doom"), trip))
        #expect(!accepts(document.replacingOccurrences(of: ",\(day("hotelAndLodging"))", with: ""), trip))
        #expect(!accepts(document.replacingOccurrences(of: "\"rationale\":\"R\",", with: ""), trip))
    }

    // MARK: Hand-built schemas

    /// Shapes only a hand-built schema can have (the decoder rejects both)
    /// keep their old meaning: a name declared twice keeps its first
    /// declaration, and a required name that is not declared can never be
    /// satisfied, so the object never closes.
    @Test
    func handBuiltSchemaEdgeCases() {
        let duplicate = schema([("a", .string), ("a", .integer)])
        #expect(accepts("{\"a\":\"x\"}", duplicate))
        #expect(!accepts("{\"a\":1}", duplicate))
        #expect(!accepts("{\"a\":\"x\",\"a\":\"y\"}", duplicate))
        #expect(!accepts("{\"a\":\"x\",\"a\":1}", duplicate))

        let undeclared = schema([("a", .string)], required: ["zzz"])
        #expect(walk("{", undeclared) != nil)
        #expect(walk("{}", undeclared) == nil)
        #expect(walk("{\"a\":\"x\"}", undeclared) == nil)
    }

    /// The member set behind emitted keys and candidates. Its `==` and hash
    /// must be semantic (no trailing empty word), or equal positions would
    /// compare unequal.
    @Test
    func propertyMaskSetAlgebra() {
        func mask(_ members: [Int]) -> PropertyMask {
            var result = PropertyMask()
            for member in members { result.insert(member) }
            return result
        }
        let mixed = mask([0, 63, 64, 127, 128, 200])
        for member in [0, 63, 64, 127, 128, 200] { #expect(mixed.contains(member)) }
        for member in [1, 62, 65, 126, 129, 199, 201, 1_000] { #expect(!mixed.contains(member)) }
        #expect(mixed.first(where: { $0 > 63 }) == 64)
        #expect(mixed.first(where: { $0 > 200 }) == nil)
        #expect(mixed.filtered { $0 >= 128 } == mask([128, 200]))

        let lowOnly = mixed.subtracting(mask([64, 127, 128, 200]))
        #expect(lowOnly == mask([0, 63]))
        #expect(lowOnly.high.isEmpty)
        #expect(lowOnly.hashValue == mask([0, 63]).hashValue)
        #expect(mixed.subtracting(mixed).isEmpty)

        #expect(mask([3, 65]).isSubset(of: mask([3, 65, 100])))
        #expect(!mask([3, 66]).isSubset(of: mask([3, 65, 100])))
        #expect(PropertyMask.all(count: 0).isEmpty)
        #expect(PropertyMask.all(count: 64) == mask(Array(0..<64)))
        #expect(PropertyMask.all(count: 130) == mask(Array(0..<130)))
        #expect(!PropertyMask.all(count: 65).contains(65))
    }

    // MARK: Diagnostics

    /// The log line a cut-off generation leaves names keys, not bit masks: the
    /// candidates of the key being read and each open object's emitted keys.
    @Test
    func diagnosticDescriptionNamesKeys() throws {
        let s = schema([("o", nested([("a", .integer), ("ab", .integer), ("b", .integer)]))])
        let text = try #require(walk("{\"o\":{\"b\":1,\"a", s)).diagnosticDescription
        #expect(text.contains("candidates: [\"a\", \"ab\"]"), "\(text)")
        #expect(text.contains("frames: [object(emitted: [\"o\"]), object(emitted: [\"b\"])]"), "\(text)")
        #expect(!text.contains("PropertyMask"), "\(text)")
        #expect(text.contains("progress: boundary"), "\(text)")
        let midEscape = try #require(walk("{\"\\u00", schema([("é", .string)]))).diagnosticDescription
        #expect(midEscape.contains("progress: hex(digits: 2, low: false)"), "\(midEscape)")
        let inArray = try #require(walk("{\"o\":", schema([("o", .array(items: .integer, minItems: 0, maxItems: 3))])))
        #expect(inArray.walk(Array("[1,2".utf8))?.diagnosticDescription.contains("array(count: 2)") == true)
        // An unbounded array's count holds at `minItems`: past it nothing
        // reads the count, so the state does not carry it, and the line says so.
        let unbounded = try #require(walk("{\"o\":", schema([("o", .array(items: .integer, minItems: 1, maxItems: nil))])))
        #expect(unbounded.walk(Array("[".utf8))?.diagnosticDescription.contains("array(count: 0)") == true)
        #expect(unbounded.walk(Array("[1,2".utf8))?.diagnosticDescription.contains("array(count: ≥1)") == true)

        // A wide object lists eight names and counts the rest.
        let wide = schema((0..<70).map { ("p\($0)", SchemaValueType.integer) })
        let atKey = try #require(walk("{\"", wide)).diagnosticDescription
        #expect(atKey.contains("candidates: [\"p0\", \"p1\", \"p2\", \"p3\", \"p4\", \"p5\", \"p6\", \"p7\", … (+62)]"), "\(atKey)")
        #expect(!atKey.contains("\"p8\""), "\(atKey)")
        let tenEmitted = (0..<10).map { "\"p\($0)\":1" }.joined(separator: ",")
        let afterTen = try #require(walk("{" + tenEmitted, wide)).diagnosticDescription
        #expect(afterTen.contains("object(emitted: [\"p0\", \"p1\", \"p2\", \"p3\", \"p4\", \"p5\", \"p6\", \"p7\", … (+2)])"), "\(afterTen)")
    }

    // MARK: Equality

    /// States are equal at the same position of equal schemas, even when the
    /// schemas were compiled separately; the hash agrees.
    @Test
    func equalityIsByPositionAndSchema() throws {
        let object = schema([("o", nested([("k", .boolean)])), ("t", .array(items: .integer, minItems: 0, maxItems: 3))])
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
        // In an unbounded array the count holds at `minItems`, so two items in
        // and three items in are the same position; below `minItems` the count
        // still tells positions apart.
        let unbounded = SchemaConstraintState(schema: schema([("t", .array(items: .integer, minItems: 2, maxItems: nil))]))
        #expect(unbounded.walk(Array("{\"t\":[1".utf8)) != unbounded.walk(Array("{\"t\":[1,2".utf8)))
        #expect(unbounded.walk(Array("{\"t\":[1,2".utf8)) == unbounded.walk(Array("{\"t\":[1,2,3".utf8)))
        // Different paths to the same position are the same state.
        #expect(a.walk(Array("{\"o\":{\"k\":true}".utf8)) == a.walk(Array("{\"o\":{\"k\":false}".utf8)))

        // Same node layout, different schema: not equal.
        let other = schema([("o", nested([("k", .boolean)])), ("t", .array(items: .number, minItems: 0, maxItems: 3))])
        #expect(SchemaConstraintState(schema: other) != a)

        // Scalar-exact, as the automaton is: a key or value that is the same
        // string normalised differently makes a different schema, although
        // Swift's `String` calls the two equal.
        let precomposed = "\u{E9}", decomposed = "e\u{301}"
        #expect(SchemaConstraintState(root: .stringEnum([precomposed])) != SchemaConstraintState(root: .stringEnum([decomposed])))
        #expect(SchemaConstraintState(schema: schema([(precomposed, .string)])) != SchemaConstraintState(schema: schema([(decomposed, .string)])))
        #expect(SchemaConstraintState(root: .stringEnum([precomposed])) == SchemaConstraintState(root: .stringEnum([precomposed])))
    }

    // MARK: Differential test against a reference validator

    /// Seeded schemas (flat, nested, and roots of every type) × valid and
    /// mutated documents, literals spelled raw or escaped at random: the
    /// automaton accepts exactly what ``ReferenceSchemaValidator`` accepts,
    /// and everything it accepts is well-formed JSON to ``JSONGrammarState``.
    @Test
    func agreesWithTheReferenceValidator() {
        var generator = RandomSchemaGenerator(seed: 0xC0FFEE)
        var accepted = 0
        var mismatches: [String] = []
        for round in 0..<600 {
            let root: SchemaValueType = round % 3 == 2 ? generator.root() : .object(generator.object(nested: round % 2 == 1))
            let start = SchemaConstraintState(root: root)
            for k in 0..<12 {
                let valid = generator.document(for: root)
                let document = k < 4 ? Array(valid.utf8) : generator.mutate(valid)
                let automaton = start.walk(document)?.isComplete ?? false
                let reference = ReferenceSchemaValidator.validate(document, root: root)
                if automaton != reference {
                    mismatches.append("automaton=\(automaton) reference=\(reference) \(String(decoding: document, as: UTF8.self)) for \(root)")
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
