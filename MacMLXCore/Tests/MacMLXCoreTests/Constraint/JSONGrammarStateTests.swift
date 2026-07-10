import Testing

@testable import MacMLXCore

// MARK: - JSONGrammarState Tests (Track C — C1)
//
// Pure, MLX-free unit tests for the byte-level JSON pushdown automaton that
// backs the `response_format: {"type":"json_object"}` constraint. Every case
// drives the automaton over raw UTF-8 bytes exactly as the logit processor
// does at decode time.

@Suite("JSONGrammarState")
struct JSONGrammarStateTests {

    /// Walk a whole document and report whether it is accepted as a complete,
    /// well-formed top-level JSON value.
    private func accepts(_ text: String, maxDepth: Int = 64) -> Bool {
        let start = JSONGrammarState(maxDepth: maxDepth)
        guard let end = start.walk(Array(text.utf8)) else { return false }
        return end.isComplete
    }

    /// Whether the byte string can be consumed at all (may be an incomplete but
    /// still on-track prefix — not necessarily accepting).
    private func consumable(_ text: String) -> Bool {
        JSONGrammarState().walk(Array(text.utf8)) != nil
    }

    // MARK: Primitive top-level values

    @Test
    func acceptsLiterals() {
        #expect(accepts("true"))
        #expect(accepts("false"))
        #expect(accepts("null"))
        #expect(accepts("  true  "))
    }

    @Test
    func rejectsMisspelledLiterals() {
        #expect(!accepts("tru"))
        #expect(!accepts("trues"))
        #expect(!accepts("True"))
        #expect(!accepts("nul"))
        #expect(!consumable("tx"))
    }

    @Test
    func acceptsNumbers() {
        for n in ["0", "-0", "12", "-12", "3.14", "-3.14", "0.5",
                  "1e10", "1E10", "1e+10", "1e-10", "2.5e3", "123456789",
                  "-0.0", "1.0e0"] {
            #expect(accepts(n), "expected \(n) to be accepted")
        }
    }

    @Test
    func rejectsMalformedNumbers() {
        for n in ["01", "1.", ".5", "-", "1e", "1e+", "+1", "1..2",
                  "00", "1.2.3", "0x1", "- 1", "1 2", "٣"] {
            #expect(!accepts(n), "expected \(n) to be rejected")
        }
    }

    @Test
    func acceptsTopLevelStringWithEscapes() {
        #expect(accepts("\"hello\""))
        #expect(accepts("\"a\\nb\""))
        #expect(accepts("\"tab\\tend\""))
        #expect(accepts("\"quote\\\"inside\""))
        #expect(accepts("\"unicode\\u00e9\""))
        #expect(accepts("\"slash\\/\""))
    }

    @Test
    func rejectsBadStrings() {
        #expect(!accepts("\"unterminated"))
        #expect(!accepts("\"bad\\xescape\""))
        #expect(!accepts("\"short\\u12\""))
        #expect(!accepts("\"\u{01}\""))          // raw control char
        #expect(!accepts("noquotes"))
    }

    @Test
    func acceptsUnicodeStringContent() {
        // Multi-byte UTF-8 scalars are legal, unescaped, inside a string.
        #expect(accepts("\"café\""))
        #expect(accepts("\"日本語\""))
        #expect(accepts("\"emoji 😀\""))
    }

    @Test
    func enforcesSurrogatePairing() {
        // A complete `\u` surrogate pair (😀 = U+1F600 = 😀) is
        // accepted — and would parse under JSONSerialization.
        #expect(accepts("\"\\uD83D\\uDE00\""))
        // A lone high surrogate is rejected (JSONSerialization would reject the
        // unpaired surrogate — the gap this pairing check closes).
        #expect(!accepts("\"\\uD83D\""))
        // A lone low surrogate is rejected.
        #expect(!accepts("\"\\uDE00\""))
        // A high surrogate not followed by a low surrogate `\u` is rejected.
        #expect(!accepts("\"\\uD83D\\u0041\""))
        #expect(!accepts("\"\\uD83Dx\""))
        // A plain BMP `\u` escape is unaffected.
        #expect(accepts("\"\\u00e9\""))
    }

    // MARK: Objects

    @Test
    func acceptsObjects() {
        #expect(accepts("{}"))
        #expect(accepts("{ }"))
        #expect(accepts("{\"a\":1}"))
        #expect(accepts("{\"a\": 1, \"b\": 2}"))
        #expect(accepts("{\"nested\":{\"x\":true}}"))
        #expect(accepts("{\"list\":[1,2,3]}"))
        #expect(accepts("{\n  \"k\" : \"v\"\n}"))
    }

    @Test
    func rejectsBadObjects() {
        #expect(!accepts("{\"a\":1,}"))          // trailing comma
        #expect(!accepts("{\"a\"}"))             // missing colon+value
        #expect(!accepts("{a:1}"))               // unquoted key
        #expect(!accepts("{\"a\":}"))            // missing value
        #expect(!accepts("{\"a\":1"))            // unclosed
        #expect(!accepts("{,}"))                 // leading comma
        #expect(!accepts("{\"a\":1 \"b\":2}"))   // missing comma
    }

    // MARK: Arrays

    @Test
    func acceptsArrays() {
        #expect(accepts("[]"))
        #expect(accepts("[ ]"))
        #expect(accepts("[1]"))
        #expect(accepts("[1, 2, 3]"))
        #expect(accepts("[true, false, null]"))
        #expect(accepts("[\"a\", \"b\"]"))
        #expect(accepts("[[1],[2,3],[]]"))
        #expect(accepts("[{\"a\":1}, {\"b\":2}]"))
    }

    @Test
    func rejectsBadArrays() {
        #expect(!accepts("[1,]"))                // trailing comma
        #expect(!accepts("[,1]"))                // leading comma
        #expect(!accepts("[1 2]"))               // missing comma
        #expect(!accepts("[1"))                  // unclosed
        #expect(!accepts("[1,2,]"))
    }

    // MARK: Nesting & depth

    @Test
    func acceptsDeepNesting() {
        #expect(accepts("[[[[[1]]]]]"))
        #expect(accepts("{\"a\":{\"b\":{\"c\":{\"d\":1}}}}"))
    }

    @Test
    func rejectsBeyondMaxDepth() {
        // maxDepth 2 admits two open containers but not a third.
        #expect(accepts("[[1]]", maxDepth: 2))
        #expect(!consumableDepth("[[[", maxDepth: 2))
    }

    private func consumableDepth(_ text: String, maxDepth: Int) -> Bool {
        JSONGrammarState(maxDepth: maxDepth).walk(Array(text.utf8)) != nil
    }

    // MARK: Completion / accept-state semantics

    @Test
    func incompletePrefixesAreNotComplete() {
        let s = JSONGrammarState()
        #expect(s.walk(Array("{\"a\":".utf8))?.isComplete == false)
        #expect(s.walk(Array("[1,".utf8))?.isComplete == false)
        #expect(s.walk(Array("tr".utf8))?.isComplete == false)
        #expect(s.walk(Array("".utf8))?.isComplete == false)   // empty: nothing parsed
    }

    @Test
    func topLevelNumberIsCompleteWithoutTerminator() {
        // A bare number has no closing delimiter, so its terminal digit
        // sub-state must itself count as accepting at the top level.
        let s = JSONGrammarState()
        #expect(s.walk(Array("123".utf8))?.isComplete == true)
        #expect(s.walk(Array("3.14".utf8))?.isComplete == true)
        #expect(s.walk(Array("1e5".utf8))?.isComplete == true)
        #expect(s.walk(Array("-".utf8))?.isComplete == false)   // needs a digit
    }

    @Test
    func trailingContentAfterCompleteValueIsRejected() {
        #expect(!accepts("{}x"))
        #expect(!accepts("truefalse"))
        #expect(!accepts("1 2"))
        #expect(!accepts("[]{}"))
        // Trailing whitespace is fine.
        #expect(accepts("{}   \n"))
    }

    @Test
    func whitespaceHandling() {
        #expect(accepts("   {}   "))
        #expect(accepts("\t\n[ 1 ,\r 2 ]\n"))
        // No whitespace permitted inside a literal or number token.
        #expect(!accepts("n ull"))
        #expect(!accepts("1 0"))
    }
}
