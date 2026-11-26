import Testing
import Foundation
@testable import MacMLXCore

// Wrapped in a @Suite struct so module-scope test names don't clash.
@Suite
struct JSONValueTests {

    @Test
    func nestedObjectRoundTripsThroughCodable() throws {
        // Note: whole-number doubles (e.g. 2.0) re-encode as integers and
        // would decode back as `.int`, so the fixture uses non-integer
        // doubles (0.7, 1.5) to keep the round-trip exact.
        let original: JSONValue = .object([
            "enable_thinking": .bool(true),
            "temperature": .double(0.7),
            "max": .int(42),
            "name": .string("qwen"),
            "empty": .null,
            "stops": .array([.string("a"), .int(1), .bool(false)]),
            "nested": .object(["deep": .array([.double(1.5)])]),
        ])

        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(JSONValue.self, from: data)

        #expect(decoded == original)
    }

    @Test
    func toSendableUnwrapsScalarsNullAndCollections() {
        #expect((JSONValue.string("x").toSendable() as? String) == "x")
        #expect((JSONValue.int(3).toSendable() as? Int) == 3)
        #expect((JSONValue.double(1.5).toSendable() as? Double) == 1.5)
        #expect((JSONValue.bool(true).toSendable() as? Bool) == true)
        #expect(JSONValue.null.toSendable() is NSNull)

        let array = JSONValue.array([.int(1), .string("a")]).toSendable() as? [any Sendable]
        #expect(array?.count == 2)

        let object = JSONValue.object(["k": .bool(true)]).toSendable() as? [String: any Sendable]
        #expect((object?["k"] as? Bool) == true)
    }

    // MARK: Nesting

    /// `depth` containers around a null: objects, arrays, or the two in turn.
    private func nested(_ depth: Int, open: [String], close: [String]) -> Data {
        var text = ""
        for level in 0..<depth { text += open[level % open.count] }
        text += "null"
        for level in (0..<depth).reversed() { text += close[level % close.count] }
        return Data(text.utf8)
    }

    private static let shapes: [(open: [String], close: [String])] = [
        (["{\"a\":"], ["}"]),
        (["["], ["]"]),
        (["{\"a\":", "["], ["}", "]"]),
    ]

    /// The deepest text that decodes does so on the thread a test runs on,
    /// which is the kind of thread a request is decoded on.
    @Test
    func aValueAtTheNestingLimitDecodes() throws {
        for shape in Self.shapes {
            let data = nested(JSONValue.maximumNestingDepth, open: shape.open, close: shape.close)
            var value = try JSONDecoder().decode(JSONValue.self, from: data)
            var depth = 0
            while true {
                if case .object(let object) = value, let inner = object["a"] {
                    value = inner
                } else if case .array(let array) = value, let inner = array.first {
                    value = inner
                } else {
                    break
                }
                depth += 1
            }
            #expect(depth == JSONValue.maximumNestingDepth)
            #expect(value == .null)
        }
    }

    /// One level more is an error, and the error says why. It used to decode,
    /// and at a few hundred levels the recursion ran off the end of the stack.
    @Test
    func aValuePastTheNestingLimitIsAnError() {
        for shape in Self.shapes {
            for depth in [JSONValue.maximumNestingDepth + 1, 500] {
                let data = nested(depth, open: shape.open, close: shape.close)
                #expect {
                    try JSONDecoder().decode(JSONValue.self, from: data)
                } throws: { error in
                    guard case DecodingError.dataCorrupted(let context) = error else { return false }
                    return context.debugDescription.contains("nested deeper")
                }
            }
        }
    }

    /// The limit is on where a value sits, and an empty container at the limit
    /// has nothing below it, so it decodes like any other value there.
    @Test
    func anEmptyContainerAtTheLimitDecodes() throws {
        let depth = JSONValue.maximumNestingDepth
        let data = Data((String(repeating: "[", count: depth) + "[]" + String(repeating: "]", count: depth)).utf8)
        _ = try JSONDecoder().decode(JSONValue.self, from: data)
        let oneMore = Data("[".utf8) + data + Data("]".utf8)
        #expect(throws: DecodingError.self) { try JSONDecoder().decode(JSONValue.self, from: oneMore) }
    }

    /// The limit counts from the top of the text, not from the value: a value
    /// inside a request counts the request's own containers.
    @Test
    func theLimitCountsFromTheTopOfTheText() throws {
        struct Wrapper: Decodable { let tools: [JSONValue] }
        let fits = nested(JSONValue.maximumNestingDepth - 2, open: ["["], close: ["]"])
        let tooDeep = nested(JSONValue.maximumNestingDepth - 1, open: ["["], close: ["]"])
        func wrapped(_ data: Data) -> Data { Data(#"{"tools":["#.utf8) + data + Data("]}".utf8) }
        _ = try JSONDecoder().decode(Wrapper.self, from: wrapped(fits))
        #expect(throws: DecodingError.self) { try JSONDecoder().decode(Wrapper.self, from: wrapped(tooDeep)) }
    }
}
