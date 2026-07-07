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
}
