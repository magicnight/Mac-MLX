import Foundation
import Testing

@testable import MacMLXCore

import enum MacMLXCore.JSONValue

/// `JSONObject` keeps its keys in order, and the order is not part of its
/// value.
@Suite("JSONObject")
struct JSONObjectTests {

    @Test
    func aLiteralKeepsTheOrderItIsWrittenIn() {
        let object: JSONObject = ["type": .string("object"), "properties": .object([:]), "a": .int(1)]
        #expect(object.keys == ["type", "properties", "a"])
        #expect(object.values == [.string("object"), .object([:]), .int(1)])
        #expect(object.map(\.key) == ["type", "properties", "a"])
    }

    @Test
    func aPlainDictionaryComesOutSorted() {
        let dictionary: [String: JSONValue] = ["type": .int(1), "b": .int(2), "a": .int(3)]
        #expect(JSONObject(dictionary).keys == ["a", "b", "type"])
        guard case .object(let object) = JSONValue.object(dictionary) else {
            Issue.record("not an object")
            return
        }
        #expect(object.keys == ["a", "b", "type"])
    }

    @Test
    func theOrderIsNotPartOfTheValue() {
        let forward: JSONObject = ["a": .int(1), "b": .int(2)]
        let backward: JSONObject = ["b": .int(2), "a": .int(1)]
        #expect(forward == backward)
        #expect(forward.hashValue == backward.hashValue)
        #expect(JSONValue.object(forward) == JSONValue.object(backward))
        #expect(forward != ["a": .int(1), "b": .int(3)])
        #expect(forward != ["a": .int(1)])
    }

    @Test
    func settingAKeyKeepsItsPlaceAndANewKeyGoesLast() {
        var object: JSONObject = ["a": .int(1), "b": .int(2)]
        object["a"] = .int(10)
        object["c"] = .int(3)
        #expect(object.keys == ["a", "b", "c"])
        #expect(object["a"] == .int(10))
        object["a"] = nil
        #expect(object.keys == ["b", "c"])
        #expect(object.count == 2)
        #expect(object.removeValue(forKey: "b") == .int(2))
        #expect(object.removeValue(forKey: "b") == nil)
        #expect(object.keys == ["c"])
    }

    @Test
    func aRepeatedKeyKeepsItsFirstPlaceAndItsLastValue() {
        let object = JSONObject([("b", JSONValue.int(1)), ("a", .int(2)), ("b", .int(3))])
        #expect(object.keys == ["b", "a"])
        #expect(object["b"] == .int(3))
    }

    @Test
    func mappingAndFilteringKeepTheOrder() {
        let object: JSONObject = ["z": .int(1), "a": .int(2), "m": .int(3)]
        let mapped: JSONObject = object.mapValues { _ in JSONValue.null }
        #expect(mapped.keys == ["z", "a", "m"])
        #expect(object.filter { $0.key != "a" }.keys == ["z", "m"])
        let plain: [String: Int] = object.mapValues { _ in 0 }
        #expect(plain == ["z": 0, "a": 0, "m": 0])
        #expect(object.dictionary == ["z": .int(1), "a": .int(2), "m": .int(3)])
    }

    /// Without a `JSONKeyOrder` the decoder has no order to give, and the keys
    /// come out sorted rather than in the order of a hash table.
    @Test
    func decodingWithoutAScanSortsTheKeys() throws {
        let data = Data(#"{"type":"object","b":{"z":1,"y":2},"a":[{"q":1,"p":2}]}"#.utf8)
        let value = try JSONDecoder().decode(JSONValue.self, from: data)
        #expect(value.orderedDescription == "{a:[{p:2,q:1}],b:{y:2,z:1},type:object}")
    }

    @Test
    func encodingRoundTrips() throws {
        let object: JSONObject = ["type": .string("object"), "n": .null, "list": .array([.int(1), .object(["k": .bool(true)])])]
        let decoded = try JSONDecoder().decode(JSONObject.self, from: JSONEncoder().encode(object))
        #expect(decoded == object)
    }
}
