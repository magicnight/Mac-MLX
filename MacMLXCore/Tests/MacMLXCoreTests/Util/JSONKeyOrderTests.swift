import Foundation
import Testing

@testable import MacMLXCore

import enum MacMLXCore.JSONValue

/// `JSONKeyOrder.decode` gives every `JSONObject` the key order of the JSON
/// text, wherever in the decoded type the object sits.
@Suite("JSONKeyOrder")
struct JSONKeyOrderTests {

    private struct Envelope: Decodable {
        let model: String
        let tools: [JSONValue]?
        let kwargs: [String: JSONValue]?
        let inner: Inner?

        struct Inner: Decodable {
            let input_schema: JSONValue
        }
    }

    private func decode(_ text: String) throws -> Envelope {
        try JSONKeyOrder.decode(Envelope.self, from: Data(text.utf8))
    }

    @Test
    func objectsKeepTheOrderOfTheText() throws {
        let envelope = try decode(
            #"""
            {"tools":[{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object","properties":{"zeta":{"type":"string","description":"z"},"alpha":{"enum":["b","a"],"type":"string"}},"required":["zeta"]}}}],"model":"m"}
            """#)
        #expect(
            envelope.tools?.first?.orderedDescription
                == "{type:function,function:{name:f,description:d,parameters:{type:object,"
                + "properties:{zeta:{type:string,description:z},alpha:{enum:[b,a],type:string}},required:[zeta]}}}")
    }

    @Test
    func theOrderReachesEveryPlaceAJSONValueSits() throws {
        let envelope = try decode(
            #"""
            {
              "model": "m",
              "kwargs": { "z": { "b": 1, "a": 2 }, "1": { "y": 1, "x": 2 }, "0": [ [ { "d": 1, "c": 2 } ] ] },
              "inner": { "input_schema": { "type": "object", "properties": {} , "additionalProperties" : false } },
              "tools": [ 1, [ { "b": 1, "a": 2 }, { "y": 1, "x": 2 } ], { "t": 1, "s": [ { "k": 1, "j": 2 } ] } ]
            }
            """#)
        #expect(envelope.kwargs?["z"]?.orderedDescription == "{b:1,a:2}")
        #expect(envelope.kwargs?["1"]?.orderedDescription == "{y:1,x:2}")
        #expect(envelope.kwargs?["0"]?.orderedDescription == "[[{d:1,c:2}]]")
        #expect(envelope.inner?.input_schema.orderedDescription == "{type:object,properties:{},additionalProperties:false}")
        #expect(envelope.tools?.map(\.orderedDescription) == ["1", "[{b:1,a:2},{y:1,x:2}]", "{t:1,s:[{k:1,j:2}]}"])
    }

    /// Keys with escapes, a key that looks like an array position, and string
    /// values full of the characters the scan steers by.
    @Test
    func awkwardKeysAndStringsDoNotThrowTheScanOff() throws {
        let envelope = try decode(
            #"""
            {"model":"m","tools":[{"z\"q":1,"Index 0":{"b":1,"a":2},"\u00e9":2,"a\\":3,"text":"} ] , \" { [ \\","0":{"y":"\\\"","x":2},"é2":{"n":1,"m":2}}]}
            """#)
        let tool = try #require(envelope.tools?.first)
        guard case .object(let object) = tool else {
            Issue.record("not an object")
            return
        }
        #expect(object.keys == ["z\"q", "Index 0", "é", "a\\", "text", "0", "é2"])
        #expect(object["Index 0"]?.orderedDescription == "{b:1,a:2}")
        #expect(object["0"]?.orderedDescription == #"{y:\",x:2}"#)
        #expect(object["é2"]?.orderedDescription == "{n:1,m:2}")
        #expect(object["text"] == .string(#"} ] , " { [ \"#))
    }

    @Test
    func aTopLevelValueIsOrderedToo() throws {
        let value = try JSONKeyOrder.decode(JSONValue.self, from: Data(#" {"b":[{"d":1,"c":2}],"a":null} "#.utf8))
        #expect(value.orderedDescription == "{b:[{d:1,c:2}],a:null}")
    }

    /// `JSONDecoder` takes a UTF-8 byte order mark, so the scan steps over it.
    @Test
    func aByteOrderMarkIsSteppedOver() throws {
        let data = Data([0xEF, 0xBB, 0xBF]) + Data(#"{"b":{"d":1,"c":2},"a":1}"#.utf8)
        #expect(try JSONKeyOrder.decode(JSONValue.self, from: data).orderedDescription == "{b:{d:1,c:2},a:1}")
    }

    /// A schema nested as deep as `JSONValue` decodes still decodes in order.
    @Test
    func aDeeplyNestedObjectDecodes() throws {
        let depth = JSONValue.maximumNestingDepth
        let text = String(repeating: #"{"b":1,"a":"#, count: depth) + "null" + String(repeating: "}", count: depth)
        var value = try JSONKeyOrder.decode(JSONValue.self, from: Data(text.utf8))
        for _ in 0..<depth {
            guard case .object(let object) = value, let next = object["a"] else {
                Issue.record("lost the nesting")
                return
            }
            #expect(object.keys == ["b", "a"])
            value = next
        }
        #expect(value == .null)
    }

    /// The decoded value is the one `JSONDecoder` gives; only the order is new.
    @Test
    func theValueIsWhatJSONDecoderDecodes() throws {
        let data = Data(#"{"model":"m","tools":[{"b":1.5,"a":[true,null,"s",{"y":-2,"x":1e3}]}]}"#.utf8)
        let ordered = try JSONKeyOrder.decode(Envelope.self, from: data)
        let plain = try JSONDecoder().decode(Envelope.self, from: data)
        #expect(ordered.tools == plain.tools)
    }

    /// The scan never rejects: malformed text is the decoder's to report.
    @Test
    func malformedTextIsTheDecodersError() {
        for text in [#"{"model":"m","tools":[{"b":1,"a":2"#, #"{"model":"m" "tools":[]}"#, "", "{\"model\":\"m"] {
            #expect(throws: DecodingError.self) {
                _ = try JSONKeyOrder.decode(Envelope.self, from: Data(text.utf8))
            }
        }
    }

    /// What the scan keeps is bounded by the text. Long keys nested deep once
    /// cost a path per level, each as long as everything above it; this text
    /// would have kept hundreds of megabytes that way.
    @Test
    func deepNestingUnderLongKeysScansInLinearSpaceAndTime() {
        let key = String(repeating: "k", count: 20_000)
        let depth = 400
        let text = String(repeating: "{\"b\":1,\"\(key)\":", count: depth) + "1" + String(repeating: "}", count: depth)
        let clock = ContinuousClock()
        let elapsed = clock.measure { _ = JSONKeyOrder(scanning: Data(text.utf8)) }
        #expect(elapsed < .seconds(5))
    }

    /// Deeper than the scan follows: no crash, and the decoder still decides.
    @Test
    func deepNestingDoesNotOverflow() {
        let depth = 100_000
        let text = String(repeating: "[", count: depth) + String(repeating: "]", count: depth)
        #expect(throws: (any Error).self) {
            _ = try JSONKeyOrder.decode(JSONValue.self, from: Data(text.utf8))
        }
        _ = JSONKeyOrder(scanning: Data(String(repeating: "{\"a\":", count: depth).utf8))
    }
}
