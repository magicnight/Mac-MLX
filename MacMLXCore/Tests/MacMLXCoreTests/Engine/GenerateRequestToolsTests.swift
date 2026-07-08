import Testing
import Foundation
@testable import MacMLXCore

@Suite("GenerateRequest.tools")
struct GenerateRequestToolsTests {

    @Test("tools round-trips through Codable")
    func toolsRoundTrip() throws {
        let tools: [JSONValue] = [
            .object([
                "type": .string("function"),
                "function": .object([
                    "name": .string("get_weather"),
                    "description": .string("Look up weather"),
                    "parameters": .object(["type": .string("object")]),
                ]),
            ]),
        ]
        let req = GenerateRequest(
            model: "m",
            messages: [ChatMessage(role: .user, content: "hi")],
            tools: tools
        )
        let data = try JSONEncoder().encode(req)
        let back = try JSONDecoder().decode(GenerateRequest.self, from: data)
        #expect(back == req)
        #expect(back.tools?.count == 1)
    }

    /// Pre-v0.5 request JSON has no `tools` (or `templateKwargs`) key. Both are
    /// optional, so the synthesised decoder must default them to nil.
    @Test("legacy request JSON without a tools key decodes with nil tools")
    func legacyDecodesWithNilTools() throws {
        let legacy = """
        {"model":"m",\
        "messages":[{"id":"1FAA0000-0000-0000-0000-000000000001","role":"user","content":"hi"}],\
        "parameters":{"temperature":0.7,"topP":0.95,"maxTokens":2048,"stream":true}}
        """
        let decoded = try JSONDecoder().decode(GenerateRequest.self, from: Data(legacy.utf8))
        #expect(decoded.tools == nil)
        #expect(decoded.templateKwargs == nil)
        #expect(decoded.messages.count == 1)
    }

    @Test("toolSpecs(from:) converts object specs and drops non-objects")
    func toolSpecsConvertsObjectsAndDropsNonObjects() {
        let specs = MLXSwiftEngine.toolSpecs(from: [
            .object(["name": .string("a"), "n": .int(1)]),
            .string("not-an-object"),   // dropped
            .array([.int(1)]),          // dropped
            .object(["name": .string("b")]),
        ])
        #expect(specs.count == 2)
        #expect((specs[0]["name"] as? String) == "a")
        #expect((specs[0]["n"] as? Int) == 1)
        #expect((specs[1]["name"] as? String) == "b")
    }
}
