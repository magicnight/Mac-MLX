import Testing
import Foundation
import MCP
@testable import MacMLXCore

// Note: this file deliberately does NOT import MLXLMCommon so bare `JSONValue`
// resolves unambiguously to macMLX's type. The upstream
// `MLXLMCommon.JSONValue → JSONValue` converter is covered in
// `ToolCallDetectionTests` (which needs the MLXLMCommon import for `ToolCall`).

@Suite("ToolValueBridge")
struct ToolValueBridgeTests {

    // MARK: - MCP.Value → macMLX JSONValue

    @Test("MCP scalars, null and nested collections convert to macMLX JSONValue")
    func mcpValueToJSONValueCoversAllCases() {
        let mcp: MCP.Value = .object([
            "flag": .bool(true),
            "count": .int(7),
            "ratio": .double(1.5),
            "label": .string("hi"),
            "empty": .null,
            "list": .array([.int(1), .string("a")]),
            "nested": .object(["deep": .array([.bool(false)])]),
        ])

        let json = ToolValueBridge.jsonValue(from: mcp)

        let expected: JSONValue = .object([
            "flag": .bool(true),
            "count": .int(7),
            "ratio": .double(1.5),
            "label": .string("hi"),
            "empty": .null,
            "list": .array([.int(1), .string("a")]),
            "nested": .object(["deep": .array([.bool(false)])]),
        ])
        #expect(json == expected)
    }

    @Test("MCP .data becomes a base64 string (mime type dropped)")
    func mcpDataConvertsToBase64String() {
        let bytes = Data([0x00, 0x01, 0xFE, 0xFF])
        let mcp: MCP.Value = .data(mimeType: "application/octet-stream", bytes)

        let json = ToolValueBridge.jsonValue(from: mcp)

        #expect(json == .string(bytes.base64EncodedString()))
    }

    // MARK: - macMLX JSONValue → MCP.Value

    @Test("macMLX JSONValue converts to MCP.Value across all cases")
    func jsonValueToMCPValueCoversAllCases() {
        let json: JSONValue = .object([
            "flag": .bool(true),
            "count": .int(7),
            "ratio": .double(1.5),
            "label": .string("hi"),
            "empty": .null,
            "list": .array([.int(1), .string("a")]),
        ])

        let mcp = ToolValueBridge.mcpValue(from: json)

        let expected: MCP.Value = .object([
            "flag": .bool(true),
            "count": .int(7),
            "ratio": .double(1.5),
            "label": .string("hi"),
            "empty": .null,
            "list": .array([.int(1), .string("a")]),
        ])
        #expect(mcp == expected)
    }

    @Test("macMLX → MCP → macMLX round-trips a realistic argument dictionary")
    func jsonValueRoundTripsThroughMCPValue() {
        let original: JSONValue = .object([
            "location": .string("Paris"),
            "days": .int(3),
            "units": .string("metric"),
            "flags": .array([.bool(true), .null]),
        ])

        let back = ToolValueBridge.jsonValue(from: ToolValueBridge.mcpValue(from: original))

        #expect(back == original)
    }

    // MARK: - MCP.Tool → OpenAI function spec

    @Test("MCP Tool converts to the OpenAI function spec shape with the schema inlined")
    func toolConvertsToOpenAIFunctionSpec() {
        let schema: MCP.Value = .object([
            "type": .string("object"),
            "properties": .object([
                "location": .object([
                    "type": .string("string"),
                    "description": .string("City name"),
                ]),
            ]),
            "required": .array([.string("location")]),
        ])
        let tool = MCP.Tool(
            name: "get_weather",
            description: "Look up the weather for a city",
            inputSchema: schema
        )

        let spec = ToolValueBridge.openAIToolSpec(from: tool)

        let expected: JSONValue = .object([
            "type": .string("function"),
            "function": .object([
                "name": .string("get_weather"),
                "description": .string("Look up the weather for a city"),
                "parameters": ToolValueBridge.jsonValue(from: schema),
            ]),
        ])
        #expect(spec == expected)
    }

    @Test("A nil tool description becomes an empty string in the function spec")
    func toolWithNilDescriptionUsesEmptyString() {
        let tool = MCP.Tool(
            name: "ping",
            description: nil,
            inputSchema: .object(["type": .string("object")])
        )

        let spec = ToolValueBridge.openAIToolSpec(from: tool)

        guard case .object(let root) = spec,
              case .object(let function)? = root["function"],
              case .string(let description)? = function["description"]
        else {
            Issue.record("Expected nested function.description string, got \(spec)")
            return
        }
        #expect(description == "")
    }
}
