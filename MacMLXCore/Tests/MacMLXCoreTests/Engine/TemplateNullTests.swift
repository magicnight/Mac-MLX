import Foundation
import Jinja
import MLXLLM
import MLXLMCommon
import Testing

@testable import MacMLXCore

import enum MacMLXCore.JSONValue

// MARK: - JSON null reaching the chat template
//
// Pure, MLX-free. swift-jinja reads Swift `nil` as null and throws on
// `NSNull` ("Cannot convert value of type NSNull to Jinja Value"), so a JSON
// null in a tool schema, a template kwarg or a replayed tool call's arguments
// must reach `applyChatTemplate` as `nil`. Each test converts
// `MLXSwiftEngine.userInput(for:isVLM:messageGenerator:)`'s output the way
// swift-transformers' `applyChatTemplate` does (`Value(any:)` on every
// message, tool and context entry) and renders it.

@Suite("TemplateNull")
struct TemplateNullTests {

    /// The tool of issue #175's reproduction, plus a null inside an array.
    private let toolWithNulls: JSONValue = .object([
        "type": .string("function"),
        "function": .object([
            "name": .string("x"),
            "parameters": .object([
                "type": .string("object"),
                "properties": .object([
                    "a": .object(["type": .string("string"), "default": .null]),
                    "b": .object(["enum": .array([.string("on"), .null])]),
                ]),
            ]),
        ]),
    ])

    private func callingTool(arguments: [String: JSONValue]) -> [ChatMessage] {
        [
            ChatMessage(role: .user, content: "Weather in Paris?"),
            ChatMessage(
                role: .assistant, content: "",
                toolCalls: [ToolCallRequest(id: "c1", name: "get_weather", arguments: arguments)]),
            ChatMessage(role: .tool, content: "18C", toolCallID: "c1"),
        ]
    }

    private let argumentsWithNulls: [String: JSONValue] = [
        "city": .string("Paris"),
        "unit": .null,
        "filter": .object(["tag": .null, "pages": .array([.int(1), .null])]),
    ]

    /// The render context `applyChatTemplate` builds from `input`, with the
    /// messages a processor holding `generator` would hand it.
    private func context(
        _ input: UserInput, generator: any MessageGenerator = DefaultMessageGenerator()
    ) throws -> [String: Jinja.Value] {
        var context: [String: Jinja.Value] = [
            "messages": .array(try generator.generate(from: input).map { try Jinja.Value(any: $0) })
        ]
        if let tools = input.tools {
            context["tools"] = .array(try tools.map { try Jinja.Value(any: $0) })
        }
        for (key, value) in input.additionalContext ?? [:] {
            context[key] = try Jinja.Value(any: value)
        }
        return context
    }

    /// `expression | tojson` rendered over `context`, parsed back.
    private func json(_ expression: String, in context: [String: Jinja.Value]) throws -> Any {
        let rendered = try Jinja.Template("{{ \(expression) | tojson }}").render(context)
        return try JSONSerialization.jsonObject(with: Data(rendered.utf8), options: [.fragmentsAllowed])
    }

    private func value(_ root: Any, at path: [Any]) -> Any? {
        path.reduce(Optional(root)) { node, step in
            if let key = step as? String { return (node as? [String: Any])?[key] }
            if let index = step as? Int, let array = node as? [Any], array.indices.contains(index) {
                return array[index]
            }
            return nil
        }
    }

    // MARK: Tool schemas

    @Test
    func aNullInAToolSchemaRendersAsNull() throws {
        let request = GenerateRequest(
            model: "m", messages: [ChatMessage(role: .user, content: "hi")], tools: [toolWithNulls])
        let input = MLXSwiftEngine.userInput(
            for: request, isVLM: false, messageGenerator: DefaultMessageGenerator(), logsDroppedInputs: false)
        let tools = try json("tools", in: try context(input))
        let properties: [Any] = [0, "function", "parameters", "properties"]
        #expect(value(tools, at: properties + ["a", "default"]) is NSNull)
        #expect(value(tools, at: properties + ["a", "type"]) as? String == "string")
        #expect(value(tools, at: properties + ["b", "enum", 0]) as? String == "on")
        #expect(value(tools, at: properties + ["b", "enum", 1]) is NSNull)
    }

    // MARK: Template kwargs

    @Test
    func aNullTemplateKwargRendersAsNull() throws {
        let request = GenerateRequest(
            model: "m", messages: [ChatMessage(role: .user, content: "hi")],
            templateKwargs: ["reasoning_effort": .null, "enable_thinking": .bool(true)])
        let input = MLXSwiftEngine.userInput(
            for: request, isVLM: false, messageGenerator: DefaultMessageGenerator(), logsDroppedInputs: false)
        let context = try context(input)
        #expect(context["reasoning_effort"] == .null)
        #expect(context["enable_thinking"] == .boolean(true))
    }

    // MARK: Replayed tool calls

    @Test(arguments: [0, 1])
    func aNullToolCallArgumentRendersAsNull(generatorIndex: Int) throws {
        let generators: [any MessageGenerator] = [DefaultMessageGenerator(), NoSystemMessageGenerator()]
        let generator = generators[generatorIndex]
        let request = GenerateRequest(model: "m", messages: callingTool(arguments: argumentsWithNulls))
        let input = MLXSwiftEngine.userInput(
            for: request, isVLM: false, messageGenerator: generator, logsDroppedInputs: false)
        let messages = try json("messages", in: try context(input, generator: generator))
        let arguments: [Any] = [1, "tool_calls", 0, "function", "arguments"]
        #expect(value(messages, at: arguments + ["city"]) as? String == "Paris")
        #expect(value(messages, at: arguments + ["unit"]) is NSNull)
        #expect(value(messages, at: arguments + ["filter", "tag"]) is NSNull)
        #expect(value(messages, at: arguments + ["filter", "pages", 0]) as? Int == 1)
        #expect(value(messages, at: arguments + ["filter", "pages", 1]) is NSNull)
        #expect(value(messages, at: [1, "tool_calls", 0, "id"]) as? String == "c1")
        #expect(value(messages, at: [2, "tool_call_id"]) as? String == "c1")
    }

    /// A tool call without a null renders from the `.chat` input it always
    /// did: only a request that would fail takes the generator's dictionaries.
    @Test
    func aToolCallWithoutANullStillBuildsTheChatInput() {
        let request = GenerateRequest(
            model: "m",
            messages: callingTool(arguments: ["city": .string("Paris"), "days": .array([.int(1), .int(2)])]))
        let input = MLXSwiftEngine.userInput(
            for: request, isVLM: false, messageGenerator: DefaultMessageGenerator(), logsDroppedInputs: false)
        guard case .chat = input.prompt else {
            Issue.record("expected a .chat prompt, got \(input.prompt)")
            return
        }
    }

    // MARK: JSONValue

    /// `toSendable()` feeds `JSONSerialization`, which needs `NSNull`.
    @Test
    func toSendableKeepsNSNullForJSONSerialization() throws {
        let object = JSONValue.object(["a": .null, "b": .array([.null, .int(1)])]).toSendable()
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        #expect(String(decoding: data, as: UTF8.self) == #"{"a":null,"b":[null,1]}"#)
    }

    @Test
    func toTemplateValueDiffersFromToSendableOnlyAtNulls() throws {
        let value = JSONValue.object([
            "s": .string("x"), "i": .int(2), "d": .double(1.5), "t": .bool(true),
            "n": .null, "list": .array([.null, .string("y")]),
        ])
        let template = try #require(value.toTemplateValue() as? [String: any Sendable])
        #expect(template["s"] as? String == "x")
        #expect(template["i"] as? Int == 2)
        #expect(template["d"] as? Double == 1.5)
        #expect(template["t"] as? Bool == true)
        #expect(!(template["n"] is NSNull))
        #expect(try Jinja.Value(any: template["n"]) == .null)
        let list = try #require(template["list"] as? [any Sendable])
        #expect(try Jinja.Value(any: list[0]) == .null)
        #expect(list[1] as? String == "y")
    }
}
