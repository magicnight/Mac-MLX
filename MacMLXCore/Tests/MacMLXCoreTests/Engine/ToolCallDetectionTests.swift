import Testing
import Foundation
import MLXLMCommon
@testable import MacMLXCore

@Suite("Tool-call detection")
struct ToolCallDetectionTests {

    @Test("FinishReason.toolCalls raw value matches OpenAI")
    func finishReasonToolCallsRawValue() {
        #expect(FinishReason.toolCalls.rawValue == "tool_calls")
    }

    @Test("GenerateChunk carries tool calls and round-trips; a plain chunk omits the key")
    func generateChunkToolCallsRoundTrip() throws {
        let call = ToolCallRequest(
            id: "call_1", name: "get_weather", arguments: ["city": .string("Paris")])
        let chunk = GenerateChunk(
            text: "",
            finishReason: .toolCalls,
            usage: TokenUsage(promptTokens: 3, completionTokens: 4),
            toolCalls: [call]
        )
        let data = try JSONEncoder().encode(chunk)
        let back = try JSONDecoder().decode(GenerateChunk.self, from: data)
        #expect(back == chunk)
        #expect(back.toolCalls?.first?.name == "get_weather")

        // A plain text chunk must not carry the tool-calls key (wire back-compat).
        let plain = GenerateChunk(text: "hi")
        let plainJSON = String(decoding: try JSONEncoder().encode(plain), as: UTF8.self)
        #expect(!plainJSON.contains("toolCalls"))
    }

    @Test("Streaming processor extracts a JSON-format tool call and strips it from display")
    func processorExtractsJSONToolCall() {
        let (display, calls) = MLXSwiftEngine.processToolCallStream(
            format: .json,
            pieces: [
                "Sure. ",
                "<tool_call>",
                #"{"name": "get_weather", "arguments": {"city": "Paris"}}"#,
                "</tool_call>",
            ]
        )
        #expect(display == "Sure. ")
        #expect(calls.count == 1)
        #expect(calls.first?.name == "get_weather")
        // This file imports MLXLMCommon, so the bare type name `JSONValue` would
        // be ambiguous; leading-dot member syntax resolves against the (macMLX)
        // argument value's type instead.
        #expect(calls.first?.arguments["city"] == .string("Paris"))
        #expect(calls.first?.id.hasPrefix("call_") == true)
    }

    @Test("Upstream ToolCall converts to ToolCallRequest; a missing id is synthesised")
    func toolCallRequestConversion() {
        let withID = ToolCall(
            function: .init(name: "f", arguments: ["k": MLXLMCommon.JSONValue.int(2)]),
            id: "call_abc")
        let converted = MLXSwiftEngine.toolCallRequest(from: withID)
        #expect(converted.id == "call_abc")
        #expect(converted.name == "f")
        #expect(converted.arguments["k"] == .int(2))

        let noID = ToolCall(
            function: .init(name: "g", arguments: [String: MLXLMCommon.JSONValue]()),
            id: nil)
        let synthesised = MLXSwiftEngine.toolCallRequest(from: noID)
        #expect(synthesised.id.hasPrefix("call_"))
        #expect(synthesised.name == "g")
    }

    @Test(
        "makeToolProcessor gates on hasTools AND off response_format, not merely on format",
        arguments: [
            // (format, hasTools, hasResponseFormat, expectNonNil)
            (ToolCallFormat?.some(.json), true, false, true),
            (ToolCallFormat?.some(.json), false, false, false),   // the hasTools regression
            (ToolCallFormat?.none, true, false, false),
            (ToolCallFormat?.none, false, false, false),
            // response_format present → never engage the processor even with tools:
            // a grammar-constrained turn emits bare JSON the `.json` processor would
            // buffer and mis-drain as an empty tool call.
            (ToolCallFormat?.some(.json), true, true, false),
            (ToolCallFormat?.some(.xmlFunction), true, true, false),
        ]
    )
    func makeToolProcessorGatesOnHasTools(
        format: ToolCallFormat?, hasTools: Bool, hasResponseFormat: Bool, expectNonNil: Bool
    ) {
        let processor = MLXSwiftEngine.makeToolProcessor(
            format: format, hasTools: hasTools, hasResponseFormat: hasResponseFormat)
        #expect((processor != nil) == expectNonNil)
    }

    @Test("Upstream MLXLMCommon.JSONValue converts faithfully to macMLX JSONValue")
    func upstreamJSONValueConvertsFaithfully() {
        let upstream: MLXLMCommon.JSONValue = .object([
            "city": .string("Tokyo"),
            "count": .int(2),
            "verbose": .bool(false),
            "score": .double(0.25),
            "maybe": .null,
            "tags": .array([.string("x"), .string("y")]),
        ])

        // `json` is macMLX's `JSONValue` (inferred from the converter's return
        // type); the leading-dot literal below resolves against it, avoiding the
        // ambiguous bare type name.
        let json = ToolValueBridge.jsonValue(from: upstream)

        #expect(json == .object([
            "city": .string("Tokyo"),
            "count": .int(2),
            "verbose": .bool(false),
            "score": .double(0.25),
            "maybe": .null,
            "tags": .array([.string("x"), .string("y")]),
        ]))
    }
}
