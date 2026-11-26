import Foundation
import Jinja
import MLXLLM
import MLXLMCommon
import Testing

@testable import MacMLXCore

import enum MacMLXCore.JSONValue

/// A tool schema must reach the chat template with its keys in the order the
/// client sent them. Python renders `tools | tojson` with `json.dumps` and no
/// `sort_keys`, so the checkpoints were trained on
/// `{"type": "function", "function": {"name": …, "description": …,
/// "parameters": …}}`; a render with the keys sorted is the same JSON and a
/// different prompt. macMLX sorted them until this was fixed: the request
/// decoded into Swift dictionaries, and swift-transformers sorts a dictionary's
/// keys when it builds the template context.
///
/// The tests take the path a server request takes: the request JSON is decoded
/// with `JSONKeyOrder.decode`, `MLXSwiftEngine.userInput` builds the input, and
/// `TokenizerBridge.applyChatTemplate` renders it. The fixture is a byte-level
/// BPE tokenizer with one token per printable ASCII character and no merges,
/// so decoding the returned ids gives the rendered text back; its template is
/// the tools loop the Qwen templates use (`tool | tojson`). Each expected
/// string is what `json.dumps(tool, ensure_ascii=False)` prints for the same
/// request. No model, no Metal.
@Suite("TokenizerBridge tool key order")
struct TokenizerBridgeToolKeyOrderTests {

    private struct Request: Decodable {
        let tools: [JSONValue]
    }

    private func load() async throws -> any MLXLMCommon.Tokenizer {
        let dir = try #require(
            Bundle.module.url(
                forResource: "ascii_bytelevel_chat_tokenizer", withExtension: nil, subdirectory: "Fixtures"))
        return try await HuggingFaceTokenizerLoader().load(from: dir)
    }

    /// The prompt for a request body's tools and one user message, as text.
    private func render(_ body: String, templateKwargs: [String: JSONValue]? = nil) async throws -> String {
        let tokenizer = try await load()
        let tools = try JSONKeyOrder.decode(Request.self, from: Data(body.utf8)).tools
        let request = GenerateRequest(
            model: "m", messages: [ChatMessage(role: .user, content: "hi")], templateKwargs: templateKwargs,
            tools: tools)
        let generator = DefaultMessageGenerator()
        let input = MLXSwiftEngine.userInput(
            for: request, isVLM: false, messageGenerator: generator, logsDroppedInputs: false)
        let ids = try tokenizer.applyChatTemplate(
            messages: generator.generate(from: input), tools: input.tools, additionalContext: input.additionalContext)
        return tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
    }

    @Test
    func toolsRenderInTheOrderTheClientSentThem() async throws {
        let rendered = try await render(
            #"""
            {"tools":[{"type":"function","function":{"name":"get_weather","description":"Get the weather","parameters":{"type":"object","properties":{"location":{"type":"string","description":"City name"},"unit":{"type":"string","enum":["c","f"]}},"required":["location"]}}}]}
            """#)
        #expect(
            rendered
                == #"<tools>{"type": "function", "function": {"name": "get_weather", "description": "Get the weather", "#
                + #""parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City name"}, "#
                + #""unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["location"]}}}</tools>hi"#)
    }

    /// The order is the client's, not a canonical one: a spec written in
    /// neither sorted nor OpenAI's documented order renders as written, and so
    /// does each tool of several.
    @Test
    func anUnusualOrderIsKeptAsWritten() async throws {
        let rendered = try await render(
            #"""
            {"tools":[{"function":{"parameters":{"required":["b"],"type":"object","properties":{"b":{"type":"integer","minimum":0},"a":{"type":["string","null"],"default":null}}},"name":"f","description":"d"},"type":"function"},{"type":"function","function":{"name":"g","strict":true,"parameters":{}}}]}
            """#)
        #expect(
            rendered
                == #"<tools>{"function": {"parameters": {"required": ["b"], "type": "object", "properties": "#
                + #"{"b": {"type": "integer", "minimum": 0}, "a": {"type": ["string", "null"], "default": null}}}, "#
                + #""name": "f", "description": "d"}, "type": "function"}"#
                + #"{"type": "function", "function": {"name": "g", "strict": true, "parameters": {}}}</tools>hi"#)
    }

    /// Template kwargs travel in the same context and are not disturbed.
    @Test
    func templateKwargsStillReachTheTemplateBesideTheTools() async throws {
        let rendered = try await render(
            #"{"tools":[{"type":"function","function":{"name":"f"}}]}"#, templateKwargs: ["enable_thinking": .bool(true)])
        #expect(rendered == #"<tools>{"type": "function", "function": {"name": "f"}}</tools>hi<think>"#)
    }

    // MARK: The context

    @Test
    func theContextHasToolsOnlyWhenTheRequestHasThem() {
        #expect(MLXSwiftEngine.templateContext(kwargs: nil, tools: nil) == nil)
        let kwargsOnly = MLXSwiftEngine.templateContext(kwargs: ["enable_thinking": .bool(true)], tools: nil)
        #expect(kwargsOnly?.keys.sorted() == ["enable_thinking"])
        let both = MLXSwiftEngine.templateContext(
            kwargs: ["enable_thinking": .bool(true)], tools: [.object(["type": .string("function")])])
        #expect(both?.keys.sorted() == ["enable_thinking", "tools"])
    }

    /// The ordered list has the elements `toolSpecs(from:)` keeps: a spec that
    /// is not an object is dropped from both.
    @Test
    func theOrderedListMatchesTheToolSpecs() throws {
        let tools: [JSONValue] = [.string("bogus"), .object(["type": .string("function")]), .null]
        let context = try #require(MLXSwiftEngine.templateContext(kwargs: nil, tools: tools))
        let rendered = try #require(context["tools"] as? Jinja.Value)
        guard case .array(let elements) = rendered else {
            Issue.record("tools is not a list")
            return
        }
        #expect(elements.count == MLXSwiftEngine.toolSpecs(from: tools).count)
        #expect(elements.count == 1)
    }

    /// A `tools` template kwarg is the user's own override and wins, as it did
    /// before the context carried the request's tools.
    @Test
    func aToolsTemplateKwargIsLeftAlone() throws {
        let context = try #require(
            MLXSwiftEngine.templateContext(
                kwargs: ["tools": .string("mine")], tools: [.object(["type": .string("function")])]))
        #expect(context["tools"] as? String == "mine")
    }
}
