import Foundation
import MLXLMCommon
import Testing

@testable import MacMLXCore

// MARK: - Replayed reasoning reaching the chat template
//
// Pure, MLX-free: how `MLXSwiftEngine.userInput(for:isVLM:messageGenerator:)`
// carries an assistant turn's reasoning to the template — the message
// generator's dictionaries with `reasoning_content` set on the matching
// assistant dictionary — and that a request without reasoning still builds
// the `.chat` input it always did.

@Suite("ReasoningInjection")
struct ReasoningInjectionTests {

    private let call = ToolCallRequest(id: "c1", name: "Bash", arguments: ["command": .string("ls")])

    /// A tool loop: two assistant turns, the first with reasoning.
    private var conversation: [ChatMessage] {
        [
            ChatMessage(role: .user, content: "List the files."),
            ChatMessage(role: .assistant, content: "", toolCalls: [call], reasoning: "Run ls."),
            ChatMessage(role: .tool, content: "a.txt", toolCallID: "c1"),
            ChatMessage(role: .assistant, content: "One file.", reasoning: nil),
        ]
    }

    private func request(_ messages: [ChatMessage], system: String? = "Be brief.") -> GenerateRequest {
        GenerateRequest(
            model: "m", messages: messages, systemPrompt: system,
            templateKwargs: ["enable_thinking": .bool(true)],
            tools: [.object(["type": .string("function"), "function": .object(["name": .string("Bash")])])])
    }

    private func chat(_ messages: [ChatMessage]) -> [Chat.Message] {
        messages.map { MLXSwiftEngine.upstreamChatMessage(from: $0, images: []) }
    }

    private func dicts(_ input: UserInput) throws -> [[String: any Sendable]] {
        guard case .messages(let messages) = input.prompt else {
            Issue.record("expected a .messages prompt, got \(input.prompt)")
            return []
        }
        return messages
    }

    private func roles(_ messages: [[String: any Sendable]]) -> [String] {
        messages.compactMap { $0["role"] as? String }
    }

    private func reasoning(_ messages: [[String: any Sendable]]) -> [String?] {
        messages.map { $0["reasoning_content"] as? String }
    }

    // MARK: injectingReasoning

    @Test
    func theKthAssistantDictionaryGetsTheKthReasoning() throws {
        let all = request(conversation).allMessages
        let generated = DefaultMessageGenerator().generate(messages: chat(all))
        let injected = try #require(MLXSwiftEngine.injectingReasoning(
            generated, assistantReasoning: ["Run ls.", nil]))
        #expect(roles(injected) == ["system", "user", "assistant", "tool", "assistant"])
        #expect(reasoning(injected) == [nil, nil, "Run ls.", nil, nil])
        // Everything else is the generator's own output.
        #expect(injected[2]["tool_calls"] != nil)
        #expect(injected[4]["content"] as? String == "One file.")
    }

    /// Gemma's generator drops system messages, so dictionaries and messages
    /// are not one to one; assistant turns still line up.
    @Test
    func aGeneratorThatDropsSystemMessagesStillLinesUp() throws {
        let all = request(conversation).allMessages
        let generated = NoSystemMessageGenerator().generate(messages: chat(all))
        let injected = try #require(MLXSwiftEngine.injectingReasoning(
            generated, assistantReasoning: [nil, "Then answer."]))
        #expect(roles(injected) == ["user", "assistant", "tool", "assistant"])
        #expect(reasoning(injected) == [nil, nil, nil, "Then answer."])
    }

    @Test
    func emptyReasoningAddsNoKey() throws {
        let generated = DefaultMessageGenerator().generate(messages: chat(conversation))
        let injected = try #require(MLXSwiftEngine.injectingReasoning(generated, assistantReasoning: ["", nil]))
        #expect(reasoning(injected) == [nil, nil, nil, nil])
    }

    @Test
    func aCountMismatchInjectsNothing() {
        let generated = DefaultMessageGenerator().generate(messages: chat(conversation))
        #expect(MLXSwiftEngine.injectingReasoning(generated, assistantReasoning: ["Run ls."]) == nil)
        #expect(MLXSwiftEngine.injectingReasoning(generated, assistantReasoning: ["a", "b", "c"]) == nil)
    }

    // MARK: userInput

    @Test
    func reasoningBuildsTheGeneratorsMessagesWithToolsAndKwargs() throws {
        let input = MLXSwiftEngine.userInput(
            for: request(conversation), isVLM: false, messageGenerator: NoSystemMessageGenerator(),
            logsDroppedInputs: false)
        let messages = try dicts(input)
        #expect(roles(messages) == ["user", "assistant", "tool", "assistant"])
        #expect(reasoning(messages) == [nil, "Run ls.", nil, nil])
        #expect(input.tools?.count == 1)
        #expect(input.additionalContext?["enable_thinking"] as? Bool == true)
    }

    /// No reasoning: the `.chat` input of before, so the prompt is unchanged.
    @Test
    func withoutReasoningTheInputIsChat() {
        let plain = conversation.map {
            ChatMessage(id: $0.id, role: $0.role, content: $0.content, toolCallID: $0.toolCallID, toolCalls: $0.toolCalls)
        }
        let input = MLXSwiftEngine.userInput(
            for: request(plain), isVLM: false, messageGenerator: DefaultMessageGenerator(), logsDroppedInputs: false)
        guard case .chat(let messages) = input.prompt else {
            Issue.record("expected a .chat prompt")
            return
        }
        #expect(messages.map(\.role) == [.system, .user, .assistant, .tool, .assistant])
        #expect(input.tools?.count == 1)
    }

    /// A VLM (no message generator) keeps `.chat`: its processor handles the
    /// images there, so reasoning is not rendered on a VLM.
    @Test
    func withoutAGeneratorTheInputIsChat() {
        let input = MLXSwiftEngine.userInput(
            for: request(conversation), isVLM: true, messageGenerator: nil, logsDroppedInputs: false)
        guard case .chat = input.prompt else {
            Issue.record("expected a .chat prompt")
            return
        }
    }
}
