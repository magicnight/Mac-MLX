import Foundation
import Testing

@testable import MacMLXCore

// MARK: - MidConversationSystemFolding Tests
//
// Pure, MLX-free: where system turns end up once a conversation is laid out
// for a template that accepts a system turn only first, and that the layout
// of a history stays a prefix of the layout of the same history extended (the
// prompt cache's reuse depends on it).

@Suite("MidConversationSystemFolding")
struct MidConversationSystemFoldingTests {

    private func system(_ text: String) -> ChatMessage { ChatMessage(role: .system, content: text) }
    private func user(_ text: String) -> ChatMessage { ChatMessage(role: .user, content: text) }
    private func assistant(_ text: String) -> ChatMessage { ChatMessage(role: .assistant, content: text) }
    private func toolCall(_ id: String) -> ChatMessage {
        ChatMessage(role: .assistant, content: "", toolCalls: [ToolCallRequest(id: id, name: "Bash", arguments: [:])])
    }
    private func toolResult(_ id: String, _ text: String) -> ChatMessage {
        ChatMessage(role: .tool, content: text, toolCallID: id)
    }

    private let image = ImageAttachment(fileURL: URL(filePath: "/tmp/x.png"), mimeType: "image/png")

    // MARK: Placement

    @Test
    func leadingSystemTurnsJoinIntoThePromptInOrder() {
        let (prompt, messages) = MidConversationSystemFolding.fold([
            system("first"), system("second"), user("hi"),
        ])
        #expect(prompt == "first\n\nsecond")
        #expect(messages.map(\.role) == [.user])
        #expect(messages.map(\.content) == ["hi"])
    }

    @Test
    func noSystemTurnsLeavesMessagesUntouched() {
        let input = [user("a"), assistant("b"), user("c")]
        let (prompt, messages) = MidConversationSystemFolding.fold(input)
        #expect(prompt == nil)
        #expect(messages == input)
    }

    @Test
    func systemTurnAfterAUserTurnIsAppendedToIt() {
        let first = ChatMessage(role: .user, content: "question", images: [image])
        let (prompt, messages) = MidConversationSystemFolding.fold([
            system("lead"), first, system("note 1"), system("note 2"), assistant("b"),
        ])
        #expect(prompt == "lead")
        #expect(messages.map(\.role) == [.user, .assistant])
        #expect(messages[0].content == "question\n\nnote 1\n\nnote 2")
        // The folded turn is the same turn: id and images survive.
        #expect(messages[0].id == first.id)
        #expect(messages[0].images == first.images)
    }

    @Test
    func systemTurnAfterAnAssistantTurnOpensAUserTurn() {
        let (prompt, messages) = MidConversationSystemFolding.fold([
            user("a"), assistant("b"), system("late"),
        ])
        #expect(prompt == nil)
        #expect(messages.map(\.role) == [.user, .assistant, .user])
        #expect(messages.map(\.content) == ["a", "b", "late"])
    }

    @Test
    func systemTurnAfterAToolTurnOpensAUserTurn() {
        let (_, messages) = MidConversationSystemFolding.fold([
            user("a"), toolCall("c1"), toolResult("c1", "out"), system("<total_tokens>1</total_tokens>"),
        ])
        #expect(messages.map(\.role) == [.user, .assistant, .tool, .user])
        #expect(messages[2].content == "out")
        #expect(messages[3].content == "<total_tokens>1</total_tokens>")
    }

    @Test
    func userTurnDirectlyAfterAnOpenedTurnIsMergedIntoIt() {
        let next = ChatMessage(role: .user, content: "question", images: [image])
        let (_, messages) = MidConversationSystemFolding.fold([
            user("a"), assistant("b"), system("note 1"), system("note 2"), next, user("separate"),
        ])
        #expect(messages.map(\.role) == [.user, .assistant, .user, .user])
        #expect(messages[2].content == "note 1\n\nnote 2\n\nquestion")
        // The merged turn is the user turn: its id and images.
        #expect(messages[2].id == next.id)
        #expect(messages[2].images == next.images)
        // Only the turn directly after merges.
        #expect(messages[3].content == "separate")
    }

    @Test
    func systemTurnAfterALeadingAssistantTurnNeverReachesThePrompt() {
        let (prompt, messages) = MidConversationSystemFolding.fold([
            system("lead"), assistant("b"), system("late"),
        ])
        #expect(prompt == "lead")
        #expect(messages.map(\.role) == [.assistant, .user])
        #expect(messages.map(\.content) == ["b", "late"])
    }

    @Test
    func systemOnlyInputIsAllPrompt() {
        let (prompt, messages) = MidConversationSystemFolding.fold([system("one"), system("two")])
        #expect(prompt == "one\n\ntwo")
        #expect(messages.isEmpty)
    }

    @Test
    func imagesOnSystemTurnsBecomePlaceholders() {
        let (prompt, messages) = MidConversationSystemFolding.fold([
            ChatMessage(role: .system, content: "lead", images: [image]),
            user("a"),
            ChatMessage(role: .system, content: "", images: [image, image]),
            assistant("b"),
        ])
        #expect(prompt == "lead\n[image omitted from a system message]")
        #expect(messages[0].content == "a\n\n[image omitted from a system message]\n"
            + "[image omitted from a system message]")
        #expect(messages.allSatisfy { $0.images.isEmpty })
    }

    @Test
    func emptyTextsAddNoStrayBlankLines() {
        let (prompt, messages) = MidConversationSystemFolding.fold([
            system(""), user("a"), assistant("b"), system(""), system("note"), user(""),
        ])
        #expect(prompt == nil)
        // The empty system turn opens nothing; the empty user turn merges in
        // without a trailing separator.
        #expect(messages.map(\.content) == ["a", "b", "note"])
    }

    // MARK: Prefix stability

    /// The folded rendering of `history` must be a prefix of the folded
    /// rendering of `history + more`, message by message, except that the
    /// last message of `history` may grow by appended text.
    private func expectPrefixStable(
        _ history: [ChatMessage],
        extendedBy more: [ChatMessage],
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let short = MidConversationSystemFolding.fold(history)
        let long = MidConversationSystemFolding.fold(history + more)
        #expect(short.systemPrompt == long.systemPrompt, sourceLocation: sourceLocation)
        guard let lastShort = short.messages.last else { return }
        guard long.messages.count >= short.messages.count else {
            Issue.record("the extended rendering is shorter", sourceLocation: sourceLocation)
            return
        }
        for (a, b) in zip(short.messages.dropLast(), long.messages) {
            #expect(a.role == b.role && a.content == b.content && a.images == b.images
                && a.toolCallID == b.toolCallID && a.toolCalls == b.toolCalls,
                "message changed: \(a.content) → \(b.content)", sourceLocation: sourceLocation)
        }
        let grown = long.messages[short.messages.count - 1]
        #expect(grown.role == lastShort.role, sourceLocation: sourceLocation)
        #expect(grown.content.hasPrefix(lastShort.content), sourceLocation: sourceLocation)
        #expect(grown.toolCallID == lastShort.toolCallID && grown.toolCalls == lastShort.toolCalls,
                sourceLocation: sourceLocation)
        #expect(Array(grown.images.prefix(lastShort.images.count)) == lastShort.images,
                sourceLocation: sourceLocation)
    }

    /// Claude Code's first request: three user text blocks (joined), then a
    /// system turn carrying its reminder; the second replays both and adds the
    /// assistant tool call, the tool result, and a trailing system turn.
    @Test
    func claudeCodeTurnsArePrefixStable() {
        let turn1 = [user("block 1\nblock 2\nblock 3"), system("<system-reminder>CLAUDE.md…</system-reminder>")]
        let turn2Additions = [toolCall("toolu_1"), toolResult("toolu_1", "hi"), system("<total_tokens>9</total_tokens>")]
        let turn3Additions = [assistant("Done."), user("next task"), system("<system-reminder>…</system-reminder>")]
        expectPrefixStable(turn1, extendedBy: turn2Additions)
        expectPrefixStable(turn1 + turn2Additions, extendedBy: turn3Additions)

        // The reminder stays in the first user turn on every turn.
        let (_, messages) = MidConversationSystemFolding.fold(turn1 + turn2Additions)
        #expect(messages.map(\.role) == [.user, .assistant, .tool, .user])
        #expect(messages[0].content.hasSuffix("CLAUDE.md…</system-reminder>"))
        #expect(messages[3].content == "<total_tokens>9</total_tokens>")
    }

    /// Codex: developer items lead (the system prompt), and a later one arrives
    /// after a tool output, before the next user message.
    @Test
    func codexDeveloperItemsArePrefixStable() {
        let turn1 = [system("dev one"), system("dev two"), user("Run echo")]
        let turn2Additions = [toolCall("call_1"), toolResult("call_1", "hi\n"), system("permissions changed")]
        let turn3Additions = [user("now list files"), toolCall("call_2"), toolResult("call_2", "a b")]
        expectPrefixStable(turn1, extendedBy: turn2Additions)
        expectPrefixStable(turn1 + turn2Additions, extendedBy: turn3Additions)
        expectPrefixStable(turn1 + turn2Additions + turn3Additions, extendedBy: [system("note"), user("more")])

        let (prompt, messages) = MidConversationSystemFolding.fold(turn1 + turn2Additions + turn3Additions)
        #expect(prompt == "dev one\n\ndev two")
        #expect(messages.map(\.role) == [.user, .assistant, .tool, .user, .assistant, .tool])
        #expect(messages[3].content == "permissions changed\n\nnow list files")
    }

    /// Every split point of one mixed conversation keeps the property.
    @Test
    func everySplitOfAMixedConversationIsPrefixStable() {
        let conversation = [
            system("lead"), user("a"), system("s1"), toolCall("c1"), toolResult("c1", "r1"),
            system("s2"), system("s3"), user("b"), user("c"), assistant("d"), system("s4"),
            toolCall("c2"), toolResult("c2", "r2"), user("e"), system("s5"),
        ]
        // From the first item on: an empty history has no system prompt yet.
        for split in 1...conversation.count {
            expectPrefixStable(Array(conversation.prefix(split)), extendedBy: Array(conversation.dropFirst(split)))
        }
    }
}
