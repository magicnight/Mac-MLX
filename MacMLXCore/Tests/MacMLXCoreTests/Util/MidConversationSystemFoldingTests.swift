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

    /// `/v1/responses`' placement unless `appendsToToolTurns` (`/v1/messages`).
    private func fold(
        _ messages: [ChatMessage], appendsToToolTurns: Bool = false
    ) -> (systemPrompt: String?, messages: [ChatMessage]) {
        MidConversationSystemFolding.fold(messages, appendsToToolTurns: appendsToToolTurns)
    }

    private let image = ImageAttachment(fileURL: URL(filePath: "/tmp/x.png"), mimeType: "image/png")

    // MARK: Placement

    @Test
    func leadingSystemTurnsJoinIntoThePromptInOrder() {
        let (prompt, messages) = fold([
            system("first"), system("second"), user("hi"),
        ])
        #expect(prompt == "first\n\nsecond")
        #expect(messages.map(\.role) == [.user])
        #expect(messages.map(\.content) == ["hi"])
    }

    @Test
    func noSystemTurnsLeavesMessagesUntouched() {
        let input = [user("a"), assistant("b"), user("c")]
        let (prompt, messages) = fold(input)
        #expect(prompt == nil)
        #expect(messages == input)
    }

    @Test
    func systemTurnAfterAUserTurnIsAppendedToIt() throws {
        let first = ChatMessage(role: .user, content: "question", images: [image])
        let (prompt, messages) = fold([
            system("lead"), first, system("note 1"), system("note 2"), assistant("b"),
        ])
        #expect(prompt == "lead")
        try #require(messages.map(\.role) == [.user, .assistant])
        #expect(messages[0].content == "question\n\nnote 1\n\nnote 2")
        // The folded turn is the same turn: id and images survive.
        #expect(messages[0].id == first.id)
        #expect(messages[0].images == first.images)
    }

    @Test
    func systemTurnAfterAnAssistantTurnOpensAUserTurn() throws {
        let (prompt, messages) = fold([
            user("a"), assistant("b"), system("late"),
        ])
        #expect(prompt == nil)
        try #require(messages.map(\.role) == [.user, .assistant, .user])
        #expect(messages.map(\.content) == ["a", "b", "late"])
    }

    /// `/v1/responses`: a developer item after a tool output is an
    /// instruction, kept out of the tool response in a user turn of its own.
    @Test
    func systemTurnAfterAToolTurnOpensAUserTurn() throws {
        let (_, messages) = fold([
            user("a"), toolCall("c1"), toolResult("c1", "out"), system("permissions changed"),
        ])
        try #require(messages.map(\.role) == [.user, .assistant, .tool, .user])
        #expect(messages[2].content == "out")
        #expect(messages[3].content == "permissions changed")
    }

    /// `/v1/messages`, Claude Code's shape: a system turn right after the tool
    /// result joins it rather than opening a user turn, which a thinking
    /// template would take for a new query.
    @Test
    func anthropicSystemTurnAfterAToolTurnIsAppendedToIt() throws {
        let (_, messages) = fold([
            user("a"), toolCall("c1"), toolResult("c1", "out"), system("<total_tokens>1</total_tokens>"),
            system("<total_tokens>2</total_tokens>"),
        ], appendsToToolTurns: true)
        try #require(messages.map(\.role) == [.user, .assistant, .tool])
        #expect(messages[2].content == "out\n\n<total_tokens>1</total_tokens>\n\n<total_tokens>2</total_tokens>")
        #expect(messages[2].toolCallID == "c1")
    }

    /// After several tool results, the last one takes the system text; a user
    /// turn after it stays a turn of its own.
    @Test
    func anthropicSystemTurnAfterSeveralToolTurnsJoinsTheLast() throws {
        let calls = ChatMessage(role: .assistant, content: "", toolCalls: [
            ToolCallRequest(id: "c1", name: "Bash", arguments: [:]),
            ToolCallRequest(id: "c2", name: "Bash", arguments: [:]),
        ])
        let (_, messages) = fold([
            user("a"), calls, toolResult("c1", "one"), toolResult("c2", "two"), system("note"), user("next"),
        ], appendsToToolTurns: true)
        try #require(messages.map(\.role) == [.user, .assistant, .tool, .tool, .user])
        #expect(messages.map(\.content) == ["a", "", "one", "two\n\nnote", "next"])
    }

    @Test
    func userTurnDirectlyAfterAnOpenedTurnIsMergedIntoIt() throws {
        let next = ChatMessage(role: .user, content: "question", images: [image])
        let (_, messages) = fold([
            user("a"), assistant("b"), system("note 1"), system("note 2"), next, user("separate"),
        ])
        try #require(messages.map(\.role) == [.user, .assistant, .user, .user])
        #expect(messages[2].content == "note 1\n\nnote 2\n\nquestion")
        // The merged turn is the user turn: its id and images.
        #expect(messages[2].id == next.id)
        #expect(messages[2].images == next.images)
        // Only the turn directly after merges.
        #expect(messages[3].content == "separate")
    }

    @Test
    func systemTurnAfterALeadingAssistantTurnNeverReachesThePrompt() throws {
        let (prompt, messages) = fold([
            system("lead"), assistant("b"), system("late"),
        ])
        #expect(prompt == "lead")
        try #require(messages.map(\.role) == [.assistant, .user])
        #expect(messages.map(\.content) == ["b", "late"])
    }

    @Test
    func systemOnlyInputIsAllPrompt() {
        let (prompt, messages) = fold([system("one"), system("two")])
        #expect(prompt == "one\n\ntwo")
        #expect(messages.isEmpty)
    }

    @Test
    func imagesOnSystemTurnsBecomePlaceholders() {
        let (prompt, messages) = fold([
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
        let (prompt, messages) = fold([
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
        appendsToToolTurns: Bool = false,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let short = fold(history, appendsToToolTurns: appendsToToolTurns)
        let long = fold(history + more, appendsToToolTurns: appendsToToolTurns)
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
    func claudeCodeTurnsArePrefixStable() throws {
        let turn1 = [user("block 1\nblock 2\nblock 3"), system("<system-reminder>CLAUDE.md…</system-reminder>")]
        let turn2Additions = [toolCall("toolu_1"), toolResult("toolu_1", "hi"), system("<total_tokens>9</total_tokens>")]
        let turn3Additions = [assistant("Done."), user("next task"), system("<system-reminder>…</system-reminder>")]
        expectPrefixStable(turn1, extendedBy: turn2Additions, appendsToToolTurns: true)
        expectPrefixStable(turn1 + turn2Additions, extendedBy: turn3Additions, appendsToToolTurns: true)

        // The reminder stays in the first user turn on every turn, and the
        // trailing system turn joins the tool result.
        let (_, messages) = fold(turn1 + turn2Additions, appendsToToolTurns: true)
        try #require(messages.map(\.role) == [.user, .assistant, .tool])
        #expect(messages[0].content.hasSuffix("CLAUDE.md…</system-reminder>"))
        #expect(messages[2].content == "hi\n\n<total_tokens>9</total_tokens>")
    }

    /// Codex: developer items lead (the system prompt), and a later one arrives
    /// after a tool output, before the next user message.
    @Test
    func codexDeveloperItemsArePrefixStable() throws {
        let turn1 = [system("dev one"), system("dev two"), user("Run echo")]
        let turn2Additions = [toolCall("call_1"), toolResult("call_1", "hi\n"), system("permissions changed")]
        let turn3Additions = [user("now list files"), toolCall("call_2"), toolResult("call_2", "a b")]
        expectPrefixStable(turn1, extendedBy: turn2Additions)
        expectPrefixStable(turn1 + turn2Additions, extendedBy: turn3Additions)
        expectPrefixStable(turn1 + turn2Additions + turn3Additions, extendedBy: [system("note"), user("more")])

        let (prompt, messages) = fold(turn1 + turn2Additions + turn3Additions)
        #expect(prompt == "dev one\n\ndev two")
        try #require(messages.map(\.role) == [.user, .assistant, .tool, .user, .assistant, .tool])
        #expect(messages[2].content == "hi\n")
        #expect(messages[3].content == "permissions changed\n\nnow list files")
    }

    /// A user turn merged into the turn a system turn opened brings its
    /// images. The text only grows by appending, but a VLM's template may
    /// render the images ahead of the text, so the rendered tokens of that
    /// last message can shift. Only the last message: every earlier one is
    /// unchanged, which `expectPrefixStable` still checks.
    @Test
    func mergedUserTurnWithImagesOnlyChangesTheLastMessage() throws {
        let history = [user("a"), assistant("b"), system("note")]
        let next = ChatMessage(role: .user, content: "look", images: [image])
        expectPrefixStable(history, extendedBy: [next])

        let (_, messages) = fold(history + [next])
        try #require(messages.map(\.role) == [.user, .assistant, .user])
        #expect(messages[2].content == "note\n\nlook")
        #expect(messages[2].images == [image])
    }

    /// Every split point of one mixed conversation keeps the property.
    @Test
    func everySplitOfAMixedConversationIsPrefixStable() {
        let conversation = [
            system("lead"), user("a"), system("s1"), toolCall("c1"), toolResult("c1", "r1"),
            system("s2"), system("s3"), user("b"), user("c"), assistant("d"), system("s4"),
            toolCall("c2"), toolResult("c2", "r2"), system("s6"), user("e"), system("s5"),
            toolCall("c3"), toolResult("c3", "r3"), system("<total_tokens>1</total_tokens>"),
        ]
        // From the first item on: an empty history has no system prompt yet.
        // Both placements of a system turn after a tool turn.
        for appendsToToolTurns in [false, true] {
            for split in 1...conversation.count {
                expectPrefixStable(
                    Array(conversation.prefix(split)), extendedBy: Array(conversation.dropFirst(split)),
                    appendsToToolTurns: appendsToToolTurns)
            }
        }
    }

    /// Folding touches system and user turns only: an assistant turn's
    /// reasoning stays on that turn, wherever system turns fall around it.
    @Test
    func assistantReasoningStaysOnItsTurn() throws {
        let call = ChatMessage(
            role: .assistant, content: "", toolCalls: [ToolCallRequest(id: "c1", name: "Bash", arguments: [:])],
            reasoning: "run it")
        let answer = ChatMessage(role: .assistant, content: "done", reasoning: "it worked")
        let (_, messages) = fold([
            system("lead"), user("a"), system("s1"), call, system("s2"), toolResult("c1", "r"),
            system("s3"), answer, system("s4"),
        ], appendsToToolTurns: true)
        try #require(messages.map(\.role) == [.user, .assistant, .user, .tool, .assistant, .user])
        #expect(messages.map(\.reasoning) == [nil, "run it", nil, nil, "it worked", nil])
        #expect(messages[1].id == call.id)
        #expect(messages[4].id == answer.id)
    }
}
