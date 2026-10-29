import Foundation
import Testing

@testable import MacMLXCore

// MARK: - MidConversationSystemFolding Tests
//
// Pure, MLX-free: where system turns end up once a conversation is laid out
// for a template that accepts a system turn only first.

@Suite("MidConversationSystemFolding")
struct MidConversationSystemFoldingTests {

    private func system(_ text: String) -> ChatMessage { ChatMessage(role: .system, content: text) }
    private func user(_ text: String) -> ChatMessage { ChatMessage(role: .user, content: text) }
    private func assistant(_ text: String) -> ChatMessage { ChatMessage(role: .assistant, content: text) }

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
    func midConversationSystemTurnIsPrependedToTheNextUserTurn() {
        let next = ChatMessage(role: .user, content: "question", images: [
            ImageAttachment(fileURL: URL(filePath: "/tmp/x.png"), mimeType: "image/png"),
        ])
        let (prompt, messages) = MidConversationSystemFolding.fold([
            system("lead"), user("a"), assistant("b"), system("note 1"), system("note 2"), next,
        ])
        #expect(prompt == "lead")
        #expect(messages.map(\.role) == [.user, .assistant, .user])
        #expect(messages[2].content == "note 1\n\nnote 2\n\nquestion")
        // The folded turn is the same turn: id and images survive.
        #expect(messages[2].id == next.id)
        #expect(messages[2].images == next.images)
    }

    @Test
    func trailingSystemTurnIsAppendedToTheLastUserTurn() {
        let (prompt, messages) = MidConversationSystemFolding.fold([
            user("a"), assistant("b"), system("late"),
        ])
        #expect(prompt == nil)
        #expect(messages.map(\.content) == ["a\n\nlate", "b"])
    }

    @Test
    func systemTurnWithNoUserTurnAnywhereGoesToThePrompt() {
        let (prompt, messages) = MidConversationSystemFolding.fold([
            system("lead"), assistant("b"), system("late"),
        ])
        #expect(prompt == "lead\n\nlate")
        #expect(messages.map(\.role) == [.assistant])
    }

    @Test
    func imagesOnSystemTurnsBecomePlaceholders() {
        let image = ImageAttachment(fileURL: URL(filePath: "/tmp/x.png"), mimeType: "image/png")
        let (prompt, messages) = MidConversationSystemFolding.fold([
            ChatMessage(role: .system, content: "lead", images: [image]),
            user("a"),
            ChatMessage(role: .system, content: "", images: [image, image]),
            user("b"),
        ])
        #expect(prompt == "lead\n[image omitted from a system message]")
        #expect(messages[1].content == "[image omitted from a system message]\n"
            + "[image omitted from a system message]\n\nb")
        #expect(messages.allSatisfy { $0.images.isEmpty })
    }

    @Test
    func emptyTextsAddNoStrayBlankLines() {
        let (prompt, messages) = MidConversationSystemFolding.fold([
            system(""), user("a"), system("note"), user(""),
        ])
        #expect(prompt == nil)
        #expect(messages.map(\.content) == ["a", "note"])
    }
}
