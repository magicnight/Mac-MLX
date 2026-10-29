// Copyright © 2026 macMLX. English comments only.

/// Places a conversation's system turns where every chat template accepts them.
///
/// Agent clients interleave instructions with the conversation: the Responses
/// API carries `system` / `developer` items anywhere in `input`, and Claude
/// Code sends `role:"system"` turns between (and after) its turns. Many chat
/// templates (Gemma's among them) reject a system turn anywhere but first, and
/// macMLX renders the system prompt once, ahead of the messages
/// (`GenerateRequest.systemPrompt`). So:
///
///  - the system turns before the first other turn are joined, in order, with
///    a blank line into the system prompt;
///  - a later system turn whose previous message is a user turn is appended to
///    that turn's text, after a blank line;
///  - any other later system turn (after an assistant or tool turn) opens a
///    new user turn at its own position carrying its text. Further system
///    turns append to it like any user turn, and a user turn directly after it
///    is merged into it: its text follows the system text, and the merged
///    turn keeps that user turn's id and images.
///
/// The rule is position-stable, because the prompt cache depends on it:
/// clients replay the whole conversation every turn, and macMLX reuses the
/// longest common token prefix between consecutive prompts. A system turn
/// only ever changes the message just before it (or the one it opens), never
/// an earlier one, and never the system prompt once the conversation has
/// begun — so the rendering of a history is a prefix of the rendering of that
/// history extended, except that its last message may grow by appended text.
/// (Folding into the NEXT user turn would move a trailing reminder from one
/// turn to another as the conversation grows and re-prefill everything after
/// the system prompt every turn.)
///
/// A system turn's images cannot go where its text goes — the system prompt
/// and a folded user turn's text are text — so each becomes the placeholder
/// ``omittedImage`` at the end of the turn's text. A system turn with no text
/// and no images is dropped.
///
/// Pure: the caller decides which wire roles count as system and builds
/// `.system` messages for them.
enum MidConversationSystemFolding {

    /// The separator between joined instruction texts.
    static let separator = "\n\n"

    /// What an image on a system turn becomes.
    static let omittedImage = "[image omitted from a system message]"

    /// Splits `messages` into a system prompt and system-free messages.
    ///
    /// - Returns: `systemPrompt` is nil when no text ends up there; `messages`
    ///   holds every non-system message in order, plus the user turns opened
    ///   by system turns. A user turn that gains folded text keeps its id,
    ///   images and tool fields.
    static func fold(_ messages: [ChatMessage]) -> (systemPrompt: String?, messages: [ChatMessage]) {
        var systemParts: [String] = []
        var out: [ChatMessage] = []
        // Whether `out.last` is a user turn a system turn opened, which a user
        // turn directly after it merges into.
        var lastWasOpenedBySystem = false

        for message in messages {
            if message.role == .system {
                let text = systemText(message)
                guard !text.isEmpty else { continue }
                if out.isEmpty {
                    systemParts.append(text)
                } else if let last = out.last, last.role == .user {
                    out[out.count - 1] = replacingContent(of: last, with: join([last.content, text]))
                } else {
                    out.append(ChatMessage(role: .user, content: text))
                    lastWasOpenedBySystem = true
                }
                continue
            }
            if message.role == .user, lastWasOpenedBySystem, let opened = out.last {
                out[out.count - 1] = replacingContent(of: message, with: join([opened.content, message.content]))
            } else {
                out.append(message)
            }
            lastWasOpenedBySystem = false
        }

        let prompt = join(systemParts)
        return (prompt.isEmpty ? nil : prompt, out)
    }

    /// A system turn's text, one placeholder line per image after it.
    private static func systemText(_ message: ChatMessage) -> String {
        ([message.content] + message.images.map { _ in omittedImage })
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
    }

    /// Joins texts with the separator, skipping empty ones so an empty turn
    /// adds no stray blank line.
    private static func join(_ parts: [String]) -> String {
        parts.filter { !$0.isEmpty }.joined(separator: separator)
    }

    private static func replacingContent(of message: ChatMessage, with content: String) -> ChatMessage {
        ChatMessage(
            id: message.id,
            role: message.role,
            content: content,
            images: message.images,
            toolCallID: message.toolCallID,
            toolCalls: message.toolCalls
        )
    }
}
