// Copyright © 2026 macMLX. English comments only.

/// Places a conversation's system turns where every chat template accepts them.
///
/// Agent clients interleave instructions with the conversation: the Responses
/// API carries `system` / `developer` items anywhere in `input`, and Claude
/// Code injects reminders between turns. Many chat templates (Gemma's among
/// them) reject a system turn anywhere but first, and macMLX renders the system
/// prompt once, ahead of the messages (`GenerateRequest.systemPrompt`). So:
///
///  - the system turns before the first other turn are joined, in order, with
///    a blank line into the system prompt;
///  - a later system turn is folded into the text of the next user turn
///    (prepended, separated by a blank line); several in a row fold together;
///  - one with no user turn after it is appended to the last user turn
///    instead, and with no user turn at all, to the system prompt — the only
///    place left that every template renders.
///
/// A system turn's images cannot go where its text goes — the system prompt
/// and a folded user turn's text are text — so each becomes the placeholder
/// ``omittedImage`` at the end of the turn's text.
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
    ///   holds every non-system message in order, user turns carrying any
    ///   folded instructions. A folded user turn keeps its id, images and tool
    ///   fields.
    static func fold(_ messages: [ChatMessage]) -> (systemPrompt: String?, messages: [ChatMessage]) {
        var systemParts: [String] = []
        var out: [ChatMessage] = []
        var pending: [String] = []
        var seenNonSystem = false

        for message in messages {
            if message.role == .system {
                let text = systemText(message)
                if seenNonSystem {
                    pending.append(text)
                } else {
                    systemParts.append(text)
                }
                continue
            }
            seenNonSystem = true
            if message.role == .user, !pending.isEmpty {
                out.append(replacingContent(of: message, with: join(pending + [message.content])))
                pending.removeAll()
            } else {
                out.append(message)
            }
        }

        if !pending.isEmpty {
            if let lastUser = out.lastIndex(where: { $0.role == .user }) {
                out[lastUser] = replacingContent(of: out[lastUser], with: join([out[lastUser].content] + pending))
            } else {
                systemParts.append(contentsOf: pending)
            }
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
