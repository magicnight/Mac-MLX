// Copyright © 2026 macMLX. English comments only.

import MLXLMCommon

/// The generator a full-path VLM load renders with (``FullPathVLM``): a
/// message that carries an image or a video goes through its family's
/// generator, which lays out the media placeholders the processor expands;
/// any other message renders as the text load renders it, through
/// `DefaultMessageGenerator` (string content plus tool metadata), so a
/// text-only prompt is the same tokens on both loads.
///
/// Upstream's family generators turn every non-system message into a list
/// of content parts, media or not. Gemma 4's template keeps the model turn
/// open after a tool result only when the assistant message that made the
/// call has empty content, and a one-part list is not empty: rendered from
/// parts, a prompt ending in a tool result closed the turn with `<turn|>`
/// and opened none, so the model answered from outside any turn.
///
/// The engine hands the processor this generator's `.messages`, which the
/// processors pass through unchanged, rather than `.chat`, which they would
/// render with their family's own generator.
struct FullPathMessageGenerator: MessageGenerator {

    /// The family's generator, for messages with media.
    let media: any MessageGenerator

    /// The text load's generator, for everything else.
    let text: any MessageGenerator

    init(media: any MessageGenerator, text: any MessageGenerator = DefaultMessageGenerator()) {
        self.media = media
        self.text = text
    }

    func generate(message: Chat.Message) -> Message {
        message.images.isEmpty && message.videos.isEmpty
            ? text.generate(message: message) : media.generate(message: message)
    }

    func generate(messages: [Chat.Message]) -> [Message] {
        messages.map { generate(message: $0) }
    }
}
