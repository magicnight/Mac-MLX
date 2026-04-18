// ChatMessageView.swift
// macMLX

import SwiftUI
import MacMLXCore

struct ChatMessageView: View {

    let message: UIChatMessage
    /// Actions available from the right-click context menu (#11). All
    /// optional — callers that don't want a menu can leave them nil.
    var onCopy: (() -> Void)? = nil
    var onEdit: (() -> Void)? = nil
    var onRegenerate: (() -> Void)? = nil
    var onDelete: (() -> Void)? = nil
    /// Truncate the conversation at this message (keep this + earlier,
    /// drop everything later). Powers "Rewind to here" in v0.3.2 history
    /// management — same underlying semantics as Git's soft reset but
    /// at the per-message level.
    var onTruncate: (() -> Void)? = nil

    var body: some View {
        HStack(alignment: .bottom, spacing: 0) {
            if message.role == .user {
                Spacer(minLength: 60)
                bubble
            } else {
                bubble
                Spacer(minLength: 60)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 4)
        .contextMenu { contextMenuItems }
    }

    // MARK: - Context menu (#11)

    @ViewBuilder
    private var contextMenuItems: some View {
        if let onCopy {
            Button {
                onCopy()
            } label: {
                Label("Copy", systemImage: "doc.on.doc")
            }
        }

        if message.role == .user, let onEdit {
            Button {
                onEdit()
            } label: {
                Label("Edit…", systemImage: "pencil")
            }
        }

        if message.role == .assistant, let onRegenerate {
            Button {
                onRegenerate()
            } label: {
                Label("Regenerate", systemImage: "arrow.clockwise")
            }
        }

        if let onTruncate {
            Button {
                onTruncate()
            } label: {
                Label("Rewind to here", systemImage: "arrow.uturn.backward")
            }
            .help("Keep this message and every earlier message, drop later ones.")
        }

        if onDelete != nil && onCopy != nil {
            Divider()
        }

        if let onDelete {
            Button(role: .destructive) {
                onDelete()
            } label: {
                Label("Delete", systemImage: "trash")
            }
        }
    }

    @ViewBuilder
    private var bubble: some View {
        VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
            renderedContent
                .textSelection(.enabled)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(bubbleBackground, in: RoundedRectangle(cornerRadius: 14))
                .foregroundStyle(message.role == .user ? Color.white : Color.primary)

            // Timestamp + token count
            HStack(spacing: 6) {
                Text(message.timestamp, style: .time)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)

                if let count = message.tokenCount {
                    Text("·")
                        .foregroundStyle(.tertiary)
                        .font(.caption2)
                    Text("\(count) tok")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
        }
    }

    private var bubbleBackground: Color {
        switch message.role {
        case .user:      return .accentColor
        case .assistant: return Color(.secondarySystemFill)
        case .system:    return Color(.tertiarySystemFill)
        }
    }

    /// Append blinking cursor while generating.
    private var messageContent: String {
        message.isGenerating ? message.content + " █" : message.content
    }

    // MARK: - Rendered content (#10 Markdown)

    /// Assistant messages are rendered with inline-only Markdown: bold,
    /// italic, inline code spans, links — and crucially, paragraph breaks
    /// are preserved.
    ///
    /// Why `.inlineOnlyPreservingWhitespace` and not `.full`:
    /// - `.full` parses block-level structure (headers, lists, code
    ///   fences) but SwiftUI's `Text(AttributedString)` cannot *render*
    ///   block structure. It flattens the entire attributed string into
    ///   a single run AND consumes the `\n\n` separators that delimited
    ///   paragraphs in the source. Net effect: a well-structured LLM
    ///   reply collapses to one wall of prose. This was a v0.2 regression
    ///   from v0.1's plain `Text(content)` which naturally preserved
    ///   newlines.
    /// - `.inlineOnlyPreservingWhitespace` keeps every whitespace
    ///   character in the output (so `\n\n` still renders as a blank
    ///   line) while still lighting up inline `**bold**` / `*italic*` /
    ///   `` `code` `` / `[link](url)`. Block markers like `# Heading`
    ///   or `- item` pass through as literal text — acceptable, and
    ///   better than losing the structure entirely.
    /// - `.returnPartiallyParsedIfPossible` remains the right failure
    ///   policy for streaming: a chunk arriving mid-`**bold**` parses
    ///   cleanly as partial bold, no flicker between tokens.

    /// Renders the message with `<think>` blocks broken out into
    /// collapsible sub-views. User and system messages bypass
    /// segmentation (they never emit think tags in practice; showing
    /// one as literal text is fine if a user pastes it).
    @ViewBuilder
    private var renderedContent: some View {
        if message.role == .assistant {
            let segments = MessageSegmenter.parse(messageContent)
            VStack(alignment: .leading, spacing: 6) {
                ForEach(Array(segments.enumerated()), id: \.offset) { _, seg in
                    switch seg {
                    case .text(let s):
                        inlineMarkdown(s)
                    case .think(let s, let isClosed):
                        ThinkBlockView(
                            content: s,
                            isStreaming: !isClosed && message.isGenerating
                        )
                    }
                }
            }
        } else {
            Text(messageContent)
        }
    }

    /// Inline Markdown pass, extracted so every assistant text segment
    /// can reuse it. Same parse options as pre-v0.3.6 behaviour.
    @ViewBuilder
    private func inlineMarkdown(_ text: String) -> some View {
        if let attributed = try? AttributedString(
            markdown: text,
            options: AttributedString.MarkdownParsingOptions(
                allowsExtendedAttributes: false,
                interpretedSyntax: .inlineOnlyPreservingWhitespace,
                failurePolicy: .returnPartiallyParsedIfPossible
            )
        ) {
            Text(attributed)
        } else {
            Text(text)
        }
    }
}

#Preview {
    VStack {
        ChatMessageView(message: UIChatMessage(role: .user, content: "What is MLX?"))
        ChatMessageView(message: UIChatMessage(
            role: .assistant,
            content: """
            **MLX** is Apple's array framework for machine learning on Apple Silicon.

            ## Key features
            - Unified memory (CPU + GPU + ANE)
            - Lazy evaluation
            - Multi-device support

            Use `import MLX` to get started. Here is a tiny example:

            ```swift
            import MLX
            let x = MLXArray([1, 2, 3])
            print(x.sum())
            ```
            """
        ))
        ChatMessageView(message: UIChatMessage(
            role: .assistant,
            content: "Streaming **bol",
            isGenerating: true
        ))
    }
    .frame(width: 520)
    .padding()
}
