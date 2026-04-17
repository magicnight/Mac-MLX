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

    /// Assistant messages are rendered as Markdown — LLM outputs are
    /// naturally markdown-heavy (code, lists, headers). User messages
    /// stay plain: users rarely type intentional markdown and literal
    /// asterisks shouldn't collapse to bold.
    ///
    /// `AttributedString.MarkdownParsingOptions.failurePolicy =
    /// .returnPartiallyParsedIfPossible` is key during streaming — when a
    /// chunk arrives mid-`**bold**`, the partial `**bold` chunk parses as
    /// literal characters and won't throw, so rendering stays smooth.
    @ViewBuilder
    private var renderedContent: some View {
        let text = messageContent
        if message.role == .assistant,
           let attributed = try? AttributedString(
               markdown: text,
               options: AttributedString.MarkdownParsingOptions(
                   allowsExtendedAttributes: false,
                   interpretedSyntax: .full,
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
