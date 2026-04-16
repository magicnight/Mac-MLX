// ChatMessageView.swift
// macMLX

import SwiftUI
import MacMLXCore

struct ChatMessageView: View {

    let message: UIChatMessage

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
    }

    @ViewBuilder
    private var bubble: some View {
        VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
            Text(messageContent)
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
}

#Preview {
    VStack {
        ChatMessageView(message: UIChatMessage(role: .user, content: "What is MLX?"))
        ChatMessageView(message: UIChatMessage(role: .assistant, content: "MLX is Apple's array framework for machine learning on Apple Silicon.", isGenerating: false))
        ChatMessageView(message: UIChatMessage(role: .assistant, content: "Generating", isGenerating: true))
    }
    .frame(width: 500)
}
