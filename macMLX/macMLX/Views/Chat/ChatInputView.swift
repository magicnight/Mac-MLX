// ChatInputView.swift
// macMLX

import SwiftUI

struct ChatInputView: View {

    @Binding var text: String
    let isGenerating: Bool
    let isModelLoaded: Bool
    let onSend: () -> Void
    let onStop: () -> Void

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            // Auto-growing TextField keeps the cursor vertically centered
            // on a single line and expands to up to 5 lines. macOS 14+.
            TextField(
                isModelLoaded ? "Message…" : "Load a model first",
                text: $text,
                axis: .vertical
            )
            .textFieldStyle(.plain)
            .lineLimit(1...5)
            .font(.body)
            .disabled(isGenerating || !isModelLoaded)
            .onSubmit {
                // Cmd+Return still sends via the Send button's keyboard
                // shortcut. Plain Return inserts newline (default for
                // axis:.vertical). Shift+Return is identical — TextField
                // handles it.
                if canSend { onSend() }
            }

            // Send / Stop button
            if isGenerating {
                Button(action: onStop) {
                    Image(systemName: "stop.fill")
                        .foregroundStyle(.white)
                        .frame(width: 32, height: 32)
                        .background(Color.red, in: Circle())
                }
                .buttonStyle(.plain)
                .keyboardShortcut(.escape, modifiers: [])
            } else {
                Button(action: onSend) {
                    Image(systemName: "arrow.up")
                        .foregroundStyle(.white)
                        .frame(width: 32, height: 32)
                        .background(
                            canSend ? Color.accentColor : Color.secondary,
                            in: Circle()
                        )
                }
                .buttonStyle(.plain)
                .disabled(!canSend)
                .keyboardShortcut(.return, modifiers: .command)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 14)
                .stroke(Color.secondary.opacity(0.3), lineWidth: 1)
                .background(.background, in: RoundedRectangle(cornerRadius: 14))
        )
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    private var canSend: Bool {
        !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        && isModelLoaded
        && !isGenerating
    }
}

#Preview {
    VStack {
        ChatInputView(
            text: .constant("Hello!"),
            isGenerating: false,
            isModelLoaded: true,
            onSend: {},
            onStop: {}
        )
        ChatInputView(
            text: .constant(""),
            isGenerating: true,
            isModelLoaded: true,
            onSend: {},
            onStop: {}
        )
    }
    .frame(width: 500)
}
