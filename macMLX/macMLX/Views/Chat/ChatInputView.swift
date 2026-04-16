// ChatInputView.swift
// macMLX

import SwiftUI

struct ChatInputView: View {

    @Binding var text: String
    let isGenerating: Bool
    let isModelLoaded: Bool
    let onSend: () -> Void
    let onStop: () -> Void

    // Approximate max height for 5 lines of text
    private let maxHeight: CGFloat = 110

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            // Multi-line input
            TextEditor(text: $text)
                .font(.body)
                .frame(minHeight: 36, maxHeight: maxHeight)
                .fixedSize(horizontal: false, vertical: true)
                .scrollContentBackground(.hidden)
                .background(.clear)
                .disabled(isGenerating || !isModelLoaded)
                .overlay(alignment: .topLeading) {
                    if text.isEmpty {
                        Text(isModelLoaded ? "Message…" : "Load a model first")
                            .foregroundStyle(.tertiary)
                            .font(.body)
                            .padding(.top, 8)
                            .padding(.leading, 4)
                            .allowsHitTesting(false)
                    }
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
