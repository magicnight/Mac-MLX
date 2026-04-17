// EditMessageSheet.swift
// macMLX
//
// Sheet presented when the user picks "Edit…" from a user-message
// context menu (#11). Saving the edit truncates the conversation to
// this message and re-runs generation.

import SwiftUI

struct EditMessageSheet: View {

    @Binding var text: String
    let onCancel: () -> Void
    let onSave: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Edit message")
                .font(.headline)

            Text("Saving replaces the message and regenerates the assistant's reply from this point.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            TextEditor(text: $text)
                .font(.body)
                .frame(minHeight: 140)
                .padding(6)
                .background(.quaternary, in: RoundedRectangle(cornerRadius: 6))
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(Color(.separatorColor), lineWidth: 1)
                )

            HStack {
                Spacer()
                Button("Cancel", role: .cancel, action: onCancel)
                    .keyboardShortcut(.cancelAction)
                Button("Save", action: onSave)
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
                    .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(20)
        .frame(width: 480)
    }
}

#Preview {
    @Previewable @State var text = "What is MLX used for?"
    return EditMessageSheet(
        text: $text,
        onCancel: {},
        onSave: {}
    )
}
