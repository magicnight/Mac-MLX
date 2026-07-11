// CopyIDButton.swift
// macMLX
//
// One-click "copy model ID to pasteboard" button, reused by the model
// library row and the model card (Track F #3 — small but high-frequency:
// model ids are what callers paste into `/v1/chat/completions` requests).
// Mirrors ChatView's existing toolbar copy-model-ID button: same
// animated checkmark feedback, same NSPasteboard call.

import AppKit
import SwiftUI

struct CopyIDButton: View {
    let id: String

    @State private var justCopied = false
    /// Backs the "flip back to the clipboard icon after a delay" timer.
    /// A rapid second click cancels the first click's still-pending reset
    /// so it can't fire mid-way through the second click's own delay
    /// window and hide the checkmark early.
    @State private var resetTask: Task<Void, Never>? = nil

    var body: some View {
        Button {
            let pb = NSPasteboard.general
            pb.clearContents()
            pb.setString(id, forType: .string)
            withAnimation(.easeOut(duration: 0.15)) {
                justCopied = true
            }
            resetTask?.cancel()
            resetTask = Task { @MainActor in
                try? await Task.sleep(for: .milliseconds(1200))
                guard !Task.isCancelled else { return }
                withAnimation(.easeOut(duration: 0.15)) {
                    justCopied = false
                }
            }
        } label: {
            Image(systemName: justCopied ? "checkmark" : "doc.on.doc")
        }
        .buttonStyle(.plain)
        .foregroundStyle(.secondary)
        .help(justCopied ? "Copied!" : "Copy model ID (\(id))")
    }
}

#Preview {
    CopyIDButton(id: "mlx-community/Qwen3-8B-4bit")
        .padding()
}
