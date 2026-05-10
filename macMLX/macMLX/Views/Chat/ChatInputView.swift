// ChatInputView.swift
// macMLX

import SwiftUI
import MacMLXCore
import UniformTypeIdentifiers

struct ChatInputView: View {

    @Binding var text: String
    /// VLM image attachments staged for the next user message.
    @Binding var attachedImages: [ImageAttachment]
    let isGenerating: Bool
    let isModelLoaded: Bool
    /// True when the loaded model accepts images (VLM). Drives the
    /// paperclip button's enabled state.
    let canAttachImages: Bool
    let onSend: () -> Void
    let onStop: () -> Void

    @State private var isFileImporterPresented = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if !attachedImages.isEmpty {
                thumbnailStrip
            }
            inputRow
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
        .fileImporter(
            isPresented: $isFileImporterPresented,
            allowedContentTypes: [.image, .png, .jpeg, .gif, .webP, .heic, .bmp],
            allowsMultipleSelection: true
        ) { result in
            switch result {
            case .success(let urls):
                for url in urls {
                    if let mime = ImageAttachment.mimeType(forPathExtension: url.pathExtension) {
                        attachedImages.append(ImageAttachment(fileURL: url, mimeType: mime))
                    }
                }
            case .failure:
                // Silent — fileImporter surfaces its own error UI.
                break
            }
        }
    }

    // MARK: - Subviews

    private var thumbnailStrip: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(attachedImages, id: \.fileURL) { att in
                    ZStack(alignment: .topTrailing) {
                        AsyncThumbnailImage(url: att.fileURL)
                            .frame(width: 56, height: 56)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                            .overlay(
                                RoundedRectangle(cornerRadius: 8)
                                    .stroke(Color.secondary.opacity(0.3), lineWidth: 1)
                            )

                        Button {
                            attachedImages.removeAll { $0.fileURL == att.fileURL }
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .font(.system(size: 16))
                                .foregroundStyle(.white, .black.opacity(0.7))
                        }
                        .buttonStyle(.plain)
                        .offset(x: 6, y: -6)
                    }
                }
            }
            .padding(.vertical, 2)
        }
    }

    private var inputRow: some View {
        HStack(alignment: .bottom, spacing: 8) {
            // Image picker button (paperclip). Disabled when the loaded
            // model can't take images. Tooltip explains why.
            Button {
                isFileImporterPresented = true
            } label: {
                Image(systemName: "photo.on.rectangle")
                    .foregroundStyle(canAttachImages ? .secondary : Color.secondary.opacity(0.4))
                    .frame(width: 28, height: 28)
            }
            .buttonStyle(.plain)
            .disabled(!canAttachImages || isGenerating || !isModelLoaded)
            .help(
                canAttachImages
                ? "Attach image (jpeg, png, webp, gif, heic, bmp)"
                : "Load a vision-capable model (Qwen-VL, Gemma-3, SmolVLM, …) to attach images"
            )

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
    }

    private var canSend: Bool {
        let hasContent = !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || !attachedImages.isEmpty
        return hasContent && isModelLoaded && !isGenerating
    }
}

/// Tiny disk-image thumbnail loader. Uses NSImage on the main actor —
/// images are small (≤120pt) so synchronous decode is fine. Gracefully
/// degrades to a placeholder glyph if the file can't be read.
struct AsyncThumbnailImage: View {
    let url: URL

    var body: some View {
        if let nsImage = NSImage(contentsOf: url) {
            Image(nsImage: nsImage)
                .resizable()
                .aspectRatio(contentMode: .fill)
        } else {
            Image(systemName: "photo")
                .imageScale(.large)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color.secondary.opacity(0.1))
        }
    }
}

#Preview {
    VStack {
        ChatInputView(
            text: .constant("Hello!"),
            attachedImages: .constant([]),
            isGenerating: false,
            isModelLoaded: true,
            canAttachImages: true,
            onSend: {},
            onStop: {}
        )
        ChatInputView(
            text: .constant(""),
            attachedImages: .constant([]),
            isGenerating: true,
            isModelLoaded: true,
            canAttachImages: false,
            onSend: {},
            onStop: {}
        )
    }
    .frame(width: 500)
}
