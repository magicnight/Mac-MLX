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
    /// True when a speech-to-text model is installed and unambiguously
    /// resolvable. Drives the waveform button's enabled state — the button is
    /// always visible so the feature is discoverable, and its tooltip explains
    /// what is missing rather than the control silently not being there.
    let canTranscribeAudio: Bool
    /// A transcription is running: the waveform button becomes a stop control.
    let isTranscribingAudio: Bool
    /// Why the last transcription produced no text, or `nil`. Shown inline
    /// above the composer so a failure is never swallowed.
    let transcriptionError: String?
    let onSend: () -> Void
    let onStop: () -> Void
    /// User picked an audio file to transcribe.
    let onPickAudio: (URL) -> Void
    /// User asked to abandon the running transcription.
    let onCancelTranscription: () -> Void
    /// User dismissed the transcription error.
    let onDismissTranscriptionError: () -> Void

    @State private var isFileImporterPresented = false
    @State private var isAudioImporterPresented = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if !attachedImages.isEmpty {
                thumbnailStrip
            }
            if isTranscribingAudio {
                transcribingStatusRow
            } else if let transcriptionError {
                transcriptionErrorRow(transcriptionError)
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
        .fileImporter(
            isPresented: $isAudioImporterPresented,
            allowedContentTypes: [.audio, .mp3, .wav, .mpeg4Audio, .aiff],
            allowsMultipleSelection: false
        ) { result in
            // One file at a time: the result replaces the composer's contents
            // conceptually, and batching several transcripts into one message
            // is not an ask anyone has made.
            if case .success(let urls) = result, let url = urls.first {
                onPickAudio(url)
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

    /// Live status while the model works. Transcription has no token stream to
    /// watch, so without this the composer would sit inert for however long the
    /// file takes and read as a button that did nothing.
    private var transcribingStatusRow: some View {
        HStack(spacing: 6) {
            ProgressView()
                .controlSize(.small)
            Text("Transcribing…")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer(minLength: 4)
            Button("Cancel", action: onCancelTranscription)
                .buttonStyle(.plain)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 2)
    }

    /// Inline failure notice for the last transcription. Dismissible, and
    /// placed above the composer so it reads as belonging to the input rather
    /// than to the conversation.
    private func transcriptionErrorRow(_ message: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.caption2)
                .foregroundStyle(.orange)
            Text(message)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 4)
            Button(action: onDismissTranscriptionError) {
                Image(systemName: "xmark")
                    .font(.caption2)
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .help("Dismiss")
        }
        .padding(.horizontal, 2)
    }

    /// Audio-transcription button. Flips to a stop control while a
    /// transcription runs, so the action is always cancellable — the model can
    /// take a while on a long file and there must be a way out that is not
    /// "quit the app".
    private var transcribeButton: some View {
        Button {
            if isTranscribingAudio {
                onCancelTranscription()
            } else {
                isAudioImporterPresented = true
            }
        } label: {
            if isTranscribingAudio {
                Image(systemName: "stop.circle")
                    .foregroundStyle(.orange)
                    .frame(width: 28, height: 28)
            } else {
                Image(systemName: "waveform")
                    .foregroundStyle(canTranscribeAudio ? .secondary : Color.secondary.opacity(0.4))
                    .frame(width: 28, height: 28)
            }
        }
        .buttonStyle(.plain)
        .disabled(!isTranscribingAudio && !canTranscribeAudio)
        .help(
            isTranscribingAudio
            ? "Stop transcribing"
            : (canTranscribeAudio
               ? "Transcribe an audio file into the message box"
               : "Install a speech-to-text model (Whisper, Parakeet, …) to transcribe audio")
        )
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

            // Audio transcription. Independent of `isModelLoaded` on purpose:
            // it runs on `AudioEngine`, not the chat model, so it works before
            // a chat model is loaded — transcribing a file and then picking a
            // model to ask about it is a reasonable order to work in.
            transcribeButton

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
            canTranscribeAudio: true,
            isTranscribingAudio: false,
            transcriptionError: nil,
            onSend: {},
            onStop: {},
            onPickAudio: { _ in },
            onCancelTranscription: {},
            onDismissTranscriptionError: {}
        )
        ChatInputView(
            text: .constant(""),
            attachedImages: .constant([]),
            isGenerating: true,
            isModelLoaded: true,
            canAttachImages: false,
            canTranscribeAudio: false,
            isTranscribingAudio: false,
            transcriptionError: nil,
            onSend: {},
            onStop: {},
            onPickAudio: { _ in },
            onCancelTranscription: {},
            onDismissTranscriptionError: {}
        )
        // Transcribing, and the failure state.
        ChatInputView(
            text: .constant(""),
            attachedImages: .constant([]),
            isGenerating: false,
            isModelLoaded: true,
            canAttachImages: false,
            canTranscribeAudio: true,
            isTranscribingAudio: true,
            transcriptionError: nil,
            onSend: {},
            onStop: {},
            onPickAudio: { _ in },
            onCancelTranscription: {},
            onDismissTranscriptionError: {}
        )
        ChatInputView(
            text: .constant(""),
            attachedImages: .constant([]),
            isGenerating: false,
            isModelLoaded: true,
            canAttachImages: false,
            canTranscribeAudio: true,
            isTranscribingAudio: false,
            transcriptionError: AudioTranscriptionViewModel.noSpeechMessage,
            onSend: {},
            onStop: {},
            onPickAudio: { _ in },
            onCancelTranscription: {},
            onDismissTranscriptionError: {}
        )
    }
    .frame(width: 500)
}
