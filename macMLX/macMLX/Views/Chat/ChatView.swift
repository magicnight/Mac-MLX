// ChatView.swift
// macMLX
//
// Streaming chat interface. Replaces the Stage 4 Task 6 stub.

import SwiftUI
import MacMLXCore

struct ChatView: View {

    @Environment(AppState.self) private var appState

    // Fix #1: the view model is owned by AppState (not by this view's @State),
    // so switching to another sidebar tab no longer tears down the VM and its
    // streaming Task. Just read the shared instance here.
    var body: some View {
        ChatContent(viewModel: appState.chat)
    }
}

// MARK: - ChatContent

private struct ChatContent: View {

    @Bindable var viewModel: ChatViewModel
    @Environment(AppState.self) private var appState
    @State private var scrollProxy: ScrollViewProxy? = nil

    private var isModelLoaded: Bool {
        appState.coordinator.status.isLoaded
    }

    var body: some View {
        VStack(spacing: 0) {
            // Toolbar
            chatToolbar
            Divider()

            // No model loaded banner
            if !isModelLoaded {
                noModelBanner
            }

            // Message list
            messageList

            // Input area
            ChatInputView(
                text: $viewModel.inputText,
                isGenerating: viewModel.isGenerating,
                isModelLoaded: isModelLoaded,
                onSend: {
                    Task { await viewModel.send() }
                },
                onStop: {
                    viewModel.stopGeneration()
                }
            )
        }
        .navigationTitle("Chat")
        .onChange(of: viewModel.messages.count) { _, _ in
            scrollToBottom()
        }
        .onChange(of: viewModel.messages.last?.content) { _, _ in
            scrollToBottom()
        }
    }

    // MARK: - Toolbar

    private var chatToolbar: some View {
        HStack {
            // Model selector
            Picker(selection: .constant(appState.coordinator.currentModel?.id ?? "")) {
                if let model = appState.coordinator.currentModel {
                    Text(model.displayName).tag(model.id)
                } else {
                    Text("No model loaded").tag("")
                }
            } label: {
                Label("Model", systemImage: "cpu")
            }
            .pickerStyle(.menu)
            .frame(maxWidth: 220)

            Spacer()

            // Token counter
            if appState.coordinator.tokensGeneratedTotal > 0 {
                Text("\(appState.coordinator.tokensGeneratedTotal.formatted()) tokens")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            // New Chat button
            Button {
                viewModel.clearHistory()
            } label: {
                Label("New Chat", systemImage: "plus.bubble")
            }
            .buttonStyle(.bordered)
            .keyboardShortcut("n", modifiers: .command)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    // MARK: - No model banner

    private var noModelBanner: some View {
        HStack {
            Image(systemName: "exclamationmark.circle")
                .foregroundStyle(.orange)
            Text("No model loaded. Go to Models to load one.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 6)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.orange.opacity(0.08))
    }

    // MARK: - Message list

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    // System prompt display
                    if !viewModel.systemPrompt.isEmpty {
                        systemPromptRow
                    }

                    ForEach(viewModel.messages) { message in
                        ChatMessageView(message: message)
                            .id(message.id)
                    }

                    // Anchor for scrolling
                    Color.clear
                        .frame(height: 1)
                        .id("bottom")
                }
                .padding(.vertical, 8)
            }
            .onAppear { scrollProxy = proxy }
        }
    }

    private var systemPromptRow: some View {
        HStack {
            Image(systemName: "text.bubble")
                .font(.caption)
                .foregroundStyle(.secondary)
            Text("System: \(viewModel.systemPrompt)")
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.tail)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 6)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.tertiarySystemFill))
    }

    // MARK: - Scroll helper

    private func scrollToBottom() {
        withAnimation(.easeOut(duration: 0.15)) {
            scrollProxy?.scrollTo("bottom", anchor: .bottom)
        }
    }
}

#Preview {
    ChatView()
        .environment(AppState())
        .frame(width: 600, height: 500)
}
