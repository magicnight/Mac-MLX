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
    @State private var showInspector: Bool = false

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
        // Parameters Inspector (#15) — collapsible right pane, toggled
        // via the slider-icon button in chatToolbar.
        .inspector(isPresented: $showInspector) {
            ParametersInspector()
                .inspectorColumnWidth(min: 280, ideal: 320, max: 420)
        }
        // Edit sheet (#11). Bound to viewModel.editingMessageID so any
        // right-click → Edit opens it; Save/Cancel dismiss.
        .sheet(
            isPresented: Binding(
                get: { viewModel.editingMessageID != nil },
                set: { if !$0 { viewModel.cancelEdit() } }
            )
        ) {
            EditMessageSheet(
                text: $viewModel.editingText,
                onCancel: { viewModel.cancelEdit() },
                onSave: {
                    _ = Task { @MainActor in
                        await viewModel.commitEdit()
                    }
                }
            )
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

            // Parameters Inspector toggle (#15)
            Button {
                showInspector.toggle()
            } label: {
                Label("Parameters", systemImage: "slider.horizontal.3")
                    .labelStyle(.iconOnly)
            }
            .buttonStyle(.bordered)
            .help("Parameters Inspector")
            .keyboardShortcut("i", modifiers: [.command, .option])

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
                        let messageCopy = message
                        ChatMessageView(
                            message: message,
                            onCopy: { viewModel.copyToPasteboard(messageCopy.content) },
                            onEdit: message.role == .user
                                ? { viewModel.startEdit(messageCopy) }
                                : nil,
                            onRegenerate: message.role == .assistant
                                ? {
                                    // Discard the returned Task so the
                                    // closure resolves as () -> Void.
                                    _ = Task { @MainActor in
                                        await viewModel.regenerate(from: messageCopy.id)
                                    }
                                }
                                : nil,
                            onDelete: { viewModel.delete(messageCopy.id) }
                        )
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
