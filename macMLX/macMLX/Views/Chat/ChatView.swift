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
    /// Conversation-history sidebar (v0.3.2). Default **collapsed** so
    /// existing users' first impression of the tab is unchanged; user
    /// opens it via the toolbar button (⌘⌃S) when they want to browse.
    @State private var showConversationSidebar: Bool = false
    /// Local models available for the toolbar model switcher (#5).
    /// Refreshed on appear + when the model directory changes.
    @State private var availableModels: [LocalModel] = []
    /// Model currently being loaded via the toolbar switcher — keeps the
    /// menu greyed while load is in flight so the user can't stack swaps.
    @State private var switchingToModelID: String? = nil

    private var isModelLoaded: Bool {
        appState.coordinator.status.isLoaded
    }

    var body: some View {
        HStack(spacing: 0) {
            if showConversationSidebar {
                ConversationSidebar(viewModel: viewModel)
                    .frame(minWidth: 200, idealWidth: 240, maxWidth: 320)
                    .transition(.move(edge: .leading).combined(with: .opacity))
                Divider()
            }
            mainColumn
        }
        .animation(.easeInOut(duration: 0.18), value: showConversationSidebar)
    }

    private var mainColumn: some View {
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
        .task {
            await reloadAvailableModels()
        }
        .onChange(of: appState.currentSettings.modelDirectory) { _, _ in
            Task { await reloadAvailableModels() }
        }
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

    // MARK: - Model switcher (#5)

    private var modelSwitcher: some View {
        Menu {
            if availableModels.isEmpty {
                Text("No local models")
            }
            ForEach(availableModels) { model in
                Button {
                    switchToModel(model)
                } label: {
                    if model.id == appState.coordinator.currentModel?.id {
                        Label(model.displayName, systemImage: "checkmark")
                    } else {
                        Text(model.displayName)
                    }
                }
            }
            Divider()
            Button("Refresh") {
                Task { await reloadAvailableModels() }
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "cpu")
                Text(currentModelLabel)
                    .lineLimit(1)
                    .truncationMode(.middle)
                if switchingToModelID != nil {
                    ProgressView()
                        .controlSize(.small)
                }
            }
        }
        .menuStyle(.borderlessButton)
        .frame(maxWidth: 260)
        // Block swapping mid-generation — coordinator state would get
        // weird if we unload while a token stream is in flight.
        .disabled(viewModel.isGenerating || switchingToModelID != nil)
    }

    private var currentModelLabel: String {
        if let id = switchingToModelID {
            return "Loading \(id)…"
        }
        if let current = appState.coordinator.currentModel {
            return current.displayName
        }
        return availableModels.isEmpty ? "No local models" : "Pick a model…"
    }

    private func switchToModel(_ model: LocalModel) {
        if model.id == appState.coordinator.currentModel?.id { return }
        switchingToModelID = model.id
        Task {
            _ = await appState.coordinator.load(model)
            switchingToModelID = nil
            // Start a fresh conversation so the new model doesn't inherit
            // tokens produced by the previous model's tokenizer/template.
            viewModel.createNew()
        }
    }

    private func reloadAvailableModels() async {
        let dir = appState.currentSettings.modelDirectory
        availableModels = (try? await appState.library.scan(dir)) ?? []
    }

    // MARK: - Toolbar

    private var chatToolbar: some View {
        HStack {
            // Conversation sidebar toggle (v0.3.2). Collapsed by default;
            // user opens when they want to browse / switch conversations.
            Button {
                showConversationSidebar.toggle()
            } label: {
                Label("Conversations", systemImage: "sidebar.leading")
                    .labelStyle(.iconOnly)
            }
            .buttonStyle(.bordered)
            .help("Show/hide conversation history (⌘⌃S)")
            .keyboardShortcut("s", modifiers: [.command, .control])

            // Model switcher (#5) — pre-v0.3.1 this was a `.constant`
            // Picker that only displayed the loaded model with no way to
            // switch. Now a real Menu: lists local models, checkmarks the
            // one loaded, loads on tap.
            modelSwitcher

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
                            onDelete: { viewModel.delete(messageCopy.id) },
                            onTruncate: { viewModel.truncateAfter(messageCopy.id) }
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
