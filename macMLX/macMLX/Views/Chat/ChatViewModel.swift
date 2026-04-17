// ChatViewModel.swift
// macMLX
//
// @Observable view model for ChatView. Manages message history (now
// persisted to disk via ConversationStore — #9) and drives streaming
// generation via EngineCoordinator. Lives on AppState so it survives
// sidebar tab switches (see #1).

import Foundation
import MacMLXCore

/// A chat message as tracked in the UI (different from MacMLXCore's
/// ChatMessage / StoredMessage — adds `isGenerating` and `tokenCount`
/// for live streaming display).
struct UIChatMessage: Identifiable {
    let id: UUID
    let role: MessageRole
    var content: String
    let timestamp: Date
    var tokenCount: Int?
    var isGenerating: Bool

    init(
        id: UUID = UUID(),
        role: MessageRole,
        content: String,
        timestamp: Date = Date(),
        tokenCount: Int? = nil,
        isGenerating: Bool = false
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.timestamp = timestamp
        self.tokenCount = tokenCount
        self.isGenerating = isGenerating
    }

    /// Restore from persistence.
    init(stored: StoredMessage) {
        self.id = stored.id
        self.role = stored.role
        self.content = stored.content
        self.timestamp = stored.timestamp
        self.tokenCount = stored.tokenCount
        self.isGenerating = false
    }

    /// Dehydrate for persistence. Transient `isGenerating` is dropped.
    var asStored: StoredMessage {
        StoredMessage(
            id: id,
            role: role,
            content: content,
            timestamp: timestamp,
            tokenCount: tokenCount
        )
    }
}

@Observable
@MainActor
final class ChatViewModel {

    // MARK: - State

    var messages: [UIChatMessage] = []
    var inputText: String = ""
    var isGenerating: Bool = false
    var systemPrompt: String = "You are a helpful assistant."

    // MARK: - Private

    private let coordinator: EngineCoordinator
    private let store: ConversationStore
    /// Persistent header for the active conversation. Refreshed whenever
    /// we save (updatedAt bump) or start a new chat (fresh UUID).
    private var current: Conversation
    private var generationTask: Task<Void, Never>? = nil

    // MARK: - Init

    init(coordinator: EngineCoordinator, store: ConversationStore) {
        self.coordinator = coordinator
        self.store = store
        // Start with a fresh empty conversation. persist() reads the
        // VM's `systemPrompt` each time it runs, so we don't need to
        // bake it into the initial Conversation here (doing so would
        // require reading `self.systemPrompt` before other stored
        // properties are fully initialised — Swift 6 rejects that).
        self.current = Conversation()

        // Load the latest persisted conversation in the background.
        Task { [weak self] in
            guard let self else { return }
            if let loaded = try? await store.loadLatest() {
                await MainActor.run {
                    self.adopt(loaded)
                }
            }
        }
    }

    // MARK: - Send

    func send() async {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty,
              let currentModel = coordinator.currentModel else { return }

        inputText = ""
        isGenerating = true

        // Append user message
        messages.append(UIChatMessage(role: .user, content: text))

        // Build request from current conversation history
        let coreMessages: [ChatMessage] = messages
            .filter { !$0.isGenerating }
            .map { ChatMessage(role: $0.role, content: $0.content) }

        let request = GenerateRequest(
            model: currentModel.id,
            messages: coreMessages,
            systemPrompt: systemPrompt.isEmpty ? nil : systemPrompt,
            parameters: GenerationParameters()
        )

        // Append placeholder assistant message
        let assistantMsg = UIChatMessage(role: .assistant, content: "", isGenerating: true)
        messages.append(assistantMsg)
        let assistantIdx = messages.count - 1

        // Stream tokens
        generationTask = Task { [weak self] in
            guard let self else { return }
            let stream = self.coordinator.generate(request)
            do {
                for try await chunk in stream {
                    guard !Task.isCancelled else { break }
                    self.messages[assistantIdx].content += chunk.text
                    if let usage = chunk.usage {
                        self.messages[assistantIdx].tokenCount = usage.completionTokens
                    }
                }
            } catch {
                self.messages[assistantIdx].content += "\n[Error: \(error.localizedDescription)]"
            }
            self.messages[assistantIdx].isGenerating = false
            self.isGenerating = false
            self.persist()
        }
        await generationTask?.value
    }

    // MARK: - Stop

    func stopGeneration() {
        generationTask?.cancel()
        generationTask = nil
        for idx in messages.indices where messages[idx].isGenerating {
            messages[idx].isGenerating = false
        }
        isGenerating = false
        // Persist whatever we got before cancel — not a wasted exchange.
        persist()
    }

    // MARK: - New chat

    /// Clears the current in-memory state and allocates a fresh
    /// conversation ID. The previous conversation stays on disk (user
    /// can scroll their history in the future sidebar — #9 follow-up).
    func clearHistory() {
        stopGeneration()
        messages = []
        inputText = ""
        current = Conversation(systemPrompt: systemPrompt)
        // Don't persist the empty conversation — only written on first
        // real message.
    }

    // MARK: - Private helpers

    /// Replace in-memory state with a loaded-from-disk conversation.
    private func adopt(_ conversation: Conversation) {
        // If the user has already started typing in the current session
        // (non-empty messages), don't clobber their work.
        guard messages.isEmpty else { return }
        current = conversation
        messages = conversation.messages.map(UIChatMessage.init(stored:))
        if !conversation.systemPrompt.isEmpty {
            systemPrompt = conversation.systemPrompt
        }
    }

    /// Serialise the current conversation to disk. Fire-and-forget — we
    /// don't block the UI on I/O and we tolerate transient failures.
    private func persist() {
        var conv = current
        conv.messages = messages
            .filter { !$0.isGenerating }
            .map(\.asStored)
        conv.systemPrompt = systemPrompt
        conv.modelID = coordinator.currentModel?.id
        current = conv

        Task { [store] in
            try? await store.save(conv)
        }
    }
}
