// ChatViewModel.swift
// macMLX
//
// @Observable view model for ChatView. Manages message history (now
// persisted to disk via ConversationStore — #9) and drives streaming
// generation via EngineCoordinator. Lives on AppState so it survives
// sidebar tab switches (see #1).

import AppKit
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

    /// All saved conversations, newest-first. Refreshed on `reloadConversationList()`
    /// — fires after every `persist()`, `switchTo(_:)`, `deleteConversation(_:)`,
    /// and once on init. Drives the v0.3.2 conversation sidebar.
    private(set) var conversations: [Conversation] = []

    /// ID of the currently-open conversation. Nil until first save.
    /// Used by the sidebar to highlight the active row.
    var currentConversationID: UUID { current.id }

    /// Convenience passthrough to the parameters VM so existing ChatView
    /// code that reads `viewModel.systemPrompt` keeps working. The source
    /// of truth is `parameters.parameters.systemPrompt` (editable from the
    /// Parameters Inspector).
    var systemPrompt: String {
        parameters.parameters.systemPrompt
    }

    // MARK: - Private

    private let coordinator: EngineCoordinator
    private let store: ConversationStore
    /// Parameters VM — source of truth for temperature / topP / maxTokens
    /// / systemPrompt. ChatViewModel reads its `.parameters` snapshot on
    /// every send so the user's latest slider tweaks take effect on the
    /// next generation. Injected (not held as AppState ref) to avoid a
    /// retain cycle.
    private let parameters: ParametersViewModel
    /// Persistent header for the active conversation. Refreshed whenever
    /// we save (updatedAt bump) or start a new chat (fresh UUID).
    private var current: Conversation
    private var generationTask: Task<Void, Never>? = nil

    // MARK: - Init

    init(
        coordinator: EngineCoordinator,
        store: ConversationStore,
        parameters: ParametersViewModel
    ) {
        self.coordinator = coordinator
        self.store = store
        self.parameters = parameters
        // Start with a fresh empty conversation. persist() reads the
        // VM's `systemPrompt` each time it runs, so we don't need to
        // bake it into the initial Conversation here (doing so would
        // require reading `self.systemPrompt` before other stored
        // properties are fully initialised — Swift 6 rejects that).
        self.current = Conversation()

        // Load the latest persisted conversation in the background, and
        // populate the sidebar list.
        Task { [weak self] in
            guard let self else { return }
            if let loaded = try? await store.loadLatest() {
                await MainActor.run {
                    self.adopt(loaded)
                }
            }
            await self.reloadConversationList()
        }
    }

    // MARK: - Conversation management (#v0.3.2)

    /// Refresh `conversations` from disk. Fired after every save and
    /// after sidebar-driven mutations (rename / delete / switch).
    func reloadConversationList() async {
        conversations = (try? await store.list()) ?? []
    }

    /// Switch the active conversation. Saves the current state first
    /// (not destructive) and adopts the target. Cancels any in-flight
    /// generation.
    func switchTo(_ conversationID: UUID) async {
        guard conversationID != current.id else { return }
        stopGeneration()
        // Flush current to disk before loading a different one so the
        // user doesn't lose a half-typed-but-auto-saved pass.
        persistNow()

        // Find target in our cached list first; fall back to a fresh
        // `list()` in case the cache is stale.
        var target = conversations.first(where: { $0.id == conversationID })
        if target == nil {
            await reloadConversationList()
            target = conversations.first(where: { $0.id == conversationID })
        }
        guard let target else { return }

        // Clear state before adopting so `adopt(_:)` doesn't short-circuit
        // on the "don't clobber user's work" guard.
        messages = []
        inputText = ""
        adopt(target)
    }

    /// Create a fresh conversation. Current state is persisted first; the
    /// new conversation is blank until first message (no empty rows in
    /// the sidebar).
    func createNew() {
        stopGeneration()
        persistNow()
        messages = []
        inputText = ""
        current = Conversation(
            modelID: coordinator.currentModel?.id,
            systemPrompt: parameters.parameters.systemPrompt
        )
        // Reload so sidebar reflects the just-persisted previous convo.
        Task { await reloadConversationList() }
    }

    /// Rename a conversation by ID. Writes the rename through
    /// ConversationStore and refreshes the sidebar.
    func rename(_ conversationID: UUID, to newTitle: String) async {
        let trimmed = newTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        if conversationID == current.id {
            current.title = trimmed
            persistNow()
        } else if var target = conversations.first(where: { $0.id == conversationID }) {
            target.title = trimmed
            try? await store.save(target)
        }
        await reloadConversationList()
    }

    /// Delete a conversation by ID. If it's the currently-open one, we
    /// transition to the next most recent (or a fresh empty conversation
    /// if none remain).
    func deleteConversation(_ conversationID: UUID) async {
        try? await store.delete(id: conversationID)

        if conversationID == current.id {
            stopGeneration()
            messages = []
            await reloadConversationList()
            // Jump to the next most recent, or new-chat if store is empty.
            if let fallback = conversations.first {
                // Can't call `switchTo` (it'd flush the just-deleted convo
                // back to disk). Adopt directly.
                current = fallback
                messages = fallback.messages.map(UIChatMessage.init(stored:))
            } else {
                current = Conversation(
                    modelID: coordinator.currentModel?.id,
                    systemPrompt: parameters.parameters.systemPrompt
                )
            }
        } else {
            await reloadConversationList()
        }
    }

    /// Truncate the current conversation after `messageID` — the named
    /// message and every earlier one stay; everything after is dropped.
    /// Used by the "Rewind to here" context menu entry (v0.3.2 #).
    /// Does not regenerate — user is in charge of what to do next
    /// (often: edit the last user message and resend).
    func truncateAfter(_ messageID: UUID) {
        guard let idx = messages.firstIndex(where: { $0.id == messageID }) else { return }
        stopGeneration()
        if idx + 1 < messages.count {
            messages.removeSubrange((idx + 1)...)
        }
        persist()
    }

    // MARK: - Edit state (#11)

    /// ID of the message currently being edited (nil = no edit in progress).
    /// ChatContent's .sheet presentation binds against this.
    var editingMessageID: UUID? = nil
    /// Live-edited buffer for the sheet. Copied out of the message on
    /// `startEdit(_:)` and copied back on `commitEdit()`.
    var editingText: String = ""

    // MARK: - Send

    /// Submit `inputText` as a new user message and start streaming the
    /// assistant's reply.
    func send() async {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty,
              coordinator.currentModel != nil else { return }

        inputText = ""
        messages.append(UIChatMessage(role: .user, content: text))
        await generate()
    }

    // MARK: - Regenerate / Edit / Delete (#11)

    /// Re-run inference for the assistant message identified by
    /// `messageID`. Removes that message (and anything after it) then
    /// re-submits the conversation up to the last user turn.
    func regenerate(from messageID: UUID) async {
        stopGeneration()
        guard let idx = messages.firstIndex(where: { $0.id == messageID }) else { return }
        messages.removeSubrange(idx...)
        persist()
        await generate()
    }

    /// Begin editing `message` — opens the edit sheet.
    /// Only user messages are editable (assistant turns come from the
    /// engine and regenerating is the equivalent action).
    func startEdit(_ message: UIChatMessage) {
        guard message.role == .user else { return }
        editingMessageID = message.id
        editingText = message.content
    }

    /// Cancel the in-flight edit without modifying messages.
    func cancelEdit() {
        editingMessageID = nil
        editingText = ""
    }

    /// Commit the current edit. If the edited message is followed by
    /// assistant turns, those are discarded and generation re-runs
    /// against the new content — the natural "re-ask from here" flow.
    func commitEdit() async {
        guard let id = editingMessageID else { return }
        let newContent = editingText.trimmingCharacters(in: .whitespacesAndNewlines)
        editingMessageID = nil
        editingText = ""
        guard !newContent.isEmpty,
              let idx = messages.firstIndex(where: { $0.id == id }) else { return }
        stopGeneration()
        messages[idx].content = newContent

        // Drop everything after the edited message (no stale assistant
        // replies that no longer match the new question).
        if idx + 1 < messages.count {
            messages.removeSubrange((idx + 1)...)
        }
        persist()
        // Only regenerate if we're at/past a user turn (should always be
        // true here since edit only applies to user messages).
        await generate()
    }

    /// Remove a message by ID. No automatic regeneration.
    func delete(_ messageID: UUID) {
        messages.removeAll { $0.id == messageID }
        persist()
    }

    /// Copy `text` to the pasteboard. Exposed here so ChatMessageView
    /// doesn't have to import AppKit.
    func copyToPasteboard(_ text: String) {
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(text, forType: .string)
    }

    // MARK: - Private generation driver

    /// Build a GenerateRequest from the current messages + parameters,
    /// append a placeholder assistant message, stream tokens into it,
    /// and persist when done. Shared by `send()` and `regenerate(from:)`.
    private func generate() async {
        guard let currentModel = coordinator.currentModel else {
            await LogManager.shared.warning(
                "generate() aborted: no currentModel on coordinator",
                category: .inference
            )
            return
        }

        isGenerating = true

        let coreMessages: [ChatMessage] = messages
            .filter { !$0.isGenerating }
            .map { ChatMessage(role: $0.role, content: $0.content) }

        let params = parameters.parameters
        let request = GenerateRequest(
            model: currentModel.id,
            messages: coreMessages,
            systemPrompt: params.systemPrompt.isEmpty ? nil : params.systemPrompt,
            parameters: params.asGenerationParameters()
        )

        await LogManager.shared.info(
            "Starting generation for \(currentModel.id) (messages=\(coreMessages.count), maxTokens=\(params.maxTokens))",
            category: .inference
        )

        let assistantMsg = UIChatMessage(role: .assistant, content: "", isGenerating: true)
        messages.append(assistantMsg)
        let assistantIdx = messages.count - 1

        generationTask = Task { [weak self] in
            guard let self else { return }
            let stream = self.coordinator.generate(request)
            var chunkCount = 0
            var totalChars = 0
            do {
                for try await chunk in stream {
                    guard !Task.isCancelled else { break }
                    chunkCount += 1
                    totalChars += chunk.text.count
                    self.messages[assistantIdx].content += chunk.text
                    if let usage = chunk.usage {
                        self.messages[assistantIdx].tokenCount = usage.completionTokens
                    }
                }
                await LogManager.shared.info(
                    "Generation finished: chunks=\(chunkCount), chars=\(totalChars), tokens=\(self.messages[assistantIdx].tokenCount ?? 0)",
                    category: .inference
                )
            } catch {
                await LogManager.shared.error(
                    "Generation failed after \(chunkCount) chunks: \(error.localizedDescription)",
                    category: .inference
                )
                self.messages[assistantIdx].content += "\n[Error: \(error.localizedDescription)]"
            }
            self.messages[assistantIdx].isGenerating = false
            self.isGenerating = false
            self.persist()

            // Visible fallback when the stream truly produced nothing.
            // Otherwise the user sees an empty bubble with no feedback —
            // ambiguous between "pending" and "completed-with-zero-output".
            if chunkCount == 0 && self.messages[assistantIdx].content.isEmpty {
                self.messages[assistantIdx].content = "[No output — model returned zero tokens. Check Logs tab for details.]"
            }
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

    /// Compat alias for `createNew()` — kept so existing New Chat button
    /// callsites don't need to change.
    func clearHistory() { createNew() }

    // MARK: - Private helpers

    /// Replace in-memory state with a loaded-from-disk conversation.
    /// Does NOT restore the conversation's systemPrompt — per-conversation
    /// systemPrompt override is v0.2 Batch 2 territory. For now the
    /// Inspector-driven ModelParameters.systemPrompt is the source of truth.
    private func adopt(_ conversation: Conversation) {
        // If the user has already started typing in the current session
        // (non-empty messages), don't clobber their work.
        guard messages.isEmpty else { return }
        current = conversation
        messages = conversation.messages.map(UIChatMessage.init(stored:))
    }

    /// Serialise the current conversation to disk. Fire-and-forget — we
    /// don't block the UI on I/O and we tolerate transient failures.
    private func persist() {
        var conv = current
        conv.messages = messages
            .filter { !$0.isGenerating }
            .map(\.asStored)
        conv.systemPrompt = parameters.parameters.systemPrompt
        conv.modelID = coordinator.currentModel?.id
        current = conv

        Task { [store] in
            try? await store.save(conv)
            // Reload the sidebar list so a save bumping updatedAt
            // re-sorts the sidebar in real time.
            await self.reloadConversationList()
        }
    }

    /// Like `persist()` but skips the save when there's nothing to save
    /// (empty blank conversation that never got a message). Used by the
    /// switch/createNew/delete paths to flush outgoing state without
    /// spamming empty rows into the sidebar.
    private func persistNow() {
        // Only persist if the conversation has at least one stored message.
        let storable = messages.filter { !$0.isGenerating }
        guard !storable.isEmpty else { return }
        persist()
    }
}
