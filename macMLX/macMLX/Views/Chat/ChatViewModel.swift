// ChatViewModel.swift
// macMLX
//
// @Observable view model for ChatView. Manages message history (in-memory,
// v0.1 only), drives streaming generation via EngineCoordinator.

import Foundation
import MacMLXCore

/// A chat message as tracked in the UI (different from MacMLXCore ChatMessage —
/// adds isGenerating + tokenCount for live display).
struct UIChatMessage: Identifiable {
    let id: UUID
    let role: MessageRole
    var content: String
    let timestamp: Date
    var tokenCount: Int?
    var isGenerating: Bool

    init(role: MessageRole, content: String, isGenerating: Bool = false) {
        self.id = UUID()
        self.role = role
        self.content = content
        self.timestamp = Date()
        self.tokenCount = nil
        self.isGenerating = isGenerating
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

    private let appState: AppState
    private var generationTask: Task<Void, Never>? = nil

    // MARK: - Init

    init(appState: AppState) {
        self.appState = appState
    }

    // MARK: - Send

    func send() async {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty,
              let currentModel = appState.coordinator.currentModel else { return }

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
        var assistantMsg = UIChatMessage(role: .assistant, content: "", isGenerating: true)
        messages.append(assistantMsg)
        let assistantIdx = messages.count - 1

        // Stream tokens
        generationTask = Task { [weak self] in
            guard let self else { return }
            let stream = self.appState.coordinator.generate(request)
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
        }
        await generationTask?.value
    }

    // MARK: - Stop

    func stopGeneration() {
        generationTask?.cancel()
        generationTask = nil
        // Mark any in-flight message as done
        for idx in messages.indices where messages[idx].isGenerating {
            messages[idx].isGenerating = false
        }
        isGenerating = false
    }

    // MARK: - Clear

    func clearHistory() {
        stopGeneration()
        messages = []
        inputText = ""
    }
}
