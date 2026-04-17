// AppState.swift
// macMLX
//
// Top-level @Observable state holder. Owns the shared MacMLXCore actors
// and exposes them to SwiftUI via the environment. One instance lives for
// the lifetime of the app and is created in macMLXApp's @State.

import Foundation
import MacMLXCore

@Observable
@MainActor
public final class AppState {

    // MARK: - Shared actors

    public let settings: SettingsManager
    public let library: ModelLibraryManager
    public let downloader: HFDownloader
    public let logs: LogManager
    public let coordinator: EngineCoordinator
    public let conversations: ConversationStore

    /// Single app-scoped Chat view model. Owned here (not by ChatView's
    /// @State) so message history and the in-flight generation Task survive
    /// sidebar tab switches — see issue #1.
    /// Internal rather than `public` because `ChatViewModel` is app-internal
    /// (unlike the `MacMLXCore` actor types above which are public).
    let chat: ChatViewModel

    /// Snapshot of the most recently loaded settings. Updated after each
    /// `settings.load()` / `settings.update()` call so SwiftUI bindings have
    /// something synchronous to observe.
    public private(set) var currentSettings: Settings = .default

    // MARK: - Init

    public init() {
        let settings = SettingsManager()
        let logs = LogManager()
        let coordinator = EngineCoordinator(logs: logs)
        let conversations = ConversationStore()  // defaults to ~/.mac-mlx/conversations
        self.settings = settings
        self.library = ModelLibraryManager()
        self.downloader = HFDownloader()
        self.logs = logs
        self.coordinator = coordinator
        self.conversations = conversations
        self.chat = ChatViewModel(coordinator: coordinator, store: conversations)
    }

    // MARK: - Lifecycle

    /// Bootstrap on app launch. Loads settings from disk, then primes the
    /// engine coordinator with the user's preferred engine.
    public func bootstrap() async {
        let loaded = await settings.load()
        currentSettings = loaded
        coordinator.switchTo(loaded.preferredEngine)
        await logs.log("App bootstrapped", level: .info, category: .system)
    }

    /// Persist a mutated copy of `currentSettings`. Calls `update` on the
    /// underlying actor and refreshes the local snapshot.
    public func updateSettings(_ mutate: @Sendable (inout Settings) -> Void) async {
        do {
            try await settings.update(mutate)
            currentSettings = await settings.current
        } catch {
            await logs.log(
                "Failed to persist settings: \(error.localizedDescription)",
                level: .error,
                category: .system
            )
        }
    }
}
