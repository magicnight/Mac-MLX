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
    public let hfSizeCache: HFSizeCache
    public let logs: LogManager
    public let coordinator: EngineCoordinator
    public let conversations: ConversationStore
    public let parametersStore: ModelParametersStore

    /// Local benchmark history store (issue #22). Persists to
    /// `~/.mac-mlx/benchmarks/` with the same dotfile-exemption pattern
    /// as the other stores.
    public let benchmarks: BenchmarkStore

    /// Single app-scoped Chat view model. Owned here (not by ChatView's
    /// @State) so message history and the in-flight generation Task survive
    /// sidebar tab switches — see issue #1.
    /// Internal rather than `public` because `ChatViewModel` is app-internal
    /// (unlike the `MacMLXCore` actor types above which are public).
    let chat: ChatViewModel

    /// Per-model generation parameter overrides for the Parameters
    /// Inspector panel (issue #15). Same retain story as `chat`.
    let parameters: ParametersViewModel

    /// Benchmark feature's view model (issue #22). Hoisted onto AppState
    /// so an in-progress benchmark Task survives sidebar tab switches,
    /// same rationale as `chat`.
    let benchmark: BenchmarkViewModel

    /// Long-lived model-library VM. Owned by AppState (not
    /// ModelLibraryView's @State) so tab switches don't tear down
    /// in-flight HF downloads — same pattern as `chat` (issue #1).
    ///
    /// IUO because the VM's `modelDirectoryProvider` closure needs a
    /// live reference to `self` to read `currentSettings`, which Swift
    /// forbids during stored-property init. We assign it on the last
    /// line of `init()` once all other properties are valid.
    private(set) var modelLibrary: ModelLibraryViewModel!

    /// Snapshot of the most recently loaded settings. Updated after each
    /// `settings.load()` / `settings.update()` call so SwiftUI bindings have
    /// something synchronous to observe.
    public private(set) var currentSettings: Settings = .default

    /// HTTP server instance, nil when stopped. Observable so the UI
    /// can reflect "Running on :8000" vs "Stopped".
    public private(set) var server: HummingbirdServer? = nil

    /// Port the server is currently bound to, nil when stopped.
    public private(set) var serverPort: Int? = nil

    /// True while a start/stop is in flight, so UI buttons can show
    /// a spinner and disable double-click.
    public private(set) var isServerToggling: Bool = false

    // MARK: - Init

    public init() {
        let settings = SettingsManager()
        let logs = LogManager()
        let library = ModelLibraryManager()
        let coordinator = EngineCoordinator(logs: logs)
        let conversations = ConversationStore()  // ~/.mac-mlx/conversations
        let parametersStore = ModelParametersStore()  // ~/.mac-mlx/model-params
        let benchmarks = BenchmarkStore()            // ~/.mac-mlx/benchmarks
        let parameters = ParametersViewModel(store: parametersStore)

        self.settings = settings
        self.library = library
        self.downloader = HFDownloader()
        self.hfSizeCache = HFSizeCache()
        self.logs = logs
        self.coordinator = coordinator
        self.conversations = conversations
        self.parametersStore = parametersStore
        self.benchmarks = benchmarks
        self.parameters = parameters
        self.chat = ChatViewModel(
            coordinator: coordinator,
            store: conversations,
            parameters: parameters
        )
        self.benchmark = BenchmarkViewModel(
            coordinator: coordinator,
            library: library,
            store: benchmarks,
            logs: logs
        )

        // Rehydrate per-model state after any model load, regardless of
        // which surface triggered it (Models tab, Benchmark tab, CLI via
        // OpenAI API, menu bar). Pre-v0.3 the Parameters Inspector only
        // loaded its overrides on its own onAppear, so a chat session
        // that never opened the Inspector saw dead storage — the
        // persisted per-model temperature / top_p / system prompt were
        // all ignored. Hooking into EngineCoordinator.onModelLoaded
        // makes the overrides always active.
        coordinator.onModelLoaded = { [parameters] model in
            await parameters.loadForModel(model.id)
        }

        // Constructed last: the provider closure captures `self` so all
        // other stored properties must already be valid. Weak-self keeps
        // the ownership graph acyclic (AppState -> VM -> closure -> self
        // would otherwise leak).
        self.modelLibrary = ModelLibraryViewModel(
            library: library,
            coordinator: coordinator,
            downloader: self.downloader,
            sizeCache: self.hfSizeCache,
            modelDirectoryProvider: { [weak self] in
                self?.currentSettings.modelDirectory ?? Settings.default.modelDirectory
            }
        )
    }

    // MARK: - Lifecycle

    /// Bootstrap on app launch. Loads settings from disk, then primes the
    /// engine coordinator with the user's preferred engine and pushes the
    /// stored HF endpoint into the downloader (#21).
    public func bootstrap() async {
        let loaded = await settings.load()
        currentSettings = loaded
        coordinator.switchTo(loaded.preferredEngine)
        await applyHFEndpoint(loaded.hfEndpoint)
        await logs.log("App bootstrapped", level: .info, category: .system)
        // Kick off an initial library scan now that settings are loaded —
        // otherwise the Models tab's .task fires against the default
        // Settings snapshot before bootstrap completes, and users who
        // skipped the onboarding wizard see an empty Models list until
        // they toggle Settings (which re-triggers the scan via onChange).
        await modelLibrary.loadLocalModels()

        // Auto-start the HTTP server if the user opted in. Deferred so
        // that a failure to start (e.g. port conflict) doesn't block
        // the rest of the app from becoming interactive.
        if loaded.autoStartServer {
            // Rehydrate the last-loaded model first if we have one —
            // the server needs an engine with a model to serve meaningful
            // requests. If no model was persisted, we start anyway and
            // rely on cold-swap to pull a model on first request.
            if let lastModelID = loaded.lastLoadedModel {
                let dir = loaded.modelDirectory
                if let models = try? await library.scan(dir),
                   let target = models.first(where: { $0.id == lastModelID || $0.displayName == lastModelID }) {
                    _ = await coordinator.load(target)
                }
            }
            await startServer()
        }
    }

    // MARK: - Server lifecycle

    /// Start the OpenAI-compatible HTTP server on the configured port.
    /// Idempotent — calling while running is a no-op. Logs any error
    /// to `LogManager`.
    public func startServer() async {
        guard server == nil, !isServerToggling else { return }
        guard let engine = coordinator.activeEngine else {
            await logs.log(
                "Cannot start server: no engine loaded",
                level: .warning,
                category: .system
            )
            return
        }
        isServerToggling = true
        defer { isServerToggling = false }

        // Cold-swap resolver: inbound requests naming a locally-downloaded
        // model ID that's not currently loaded can still be served — the
        // resolver pulls the model into the coordinator on demand.
        // We snapshot the directory at start time (matching ServeCommand's
        // pattern) rather than reading `currentSettings` live — a @Sendable
        // closure can't touch @MainActor state without hopping, and the
        // directory only changes via Settings, which would require a
        // server restart to re-bind anyway.
        let library = self.library
        let modelDirectory = currentSettings.modelDirectory
        let resolver: HummingbirdServer.ModelResolver = { modelID in
            let models = (try? await library.scan(modelDirectory)) ?? []
            return models.first { $0.id == modelID || $0.displayName == modelID }
        }
        // Route cold-swap through EngineCoordinator so the GUI and
        // menu bar reflect the newly-loaded model (currentModel,
        // status, onModelLoaded callback all fire correctly).
        let coord = self.coordinator
        let loadHook: HummingbirdServer.LoadHook = { model in
            let result = await coord.load(model)
            if case .failure(let err) = result {
                throw err
            }
        }
        let instance = HummingbirdServer(
            engine: engine,
            modelResolver: resolver,
            loadHook: loadHook
        )
        do {
            let actualPort = try await instance.start(
                preferredPort: currentSettings.serverPort
            )
            server = instance
            serverPort = actualPort
            await logs.log(
                "HTTP server started on http://localhost:\(actualPort)/v1",
                level: .info,
                category: .system
            )
        } catch {
            await logs.log(
                "Failed to start HTTP server: \(error.localizedDescription)",
                level: .error,
                category: .system
            )
        }
    }

    /// Stop the HTTP server. No-op when already stopped.
    public func stopServer() async {
        guard let instance = server, !isServerToggling else { return }
        isServerToggling = true
        defer { isServerToggling = false }
        await instance.stop()
        server = nil
        serverPort = nil
        await logs.log(
            "HTTP server stopped",
            level: .info,
            category: .system
        )
    }

    /// Persist + activate a new Hugging Face Hub endpoint. Safe to call
    /// while downloads are in flight — only new requests pick up the new
    /// origin. Invalid URLs (no host) are rejected silently and the
    /// previous endpoint stays in effect.
    public func setHFEndpoint(_ endpointString: String) async {
        await updateSettings { $0.hfEndpoint = endpointString }
        await applyHFEndpoint(endpointString)
    }

    private func applyHFEndpoint(_ endpointString: String) async {
        guard let url = URL(string: endpointString), url.host != nil else { return }
        await downloader.setBaseURL(url)
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
