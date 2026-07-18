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

    /// Silicon-metrics observation bridge (v0.7 W3). Drives the Activity panel:
    /// samples the hardware while that panel is visible, receives the engine's
    /// prefill/decode phase timeline (it is wired in as the pool's engine observer
    /// below), and publishes the current bottleneck verdict. App-internal like the
    /// other view models.
    let siliconMonitor: SiliconMonitor

    /// LoRA adapter directory scan (v0.5+). Backs the parameters-
    /// inspector adapter picker and the engine's per-load adapter
    /// resolution.
    public let adapterStore: AdapterStore

    /// Latest in-memory adapter list, refreshed on `bootstrap()` and
    /// after manual rescans. SwiftUI binds the parameters-inspector
    /// picker against this — the empty-array initial value is safe
    /// (the picker just shows "None" until the scan completes).
    public private(set) var availableAdapters: [LocalAdapter] = []

    /// Path under `~/.mac-mlx/adapters/` where users drop LoRA
    /// adapters. The cache subdir `.cache/` (created by
    /// `MLXSwiftEngine.applyAdapter` after a PEFT → mlx conversion)
    /// is intentionally on the same root so `du -sh ~/.mac-mlx/adapters`
    /// reports the user's storage cost truthfully.
    public var adaptersDirectory: URL { DataRoot.macMLX("adapters") }

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

    // MARK: - MCP tool routing (v0.6 wave 2)

    /// Config store for `~/.mac-mlx/mcp.json`. Read once on bootstrap.
    private let mcpConfigStore = MCPClientConfigStore()

    /// Live pool of connected MCP servers. Nil until `bootstrap()` connects
    /// at least one server, and stays nil for zero-config users so the tools
    /// feature is simply inactive (no behaviour change without `mcp.json`).
    public private(set) var mcpPool: MCPClientPool?

    /// Tool name → owning MCP server name, derived from the connected pool's
    /// tool listing. Empty until a successful connect. Drives call routing.
    public private(set) var mcpToolIndex: [String: String] = [:]

    /// OpenAI function specs for every connected tool, attached to a
    /// `GenerateRequest.tools` when a tool session runs. Empty until connect.
    public private(set) var mcpToolSpecs: [JSONValue] = []

    // MARK: - Init

    public init() {
        let settings = SettingsManager()
        let logs = LogManager()
        let library = ModelLibraryManager()
        // Create the silicon monitor BEFORE the coordinator so its (Sendable)
        // engine observer can be captured by the pool factory. Constructing it
        // opens no IOReport subscription — sampling starts only when the Activity
        // panel appears.
        let siliconMonitor = SiliconMonitor()
        let coordinator = EngineCoordinator(
            logs: logs, siliconObserver: siliconMonitor.observer)
        let conversations = ConversationStore()  // ~/.mac-mlx/conversations
        let parametersStore = ModelParametersStore()  // ~/.mac-mlx/model-params
        let benchmarks = BenchmarkStore()            // ~/.mac-mlx/benchmarks
        let parameters = ParametersViewModel(store: parametersStore)
        let adapterStore = AdapterStore()            // ~/.mac-mlx/adapters

        self.settings = settings
        self.library = library
        self.downloader = HFDownloader()
        self.hfSizeCache = HFSizeCache()
        self.logs = logs
        self.coordinator = coordinator
        self.siliconMonitor = siliconMonitor
        self.conversations = conversations
        self.parametersStore = parametersStore
        self.benchmarks = benchmarks
        self.parameters = parameters
        self.adapterStore = adapterStore
        self.chat = ChatViewModel(
            coordinator: coordinator,
            store: conversations,
            parameters: parameters
        )
        self.benchmark = BenchmarkViewModel(
            coordinator: coordinator,
            library: library,
            store: benchmarks,
            logs: logs,
            siliconMonitor: siliconMonitor
        )

        // Rehydrate per-model state after any model load, regardless of
        // which surface triggered it (Models tab, Benchmark tab, CLI via
        // OpenAI API, menu bar). Pre-v0.3 the Parameters Inspector only
        // loaded its overrides on its own onAppear, so a chat session
        // that never opened the Inspector saw dead storage — the
        // persisted per-model temperature / top_p / system prompt were
        // all ignored. Hooking into EngineCoordinator.onModelLoaded
        // makes the overrides always active.
        coordinator.onModelLoaded = { [parameters, adapterStore, coordinator, logs] model in
            await parameters.loadForModel(model.id)

            // v0.5+: if the user has a LoRA adapter pinned in this
            // model's parameters, scan + apply it on top of the base
            // model. Failures log + drop — the model still works
            // text-only, just without the adapter.
            let resolvedParams = await parameters.parameters
            guard let adapterName = resolvedParams.adapterName,
                  !adapterName.isEmpty
            else { return }

            let adaptersDir = DataRoot.macMLX("adapters")
            let scanned = (try? await adapterStore.scan(adaptersDir)) ?? []
            guard let adapter = scanned.first(where: { $0.name == adapterName }) else {
                await logs.log(
                    "LoRA adapter '\(adapterName)' configured for \(model.id) but not found under \(adaptersDir.path)",
                    level: .warning,
                    category: .engine
                )
                return
            }
            guard let engine = await coordinator.activeEngine else { return }
            do {
                try await engine.applyAdapter(adapter)
            } catch {
                await logs.log(
                    "LoRA adapter '\(adapterName)' failed to apply: \(error.localizedDescription)",
                    level: .error,
                    category: .engine
                )
            }
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
            },
            hfCacheSettingsProvider: { [weak self] in
                let s = self?.currentSettings ?? Settings.default
                return (s.scanHuggingFaceCache, s.huggingFaceCacheDirectories)
            }
        )

        // Wire the chat VM's MCP tool-routing providers. Assigned here (not at
        // `chat` construction) because they must capture the fully-initialised
        // `self`, and evaluated lazily per send — the pool doesn't exist until
        // `bootstrap()` connects it, and returns nil until then, so the chat
        // path stays on the plain non-tool flow for zero-config users.
        self.chat.toolSessionProvider = { [weak self] in self?.makeToolCallingSession() }
        self.chat.toolSpecsProvider = { [weak self] in self?.mcpToolSpecs ?? [] }
    }

    // MARK: - Lifecycle

    /// Bootstrap on app launch. Loads settings from disk, then primes the
    /// engine coordinator with the user's preferred engine and pushes the
    /// stored HF endpoint into the downloader (#21).
    public func bootstrap() async {
        let loaded = await settings.load()
        currentSettings = loaded
        // Thread the persisted prompt-cache budget into the coordinator's pool
        // factory before any model loads, so the very first engine honours the
        // user's hot/cold budgets instead of the construction-time defaults.
        coordinator.updatePromptCacheConfig(PromptCacheConfig(from: loaded))
        coordinator.switchTo(loaded.preferredEngine)
        await applyHFEndpoint(loaded.hfEndpoint)
        // Start draining the engine's phase-timeline events for the whole app
        // lifetime (independent of whether the Activity panel is open), so
        // generation state stays accurate. Hardware sampling is gated separately
        // on the panel's visibility.
        siliconMonitor.startObserving()
        await logs.log("App bootstrapped", level: .info, category: .system)
        await refreshAdapters()

        // MCP tool routing (v0.6 wave 2): connect the servers configured in
        // `~/.mac-mlx/mcp.json` in the background, so a slow or hanging server
        // can't delay app startup. Zero-config users have no `mcp.json` → this
        // is a no-op and the tools feature stays inactive.
        Task { await bootstrapMCP() }
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

    /// Re-scan `~/.mac-mlx/adapters/` and refresh `availableAdapters`.
    /// Called automatically once during `bootstrap()`. The parameters
    /// inspector exposes a "Refresh" button that calls this directly
    /// so users can drop a new adapter into the directory and pick it
    /// up without restarting the app.
    public func refreshAdapters() async {
        let dir = adaptersDirectory
        let scanned = (try? await adapterStore.scan(dir)) ?? []
        availableAdapters = scanned
        await logs.log(
            "Adapters scan: \(scanned.count) found in \(dir.path)",
            level: .debug,
            category: .system
        )
    }

    // MARK: - MCP tool routing (v0.6 wave 2)

    /// Load `~/.mac-mlx/mcp.json`, connect every configured server, and build
    /// the tool routing index + specs. Idempotent-friendly: a nil result (no
    /// config, or nothing connected) leaves the tools feature inactive.
    /// Partial connect failures are tolerated by `MCPClientPool.connectAll`.
    private func bootstrapMCP() async {
        let config = await mcpConfigStore.load()
        guard !config.mcpServers.isEmpty else { return }  // zero-config → inactive

        let pool = MCPClientPool(config: config)
        let connected = (try? await pool.connectAll()) ?? []
        guard !connected.isEmpty else {
            await logs.log(
                "MCP: no servers connected from mcp.json (\(config.mcpServers.count) configured)",
                level: .warning,
                category: .system
            )
            await pool.disconnectAll()
            return
        }

        let toolsByServer = await pool.listAllTools()
        let (index, specs) = ToolValueBridge.toolIndexAndSpecs(from: toolsByServer)

        mcpPool = pool
        mcpToolIndex = index
        mcpToolSpecs = specs
        await logs.log(
            "MCP: connected \(connected.count) server(s) — \(connected.joined(separator: ", ")); \(index.count) tool(s) available",
            level: .info,
            category: .system
        )
    }

    /// Build a `ToolCallingSession` configured for the current MCP setup, or
    /// nil when no servers are connected (tools feature inactive → callers use
    /// the plain generation path). The session injects the coordinator's
    /// `generate` as the model-turn driver (bridged onto the main actor) and
    /// the live pool for tool dispatch. MacMLXCore stays free of app imports —
    /// only the closure crosses the boundary.
    func makeToolCallingSession() -> ToolCallingSession? {
        guard let pool = mcpPool, !mcpToolIndex.isEmpty else { return nil }
        let coordinator = self.coordinator
        let generate: @Sendable (GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error> = { request in
            AsyncThrowingStream { continuation in
                // Hop onto the main actor to call the @MainActor coordinator,
                // then relay chunks to the session's stream. Honour the yield
                // result so a cancelled consumer stops the pull, and cancel
                // this task on termination so Stop unwinds through here into
                // the engine (mirrors the coordinator's own onTermination).
                let task = Task { @MainActor in
                    do {
                        for try await chunk in coordinator.generate(request) {
                            if case .terminated = continuation.yield(chunk) { break }
                        }
                        continuation.finish()
                    } catch {
                        continuation.finish(throwing: error)
                    }
                }
                continuation.onTermination = { @Sendable _ in task.cancel() }
            }
        }
        return ToolCallingSession(generate: generate, pool: pool, toolIndex: mcpToolIndex)
    }

    /// Best-effort synchronous teardown for app termination: disconnect the MCP
    /// pool so its spawned subprocesses (npx / uvx / …) don't linger past app
    /// exit. Briefly blocks the calling thread (bounded to 3s) because
    /// `disconnectAll` is async on the pool actor and `applicationWillTerminate`
    /// gives us no async context. No-op when no pool was ever connected.
    public func teardown() {
        guard let pool = mcpPool else { return }
        mcpPool = nil
        let done = DispatchSemaphore(value: 0)
        Task.detached {
            await pool.disconnectAll()
            done.signal()
        }
        _ = done.wait(timeout: .now() + 3)
    }

    // MARK: - Server lifecycle

    /// Start the OpenAI-compatible HTTP server on the configured port.
    /// Idempotent — calling while running is a no-op. Logs any error
    /// to `LogManager`.
    public func startServer() async {
        guard server == nil, !isServerToggling else { return }
        // Precondition unchanged from pre-SRV-1 behaviour: the server still
        // won't start with nothing loaded. What changes is HOW the engine
        // is handed to `HummingbirdServer` below — as a re-resolving
        // provider (SRV-1), not this one-time snapshot.
        guard await coordinator.activeEngine != nil else {
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
        // SRV-1 (CRITICAL): re-resolve the active engine on every request
        // instead of capturing it once. The GUI's cold-swap routes through
        // `ModelPool`, which mints a NEW `MLXSwiftEngine` per model — a
        // frozen reference would keep answering from the model that was
        // active when the server started (or from nothing, if that model
        // was since evicted), regardless of what's actually loaded now.
        let engineProvider: @Sendable () async -> (any InferenceEngine)? = {
            await coord.activeEngine
        }
        // POOL-3: let the server mark the active model in-flight in the
        // pool, so a concurrent GUI load can't LRU-evict a model the
        // server is mid-stream against.
        let inFlightHook: HummingbirdServer.InFlightHook = { modelID, active in
            await coord.setGenerating(modelID, active)
        }
        // SRV-5: honour the persisted bearer token on the GUI-launched
        // server too. Previously only the CLI `serve` path passed it, so a
        // user who set `serverAPIKey` still got an unauthenticated GUI
        // server (no `BearerAuthMiddleware` installed). Reads it the same way
        // `ServeCommand` does — straight from the settings snapshot.
        // A2d-2: continuous batching on by default when the resident engine
        // supports it. `ModelPool` mints a new `MLXSwiftEngine` per model, so the
        // seam must re-resolve the active engine per call (like `engineProvider`) —
        // this also makes a cold-swap drain the OUTGOING engine's cohort before the
        // swap. A non-batch resident engine resolves to `nil`, keeping the legacy
        // single-stream path; the per-model coverage gate handles uncoverable models.
        let batchServing = ProvidedBatchServing {
            await coord.activeEngine as? (any BatchGenerationServing)
        }
        let instance = HummingbirdServer(
            engineProvider: engineProvider,
            modelResolver: resolver,
            loadHook: loadHook,
            inFlightHook: inFlightHook,
            apiKey: currentSettings.serverAPIKey,
            stallTimeoutSeconds: TimeInterval(currentSettings.generationStallTimeoutSeconds),
            batchServing: batchServing
        )
        do {
            let actualPort = try await instance.start(
                preferredPort: currentSettings.serverPort
            )
            server = instance
            serverPort = actualPort
            // Share discovery with CLI (issue #—, v0.3.7): the CLI's
            // `serve` command reads this file and refuses to double-bind
            // when the GUI is already serving. `try?` because losing
            // the pid file is non-fatal — the server itself started fine.
            let record = PIDFile.Record(
                pid: Int32(ProcessInfo.processInfo.processIdentifier),
                port: actualPort,
                modelID: coordinator.currentModel?.id,
                startedAt: Date(),
                owner: .gui
            )
            try? PIDFile.write(record)
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
        // Release the shared discovery file so CLI `serve` stops
        // reporting the GUI as the owner. `try?` — absent file is fine.
        try? PIDFile.clear()
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
