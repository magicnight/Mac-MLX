// BenchmarkViewModel.swift
// macMLX
//
// @Observable @MainActor view model backing BenchmarkView (issue #22).
//
// Responsibilities:
// - Enumerate local models from ModelLibraryManager for the picker
// - Hold user-editable benchmark config (prompt tokens, gen tokens, runs)
// - Drive the BenchmarkRunner, loading the model via EngineCoordinator
//   first when needed (and timing the cold load)
// - Persist results via BenchmarkStore and expose the history for the UI

import Foundation
import Observation
import MacMLXCore

@Observable
@MainActor
final class BenchmarkViewModel {

    // MARK: - Config (user-editable)

    /// Currently selected model for the benchmark. Nil until the user
    /// picks or we auto-pick the only locally-downloaded model.
    var selectedModel: LocalModel?

    /// Approximate prompt length in tokens. Engine's tokenizer reports
    /// the real count on the terminal chunk; this is the target.
    var promptTokens: Int = 512

    /// Maximum generated tokens per iteration (`max_tokens`).
    var generationTokens: Int = 200

    /// Measured iterations (warm-up is always 1 extra, not counted).
    var runs: Int = 3

    /// User-supplied notes, attached to the result for sharing.
    var notes: String = ""

    // MARK: - State

    /// `true` while a benchmark run is in flight.
    private(set) var isRunning: Bool = false

    /// Short status line for the UI while running.
    private(set) var statusMessage: String = ""

    /// Last completed benchmark. Nil until the first run finishes.
    private(set) var lastResult: BenchmarkResult?

    /// All prior results, newest-first. Reloaded on appear + after each run.
    private(set) var history: [BenchmarkResult] = []

    /// Available local models, refreshed via `reloadModels()`.
    private(set) var availableModels: [LocalModel] = []

    /// Last surfaced error string (nil when cleared). Used for an inline
    /// banner in the UI.
    var errorMessage: String?

    // MARK: - Dependencies

    private let coordinator: EngineCoordinator
    private let library: ModelLibraryManager
    private let store: BenchmarkStore
    private let logs: LogManager

    /// In-flight benchmark task, retained so the UI can abandon it when
    /// the user clicks Cancel.
    private var runTask: Task<Void, Never>?

    // MARK: - Init

    init(
        coordinator: EngineCoordinator,
        library: ModelLibraryManager,
        store: BenchmarkStore,
        logs: LogManager
    ) {
        self.coordinator = coordinator
        self.library = library
        self.store = store
        self.logs = logs
    }

    // MARK: - Lifecycle

    /// Refresh the model picker from the library and reload history.
    /// Safe to call repeatedly — used on view appear.
    func reload(modelDirectory: URL) async {
        do {
            availableModels = try await library.scan(modelDirectory)
            // Prefer whatever's currently loaded in the coordinator;
            // otherwise pick the first local model so the picker isn't
            // empty on first visit.
            if let current = coordinator.currentModel,
               availableModels.contains(where: { $0.id == current.id }) {
                selectedModel = current
            } else if selectedModel == nil {
                selectedModel = availableModels.first
            }
        } catch {
            errorMessage = "Failed to scan model directory: \(error.localizedDescription)"
        }
        await reloadHistory()
    }

    /// Reload saved results. Called after each successful run.
    func reloadHistory() async {
        do {
            history = try await store.list()
            if lastResult == nil { lastResult = history.first }
        } catch {
            // Not fatal — the store has corrupt-file tolerance, but a
            // directory-level error still shows up via `errorMessage`.
            errorMessage = "Failed to load benchmark history: \(error.localizedDescription)"
        }
    }

    // MARK: - Running

    /// Kick off a benchmark run. Resolves on the UI thread; heavy work is
    /// delegated to a background Task. No-op if a run is already in
    /// flight, or if no model is selected.
    func start() {
        guard !isRunning else { return }
        guard let model = selectedModel else {
            errorMessage = "Pick a model first."
            return
        }
        errorMessage = nil
        isRunning = true
        statusMessage = "Preparing…"
        runTask = Task { [weak self] in
            await self?.doRun(model: model)
        }
    }

    /// Cancel an in-flight benchmark (best-effort — the active generation
    /// completes its current token before returning).
    func cancel() {
        runTask?.cancel()
        runTask = nil
        isRunning = false
        statusMessage = ""
    }

    // MARK: - Internals

    private func doRun(model: LocalModel) async {
        let loadTime = await ensureModelLoaded(model: model)
        if Task.isCancelled {
            isRunning = false
            statusMessage = ""
            return
        }
        statusMessage = "Warming up…"

        // Capture the coordinator's generate function into a @Sendable
        // closure the runner can call from its actor. The closure must
        // not retain @MainActor state — `coordinator.generate(_:)` is
        // itself @MainActor safe, and AsyncThrowingStream crosses
        // concurrency domains fine.
        let coordinator = self.coordinator
        let generate: @Sendable (GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error> = { request in
            // Force hop to @MainActor to call coordinator.generate(_:).
            // The stream values themselves are Sendable (GenerateChunk).
            AsyncThrowingStream { continuation in
                Task { @MainActor in
                    let inner = coordinator.generate(request)
                    do {
                        for try await chunk in inner {
                            continuation.yield(chunk)
                        }
                        continuation.finish()
                    } catch {
                        continuation.finish(throwing: error)
                    }
                }
            }
        }

        let runner = BenchmarkRunner(generate: generate)
        let engineVersion = coordinator.engineVersion
        let engineID = coordinator.engineID
        let promptN = promptTokens
        let genN = generationTokens
        let runsN = max(1, runs)
        let notesCopy = notes

        do {
            statusMessage = "Running \(runsN) iteration(s)…"
            let result = try await runner.run(
                modelID: model.id,
                engineID: engineID,
                engineVersion: engineVersion,
                promptTokens: promptN,
                generationTokens: genN,
                runs: runsN,
                modelLoadTimeS: loadTime,
                notes: notesCopy
            )
            if Task.isCancelled { return }
            lastResult = result
            try await store.save(result)
            await logs.log(
                "Benchmark finished: \(Int(result.generationTPS)) tok/s on \(model.id)",
                level: .info,
                category: .engine
            )
            await reloadHistory()
        } catch is CancellationError {
            // User cancelled — state already reset in cancel(). No-op.
        } catch {
            errorMessage = error.localizedDescription
            await logs.log(
                "Benchmark failed: \(error.localizedDescription)",
                level: .error,
                category: .engine
            )
        }
        isRunning = false
        statusMessage = ""
        runTask = nil
    }

    /// Ensure `model` is loaded in the coordinator. If a different model
    /// (or nothing) is loaded, triggers a load and returns the measured
    /// cold-load time in seconds. Returns 0 when the model was already
    /// loaded.
    private func ensureModelLoaded(model: LocalModel) async -> Double {
        if coordinator.currentModel?.id == model.id,
           coordinator.status.isLoaded {
            return 0
        }
        statusMessage = "Loading \(model.id)…"
        let start = Date()
        _ = await coordinator.load(model)
        return Date().timeIntervalSince(start)
    }

    // MARK: - History management

    func delete(id: UUID) async {
        try? await store.delete(id: id)
        await reloadHistory()
    }

    func clearHistory() async {
        try? await store.deleteAll()
        lastResult = nil
        await reloadHistory()
    }
}
