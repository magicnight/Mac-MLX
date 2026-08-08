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

    /// The engine seam. Production always passes the app's single
    /// `EngineCoordinator` (see `BenchmarkEngineProviding`); the protocol only
    /// exists so tests can drive the run lifecycle against a scripted engine.
    private let coordinator: any BenchmarkEngineProviding
    private let library: ModelLibraryManager
    private let store: BenchmarkStore
    private let logs: any BenchmarkLogging

    /// The shared silicon monitor. During a run this VM activates its sampling and
    /// reads its live bottleneck verdicts to attribute what limited the run — the
    /// same in-process observer the Activity panel uses, so no second observer is
    /// attached to the engine.
    private let siliconMonitor: any BenchmarkSiliconSampling

    /// In-flight benchmark task, retained so the UI can abandon it when
    /// the user clicks Cancel.
    private var runTask: Task<Void, Never>?

    /// Identity token of the run that currently owns this view model's run
    /// state. Bumped by both `start()` and `cancel()`, so at most one run is
    /// ever the owner and every earlier run is retired the instant a newer one
    /// begins (or the user cancels).
    ///
    /// Cancellation is cooperative — a cancelled run keeps winding down while
    /// its current token finishes, and used to run its unconditional epilogue
    /// long after a restarted run had taken over: clearing `isRunning` under a
    /// live run, dropping the restarted run's `runTask` (leaving Cancel with
    /// nothing to cancel), and letting the dead run's error land in the live
    /// run's banner. Each of `doRun`'s own state writes is therefore gated on
    /// still being the owner, re-checked after every suspension point (including
    /// inside `reloadHistoryOwned(generation:)`, whose store read suspends after
    /// its call site's check). What a retired run does still do is finish its
    /// in-flight inference: the stream bridging `coordinator.generate` is not
    /// wired for cancellation, so the tokens keep coming until the generation
    /// ends on its own — it just no longer writes any of this state.
    @ObservationIgnored private var currentGeneration: Int = 0

    /// Poll cadence for reading the monitor's current verdict during a run. The
    /// hardware sampler produces a fresh sample ~1 Hz; polling faster and de-duping
    /// on the sample timestamp folds each sample's verdict in exactly once.
    private static let bottleneckPollInterval: Duration = .milliseconds(200)

    // MARK: - Init

    init(
        coordinator: any BenchmarkEngineProviding,
        library: ModelLibraryManager,
        store: BenchmarkStore,
        logs: any BenchmarkLogging,
        siliconMonitor: any BenchmarkSiliconSampling
    ) {
        self.coordinator = coordinator
        self.library = library
        self.store = store
        self.logs = logs
        self.siliconMonitor = siliconMonitor
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

    /// Reload saved results. For the user-initiated paths (appear, delete, clear),
    /// where the newest read is always the one to keep. A benchmark run must use
    /// `reloadHistoryOwned(generation:)` instead, so its read cannot land after a
    /// Cancel + restart has handed the state to another run.
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
        currentGeneration &+= 1
        let generation = currentGeneration
        errorMessage = nil
        isRunning = true
        statusMessage = "Preparing…"
        runTask = Task { [weak self] in
            await self?.doRun(model: model, generation: generation)
        }
    }

    /// Cancel an in-flight benchmark (best-effort — the active generation
    /// completes its current token before returning). Retires the in-flight
    /// run so its epilogue can no longer write state that a restart may have
    /// taken over in the meantime.
    func cancel() {
        currentGeneration &+= 1
        runTask?.cancel()
        runTask = nil
        isRunning = false
        statusMessage = ""
    }

    // MARK: - Internals

    /// Whether `generation` is still the run that owns the view model's run
    /// state. False for any run retired by a later `start()` or by `cancel()`.
    private func owns(_ generation: Int) -> Bool { generation == currentGeneration }

    /// `reloadHistory()` for a benchmark run: re-checks ownership AFTER the store
    /// read suspends, not just before it. Checking only at the call site is not
    /// enough — the suspension inside `reloadHistory()` is its own hand-off point,
    /// so a run that owned the state when it called could still have been retired by
    /// a Cancel + restart by the time `store.list()` returns, and would then write
    /// `history` (and a stale error banner) over the live run's.
    private func reloadHistoryOwned(generation: Int) async {
        do {
            let loaded = try await store.list()
            guard owns(generation) else { return }
            history = loaded
            if lastResult == nil { lastResult = history.first }
        } catch {
            guard owns(generation) else { return }
            // Not fatal — the store has corrupt-file tolerance, but a
            // directory-level error still shows up via `errorMessage`.
            errorMessage = "Failed to load benchmark history: \(error.localizedDescription)"
        }
    }

    private func doRun(model: LocalModel, generation: Int) async {
        // The task body only starts after `start()` returns, so a cancel plus
        // restart can already have retired this run before its first write.
        guard owns(generation) else { return }
        let loadTime = await ensureModelLoaded(model: model, generation: generation)
        // Ownership subsumes cancellation, so there is no separate `Task.isCancelled`
        // check here: `cancel()` invalidates the generation and cancels the task in
        // the same synchronous @MainActor step, so a cancelled task is always a
        // retired one and this guard has already returned.
        guard owns(generation) else { return }
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
                let task = Task { @MainActor in
                    let inner = coordinator.generate(request)
                    do {
                        for try await chunk in inner {
                            // Honor the yield result instead of discarding it.
                            // `.terminated` means the runner stopped consuming
                            // (Cancel), so stop pulling from the engine rather
                            // than draining this generation to completion.
                            if case .terminated = continuation.yield(chunk) {
                                break
                            }
                        }
                        continuation.finish()
                    } catch {
                        continuation.finish(throwing: error)
                    }
                }
                // Propagate this stream's termination — including the consumer
                // being cancelled by the Cancel button — into the Task driving
                // generation. Without it, cancelling only the consumer left this
                // Task and the engine's generation loop running to maxTokens:
                // the GPU kept burning after Cancel, and a run restarted right
                // after competed with those zombies for a measurably lower
                // tok/s. Mirrors the same fix in `EngineCoordinator.generate`.
                continuation.onTermination = { @Sendable _ in
                    task.cancel()
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

        // Silicon attribution: for the duration of the run, sample the hardware and
        // fold the classifier's live decode verdicts into a PER-RUN collector, so the
        // result can report what limited it. Reference-counted sampling, so this is
        // independent of whether the Activity panel is open. The `defer` releases
        // sampling exactly once on every exit path (completion, early cancel, or
        // error), so it never double-decrements the shared count.
        //
        // The collector is a fresh object owned by this run and captured by this run's
        // task alone. If the user cancels and immediately restarts, the cancelled run
        // is still winding down with its own collector, so no aggregator state is
        // shared and its tail frames cannot land in the restarted run's saved
        // attribution. The two DO still share the signal source: `siliconMonitor`'s
        // `verdict`/`phase` are single-valued, so while two generations overlap both
        // collectors read whichever generation transitioned last — see the concurrent-
        // generation KNOWN LIMITATION on `SiliconMonitorModel`. Per-generation
        // attribution needs generation IDs on the observer seam and is deferred.
        //
        // Sampling starts here, before the runner does anything, so the hardware
        // sampler and the classifier are already warm when the timed window opens. The
        // collector itself stays shut until the runner reports the warm-up finished
        // (`onWarmupComplete` below): the warm-up is deliberately un-representative —
        // CPU-frequency ramp, JIT, first kernel launches. Most warm-up frames never
        // reach the collector anyway, because the warm-up is its own generation and
        // the monitor withholds a verdict until its rolling window has refilled; the
        // window catches the residual case where a slow model's warm-up runs long
        // enough to publish verdicts of its own.
        let collector = BottleneckCollector()
        let monitor = siliconMonitor
        siliconMonitor.activateSampling()
        let collectorTask = Task { @MainActor in
            while !Task.isCancelled {
                collector.collect(from: monitor)
                try? await Task.sleep(for: Self.bottleneckPollInterval)
            }
        }
        defer {
            collectorTask.cancel()
            siliconMonitor.deactivateSampling()
        }

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
                notes: notesCopy,
                onWarmupComplete: { [collector] in
                    // Pin the boundary synchronously, on the runner's own thread, so
                    // the hop below cannot push it past the first measured frames.
                    let boundary = Date()
                    Task { @MainActor in collector.openTimedWindow(at: boundary) }
                }
            )
            guard owns(generation) else { return }
            // Attach the collected attribution. `result()` is nil when the run was
            // too short to produce any decode verdict — then no attribution is
            // claimed and the UI honestly reports it as unavailable.
            let attributed = result.withBottleneck(collector.result())
            lastResult = attributed
            try await store.save(attributed)
            await logs.log(
                "Benchmark finished: \(Int(attributed.generationTPS)) tok/s on \(model.id)",
                level: .info,
                category: .engine
            )
            await reloadHistoryOwned(generation: generation)
        } catch is CancellationError {
            // User cancelled — state already reset in cancel(). No-op.
        } catch {
            await logs.log(
                "Benchmark failed: \(error.localizedDescription)",
                level: .error,
                category: .engine
            )
            guard owns(generation) else { return }
            errorMessage = error.localizedDescription
        }
        guard owns(generation) else { return }
        isRunning = false
        statusMessage = ""
        runTask = nil
    }

    /// Ensure `model` is loaded in the coordinator. If a different model
    /// (or nothing) is loaded, triggers a load and returns the measured
    /// cold-load time in seconds. Returns 0 when the model was already
    /// loaded.
    private func ensureModelLoaded(model: LocalModel, generation: Int) async -> Double {
        if coordinator.currentModel?.id == model.id,
           coordinator.status.isLoaded {
            return 0
        }
        guard owns(generation) else { return 0 }
        statusMessage = "Loading \(model.id)…"
        let start = Date()
        // `adapter: nil` is what the coordinator's own default already resolved
        // to here; it is spelled out because the seam declares the full shape.
        _ = await coordinator.load(model, adapter: nil)
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

// MARK: - Per-run bottleneck collector

/// The per-run silicon-attribution state: the decode-verdict aggregator plus the
/// last-folded sample timestamp, owned by a single benchmark run.
///
/// A reference type so the run's polling task can mutate it directly, and — the whole
/// point of it being per-run rather than a view-model property — so two overlapping
/// runs (a cancelled one still winding down while a restart begins) fold into DIFFERENT
/// collectors and cannot contaminate each other's saved attribution.
@MainActor
private final class BottleneckCollector {
    private var aggregator = BenchmarkBottleneckAggregator()
    private var lastSampleTimestamp: Date?

    /// Instant the runner's timed window opened, or nil while it has not opened yet.
    /// Nothing is folded before it, and no sample whose measurement interval reaches
    /// back before it is folded after it.
    private var timedWindowStart: Date?

    /// Open the attribution window at `instant` — the moment the runner finished its
    /// un-counted warm-up iteration. Comparing sample intervals against the instant
    /// (rather than merely flipping a flag) also discards the stale ~1 Hz sample that
    /// may still be the monitor's latest when the window opens, which would otherwise
    /// be a warm-up frame folded into a timed run's attribution.
    func openTimedWindow(at instant: Date) {
        timedWindowStart = instant
    }

    /// Fold the monitor's current verdict if it is a fresh decode-phase reading whose
    /// measurement interval lies wholly inside the timed window — the admission rule
    /// is `BenchmarkBottleneckAggregator.sampleFallsEntirelyWithinWindow`, in Core
    /// where it is unit-testable. Prefill frames publish no decode verdict and are
    /// naturally skipped; the ~1 Hz sample is de-duped on its timestamp so a faster
    /// poll folds each sample exactly once.
    ///
    /// The classifier's own per-generation gate (`SiliconMonitorModel.ingest`, which
    /// withholds a verdict until the rolling window holds this generation's frames)
    /// already suppresses most warm-up frames, since the warm-up is a separate
    /// generation that resets that counter. The window is the remaining guard, for the
    /// slow-model case where a warm-up lasts long enough to publish a verdict of its
    /// own.
    func collect(from monitor: any BenchmarkSiliconSampling) {
        guard let timedWindowStart else { return }
        guard let verdict = monitor.verdict, verdict.phase == .decode,
              let sample = monitor.latestSample,
              BenchmarkBottleneckAggregator.sampleFallsEntirelyWithinWindow(
                sample, windowStart: timedWindowStart),
              sample.timestamp != lastSampleTimestamp
        else { return }
        lastSampleTimestamp = sample.timestamp
        aggregator.add(verdict: verdict, sample: sample)
    }

    /// The run's attribution, or nil when no decode frame was folded.
    func result() -> BenchmarkBottleneck? { aggregator.result() }
}
