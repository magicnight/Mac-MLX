// SiliconMonitor.swift
// macMLX
//
// The app-layer bridge for the v0.7 silicon-metrics observation panel (W3). It
// fuses three things the panel needs and nothing else knows how to combine:
//
//   1. HARDWARE  — drives a SiliconSampler (IOReport + public pressure APIs) on a
//      ~1 Hz timer while the panel is visible, publishing the latest SiliconSample.
//   2. ENGINE PHASE — conforms (via a Sendable adapter) to the engine's
//      SiliconEngineObserver seam, so it knows when inference is in prefill vs.
//      decode and holds the last generation's phase-split throughput summary.
//   3. VERDICT — feeds every (sample, phase) pair to the W2 BottleneckClassifier
//      and publishes the current BottleneckVerdict.
//
// CONCURRENCY MODEL
// -----------------
// The class is @MainActor @Observable so SwiftUI can bind its published state
// directly. All the reduction logic lives in a pure `SiliconMonitorModel` (in
// MacMLXCore) that this shell owns and mutates on the main actor — this file only
// carries the async plumbing:
//
//   * Sampling loop: a @MainActor Task that awaits `sampler.sample()` (which hops
//     to the sampler actor and back), applies the result on the main actor via
//     `model.ingest`, then sleeps ~1 s. Started on panel appear, cancelled on
//     disappear, so IOReport is only polled while the user is watching.
//
//   * Phase events: the engine calls the SiliconPhaseObserver adapter from its own
//     isolation; the adapter yields onto an ordered AsyncStream. A long-lived
//     @MainActor consumer loop drains that stream IN ORDER and applies each event
//     to the model — guaranteeing a completion is never applied before the decode
//     transition that preceded it. The consumer runs for the app's lifetime
//     (started once via `startObserving()`), independent of panel visibility, so
//     `isGenerating` / `lastSummary` stay accurate even with the panel closed.
//
// NIL-DEFAULT COMPATIBILITY: the engine holds an OPTIONAL observer. Only the pool
// factory in EngineCoordinator passes `observer`; every other MLXSwiftEngine
// construction (e.g. Onboarding's engine check) stays observer-free and byte-for-
// byte unchanged.

import Foundation
import Observation
import MacMLXCore

@Observable
@MainActor
final class SiliconMonitor {

    // MARK: - Tunables

    /// Sampling cadence while the panel is visible. ~1 Hz is enough for a live
    /// readout and keeps the IOReport subscription cheap.
    private static let sampleInterval: Duration = .seconds(1)

    // MARK: - Published state (the panel binds these)

    /// The reduced snapshot: latest sample, verdict, phase, throughput. Mutated in
    /// place on the main actor; @Observable tracks the whole-value setter, so any
    /// field change re-renders the panel (fine at 1 Hz).
    private(set) var model = SiliconMonitorModel()

    /// True while the sampling loop is active (at least one consumer needs it).
    /// Drives the panel's "live / paused" affordance.
    private(set) var isSampling = false

    // MARK: - Flat forwarders (so the View reads `monitor.latestSample`, etc.)

    var latestSample: SiliconSample? { model.latestSample }
    var verdict: BottleneckVerdict? { model.verdict }
    var isGenerating: Bool { model.isGenerating }
    var phase: InferencePhase? { model.phase }
    var lastSummary: GenerationPhaseSummary? { model.lastSummary }
    /// `nil` before the first sample (unknown — do not paint zeros), then the
    /// sample's own IOReport availability flag.
    var ioReportAvailable: Bool? { model.ioReportAvailable }
    var ioReportUnavailableReason: String? { model.ioReportUnavailableReason }
    var prefillTokensPerSecond: Double? { model.prefillTokensPerSecond }
    var decodeTokensPerSecond: Double? { model.decodeTokensPerSecond }

    // MARK: - Dependencies

    /// The hardware source. Injectable so a headless context (or a future test
    /// harness) can substitute a scripted sampler; production uses IOReport.
    private let sampler: any SiliconSampler

    /// The engine seam. Handed to EngineCoordinator's pool factory so every model
    /// the pool mints reports its phase timeline here. `Sendable`, created once in
    /// `init`, so it is safe to read synchronously during AppState construction.
    nonisolated let observer: any SiliconEngineObserver

    /// The ordered phase-event channel the `observer` feeds and `startObserving`
    /// drains. Consumed exactly once.
    private let phaseEvents: AsyncStream<SiliconPhaseEvent>

    // MARK: - Task handles

    private var samplingTask: Task<Void, Never>?
    /// Reference count of consumers that currently need sampling. The sampling loop
    /// runs while this is positive, so the Activity panel (visible) and a benchmark
    /// run (measuring) can each turn it on independently without one stopping it for
    /// the other. The counting semantics live in `MacMLXCore` so they are unit-tested.
    private var samplingActivation = SamplingActivation()
    /// The phase-event consumer. Deliberately has no cancellation path: this monitor
    /// is an app-lifetime `AppState` property, so the loop runs until process exit
    /// (the `weak self` in `startObserving` releases it cleanly at teardown). Only
    /// the visibility-gated `samplingTask` is started/stopped.
    private var phaseConsumerTask: Task<Void, Never>?

    // MARK: - Init

    /// - Parameter sampler: the hardware source; defaults to the production
    ///   IOReport-backed sampler. Merely constructing it opens no subscription —
    ///   the first `sample()` (inside the sampling loop) does.
    init(sampler: any SiliconSampler = IOReportSiliconSampler()) {
        self.sampler = sampler
        let (stream, continuation) = AsyncStream.makeStream(of: SiliconPhaseEvent.self)
        self.phaseEvents = stream
        self.observer = SiliconPhaseObserver(continuation: continuation)
    }

    // MARK: - Phase observation lifecycle (app-lifetime)

    /// Start draining the engine phase-event stream. Idempotent; call once from
    /// AppState bootstrap. Runs for the app's lifetime so generation state stays
    /// accurate regardless of whether the panel is open.
    func startObserving() {
        guard phaseConsumerTask == nil else { return }
        phaseConsumerTask = Task { [weak self] in
            guard let events = self?.phaseEvents else { return }
            for await event in events {
                guard let self else { break }
                self.apply(event)
            }
        }
    }

    private func apply(_ event: SiliconPhaseEvent) {
        switch event {
        case .beganPrefill(let config): model.beginPrefill(config: config)
        case .beganDecode(let config): model.beginDecode(config: config)
        case .completed(let summary): model.complete(summary: summary)
        case .aborted: model.abort()
        }
    }

    // MARK: - Sampling lifecycle (reference-counted)

    /// Register a consumer that needs the ~1 Hz hardware sampling loop, starting it
    /// if it was not already running. Reference-counted so more than one surface can
    /// need sampling at once — the Activity panel calls this from `.onAppear`, and a
    /// benchmark run calls it while it measures its bottleneck. Each call must be
    /// balanced by exactly one `deactivateSampling()`. Takes an immediate first
    /// sample so the panel isn't blank for a second, then polls on the interval.
    func activateSampling() {
        guard samplingActivation.activate() else { return }  // already running
        isSampling = true
        samplingTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                let snapshot = await self.sampler.sample()
                if Task.isCancelled { return }
                self.model.ingest(snapshot)
                try? await Task.sleep(for: Self.sampleInterval)
            }
        }
    }

    /// Release a consumer of sampling. Stops the loop — so IOReport is no longer
    /// polled — only when the LAST consumer deactivates, so the panel closing while
    /// a benchmark still runs (or vice versa) does not cut sampling short. Leaves the
    /// last snapshot in place so re-opening the panel shows the most recent reading
    /// until the next sample lands.
    func deactivateSampling() {
        guard samplingActivation.deactivate() else { return }  // others still need it
        samplingTask?.cancel()
        samplingTask = nil
        isSampling = false
    }
}
