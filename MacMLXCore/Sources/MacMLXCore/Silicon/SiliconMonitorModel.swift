// Copyright © 2026 macMLX. English comments only.

import Foundation

/// The pure reduction core behind the W3 GUI observation panel: it folds the two
/// input streams the panel fuses — the engine's prefill/decode phase timeline and
/// the W1 hardware `SiliconSample` stream — into one immutable snapshot of "what to
/// display right now", and drives the W2 `BottleneckClassifier` off them.
///
/// WHY THIS LIVES IN CORE (not the SwiftUI layer):
/// the app-layer `SiliconMonitor` is a thin `@Observable @MainActor` shell that owns
/// the async plumbing (the sampling timer, the phase-event channel, the engine
/// observer adapter, view lifecycle). All of the *logic* it would otherwise carry —
/// mapping observer callbacks to a phase context, deciding when a verdict exists vs.
/// when the panel is idle, surfacing the IOReport-unavailable state honestly, and
/// computing prefill/decode tokens-per-second — is here, as synchronous mutating
/// methods on a `Sendable` value. That makes every one of those behaviours unit-
/// testable under bare `swift test` with synthetic samples and hand-driven phase
/// calls, no live IOReport and no engine required — exactly the coverage the app
/// shell cannot get (there is no app test target).
///
/// HONESTY, BAKED INTO THE SHAPE:
///   * `ioReportAvailable` is a THREE-STATE optional: `nil` before the first sample
///     (unknown — the panel must not paint zeros), `true`/`false` afterwards. A
///     `false` sample still carries thermal + memory pressure (public APIs), so the
///     panel shows those and marks only the IOReport-derived readouts unavailable.
///   * `verdict` is `nil` whenever no generation is in flight — an idle machine has
///     no inference bottleneck to attribute, so the panel shows "no active
///     generation" rather than a stale or fabricated verdict.
///   * tokens/second come only from the engine's terminal `GenerationPhaseSummary`
///     (real counts ÷ real elapsed time), and are `nil` when that time was zero —
///     never a live-sampled or divide-by-zero rate.
///
/// KNOWN LIMITATION (concurrent generations): the panel shows ONE bottleneck at a
/// time. `isGenerating` is refcounted, so two overlapping generations (a GUI chat
/// plus an HTTP request, on two pooled engines sharing this observer) never read a
/// false "Idle" mid-flight. But `phase`/`config` are single-valued and reflect the
/// most recent phase event, so a verdict during overlap may attribute the hardware
/// to whichever generation transitioned last. Correct per-generation attribution
/// needs generation IDs on the observer seam and is deferred; single-stream (the
/// primary flow) is exact.
public struct SiliconMonitorModel: Sendable {

    // MARK: - Published snapshot

    /// The most recent hardware sample, or `nil` before the first one arrives.
    public private(set) var latestSample: SiliconSample?

    /// The current bottleneck verdict, or `nil` when no generation is in flight
    /// (idle → nothing to attribute).
    public private(set) var verdict: BottleneckVerdict?

    /// The terminal summary of the most recently completed generation, kept so the
    /// panel can show the last run's prefill/decode throughput while idle. Survives
    /// an abort (an aborted run produces no new summary; the prior one stands).
    public private(set) var lastSummary: GenerationPhaseSummary?

    /// True while at least one generation is between a begin and its terminal
    /// event. Refcounted so two overlapping generations (a GUI chat plus an HTTP
    /// request, on two pooled engines sharing this observer) do not read "Idle"
    /// the moment the first of them completes while the other is still running.
    public var isGenerating: Bool { activeGenerationCount > 0 }

    /// The phase the engine is currently in, or `nil` when not generating.
    public private(set) var phase: InferencePhase?

    // MARK: - Private state

    /// Persistent across generations on purpose: its self-calibrated bandwidth
    /// ceiling accumulates over the whole session, so saturation verdicts get more
    /// accurate the longer the app runs, and resetting it would discard that
    /// calibration for no real gain. The cost is that at a generation boundary the
    /// rolling window still holds up to `rollingWindow - 1` frames from the PREVIOUS
    /// run (it is only fed while generating). Those stale frames are NOT self-
    /// healing quickly — a latched memory/thermal state is stickier under hysteresis
    /// — so `ingest` explicitly suppresses the published verdict until the window
    /// has been refreshed with this generation's own frames (see
    /// `framesSinceGenerationStart`). We keep FEEDING the classifier across the
    /// boundary (to preserve the ceiling) but do not DISPLAY its output until it is
    /// no longer contaminated.
    private var classifier = BottleneckClassifier()

    /// The generation config carried by the latest phase event, fed into every
    /// classification so the advice stays concrete.
    private var config = EngineGenerationConfig(
        kvBits: nil, kvGroupSize: nil, quantizedKVStart: nil, batchSize: 1)

    /// Number of generations currently in flight. `isGenerating` is `> 0`.
    private var activeGenerationCount = 0

    /// Frames ingested since the current run of generations began (reset when the
    /// count goes 0 → 1). While this is below `rollingWindow` the classifier's
    /// window still contains the previous run's tail, so the verdict is withheld.
    private var framesSinceGenerationStart = 0

    public init() {}

    // MARK: - Phase bridging (driven by the engine's SiliconEngineObserver events)

    /// Prefill has begun: enter the generating state in the prefill phase.
    public mutating func beginPrefill(config: EngineGenerationConfig) {
        startGeneration()
        self.config = config
        phase = .prefill
    }

    /// The first output token arrived: transition to the decode phase. If no
    /// prefill preceded it (a pathological observer), this still starts a
    /// generation so the model stays coherent; a normal prefill → decode
    /// transition within one generation does NOT double-count.
    public mutating func beginDecode(config: EngineGenerationConfig) {
        if activeGenerationCount == 0 { startGeneration() }
        self.config = config
        phase = .decode
    }

    /// The generation ran to its end: record its summary (for the idle throughput
    /// readout) and drop one active generation.
    public mutating func complete(summary: GenerationPhaseSummary) {
        lastSummary = summary
        endGeneration()
    }

    /// The generation was cancelled or abandoned: drop one active generation with
    /// no new summary. The previous `lastSummary` is intentionally left intact.
    public mutating func abort() {
        endGeneration()
    }

    /// Begin a fresh run of generations (count 0 → 1): reset the stale-frame gate
    /// so the previous run's window tail is withheld from the panel.
    private mutating func startGeneration() {
        if activeGenerationCount == 0 { framesSinceGenerationStart = 0 }
        activeGenerationCount += 1
    }

    private mutating func endGeneration() {
        activeGenerationCount = max(0, activeGenerationCount - 1)
        // Only truly idle once the LAST in-flight generation ends.
        if activeGenerationCount == 0 {
            phase = nil
            verdict = nil
        }
    }

    // MARK: - Sample ingest

    /// Fold one hardware sample into the snapshot. Always publishes it as
    /// `latestSample` (the panel's live hardware readouts show whether or not a
    /// generation is running). Only runs the classifier while a generation is in
    /// flight; otherwise clears the verdict so the panel reads "idle".
    ///
    /// Note the classifier runs even when `sample.ioReportAvailable == false`: with
    /// the IOReport-derived fields nil it still has thermal + memory pressure (public
    /// APIs), so it can still surface a memory- or thermal-bound verdict honestly
    /// rather than going blind.
    public mutating func ingest(_ sample: SiliconSample) {
        latestSample = sample
        guard isGenerating, let phase else {
            verdict = nil
            return
        }
        framesSinceGenerationStart += 1
        let context = EnginePhaseContext(
            phase: phase, tokensPerSecond: nil, config: config)
        // Always feed the classifier so its rolling window advances and its
        // session-long bandwidth ceiling stays calibrated — but withhold the
        // verdict from the panel until the window no longer contains the previous
        // run's (possibly alarming) tail. Publishing it early would surface a
        // stale, un-hedged verdict at the start of an unrelated generation.
        let fresh = classifier.classify(sample: sample, context: context)
        verdict = framesSinceGenerationStart >= BottleneckClassifier.rollingWindow ? fresh : nil
    }

    // MARK: - Derived, honesty-preserving accessors

    /// IOReport availability as a three-state value: `nil` until the first sample,
    /// then the sample's own flag. The panel keys its unavailable banner off this.
    public var ioReportAvailable: Bool? {
        latestSample.map(\.ioReportAvailable)
    }

    /// The human-readable reason IOReport is unavailable, when the latest sample
    /// reports one.
    public var ioReportUnavailableReason: String? {
        latestSample?.ioReportUnavailableReason
    }

    /// Prefill throughput (tokens/second) from the last completed generation, or
    /// `nil` when unknown (no run yet, or its prompt time was zero).
    public var prefillTokensPerSecond: Double? {
        lastSummary?.prefillTokensPerSecond
    }

    /// Decode throughput (tokens/second) from the last completed generation, or
    /// `nil` when unknown (no run yet, or its generate time was zero).
    public var decodeTokensPerSecond: Double? {
        lastSummary?.decodeTokensPerSecond
    }
}
