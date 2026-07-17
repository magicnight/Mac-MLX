// Copyright © 2026 macMLX. English comments only.

/// Fuses the W1 hardware `SiliconSample` stream with the engine's own phase
/// context to decide, frame by frame, what is limiting inference — and to say
/// so honestly and actionably.
///
/// It is a pure, deterministic state machine: feed it a sequence of
/// (`SiliconSample`, `EnginePhaseContext`) pairs and it returns a
/// `BottleneckVerdict` per frame. No IOReport, no live engine, no clock — so it
/// is fully unit-testable with synthetic sequences, and a consumer (the W3 GUI
/// panel) can drive it from real samples without any behavioural difference.
///
/// DESIGN (locked in the v0.7 silicon-track plan):
///
///   * **Priority: memory > thermal > profile.** Memory pressure precedes an
///     OOM/swap stall that dwarfs any compute/bandwidth nuance, so it wins.
///     Thermal throttling caps clocks regardless of the workload's resource
///     mix, so it wins over the compute-vs-bandwidth profile. Only when neither
///     fires do we attribute the limit to compute or bandwidth.
///
///   * **Double-threshold hysteresis (0.85 enter / 0.70 exit).** Each fractional
///     detector ENTERS its state only when its signal crosses 0.85 and only
///     LEAVES when it falls back below 0.70. The dead band between the two
///     stops a signal hovering near one threshold from flapping the verdict.
///     Thermal, an ordinal signal, uses the analogous enter-at-`serious` /
///     exit-below-`fair` pair.
///
///   * **Self-calibrating achievable bandwidth.** "Saturated" is relative to the
///     highest bandwidth this machine has actually delivered, not a datasheet
///     peak. We ratchet an observed ceiling upward and measure saturation as
///     (rolling-mean bandwidth ÷ ceiling). The ceiling is only trusted once we
///     have also seen bandwidth drop clearly below it (proof it is a real
///     ceiling, not just the current level); until then the profile falls back
///     to the phase's healthy expectation.
///
///   * **3-frame rolling mean.** Every decision uses the mean of the last three
///     samples, so a single-frame spike (a stray thermal blip, a bandwidth
///     glitch) cannot by itself move the verdict.
///
///   * **Phase-aware.** Decode is normally bandwidth-bound and prefill is
///     normally compute-bound; the verdict names the same category the hardware
///     shows but phrases its advice around whether that is the expected regime
///     for the current phase.
///
///   * **ANE is excluded.** MLX inference runs on the GPU; the ANE exposes only
///     a power proxy (no occupancy), so folding it in would only mislead. This
///     classifier never reads `anePower` or `aneBandwidth`.
public struct BottleneckClassifier: Sendable {

    // MARK: Tunables

    /// Enter threshold for the fractional detectors (bandwidth saturation, GPU
    /// occupancy, memory used-fraction): the signal must cross this to activate.
    static let enterThreshold = 0.85
    /// Exit threshold for the fractional detectors: an active state persists
    /// until the signal falls below this. The 0.85→0.70 gap is the hysteresis
    /// dead band.
    static let exitThreshold = 0.70
    /// Thermal pressure at or above which the throttle state may enter.
    static let thermalEnterLevel = ThermalPressure.serious
    /// Thermal pressure at or below which the throttle state exits (the dead
    /// band is `fair < mean < serious`).
    static let thermalExitLevel = ThermalPressure.fair
    /// Number of most-recent samples the rolling mean spans.
    static let rollingWindow = 3

    // MARK: State

    /// The last up-to-`rollingWindow` samples, oldest first.
    private var window: [SiliconSample] = []
    /// Highest bandwidth (GB/s) observed so far — the self-calibrated ceiling.
    private var achievableBandwidthGBs: Double = 0
    /// True once a bandwidth reading has been seen clearly below the ceiling,
    /// so the ceiling is a trustworthy reference rather than just "the only
    /// value we have". Until then the profile defers to the phase expectation.
    private var bandwidthCeilingTrustworthy = false
    /// Once the GPU (AGX) bandwidth channel has ever appeared, we commit to it
    /// and never mix in the coarse DRAM aggregate again — the two are different
    /// physical quantities (AGX is GPU-only; the DRAM aggregate covers CPU+GPU+ANE
    /// traffic and is structurally larger), so blending them into one ceiling
    /// would suppress saturation detection. A machine that never exposes AGX
    /// stays on the DRAM aggregate for its whole lifetime — one channel either way.
    private var bandwidthUsesGPUChannel = false

    /// Latched detector states, so hysteresis can require the exit threshold to
    /// clear a state that the enter threshold set.
    private var memoryLatched = false
    private var thermalLatched = false
    private var gpuOccupiedLatched = false
    private var bandwidthSaturatedLatched = false

    public init() {}

    // MARK: Classification

    /// Ingest one hardware sample plus its engine phase context and return the
    /// current bottleneck verdict. Mutating: the rolling window, the calibrated
    /// ceiling, and the hysteresis latches all advance by one frame.
    public mutating func classify(
        sample: SiliconSample,
        context: EnginePhaseContext
    ) -> BottleneckVerdict {
        appendToWindow(sample)
        updateBandwidthCeiling()

        let phase = context.phase

        // --- Detector signals (rolling means over the window) ---
        let meanThermal = meanThermalLevel()
        let meanUsedFraction = meanMemoryUsedFraction()
        let latestLevelPressured = latestMemoryPressured()
        let meanGPUBusy = meanGPUBusyFraction()
        let saturationRatio = bandwidthSaturationRatio()

        // --- Memory detector (highest priority) ---
        let memEnter = (meanUsedFraction.map { $0 >= Self.enterThreshold } ?? false)
            || latestLevelPressured
        let memStay = (meanUsedFraction.map { $0 >= Self.exitThreshold } ?? false)
            || latestLevelPressured
        memoryLatched = memoryLatched ? memStay : memEnter

        // --- Thermal detector ---
        let thermEnter = meanThermal >= Double(Self.thermalEnterLevel.rawValue)
        let thermStay = meanThermal > Double(Self.thermalExitLevel.rawValue)
        thermalLatched = thermalLatched ? thermStay : thermEnter

        // --- GPU occupancy detector (gates the profile) ---
        if let g = meanGPUBusy {
            gpuOccupiedLatched = gpuOccupiedLatched
                ? (g >= Self.exitThreshold)
                : (g >= Self.enterThreshold)
        } else {
            gpuOccupiedLatched = false
        }

        // --- Bandwidth saturation detector ---
        if let b = saturationRatio {
            bandwidthSaturatedLatched = bandwidthSaturatedLatched
                ? (b >= Self.exitThreshold)
                : (b >= Self.enterThreshold)
        } else {
            bandwidthSaturatedLatched = false
        }

        // --- Resolve by priority ---
        let category: BottleneckVerdict.Category
        let restsOnEstimate: Bool
        if memoryLatched {
            category = .memoryBound
            restsOnEstimate = false
        } else if thermalLatched {
            category = .thermalThrottled
            restsOnEstimate = false
        } else if gpuOccupiedLatched {
            // Profile territory: the GPU is working and neither memory nor
            // thermals are the story. Split compute vs bandwidth.
            category = profileCategory(phase: phase)
            restsOnEstimate = true
        } else {
            category = .normal
            restsOnEstimate = false
        }

        return BottleneckVerdict(
            category: category,
            phase: phase,
            advice: Self.advice(
                category: category, phase: phase, config: context.config),
            restsOnEstimatedBandwidth: restsOnEstimate
        )
    }

    /// Compute vs bandwidth attribution once the profile gate is open.
    ///
    /// While the bandwidth ceiling is not yet trustworthy we cannot tell a
    /// genuinely saturated bus from "we have only ever seen this one level", so
    /// we report the phase's healthy expectation (prefill → compute, decode →
    /// bandwidth). Once the ceiling is trustworthy the measured saturation
    /// (with hysteresis) decides, which also lets us flag the anomalies —
    /// compute-bound decode, bandwidth-bound prefill.
    private func profileCategory(phase: InferencePhase) -> BottleneckVerdict.Category {
        guard bandwidthCeilingTrustworthy else {
            switch phase {
            case .prefill: return .computeBound
            case .decode: return .bandwidthBound
            }
        }
        return bandwidthSaturatedLatched ? .bandwidthBound : .computeBound
    }

    // MARK: Window + ceiling maintenance

    private mutating func appendToWindow(_ sample: SiliconSample) {
        window.append(sample)
        if window.count > Self.rollingWindow {
            window.removeFirst(window.count - Self.rollingWindow)
        }
    }

    /// The bandwidth reading a sample contributes on the currently-committed
    /// channel. Once committed to GPU (AGX), a frame that lacks it is simply "no
    /// reading" (excluded from the means) rather than a fallback to the
    /// incommensurable DRAM aggregate — see `bandwidthUsesGPUChannel`.
    private func bandwidthReading(_ sample: SiliconSample) -> MemoryBandwidthSample? {
        bandwidthUsesGPUChannel ? sample.gpuBandwidth : sample.dramBandwidth
    }

    private mutating func updateBandwidthCeiling() {
        guard let latest = window.last else { return }
        // Commit to the GPU (AGX) channel the first time it appears — it is the
        // requestor that matters for MLX inference, at 1 GB/s-band resolution. The
        // channel scale changes, so discard any DRAM-aggregate ceiling built
        // before the switch rather than carrying an inflated reference forward.
        if latest.gpuBandwidth != nil, !bandwidthUsesGPUChannel {
            bandwidthUsesGPUChannel = true
            achievableBandwidthGBs = 0
            bandwidthCeilingTrustworthy = false
        }
        guard let reading = bandwidthReading(latest) else { return }
        let bw = reading.estimatedGBPerSecond
        if bw > achievableBandwidthGBs {
            achievableBandwidthGBs = bw
        }
        // A reading sitting clearly below the ceiling proves the ceiling is a
        // real high-water mark, so saturation ratios against it can be trusted.
        if achievableBandwidthGBs > 0, bw <= achievableBandwidthGBs * Self.exitThreshold {
            bandwidthCeilingTrustworthy = true
        }
    }

    // MARK: Rolling-mean signals

    /// Mean thermal level (raw 0…3) over the window. Thermal is always present,
    /// so this is defined whenever any sample exists.
    private func meanThermalLevel() -> Double {
        guard !window.isEmpty else { return 0 }
        let sum = window.reduce(0.0) { $0 + Double($1.thermalPressure.rawValue) }
        return sum / Double(window.count)
    }

    /// Mean non-reclaimable memory used-fraction over the samples that carry a
    /// memory-pressure reading, or `nil` when none do.
    private func meanMemoryUsedFraction() -> Double? {
        let fractions = window.compactMap { $0.memoryPressure?.usedFraction }
        guard !fractions.isEmpty else { return nil }
        return fractions.reduce(0, +) / Double(fractions.count)
    }

    /// Whether the most recent memory-pressure reading is at warning or critical
    /// — the kernel's own authoritative verdict, trusted at face value (not
    /// smoothed) because it is a discrete state, not a noisy byte count.
    private func latestMemoryPressured() -> Bool {
        guard let level = window.last?.memoryPressure?.level else { return false }
        return level == .warning || level == .critical
    }

    /// Mean GPU busy-fraction over the samples that carry a GPU utilisation
    /// reading, or `nil` when none do.
    private func meanGPUBusyFraction() -> Double? {
        let busy = window.compactMap { $0.gpuUtilization?.busyFraction }
        guard !busy.isEmpty else { return nil }
        return busy.reduce(0, +) / Double(busy.count)
    }

    /// Saturation ratio: rolling-mean bandwidth ÷ calibrated ceiling, clamped to
    /// `[0, 1]`. `nil` when no bandwidth has been observed or the ceiling is
    /// still zero (so the profile falls back to the phase expectation).
    private func bandwidthSaturationRatio() -> Double? {
        let readings = window.compactMap { bandwidthReading($0)?.estimatedGBPerSecond }
        guard !readings.isEmpty, achievableBandwidthGBs > 0 else { return nil }
        let mean = readings.reduce(0, +) / Double(readings.count)
        return min(1.0, mean / achievableBandwidthGBs)
    }

    // MARK: Advice

    /// Phase-aware, honesty-calibrated recommendation for a category. The
    /// compute/bandwidth wording hedges ("appears") because it rests on the
    /// estimated bandwidth signal; the memory/thermal wording is assertive
    /// because it comes from authoritative kernel/OS APIs.
    static func advice(
        category: BottleneckVerdict.Category,
        phase: InferencePhase,
        config: EngineGenerationConfig
    ) -> String {
        switch category {
        case .normal:
            switch phase {
            case .prefill:
                return "Prefill is progressing with no single resource pinned."
            case .decode:
                return "Decode is progressing with no single resource pinned."
            }

        case .memoryBound:
            return "Unified memory is under pressure — the model is near a "
                + "compression/swap stall or out-of-memory. Use a smaller or more "
                + "heavily quantized model, or shorten the context."

        case .thermalThrottled:
            return "Thermal throttling is capping clocks, so sustained throughput "
                + "is limited. Reduce sustained load or improve cooling."

        case .computeBound:
            switch phase {
            case .prefill:
                return "Prefill appears compute-bound (GPU pinned, memory bus has "
                    + "headroom) — the expected regime for prompt processing."
            case .decode:
                return "Decode appears compute-bound rather than bandwidth-bound — "
                    + "unusual for single-stream decode; large batches or "
                    + "speculative decoding can shift it this way."
            }

        case .bandwidthBound:
            switch phase {
            case .decode:
                let lever = config.usesQuantizedKVCache
                    ? "the KV cache is already quantized, so a lower-bit weight "
                        + "quantization is the remaining lever."
                    : "try a lower-bit KV-cache or weight quantization to raise "
                        + "tokens/sec."
                return "Decode appears memory-bandwidth-bound — the expected "
                    + "regime; " + lever
            case .prefill:
                return "Prefill appears memory-bandwidth-bound — unusual for prompt "
                    + "processing; the model may be streaming weights faster than "
                    + "it computes over them."
            }
        }
    }
}
