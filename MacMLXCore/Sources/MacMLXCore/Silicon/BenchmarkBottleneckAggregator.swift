// Copyright © 2026 macMLX. English comments only.

/// Folds the stream of per-frame `BottleneckVerdict`s produced *during* a
/// benchmark run into one dominant `BenchmarkBottleneck` attribution.
///
/// This is the pure logic behind the benchmark's "what limited this run" readout:
/// feed it the (verdict, hardware sample) pairs the W3 monitor publishes while the
/// run generates, and read the aggregated result at the end. No timers, no live
/// engine, no IOReport — so it is fully unit-testable with synthetic sequences,
/// and the app-layer collector that drives it adds no logic of its own.
///
/// DESIGN
/// ------
///   * **Decode-only.** Only `.decode` frames are counted. tokens/second measures
///     the decode steady state, and prefill is normally compute-bound while decode
///     is normally bandwidth-bound; counting prefill frames would drag the
///     attribution toward compute-bound for reasons that have nothing to do with
///     the number the benchmark reports. Prefill frames are silently ignored.
///   * **Dominant by frequency.** The attributed category is the one the most
///     decode frames agreed on. `confidence` is that category's share of all
///     decode frames, so a run that stayed in one regime reads ~1.0 and a run that
///     drifted reads lower.
///   * **Deterministic tie-break.** When two categories tie on frame count, the
///     more severe/actionable one wins (memory > thermal > bandwidth > compute >
///     normal) so the result never depends on dictionary ordering and surfaces the
///     more useful signal.
///   * **Advice from the verdict.** The dominant category's advice string is
///     carried verbatim from a decode verdict of that category — the phase- and
///     config-aware wording lives in the W2 classifier, not here.
///   * **Representative hardware.** Peaks and means of GPU occupancy and bandwidth,
///     and the peak thermal pressure, are taken over the decode frames so the UI
///     can show what the silicon was doing during the measured phase. Bandwidth is
///     kept to a single channel (GPU/AGX once it appears, else the DRAM aggregate),
///     mirroring the classifier, so peak/mean never blend incommensurable channels.
///     Missing IOReport fields are skipped, never zero-filled.
public struct BenchmarkBottleneckAggregator: Sendable {

    /// Decode-frame count per category.
    private var counts: [BottleneckVerdict.Category: Int] = [:]
    /// Total decode frames ingested (the denominator for `confidence`).
    private var decodeFrameCount = 0

    /// The most recent advice seen for each category. Advice is deterministic for
    /// a given (category, phase, config), so last-wins is a faithful representative.
    private var adviceByCategory: [BottleneckVerdict.Category: String] = [:]
    /// The most recent `restsOnEstimatedBandwidth` flag seen per category.
    private var estimateByCategory: [BottleneckVerdict.Category: Bool] = [:]

    /// Decode-frame GPU busy-fractions, for peak/mean.
    private var gpuBusyValues: [Double] = []
    /// Once the GPU (AGX) bandwidth channel has appeared among the decode frames we
    /// commit to it and never fold the coarse DRAM aggregate in again — the two are
    /// different physical quantities (AGX is GPU-only; the DRAM aggregate covers
    /// CPU+GPU+ANE traffic and is structurally larger), so averaging them would be
    /// meaningless. Mirrors `BottleneckClassifier`'s single-channel discipline, so the
    /// representative readout is drawn from the same channel the verdicts were.
    private var bandwidthUsesGPUChannel = false
    /// Decode-frame bandwidth on the GPU (AGX) channel, used once committed.
    private var gpuBandwidthValues: [Double] = []
    /// Decode-frame bandwidth on the coarse DRAM aggregate, used only while — and
    /// only if — the GPU channel never appears.
    private var dramBandwidthValues: [Double] = []
    /// Highest thermal pressure across decode frames.
    private var peakThermal: ThermalPressure?

    public init() {}

    /// Ingest one frame: a classifier verdict and the hardware sample it was drawn
    /// from. Prefill (and any non-decode) frames are ignored; only decode frames
    /// contribute to the attribution.
    public mutating func add(verdict: BottleneckVerdict, sample: SiliconSample) {
        guard verdict.phase == .decode else { return }

        decodeFrameCount += 1
        counts[verdict.category, default: 0] += 1
        adviceByCategory[verdict.category] = verdict.advice
        estimateByCategory[verdict.category] = verdict.restsOnEstimatedBandwidth

        if let busy = sample.gpuUtilization?.busyFraction {
            gpuBusyValues.append(busy)
        }
        // Commit to the GPU (AGX) channel the first time it appears and stay there —
        // never averaging it with the incommensurable DRAM aggregate. Before any AGX
        // reading the DRAM aggregate is the only signal; once AGX appears, a frame
        // lacking it is simply "no reading" on this channel, not a DRAM fallback.
        if let gpu = sample.gpuBandwidth?.estimatedGBPerSecond {
            bandwidthUsesGPUChannel = true
            gpuBandwidthValues.append(gpu)
        } else if !bandwidthUsesGPUChannel,
                  let dram = sample.dramBandwidth?.estimatedGBPerSecond {
            dramBandwidthValues.append(dram)
        }
        peakThermal = peakThermal.map { Swift.max($0, sample.thermalPressure) }
            ?? sample.thermalPressure
    }

    /// The dominant attribution, or `nil` when no decode frames were ingested (too
    /// short a run, or no sampling) — the caller then honestly reports that
    /// attribution is unavailable rather than inventing one.
    public func result() -> BenchmarkBottleneck? {
        guard decodeFrameCount > 0, let dominant = dominantCategory() else {
            return nil
        }
        let agreeing = counts[dominant] ?? 0
        // One committed channel: the GPU (AGX) values once it ever appeared, else the
        // DRAM aggregate. Never a blend of the two.
        let bandwidthValues = bandwidthUsesGPUChannel ? gpuBandwidthValues : dramBandwidthValues
        let readouts = BenchmarkBottleneck.Readouts(
            peakGPUUtilization: gpuBusyValues.max(),
            meanGPUUtilization: Self.mean(gpuBusyValues),
            peakBandwidthGBs: bandwidthValues.max(),
            meanBandwidthGBs: Self.mean(bandwidthValues),
            peakThermalPressure: peakThermal
        )
        return BenchmarkBottleneck(
            category: dominant,
            phase: .decode,
            advice: adviceByCategory[dominant] ?? "",
            confidence: Double(agreeing) / Double(decodeFrameCount),
            restsOnEstimatedBandwidth: estimateByCategory[dominant] ?? false,
            decodeFrameCount: decodeFrameCount,
            hardware: readouts
        )
    }

    // MARK: - Private

    /// The category with the most decode frames. Ties are broken by descending
    /// severity so the result is deterministic and surfaces the more actionable
    /// signal.
    private func dominantCategory() -> BottleneckVerdict.Category? {
        counts.max { lhs, rhs in
            if lhs.value != rhs.value { return lhs.value < rhs.value }
            return Self.severity(lhs.key) < Self.severity(rhs.key)
        }?.key
    }

    /// Tie-break priority: memory > thermal > bandwidth > compute > normal. Higher
    /// means more severe/actionable.
    private static func severity(_ category: BottleneckVerdict.Category) -> Int {
        switch category {
        case .memoryBound: return 4
        case .thermalThrottled: return 3
        case .bandwidthBound: return 2
        case .computeBound: return 1
        case .normal: return 0
        }
    }

    private static func mean(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        return values.reduce(0, +) / Double(values.count)
    }
}
