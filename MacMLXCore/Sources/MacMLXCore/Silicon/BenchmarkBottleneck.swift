// Copyright © 2026 macMLX. English comments only.

/// The bottleneck attribution for one benchmark run: what limited this run's
/// decode steady state, how confident that call is, and the representative
/// hardware readings behind it.
///
/// WHY THIS EXISTS (the moat)
/// -------------------------
/// A benchmark reports tokens/second; an external monitor can watch the hardware.
/// Neither alone can say *why* a given run was as fast as it was, because the
/// attribution needs the engine's own prefill/decode phase timeline fused with
/// the silicon samples — a same-process signal only the in-process engine has.
/// This value is the fused answer, produced by folding the per-frame
/// `BottleneckVerdict`s the W2 classifier emits *during the run* (see
/// `BenchmarkBottleneckAggregator`) and attached to the `BenchmarkResult`.
///
/// HONESTY, BAKED INTO THE SHAPE (matching the W1/W2/W3 discipline):
///   * `phase` is always `.decode`. tokens/second measures the decode steady
///     state, and decode is normally bandwidth-bound while prefill is normally
///     compute-bound; folding prefill frames in would misattribute the run. The
///     aggregator only ever counts decode frames, so this is `.decode` by
///     construction — carried explicitly rather than assumed, and ready for a
///     future prefill attribution without a shape change.
///   * `confidence` is the share of decode frames that agreed with the dominant
///     `category` (0…1). A run that stayed in one regime reads ~1.0; a run that
///     drifted between regimes reads lower, and the UI is expected to render that
///     uncertainty rather than assert a shaky call.
///   * `restsOnEstimatedBandwidth` forwards the dominant verdict's own honesty
///     flag: `true` when the compute-vs-bandwidth split rested on the residency-
///     *estimated* bandwidth signal (not a measurement), so the UI can mark it
///     "estimated". `false` for memory/thermal verdicts, which come from
///     authoritative kernel/OS APIs.
///   * the hardware readouts are optional per field — GPU occupancy and bandwidth
///     come from IOReport (absent when it is unavailable) while thermal pressure
///     is a public API (effectively always present). A `nil` field means "not
///     sampled", never a fabricated zero.
///
/// When a run produces no usable decode frames at all (too short to clear the
/// classifier's warm-up, or no sampling), the aggregator returns `nil` and no
/// `BenchmarkBottleneck` is attached — the UI then says attribution is
/// unavailable rather than inventing one.
public struct BenchmarkBottleneck: Sendable, Codable, Hashable {

    /// The dominant limiting resource across the run's decode frames.
    public let category: BottleneckVerdict.Category

    /// The phase this attribution describes. Always `.decode` today (see the type
    /// doc), carried explicitly for honesty and forward compatibility.
    public let phase: InferencePhase

    /// The dominant verdict's phase-aware, actionable recommendation, carried
    /// verbatim so the wording (and its hedging) lives in one place — the W2
    /// classifier — rather than being re-derived here.
    public let advice: String

    /// Share of decode frames that agreed with `category`, 0…1. `1.0` means every
    /// decode frame attributed the run to the same resource.
    public let confidence: Double

    /// True when `category` rests on the estimated bandwidth signal rather than a
    /// measured value — forwarded from the dominant verdict so the UI can flag it.
    public let restsOnEstimatedBandwidth: Bool

    /// How many decode-phase frames informed this attribution. Context for
    /// `confidence`: a high confidence over 2 frames is weaker evidence than the
    /// same confidence over 30.
    public let decodeFrameCount: Int

    /// Representative decode-phase hardware readings behind the attribution.
    public let hardware: Readouts

    public init(
        category: BottleneckVerdict.Category,
        phase: InferencePhase,
        advice: String,
        confidence: Double,
        restsOnEstimatedBandwidth: Bool,
        decodeFrameCount: Int,
        hardware: Readouts
    ) {
        self.category = category
        self.phase = phase
        self.advice = advice
        self.confidence = confidence
        self.restsOnEstimatedBandwidth = restsOnEstimatedBandwidth
        self.decodeFrameCount = decodeFrameCount
        self.hardware = hardware
    }

    /// The decode-phase hardware summary behind a `BenchmarkBottleneck`.
    ///
    /// Peaks and means are taken over the run's decode frames only (the steady
    /// state tokens/second measures), so they characterise the regime the
    /// attribution describes. Every field is optional: GPU occupancy and bandwidth
    /// are IOReport-derived and `nil` when it is unavailable; thermal pressure is a
    /// public API. A `nil` is "not sampled", not a zero.
    public struct Readouts: Sendable, Codable, Hashable {

        /// Highest GPU busy-fraction (occupancy) seen across decode frames, 0…1.
        public let peakGPUUtilization: Double?

        /// Mean GPU busy-fraction across decode frames, 0…1.
        public let meanGPUUtilization: Double?

        /// Highest estimated memory bandwidth seen across decode frames, GB/s
        /// (the GPU/AGX requestor when present, else the coarse DRAM aggregate).
        public let peakBandwidthGBs: Double?

        /// Mean estimated memory bandwidth across decode frames, GB/s.
        public let meanBandwidthGBs: Double?

        /// Highest thermal pressure observed across decode frames.
        public let peakThermalPressure: ThermalPressure?

        public init(
            peakGPUUtilization: Double?,
            meanGPUUtilization: Double?,
            peakBandwidthGBs: Double?,
            meanBandwidthGBs: Double?,
            peakThermalPressure: ThermalPressure?
        ) {
            self.peakGPUUtilization = peakGPUUtilization
            self.meanGPUUtilization = meanGPUUtilization
            self.peakBandwidthGBs = peakBandwidthGBs
            self.meanBandwidthGBs = meanBandwidthGBs
            self.peakThermalPressure = peakThermalPressure
        }
    }
}
