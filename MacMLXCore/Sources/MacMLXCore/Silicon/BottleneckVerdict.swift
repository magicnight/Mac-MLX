// Copyright © 2026 macMLX. English comments only.

/// The bottleneck classifier's answer for one sample: what is limiting
/// inference right now, in which phase, and what (if anything) to do about it.
///
/// A pure `Sendable` value — the classifier produces it, and the GUI panel (W3)
/// or any consumer renders it. It deliberately carries an `advice` string rather
/// than leaving the consumer to map category → text, so the phase-aware,
/// honesty-calibrated wording lives in one place next to the logic that decided
/// it.
public struct BottleneckVerdict: Sendable, Equatable {

    /// What is limiting throughput.
    ///
    /// The `String` raw values are pinned explicitly (not left to derive from the
    /// case names) precisely because this is a persisted form — a benchmark's
    /// bottleneck attribution stores a category (see `BenchmarkBottleneck`). Pinning
    /// them means a future case rename cannot silently change the on-disk value and
    /// orphan old history. Keep these strings stable; nothing switches on them.
    public enum Category: String, Sendable, Equatable, CaseIterable, Codable {
        /// No single resource is pinned; nothing to act on.
        case normal = "normal"
        /// Unified memory is under pressure — the highest-priority category,
        /// because it precedes an out-of-memory or a compression/swap stall that
        /// dwarfs any compute/bandwidth nuance.
        case memoryBound = "memoryBound"
        /// The OS is capping clocks to shed heat, so throughput is limited by
        /// thermals rather than by the workload's own resource mix.
        case thermalThrottled = "thermalThrottled"
        /// GPU math is the limiter: the GPU is pinned and the memory bus has
        /// headroom. The healthy, expected state for prefill.
        case computeBound = "computeBound"
        /// Memory bandwidth is the limiter: the GPU is pinned AND the memory bus
        /// is near its achievable ceiling. The healthy, expected state for decode.
        case bandwidthBound = "bandwidthBound"
    }

    /// The limiting resource.
    public let category: Category

    /// The generation phase this verdict describes.
    public let phase: InferencePhase

    /// A short, phase-aware, actionable recommendation. Honest about
    /// uncertainty: when the verdict rests on the estimated bandwidth signal
    /// (see `restsOnEstimatedBandwidth`) the wording hedges ("appears", "likely")
    /// rather than asserting a measured fact.
    public let advice: String

    /// True when the category was decided using the residency-estimated memory
    /// bandwidth (the compute-vs-bandwidth distinction), which is an estimate,
    /// not a measurement — so a consumer can render the verdict with an
    /// appropriate confidence cue. False for memory/thermal verdicts, which come
    /// from authoritative kernel/OS APIs, and for `normal`.
    public let restsOnEstimatedBandwidth: Bool

    public init(
        category: Category,
        phase: InferencePhase,
        advice: String,
        restsOnEstimatedBandwidth: Bool
    ) {
        self.category = category
        self.phase = phase
        self.advice = advice
        self.restsOnEstimatedBandwidth = restsOnEstimatedBandwidth
    }
}
