// Copyright © 2026 macMLX. English comments only.

/// An estimated memory-bandwidth reading for one requestor, in GB/s.
///
/// WHY THIS IS AN ESTIMATE, NOT A MEASUREMENT
/// ------------------------------------------
/// On the classic Apple Silicon path, DRAM bandwidth was a `Simple` byte counter you
/// could divide by the interval for an exact GB/s. That path is gone on recent macOS
/// (empirically absent on M4/M5, macOS 26.x): the only bandwidth signal IOReport
/// exposes is the `PMP0 / DCS BW` group, a **residency histogram**. Each requestor
/// channel (`AMCC RD+WR`, `AGX RD+WR`, …) reports, per window, the fraction of time
/// its instantaneous bandwidth sat in each of 32 bands labelled `"1GB/s"…"32GB/s"`
/// (or, for the `AMCC` aggregate, coarse `"32/64/96/128GB/s"` bands).
///
/// We convert that to an average by weighting each band's **midpoint** by its
/// residency fraction. This is deliberately the midpoint, not the label: a band
/// labelled `"32GB/s"` covers `(previousEdge, 32]`, so for the coarse AMCC aggregate
/// its representative value is 16 GB/s, not 32 — using the label would report 32 GB/s
/// at idle. Two honesty caveats travel with every reading:
///   * `isEstimated` is always true — the value is a residency-weighted average, and
///     for the coarse AMCC aggregate the quantum is ±16 GB/s, so sub-32 GB/s traffic
///     cannot be resolved. Per-requestor channels (`AGX`, `MACC`, `PACC`) use 1 GB/s
///     bands and are far finer.
///   * `isSaturated` warns that residency piled into the top band, whose upper edge is
///     open — true bandwidth may exceed the estimate.
public struct MemoryBandwidthSample: Sendable, Equatable {

    /// Residency-weighted average bandwidth over the window, in GB/s (10^9 B/s).
    public let estimatedGBPerSecond: Double
    /// True when the highest band carried non-trivial residency, so the real value
    /// may be higher than reported (the top band's upper edge is unbounded).
    public let isSaturated: Bool

    /// Always true — see the type doc. Present as a field so callers and UI can flag
    /// the value as an estimate without special-casing this type.
    public var isEstimated: Bool { true }

    public init(estimatedGBPerSecond: Double, isSaturated: Bool) {
        self.estimatedGBPerSecond = estimatedGBPerSecond
        self.isSaturated = isSaturated
    }

    /// Fraction of the top band's residency above which a reading is flagged saturated.
    private static let saturationThreshold = 0.01

    /// Parse an IOReport bandwidth band label such as `"  32GB/s"` into its GB/s value.
    ///
    /// Tolerates leading whitespace (IOReport right-pads these) and is case
    /// insensitive. Returns `nil` for any label that is not `<number>GB/s`, so a
    /// non-bandwidth state mixed into a row is ignored rather than miscounted.
    public static func parseBandwidthLabelGBs(_ name: String) -> Double? {
        let trimmed = name.trimmingCharacters(in: .whitespaces)
        guard let unitRange = trimmed.range(of: "GB/s", options: .caseInsensitive) else {
            return nil
        }
        let numberPart = trimmed[trimmed.startIndex..<unitRange.lowerBound]
        return Double(numberPart)
    }

    /// Estimate average bandwidth from parsed band edges and their residencies.
    ///
    /// `upperEdgesGBs` are the band labels in ascending order (the IOReport index
    /// order); `residencies` is index-aligned. The lower edge of band `i` is band
    /// `i-1`'s label (0 for the first band), and each band contributes its midpoint
    /// weighted by its residency share.
    ///
    /// Returns `nil` when the arrays are empty or mismatched. An all-zero residency
    /// row (an idle requestor) yields 0 GB/s rather than `nil`, since "idle" is a real
    /// reading, distinct from "no data".
    public static func estimate(
        upperEdgesGBs: [Double],
        residencies: [UInt64]
    ) -> MemoryBandwidthSample? {
        guard !upperEdgesGBs.isEmpty, upperEdgesGBs.count == residencies.count else {
            return nil
        }
        let total = residencies.reduce(UInt64(0), &+)
        guard total > 0 else {
            return MemoryBandwidthSample(estimatedGBPerSecond: 0, isSaturated: false)
        }

        var weighted = 0.0
        var lowerEdge = 0.0
        for index in upperEdgesGBs.indices {
            let upperEdge = upperEdgesGBs[index]
            let midpoint = (lowerEdge + upperEdge) / 2
            let share = Double(residencies[index]) / Double(total)
            weighted += midpoint * share
            lowerEdge = upperEdge
        }

        let topShare = Double(residencies[residencies.count - 1]) / Double(total)
        return MemoryBandwidthSample(
            estimatedGBPerSecond: weighted,
            isSaturated: topShare > saturationThreshold
        )
    }

    /// Estimate from raw IOReport state residencies, parsing the band labels.
    ///
    /// States whose names are not bandwidth labels are dropped (they do not belong to
    /// a bandwidth row). Returns `nil` when no band parses — the caller then treats
    /// this requestor's bandwidth as unavailable.
    public static func estimate(states: [IOReportChannelSample.StateResidency]?)
        -> MemoryBandwidthSample?
    {
        guard let states else { return nil }
        var edges: [Double] = []
        var residencies: [UInt64] = []
        for state in states {
            guard let gbs = parseBandwidthLabelGBs(state.name) else { continue }
            edges.append(gbs)
            residencies.append(state.residency)
        }
        return estimate(upperEdgesGBs: edges, residencies: residencies)
    }
}
