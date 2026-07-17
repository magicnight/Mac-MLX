// Copyright © 2026 macMLX. English comments only.

/// GPU utilisation for a window, derived from the `GPU Performance States` residency.
///
/// UNLIKE ANE, THIS IS A REAL UTILISATION FIGURE. The GPU exposes a genuine
/// performance-state residency histogram (`GPU Stats / GPU Performance States /
/// GPUPH`): one idle state (`"OFF"`) plus active DVFS points (`"P1"`, `"P2"`, …).
/// The busy fraction is the share of the window spent in any non-idle state — a true
/// occupancy percentage, not a power proxy. `idleFraction` is its complement (the
/// "residual" the GPU spent powered down).
///
/// Note this is *occupancy*, not throughput: a GPU pinned at `P1` reads as fully busy
/// whether it is saturated or lightly loaded. Pair it with `MemoryBandwidthSample`
/// (AGX) and GPU power for a fuller picture — that fusion is W2's job, not this
/// sampler's.
public struct GPUUtilizationSample: Sendable, Equatable {

    /// Fraction of the window the GPU spent in a non-idle performance state, 0…1.
    public let busyFraction: Double

    /// Fraction of the window the GPU spent idle/off (the "residual"), 0…1.
    public var idleFraction: Double { 1 - busyFraction }

    public init(busyFraction: Double) {
        self.busyFraction = busyFraction
    }

    /// True when a performance-state label denotes the powered-down / idle state.
    ///
    /// `GPUPH` uses `"OFF"`; sibling controllers use `"IDLE_OFF"` / `"DOWN"`. We match
    /// all three so the same logic survives if the fed channel ever changes.
    public static func isIdleStateName(_ name: String) -> Bool {
        let upper = name.uppercased()
        return upper == "OFF" || upper == "DOWN" || upper.contains("IDLE")
    }

    /// Compute utilisation from a performance-state channel's residencies.
    ///
    /// Returns `nil` when there are no states or the window accrued zero residency
    /// (no data), so the caller can report GPU utilisation as unavailable rather than
    /// inventing a 0%.
    public static func compute(states: [IOReportChannelSample.StateResidency]?)
        -> GPUUtilizationSample?
    {
        guard let states, !states.isEmpty else { return nil }
        var total: UInt64 = 0
        var idle: UInt64 = 0
        for state in states {
            total &+= state.residency
            if isIdleStateName(state.name) {
                idle &+= state.residency
            }
        }
        guard total > 0 else { return nil }
        let busy = Double(total - idle) / Double(total)
        return GPUUtilizationSample(busyFraction: busy)
    }
}
