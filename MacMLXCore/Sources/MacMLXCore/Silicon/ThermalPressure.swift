// Copyright © 2026 macMLX. English comments only.

import Foundation

/// Thermal pressure on the machine, read from the fully public
/// `ProcessInfo.thermalState`.
///
/// No IOReport, no private API, no elevated privileges — this is the sanctioned way
/// to observe throttling risk. A rising level means the OS is (or is about to be)
/// capping clocks, which is a first-class input to W2's "why is decode slow?"
/// reasoning: a `serious`/`critical` reading explains a throughput drop that raw GPU
/// occupancy alone would not.
///
/// Ordered so comparisons work: `.nominal < .fair < .serious < .critical`.
///
/// `Codable` is additive (the `Int` raw value already gives a stable encoded
/// form) so a benchmark's bottleneck attribution can persist a representative
/// thermal reading — see `BenchmarkBottleneck`.
public enum ThermalPressure: Int, Sendable, Equatable, Comparable, CaseIterable, Codable {
    /// No thermal pressure.
    case nominal = 0
    /// Mild pressure; fans ramping, no throttling yet.
    case fair = 1
    /// The system is shedding performance to cool down.
    case serious = 2
    /// Aggressive throttling to prevent shutdown.
    case critical = 3

    public static func < (lhs: ThermalPressure, rhs: ThermalPressure) -> Bool {
        lhs.rawValue < rhs.rawValue
    }

    /// Map a `ProcessInfo.ThermalState` to this type. `@unknown` future states degrade
    /// to `.nominal` so a new OS level never crashes the sampler.
    public static func from(_ state: ProcessInfo.ThermalState) -> ThermalPressure {
        switch state {
        case .nominal: return .nominal
        case .fair: return .fair
        case .serious: return .serious
        case .critical: return .critical
        @unknown default: return .nominal
        }
    }

    /// Current thermal pressure for this machine.
    public static func current() -> ThermalPressure {
        from(ProcessInfo.processInfo.thermalState)
    }
}
