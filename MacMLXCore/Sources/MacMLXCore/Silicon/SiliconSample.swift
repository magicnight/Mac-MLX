// Copyright © 2026 macMLX. English comments only.

import Foundation

/// One consistent snapshot of Apple Silicon hardware metrics for a sampling window.
///
/// This is the value the whole W1 layer produces and W2's bottleneck classifier will
/// consume. It is a pure, `Sendable` value type: no live CoreFoundation handles, safe
/// to hand across actors, store, diff, or fabricate in a test.
///
/// Fields sourced from IOReport (`gpuUtilization`, the bandwidth trio, the power
/// quartet) are optional and are `nil` when IOReport is unavailable or a channel is
/// absent on this chip — check `ioReportAvailable` / `ioReportUnavailableReason`.
/// Thermal pressure comes from a public API and is always present; memory pressure is
/// public and effectively always present.
///
/// HONESTY NOTES BAKED INTO THE SHAPE:
///   * `anePower` is a power PROXY. The ANE exposes no occupancy counter, so there is
///     deliberately no `aneUtilization` field — do not synthesise one.
///   * the bandwidth fields are residency-weighted estimates (`MemoryBandwidthSample`
///     carries `isEstimated` / `isSaturated`); the coarse `dramBandwidth` aggregate in
///     particular cannot resolve traffic below ~32 GB/s.
///   * Media Engine (ProRes/codec) traffic is intentionally omitted: IOReport exposes
///     its bandwidth consumption but no duty cycle, so there is nothing honest to
///     report as "utilisation" here in W1.
public struct SiliconSample: Sendable, Equatable {

    /// When this snapshot was produced.
    public let timestamp: Date
    /// Wall-clock seconds the deltas cover; 0 for the first sample (no prior window).
    public let intervalSeconds: Double

    /// Whether IOReport resolved; drives whether the private-API fields can be present.
    public let ioReportAvailable: Bool
    /// Human-readable reason IOReport was unavailable, else `nil`.
    public let ioReportUnavailableReason: String?

    /// Real GPU occupancy (see `GPUUtilizationSample`).
    public let gpuUtilization: GPUUtilizationSample?
    /// Total DRAM-controller bandwidth (AMCC aggregate; coarse — see the type doc).
    public let dramBandwidth: MemoryBandwidthSample?
    /// GPU (AGX) DRAM bandwidth, 1 GB/s-band resolution.
    public let gpuBandwidth: MemoryBandwidthSample?
    /// Neural Engine DRAM bandwidth (ANE L0 + L1 combined).
    public let aneBandwidth: MemoryBandwidthSample?

    /// CPU package power.
    public let cpuPower: PowerSample?
    /// GPU power.
    public let gpuPower: PowerSample?
    /// Neural Engine power — a PROXY for ANE activity, not a utilisation figure.
    public let anePower: PowerSample?
    /// DRAM power.
    public let dramPower: PowerSample?

    /// Thermal pressure (public API; always present).
    public let thermalPressure: ThermalPressure
    /// Unified-memory pressure (public API; `nil` only on kernel read failure).
    public let memoryPressure: MemoryPressureSample?

    public init(
        timestamp: Date,
        intervalSeconds: Double,
        ioReportAvailable: Bool,
        ioReportUnavailableReason: String?,
        gpuUtilization: GPUUtilizationSample?,
        dramBandwidth: MemoryBandwidthSample?,
        gpuBandwidth: MemoryBandwidthSample?,
        aneBandwidth: MemoryBandwidthSample?,
        cpuPower: PowerSample?,
        gpuPower: PowerSample?,
        anePower: PowerSample?,
        dramPower: PowerSample?,
        thermalPressure: ThermalPressure,
        memoryPressure: MemoryPressureSample?
    ) {
        self.timestamp = timestamp
        self.intervalSeconds = intervalSeconds
        self.ioReportAvailable = ioReportAvailable
        self.ioReportUnavailableReason = ioReportUnavailableReason
        self.gpuUtilization = gpuUtilization
        self.dramBandwidth = dramBandwidth
        self.gpuBandwidth = gpuBandwidth
        self.aneBandwidth = aneBandwidth
        self.cpuPower = cpuPower
        self.gpuPower = gpuPower
        self.anePower = anePower
        self.dramPower = dramPower
        self.thermalPressure = thermalPressure
        self.memoryPressure = memoryPressure
    }
}
