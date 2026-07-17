// Copyright © 2026 macMLX. English comments only.

import Foundation

/// Production `SiliconSampler`: reads IOReport for the private-API metrics and public
/// kernel APIs for thermal / memory pressure, folding both into a `SiliconSample`.
///
/// An `actor`, which gives it `Sendable` conformance and serialises access to the
/// non-`Sendable` `IOReportReader` it owns. The reader is created lazily on the first
/// `sample()` so merely constructing a sampler never opens a subscription.
///
/// Selection of which channel becomes which metric lives here (the reader stays a
/// generic marshaller); the actual number-crunching is delegated to the pure
/// estimators so it can be tested without a live machine.
public actor IOReportSiliconSampler: SiliconSampler {

    private var reader: IOReportReader?
    private var didCreateReader = false

    public init() {}

    public func sample() async -> SiliconSample {
        let now = Date()

        // Public-API metrics are always available and instantaneous.
        let thermal = ThermalPressure.current()
        let memoryPressure = MemoryPressureSample.current()

        let reader = ensureReader()
        guard reader.isAvailable else {
            return SiliconSample(
                timestamp: now,
                intervalSeconds: 0,
                ioReportAvailable: false,
                ioReportUnavailableReason: reader.unavailableReason,
                gpuUtilization: nil,
                dramBandwidth: nil,
                gpuBandwidth: nil,
                aneBandwidth: nil,
                cpuPower: nil,
                gpuPower: nil,
                anePower: nil,
                dramPower: nil,
                thermalPressure: thermal,
                memoryPressure: memoryPressure
            )
        }

        guard let (interval, rows) = reader.poll() else {
            // Transient sample miss: report only the always-available public metrics.
            return SiliconSample(
                timestamp: now,
                intervalSeconds: 0,
                ioReportAvailable: true,
                ioReportUnavailableReason: nil,
                gpuUtilization: nil,
                dramBandwidth: nil,
                gpuBandwidth: nil,
                aneBandwidth: nil,
                cpuPower: nil,
                gpuPower: nil,
                anePower: nil,
                dramPower: nil,
                thermalPressure: thermal,
                memoryPressure: memoryPressure
            )
        }

        return Self.assemble(
            rows: rows,
            intervalSeconds: interval,
            timestamp: now,
            thermal: thermal,
            memoryPressure: memoryPressure
        )
    }

    private func ensureReader() -> IOReportReader {
        if let reader { return reader }
        let created = IOReportReader()
        reader = created
        didCreateReader = true
        return created
    }

    // MARK: - Row → sample mapping (pure; the estimators do the math)

    /// Fold a poll's channel deltas into a `SiliconSample`. Static and pure so it can
    /// be exercised in tests with synthetic rows, no live IOReport required.
    static func assemble(
        rows: [IOReportChannelSample],
        intervalSeconds: Double,
        timestamp: Date,
        thermal: ThermalPressure,
        memoryPressure: MemoryPressureSample?
    ) -> SiliconSample {
        // Index the rows we care about by (group, subgroup, channel).
        func states(group: String, subgroup: String, channel: String)
            -> [IOReportChannelSample.StateResidency]?
        {
            rows.first {
                $0.group == group && $0.subgroup == subgroup && $0.channel == channel
            }?.states
        }
        func energyMillijoules(_ channel: String) -> Int64? {
            rows.first { $0.group == "Energy Model" && $0.channel == channel }?.simpleValue
        }
        func power(_ channel: String) -> PowerSample? {
            guard let mj = energyMillijoules(channel) else { return nil }
            return PowerSample.from(energyMillijoules: mj, intervalSeconds: intervalSeconds)
        }

        let gpuUtilization = GPUUtilizationSample.compute(
            states: states(
                group: "GPU Stats", subgroup: "GPU Performance States", channel: "GPUPH"
            )
        )
        let dramBandwidth = MemoryBandwidthSample.estimate(
            states: states(group: "PMP0", subgroup: "DCS BW", channel: "AMCC RD+WR")
        )
        let gpuBandwidth = MemoryBandwidthSample.estimate(
            states: states(group: "PMP0", subgroup: "DCS BW", channel: "AGX RD+WR")
        )
        let aneBandwidth = combinedANEBandwidth(rows: rows)

        return SiliconSample(
            timestamp: timestamp,
            intervalSeconds: intervalSeconds,
            ioReportAvailable: true,
            ioReportUnavailableReason: nil,
            gpuUtilization: gpuUtilization,
            dramBandwidth: dramBandwidth,
            gpuBandwidth: gpuBandwidth,
            aneBandwidth: aneBandwidth,
            cpuPower: power("CPU Energy"),
            gpuPower: power("GPU"),
            anePower: power("ANE"),
            dramPower: power("DRAM"),
            thermalPressure: thermal,
            memoryPressure: memoryPressure
        )
    }

    /// The ANE is split across two DRAM ports (`ANE L0 RD+WR`, `ANE L1 RD+WR`); sum
    /// their estimated bandwidths. Returns `nil` when neither port is present, and is
    /// saturated if either port is.
    private static func combinedANEBandwidth(rows: [IOReportChannelSample])
        -> MemoryBandwidthSample?
    {
        func estimate(_ channel: String) -> MemoryBandwidthSample? {
            let match = rows.first {
                $0.group == "PMP0" && $0.subgroup == "DCS BW" && $0.channel == channel
            }
            return MemoryBandwidthSample.estimate(states: match?.states)
        }
        let ports = [estimate("ANE L0 RD+WR"), estimate("ANE L1 RD+WR")].compactMap { $0 }
        guard !ports.isEmpty else { return nil }
        let total = ports.reduce(0.0) { $0 + $1.estimatedGBPerSecond }
        let saturated = ports.contains { $0.isSaturated }
        return MemoryBandwidthSample(estimatedGBPerSecond: total, isSaturated: saturated)
    }
}
