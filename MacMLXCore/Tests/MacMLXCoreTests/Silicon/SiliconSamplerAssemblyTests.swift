// Copyright © 2026 macMLX. English comments only.

import Foundation
import Testing

@testable import MacMLXCore

/// Coverage for `IOReportSiliconSampler.assemble` — the pure row → `SiliconSample`
/// mapping — plus the `IOReportFormat` code mapping. Runs synthetic channel rows so it
/// needs no live IOReport.
struct SiliconSamplerAssemblyTests {

    private func stateRow(
        group: String, subgroup: String, channel: String,
        _ states: [IOReportChannelSample.StateResidency]
    ) -> IOReportChannelSample {
        IOReportChannelSample(
            group: group, subgroup: subgroup, channel: channel,
            format: .state, simpleValue: nil, states: states
        )
    }

    private func energyRow(_ channel: String, _ millijoules: Int64) -> IOReportChannelSample {
        IOReportChannelSample(
            group: "Energy Model", subgroup: "", channel: channel,
            format: .simple, simpleValue: millijoules, states: nil
        )
    }

    @Test
    func assemblesAllMetricFamiliesFromRows() {
        let rows: [IOReportChannelSample] = [
            energyRow("CPU Energy", 4000),   // 4000 mJ / 2 s → 2 W
            energyRow("GPU", 2000),          // → 1 W
            energyRow("ANE", 0),             // → 0 W (idle proxy)
            energyRow("DRAM", 1000),         // → 0.5 W
            stateRow(
                group: "GPU Stats", subgroup: "GPU Performance States", channel: "GPUPH",
                [.init(name: "OFF", residency: 50), .init(name: "P1", residency: 50)]
            ),
            stateRow(
                group: "PMP0", subgroup: "DCS BW", channel: "AMCC RD+WR",
                [.init(name: "32GB/s", residency: 10)]  // midpoint (0,32] = 16
            ),
            stateRow(
                group: "PMP0", subgroup: "DCS BW", channel: "AGX RD+WR",
                [.init(name: "1GB/s", residency: 0), .init(name: "2GB/s", residency: 4)]
            ),
            stateRow(
                group: "PMP0", subgroup: "DCS BW", channel: "ANE L0 RD+WR",
                [.init(name: "1GB/s", residency: 4)]  // midpoint (0,1] = 0.5
            ),
            stateRow(
                group: "PMP0", subgroup: "DCS BW", channel: "ANE L1 RD+WR",
                [.init(name: "1GB/s", residency: 4)]  // midpoint 0.5 → combined 1.0
            ),
        ]

        let sample = IOReportSiliconSampler.assemble(
            rows: rows,
            intervalSeconds: 2.0,
            timestamp: Date(timeIntervalSince1970: 0),
            thermal: .fair,
            memoryPressure: nil
        )

        #expect(sample.intervalSeconds == 2.0)
        #expect(sample.ioReportAvailable == true)
        #expect(sample.ioReportUnavailableReason == nil)
        #expect(sample.thermalPressure == .fair)

        #expect(sample.gpuUtilization?.busyFraction == 0.5)
        #expect(sample.dramBandwidth?.estimatedGBPerSecond == 16)
        #expect(sample.gpuBandwidth?.estimatedGBPerSecond == 1.5)  // midpoint (1,2]
        #expect(sample.aneBandwidth?.estimatedGBPerSecond == 1.0)  // 0.5 + 0.5

        #expect(sample.cpuPower?.watts == 2.0)
        #expect(sample.gpuPower?.watts == 1.0)
        #expect(sample.anePower?.watts == 0.0)
        #expect(sample.dramPower?.watts == 0.5)
    }

    @Test
    func missingChannelsBecomeNilFields() {
        // Only a GPU utilisation row present; every other family absent → nil.
        let rows = [
            stateRow(
                group: "GPU Stats", subgroup: "GPU Performance States", channel: "GPUPH",
                [.init(name: "OFF", residency: 100)]
            )
        ]
        let sample = IOReportSiliconSampler.assemble(
            rows: rows, intervalSeconds: 1.0, timestamp: Date(),
            thermal: .nominal, memoryPressure: nil
        )
        #expect(sample.gpuUtilization?.busyFraction == 0)
        #expect(sample.dramBandwidth == nil)
        #expect(sample.gpuBandwidth == nil)
        #expect(sample.aneBandwidth == nil)
        #expect(sample.cpuPower == nil)
        #expect(sample.gpuPower == nil)
    }

    @Test
    func powerIsNilWhenIntervalIsZero() {
        // Even with an energy row, a zero interval yields no power (no window).
        let rows = [energyRow("GPU", 2000)]
        let sample = IOReportSiliconSampler.assemble(
            rows: rows, intervalSeconds: 0, timestamp: Date(),
            thermal: .nominal, memoryPressure: nil
        )
        #expect(sample.gpuPower == nil)
    }

    @Test
    func ioReportFormatMapsRawCodes() {
        #expect(IOReportFormat(rawFormat: 0) == .invalid)
        #expect(IOReportFormat(rawFormat: 1) == .simple)
        #expect(IOReportFormat(rawFormat: 2) == .state)
        #expect(IOReportFormat(rawFormat: 3) == .histogram)
        #expect(IOReportFormat(rawFormat: 4) == .simpleArray)
        #expect(IOReportFormat(rawFormat: 99) == .other)
    }
}
