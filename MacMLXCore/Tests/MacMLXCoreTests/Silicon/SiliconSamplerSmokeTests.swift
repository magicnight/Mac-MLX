// Copyright © 2026 macMLX. English comments only.

import Foundation
import XCTest

@testable import MacMLXCore

/// Gated, real-hardware smoke for the IOReport-backed sampler.
///
/// Unlike the pure `Silicon*Tests` suites (synthetic rows, always run), this opens a
/// live IOReport subscription on the machine running it and takes two real samples so
/// the second carries genuine deltas. It needs NO Metal/MLX runtime — only the private
/// IOReport symbols, which resolve on any Apple Silicon Mac.
///
/// GATED — self-skips unless `MACMLX_RUN_SILICON_SMOKE=1`. It is not a parity test; it
/// asserts only that the pipeline produces structurally sane, in-range readings, and
/// prints the observed values for the record (hardware- and load-dependent).
///
/// Run:
///   MACMLX_RUN_SILICON_SMOKE=1 swift test \
///     --filter SiliconSamplerSmokeTests
final class SiliconSamplerSmokeTests: XCTestCase {

    private func requireSmokeEnabled() throws {
        guard ProcessInfo.processInfo.environment["MACMLX_RUN_SILICON_SMOKE"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_SILICON_SMOKE=1 to run the silicon real-hardware smoke")
        }
    }

    func testSamplerProducesLiveReadings() async throws {
        try requireSmokeEnabled()

        let sampler = IOReportSiliconSampler()

        // The sampler captures a baseline when it opens, so the first sample already
        // covers a (short) window. Discard it and take a second over a real interval.
        let first = await sampler.sample()
        guard first.ioReportAvailable else {
            throw XCTSkip("IOReport unavailable: \(first.ioReportUnavailableReason ?? "unknown")")
        }

        // Give the counters a window to accumulate, then take a steady-cadence sample.
        try await Task.sleep(nanoseconds: 400_000_000)
        let sample = await sampler.sample()

        XCTAssertTrue(sample.ioReportAvailable)
        XCTAssertGreaterThan(sample.intervalSeconds, 0, "second sample must cover a window")

        // Thermal + memory pressure are public APIs and must always be present.
        XCTAssertTrue(ThermalPressure.allCases.contains(sample.thermalPressure))
        let memory = try XCTUnwrap(sample.memoryPressure, "memory pressure should read")
        XCTAssertGreaterThan(memory.totalBytes, 0)
        XCTAssertLessThanOrEqual(memory.usedBytes, memory.totalBytes)

        // Range-check the IOReport-derived fields when present (channels are chip- and
        // OS-specific, so absence is tolerated; nonsense values are not).
        if let gpu = sample.gpuUtilization {
            XCTAssertGreaterThanOrEqual(gpu.busyFraction, 0)
            XCTAssertLessThanOrEqual(gpu.busyFraction, 1)
        }
        for bandwidth in [sample.dramBandwidth, sample.gpuBandwidth, sample.aneBandwidth] {
            if let bandwidth {
                XCTAssertGreaterThanOrEqual(bandwidth.estimatedGBPerSecond, 0)
                XCTAssertTrue(bandwidth.isEstimated)
            }
        }
        for power in [sample.cpuPower, sample.gpuPower, sample.anePower, sample.dramPower] {
            if let power {
                XCTAssertGreaterThanOrEqual(power.watts, 0)
                XCTAssertLessThan(power.watts, 1000, "sanity ceiling on a laptop/desktop SoC")
            }
        }

        // Print the observed readings for the record.
        func fmt(_ value: Double?) -> String { value.map { String(format: "%.2f", $0) } ?? "n/a" }
        print("""
            [silicon smoke] interval=\(String(format: "%.3f", sample.intervalSeconds))s
              GPU busy:   \(fmt(sample.gpuUtilization.map { $0.busyFraction * 100 }))%
              DRAM BW:    \(fmt(sample.dramBandwidth?.estimatedGBPerSecond)) GB/s (est\(sample.dramBandwidth?.isSaturated == true ? ", saturated" : ""))
              GPU BW:     \(fmt(sample.gpuBandwidth?.estimatedGBPerSecond)) GB/s
              ANE BW:     \(fmt(sample.aneBandwidth?.estimatedGBPerSecond)) GB/s
              CPU power:  \(fmt(sample.cpuPower?.watts)) W
              GPU power:  \(fmt(sample.gpuPower?.watts)) W
              ANE power:  \(fmt(sample.anePower?.watts)) W (proxy only, no util%)
              DRAM power: \(fmt(sample.dramPower?.watts)) W
              thermal:    \(sample.thermalPressure)
              mem:        used \(fmt(memory.usedFraction * 100))% level=\(memory.level)
            """)
    }
}
