// Copyright © 2026 macMLX. English comments only.

import Foundation
import Testing

@testable import MacMLXCore

/// `BenchmarkBottleneck` is persisted inside a `BenchmarkResult`, so it must
/// survive a JSON round-trip exactly — including the optional hardware readouts in
/// both their present and absent forms.
struct BenchmarkBottleneckTests {

    @Test("A fully-populated bottleneck round-trips through JSON unchanged")
    func fullyPopulatedRoundTrips() throws {
        let original = BenchmarkBottleneck(
            category: .bandwidthBound,
            phase: .decode,
            advice: "Decode appears memory-bandwidth-bound — try a lower-bit quantization.",
            confidence: 0.87,
            restsOnEstimatedBandwidth: true,
            decodeFrameCount: 23,
            hardware: BenchmarkBottleneck.Readouts(
                peakGPUUtilization: 0.99,
                meanGPUUtilization: 0.96,
                peakBandwidthGBs: 118.0,
                meanBandwidthGBs: 110.5,
                peakThermalPressure: .serious
            )
        )
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(BenchmarkBottleneck.self, from: data)
        #expect(decoded == original)
    }

    @Test("A bottleneck with absent hardware readouts round-trips as nil, not zero")
    func nilReadoutsRoundTrip() throws {
        let original = BenchmarkBottleneck(
            category: .memoryBound,
            phase: .decode,
            advice: "Unified memory is under pressure.",
            confidence: 1.0,
            restsOnEstimatedBandwidth: false,
            decodeFrameCount: 5,
            hardware: BenchmarkBottleneck.Readouts(
                peakGPUUtilization: nil,
                meanGPUUtilization: nil,
                peakBandwidthGBs: nil,
                meanBandwidthGBs: nil,
                peakThermalPressure: .nominal
            )
        )
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(BenchmarkBottleneck.self, from: data)
        #expect(decoded == original)
        #expect(decoded.hardware.peakGPUUtilization == nil)
        #expect(decoded.hardware.meanBandwidthGBs == nil)
    }
}
