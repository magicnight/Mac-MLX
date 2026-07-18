// Copyright © 2026 macMLX. English comments only.

import Foundation
import Testing

@testable import MacMLXCore

/// Pure-logic coverage for the benchmark bottleneck aggregator: synthetic
/// (verdict, sample) sequences exercise the decode-only rule, dominant-by-frequency
/// attribution, the confidence share, severity tie-breaking, the representative
/// hardware readouts, and the honest empty case. No IOReport, no engine — runs
/// under bare `swift test`.
struct BenchmarkBottleneckAggregatorTests {

    // MARK: - Builders

    private func decodeVerdict(
        _ category: BottleneckVerdict.Category,
        advice: String = "decode advice",
        rests: Bool = true
    ) -> BottleneckVerdict {
        BottleneckVerdict(
            category: category, phase: .decode, advice: advice,
            restsOnEstimatedBandwidth: rests)
    }

    private func prefillVerdict(
        _ category: BottleneckVerdict.Category
    ) -> BottleneckVerdict {
        BottleneckVerdict(
            category: category, phase: .prefill, advice: "prefill advice",
            restsOnEstimatedBandwidth: false)
    }

    /// A sample exposing only the fields the aggregator reads (GPU occupancy,
    /// bandwidth, thermal). Memory/power are irrelevant here and left nil.
    private func sample(
        gpuBusy: Double? = 1.0,
        gpuBandwidthGBs: Double? = nil,
        dramBandwidthGBs: Double? = nil,
        thermal: ThermalPressure = .nominal
    ) -> SiliconSample {
        SiliconSample(
            timestamp: Date(timeIntervalSince1970: 0),
            intervalSeconds: 1.0,
            ioReportAvailable: true,
            ioReportUnavailableReason: nil,
            gpuUtilization: gpuBusy.map { GPUUtilizationSample(busyFraction: $0) },
            dramBandwidth: dramBandwidthGBs.map {
                MemoryBandwidthSample(estimatedGBPerSecond: $0, isSaturated: false)
            },
            gpuBandwidth: gpuBandwidthGBs.map {
                MemoryBandwidthSample(estimatedGBPerSecond: $0, isSaturated: false)
            },
            aneBandwidth: nil,
            cpuPower: nil,
            gpuPower: nil,
            anePower: nil,
            dramPower: nil,
            thermalPressure: thermal,
            memoryPressure: nil
        )
    }

    // MARK: - Attribution + confidence

    @Test("All-consistent decode frames give the shared category at full confidence")
    func allConsistentDecodeFrames() {
        var agg = BenchmarkBottleneckAggregator()
        for _ in 0..<4 {
            agg.add(
                verdict: decodeVerdict(.bandwidthBound, advice: "lower-bit quant"),
                sample: sample(gpuBandwidthGBs: 100))
        }
        let result = agg.result()
        #expect(result?.category == .bandwidthBound)
        #expect(result?.phase == .decode)
        #expect(result?.confidence == 1.0)
        #expect(result?.decodeFrameCount == 4)
        #expect(result?.advice == "lower-bit quant")
    }

    @Test("Mixed decode frames report the dominant category with a lower confidence")
    func mixedDecodeFramesReportDominant() {
        var agg = BenchmarkBottleneckAggregator()
        for _ in 0..<3 { agg.add(verdict: decodeVerdict(.bandwidthBound), sample: sample(gpuBandwidthGBs: 100)) }
        agg.add(verdict: decodeVerdict(.computeBound), sample: sample(gpuBandwidthGBs: 40))
        let result = agg.result()
        #expect(result?.category == .bandwidthBound)
        #expect(result?.decodeFrameCount == 4)
        #expect(result?.confidence == 0.75)  // 3 of 4 frames agreed
    }

    @Test("An empty aggregator attributes nothing")
    func emptyGivesNil() {
        let agg = BenchmarkBottleneckAggregator()
        #expect(agg.result() == nil)
    }

    // MARK: - Decode-only discipline

    @Test("Only prefill frames attribute nothing (decode-only)")
    func onlyPrefillFramesGivesNil() {
        var agg = BenchmarkBottleneckAggregator()
        for _ in 0..<3 { agg.add(verdict: prefillVerdict(.computeBound), sample: sample(gpuBandwidthGBs: 30)) }
        #expect(agg.result() == nil)
    }

    @Test("Prefill frames do not pollute the decode attribution or its confidence")
    func prefillFramesIgnored() {
        var agg = BenchmarkBottleneckAggregator()
        // Two prefill compute-bound frames (must be ignored) …
        agg.add(verdict: prefillVerdict(.computeBound), sample: sample(gpuBandwidthGBs: 30))
        agg.add(verdict: prefillVerdict(.computeBound), sample: sample(gpuBandwidthGBs: 30))
        // … then three decode bandwidth-bound frames.
        for _ in 0..<3 { agg.add(verdict: decodeVerdict(.bandwidthBound), sample: sample(gpuBandwidthGBs: 100)) }
        let result = agg.result()
        #expect(result?.category == .bandwidthBound)
        #expect(result?.decodeFrameCount == 3)   // prefill frames not counted
        #expect(result?.confidence == 1.0)        // and not diluting the confidence
    }

    // MARK: - Tie-break

    @Test("A frame-count tie is broken by descending severity")
    func tieBreaksBySeverity() {
        var agg = BenchmarkBottleneckAggregator()
        for _ in 0..<2 { agg.add(verdict: decodeVerdict(.bandwidthBound), sample: sample(gpuBandwidthGBs: 100)) }
        for _ in 0..<2 { agg.add(verdict: decodeVerdict(.memoryBound, rests: false), sample: sample(gpuBandwidthGBs: 100)) }
        let result = agg.result()
        // memory-bound outranks bandwidth-bound on the severity tie-break.
        #expect(result?.category == .memoryBound)
        #expect(result?.confidence == 0.5)
        #expect(result?.restsOnEstimatedBandwidth == false)  // carried from the memory verdict
    }

    // MARK: - Honesty flag

    @Test("The estimated-bandwidth flag is carried from the dominant verdict")
    func estimateFlagCarriedFromDominant() {
        var agg = BenchmarkBottleneckAggregator()
        for _ in 0..<3 { agg.add(verdict: decodeVerdict(.bandwidthBound, rests: true), sample: sample(gpuBandwidthGBs: 100)) }
        #expect(agg.result()?.restsOnEstimatedBandwidth == true)
    }

    // MARK: - Representative hardware

    @Test("Peaks and means summarise the decode-phase hardware")
    func representativeHardwareReadouts() {
        var agg = BenchmarkBottleneckAggregator()
        agg.add(verdict: decodeVerdict(.bandwidthBound), sample: sample(gpuBusy: 0.8, gpuBandwidthGBs: 90, thermal: .nominal))
        agg.add(verdict: decodeVerdict(.bandwidthBound), sample: sample(gpuBusy: 1.0, gpuBandwidthGBs: 100, thermal: .serious))
        let hw = agg.result()?.hardware
        #expect(hw?.peakGPUUtilization == 1.0)
        #expect(hw?.meanGPUUtilization == 0.9)
        #expect(hw?.peakBandwidthGBs == 100)
        #expect(hw?.meanBandwidthGBs == 95)
        #expect(hw?.peakThermalPressure == .serious)
    }

    @Test("Bandwidth prefers the GPU/AGX requestor over the DRAM aggregate")
    func bandwidthPrefersGPUChannel() {
        var agg = BenchmarkBottleneckAggregator()
        // Both present: the GPU (AGX) reading must be used, not the larger aggregate.
        for _ in 0..<3 {
            agg.add(
                verdict: decodeVerdict(.bandwidthBound),
                sample: sample(gpuBandwidthGBs: 100, dramBandwidthGBs: 400))
        }
        #expect(agg.result()?.hardware.peakBandwidthGBs == 100)
    }

    @Test("Once committed to the GPU channel, an intermittent AGX drop never leaks the DRAM aggregate")
    func bandwidthLatchesGPUChannelAcrossDrops() {
        var agg = BenchmarkBottleneckAggregator()
        // AGX present, then transiently absent (IOReport drops the requestor for a
        // frame — the DRAM aggregate 390 must NOT leak in), then back.
        agg.add(verdict: decodeVerdict(.bandwidthBound), sample: sample(gpuBandwidthGBs: 110, dramBandwidthGBs: 380))
        agg.add(verdict: decodeVerdict(.bandwidthBound), sample: sample(gpuBandwidthGBs: nil, dramBandwidthGBs: 390))
        agg.add(verdict: decodeVerdict(.bandwidthBound), sample: sample(gpuBandwidthGBs: 115, dramBandwidthGBs: 385))
        let hw = agg.result()?.hardware
        // GPU channel only: peak over {110, 115}, mean (110+115)/2 — the DRAM 390 excluded.
        #expect(hw?.peakBandwidthGBs == 115)
        #expect(hw?.meanBandwidthGBs == 112.5)
    }

    @Test("Bandwidth uses the DRAM aggregate only when the GPU channel never appears")
    func bandwidthUsesDRAMWhenNoGPUChannel() {
        var agg = BenchmarkBottleneckAggregator()
        for gbs in [380.0, 390.0, 385.0] {
            agg.add(verdict: decodeVerdict(.bandwidthBound), sample: sample(gpuBandwidthGBs: nil, dramBandwidthGBs: gbs))
        }
        // No AGX all run → the aggregate is the single channel; peak over {380,390,385}.
        #expect(agg.result()?.hardware.peakBandwidthGBs == 390)
    }

    @Test("A pre-commit DRAM reading is discarded once the GPU channel appears")
    func preCommitDRAMDiscardedOnGPUAppearance() {
        var agg = BenchmarkBottleneckAggregator()
        // First frame has only the DRAM aggregate; then AGX appears and commits.
        agg.add(verdict: decodeVerdict(.bandwidthBound), sample: sample(gpuBandwidthGBs: nil, dramBandwidthGBs: 400))
        agg.add(verdict: decodeVerdict(.bandwidthBound), sample: sample(gpuBandwidthGBs: 105, dramBandwidthGBs: 395))
        // Committed to GPU → only 105 counts; the pre-commit 400 is dropped, not peak.
        #expect(agg.result()?.hardware.peakBandwidthGBs == 105)
    }

    @Test("Missing IOReport fields leave the hardware readouts nil, never zero")
    func missingIOReportLeavesReadoutsNil() {
        var agg = BenchmarkBottleneckAggregator()
        // No GPU occupancy, no bandwidth (IOReport unavailable) but a memory-bound
        // verdict from public signals — thermal is still present.
        for _ in 0..<3 {
            agg.add(
                verdict: decodeVerdict(.memoryBound, rests: false),
                sample: sample(gpuBusy: nil, gpuBandwidthGBs: nil, thermal: .fair))
        }
        let hw = agg.result()?.hardware
        #expect(hw?.peakGPUUtilization == nil)
        #expect(hw?.meanGPUUtilization == nil)
        #expect(hw?.peakBandwidthGBs == nil)
        #expect(hw?.meanBandwidthGBs == nil)
        #expect(hw?.peakThermalPressure == .fair)   // public API, still present
    }
}
