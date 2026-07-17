// Copyright © 2026 macMLX. English comments only.

import Foundation
import Testing

@testable import MacMLXCore

/// Pure-logic coverage for the W3 monitor reducer: hand-driven phase events plus
/// synthetic `SiliconSample`s exercise the phase bridging, the "verdict exists only
/// while generating" rule, the honest IOReport-unavailable mapping, and the
/// prefill/decode tokens-per-second derivation — all without live IOReport or an
/// engine, under bare `swift test`.
struct SiliconMonitorModelTests {

    // MARK: - Synthetic sample builder (mirrors the W2 classifier's helper)

    /// A sample exposing the fields the reducer/classifier read, with benign
    /// defaults. Overrides let each test drive one signal.
    private func sample(
        ioReportAvailable: Bool = true,
        ioReportUnavailableReason: String? = nil,
        gpuBusy: Double? = 1.0,
        gpuBandwidthGBs: Double? = nil,
        thermal: ThermalPressure = .nominal,
        memoryUsedFraction: Double = 0.30,
        memoryLevel: MemoryPressureSample.Level = .normal
    ) -> SiliconSample {
        let total: UInt64 = 1_000_000
        let used = UInt64(Double(total) * memoryUsedFraction)
        let memory = MemoryPressureSample(
            level: memoryLevel,
            totalBytes: total,
            usedBytes: used,
            freeBytes: total - used,
            compressedBytes: 0,
            wiredBytes: 0
        )
        return SiliconSample(
            timestamp: Date(timeIntervalSince1970: 0),
            intervalSeconds: 1.0,
            ioReportAvailable: ioReportAvailable,
            ioReportUnavailableReason: ioReportUnavailableReason,
            gpuUtilization: gpuBusy.map { GPUUtilizationSample(busyFraction: $0) },
            dramBandwidth: nil,
            gpuBandwidth: gpuBandwidthGBs.map {
                MemoryBandwidthSample(estimatedGBPerSecond: $0, isSaturated: false)
            },
            aneBandwidth: nil,
            cpuPower: nil,
            gpuPower: nil,
            anePower: nil,
            dramPower: nil,
            thermalPressure: thermal,
            // Thermal + memory pressure are public APIs, present even when IOReport
            // is unavailable — matching what IOReportSiliconSampler returns.
            memoryPressure: memory
        )
    }

    private let plainConfig = EngineGenerationConfig(
        kvBits: nil, kvGroupSize: nil, quantizedKVStart: nil, batchSize: 1)

    private func summary(
        promptTokens: Int, generationTokens: Int,
        promptSeconds: Double?, generateSeconds: Double?
    ) -> GenerationPhaseSummary {
        GenerationPhaseSummary(
            config: plainConfig,
            promptTokenCount: promptTokens,
            generationTokenCount: generationTokens,
            promptSeconds: promptSeconds,
            generateSeconds: generateSeconds
        )
    }

    // MARK: - Phase bridging

    @Test("beginPrefill enters the generating state in the prefill phase")
    func beginPrefillBridges() {
        var m = SiliconMonitorModel()
        #expect(m.isGenerating == false)
        #expect(m.phase == nil)
        m.beginPrefill(config: plainConfig)
        #expect(m.isGenerating == true)
        #expect(m.phase == .prefill)
    }

    @Test("beginDecode transitions the phase to decode")
    func beginDecodeBridges() {
        var m = SiliconMonitorModel()
        m.beginPrefill(config: plainConfig)
        m.beginDecode(config: plainConfig)
        #expect(m.isGenerating == true)
        #expect(m.phase == .decode)
    }

    @Test("complete returns to idle and records the terminal summary")
    func completeBridges() {
        var m = SiliconMonitorModel()
        m.beginPrefill(config: plainConfig)
        m.beginDecode(config: plainConfig)
        m.complete(summary: summary(
            promptTokens: 100, generationTokens: 50,
            promptSeconds: 0.5, generateSeconds: 2.0))
        #expect(m.isGenerating == false)
        #expect(m.phase == nil)
        #expect(m.lastSummary != nil)
    }

    @Test("abort returns to idle without disturbing the previous summary")
    func abortKeepsPriorSummary() {
        var m = SiliconMonitorModel()
        // A completed run leaves a summary...
        m.beginPrefill(config: plainConfig)
        m.complete(summary: summary(
            promptTokens: 100, generationTokens: 50,
            promptSeconds: 0.5, generateSeconds: 2.0))
        // ...a later run that is aborted must not clear it.
        m.beginPrefill(config: plainConfig)
        m.abort()
        #expect(m.isGenerating == false)
        #expect(m.phase == nil)
        #expect(m.lastSummary != nil)
    }

    // MARK: - Verdict lifecycle (classify feed → verdict publish)

    @Test("A sample while generating publishes a phase-matched verdict")
    func generatingSamplePublishesVerdict() {
        var m = SiliconMonitorModel()
        m.beginPrefill(config: plainConfig)
        m.beginDecode(config: plainConfig)
        // Pinned GPU + saturated bandwidth over the rolling window → bandwidth-bound
        // decode (mirrors the classifier's own healthy-decode case).
        for _ in 0..<3 {
            m.ingest(sample(gpuBusy: 1.0, gpuBandwidthGBs: 100))
        }
        #expect(m.verdict != nil)
        #expect(m.verdict?.phase == .decode)
        #expect(m.verdict?.category == .bandwidthBound)
    }

    @Test("A sample while idle records hardware but publishes no verdict")
    func idleSampleHasNoVerdict() {
        var m = SiliconMonitorModel()
        m.ingest(sample(gpuBusy: 1.0, gpuBandwidthGBs: 100))
        #expect(m.latestSample != nil)   // hardware readouts still update
        #expect(m.verdict == nil)         // but no bottleneck attribution when idle
    }

    @Test("Completing a generation clears the live verdict")
    func completeClearsVerdict() {
        var m = SiliconMonitorModel()
        m.beginDecode(config: plainConfig)
        for _ in 0..<3 { m.ingest(sample(gpuBusy: 1.0, gpuBandwidthGBs: 100)) }
        #expect(m.verdict != nil)
        m.complete(summary: summary(
            promptTokens: 100, generationTokens: 50,
            promptSeconds: 0.5, generateSeconds: 2.0))
        #expect(m.verdict == nil)
    }

    @Test("Memory pressure surfaces a verdict even with IOReport unavailable")
    func classifiesFromPublicSignalsWhenIOReportUnavailable() {
        var m = SiliconMonitorModel()
        m.beginDecode(config: plainConfig)
        // No IOReport (gpu/bandwidth nil) but the kernel reports critical memory
        // pressure — a public signal the classifier still acts on.
        for _ in 0..<3 {
            m.ingest(sample(
                ioReportAvailable: false,
                ioReportUnavailableReason: "not entitled",
                gpuBusy: nil,
                gpuBandwidthGBs: nil,
                memoryUsedFraction: 0.97,
                memoryLevel: .critical))
        }
        #expect(m.verdict?.category == .memoryBound)
    }

    // MARK: - Stale-frame gate across a generation boundary (M1)

    @Test("A new generation withholds the verdict until the previous run's window frames age out")
    func staleVerdictSuppressedAtGenerationStart() {
        var m = SiliconMonitorModel()

        // Run 1 drives memory to critical → a memory-bound verdict, then completes.
        m.beginDecode(config: plainConfig)
        for _ in 0..<3 {
            m.ingest(sample(gpuBusy: 1.0, memoryUsedFraction: 0.97, memoryLevel: .critical))
        }
        #expect(m.verdict?.category == .memoryBound)
        m.complete(summary: summary(
            promptTokens: 10, generationTokens: 5, promptSeconds: 0.1, generateSeconds: 1.0))
        #expect(m.verdict == nil)

        // Run 2 is healthy, but the classifier's window still holds run 1's critical
        // tail. The panel must NOT flash a stale "near out-of-memory" alarm: the
        // first `rollingWindow - 1` frames publish no verdict at all.
        m.beginDecode(config: plainConfig)
        m.ingest(sample(gpuBusy: 1.0, gpuBandwidthGBs: 2, memoryUsedFraction: 0.30))
        #expect(m.verdict == nil)   // frame 1 — 2 stale critical frames still in window
        m.ingest(sample(gpuBusy: 1.0, gpuBandwidthGBs: 2, memoryUsedFraction: 0.30))
        #expect(m.verdict == nil)   // frame 2 — 1 stale critical frame still in window

        // Frame 3: window fully refreshed with run 2's own healthy frames. The
        // verdict returns, and it is NOT the stale memory alarm.
        m.ingest(sample(gpuBusy: 1.0, gpuBandwidthGBs: 2, memoryUsedFraction: 0.30))
        #expect(m.verdict != nil)
        #expect(m.verdict?.category != .memoryBound)
    }

    // MARK: - Concurrent-generation refcount (M2)

    @Test("Two overlapping generations do not read Idle when the first completes")
    func overlappingGenerationsRefcountIsGenerating() {
        var m = SiliconMonitorModel()
        m.beginPrefill(config: plainConfig)   // generation A starts
        m.beginPrefill(config: plainConfig)   // generation B starts (overlap)
        #expect(m.isGenerating == true)

        // A completes while B is still running — must stay "generating", not flip idle.
        m.complete(summary: summary(
            promptTokens: 10, generationTokens: 5, promptSeconds: 0.1, generateSeconds: 1.0))
        #expect(m.isGenerating == true)
        #expect(m.phase != nil)

        // B completes — now truly idle.
        m.complete(summary: summary(
            promptTokens: 10, generationTokens: 5, promptSeconds: 0.1, generateSeconds: 1.0))
        #expect(m.isGenerating == false)
        #expect(m.phase == nil)
        #expect(m.verdict == nil)
    }

    @Test("A stray terminal event without a matching begin cannot drive the count negative")
    func terminalWithoutBeginDoesNotUnderflow() {
        var m = SiliconMonitorModel()
        m.abort()   // no generation was in flight
        #expect(m.isGenerating == false)
        // A subsequent real generation still tracks correctly.
        m.beginPrefill(config: plainConfig)
        #expect(m.isGenerating == true)
        m.complete(summary: summary(
            promptTokens: 10, generationTokens: 5, promptSeconds: 0.1, generateSeconds: 1.0))
        #expect(m.isGenerating == false)
    }

    // MARK: - IOReport availability mapping (three-state)

    @Test("ioReportAvailable is nil before the first sample")
    func availabilityUnknownBeforeFirstSample() {
        let m = SiliconMonitorModel()
        #expect(m.ioReportAvailable == nil)
        #expect(m.latestSample == nil)
    }

    @Test("An unavailable sample maps to false with its reason surfaced")
    func availabilityMapsUnavailable() {
        var m = SiliconMonitorModel()
        m.ingest(sample(
            ioReportAvailable: false,
            ioReportUnavailableReason: "IOReport subscription failed"))
        #expect(m.ioReportAvailable == false)
        #expect(m.ioReportUnavailableReason == "IOReport subscription failed")
        // Thermal + memory are public, so the sample itself is still present.
        #expect(m.latestSample != nil)
    }

    @Test("An available sample maps to true with no reason")
    func availabilityMapsAvailable() {
        var m = SiliconMonitorModel()
        m.ingest(sample(ioReportAvailable: true))
        #expect(m.ioReportAvailable == true)
        #expect(m.ioReportUnavailableReason == nil)
    }

    // MARK: - Tokens/second derivation (PP / TG)

    @Test("Prefill and decode tokens/second divide real counts by real elapsed time")
    func tokensPerSecondFromSummary() {
        var m = SiliconMonitorModel()
        m.complete(summary: summary(
            promptTokens: 200, generationTokens: 100,
            promptSeconds: 0.5, generateSeconds: 2.0))
        #expect(m.prefillTokensPerSecond == 400)   // 200 / 0.5
        #expect(m.decodeTokensPerSecond == 50)     // 100 / 2.0
    }

    @Test("Tokens/second are nil when elapsed time was zero (no fabricated rate)")
    func tokensPerSecondNilOnZeroTime() {
        var m = SiliconMonitorModel()
        m.complete(summary: summary(
            promptTokens: 200, generationTokens: 100,
            promptSeconds: nil, generateSeconds: 0))
        #expect(m.prefillTokensPerSecond == nil)
        #expect(m.decodeTokensPerSecond == nil)
    }

    @Test("Tokens/second are nil before any generation completes")
    func tokensPerSecondNilBeforeAnyRun() {
        let m = SiliconMonitorModel()
        #expect(m.prefillTokensPerSecond == nil)
        #expect(m.decodeTokensPerSecond == nil)
    }
}
