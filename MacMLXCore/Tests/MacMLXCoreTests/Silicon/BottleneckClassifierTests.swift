// Copyright © 2026 macMLX. English comments only.

import Foundation
import Testing

@testable import MacMLXCore

/// Pure-logic coverage for the W2 bottleneck classifier: synthetic
/// (`SiliconSample`, phase) sequences exercise the priority ladder, the
/// double-threshold hysteresis, the self-calibrating achievable bandwidth, the
/// 3-frame smoothing, and the ANE-exclusion guarantee. No IOReport, no engine —
/// runs under bare `swift test`.
struct BottleneckClassifierTests {

    // MARK: - Synthetic sample + driver helpers

    /// Build a `SiliconSample` exposing only the fields the classifier reads,
    /// with benign defaults (GPU pinned, memory relaxed, thermals nominal) so
    /// each test overrides just the signal under exercise. `gpuBusy: nil` models
    /// a transient IOReport miss (no GPU / bandwidth reading that window).
    private func sample(
        gpuBusy: Double? = 1.0,
        gpuBandwidthGBs: Double? = nil,
        gpuBandwidthSaturated: Bool = false,
        dramBandwidthGBs: Double? = nil,
        thermal: ThermalPressure = .nominal,
        memoryUsedFraction: Double = 0.30,
        memoryLevel: MemoryPressureSample.Level = .normal,
        anePowerWatts: Double? = nil,
        aneBandwidthGBs: Double? = nil
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
            ioReportAvailable: true,
            ioReportUnavailableReason: nil,
            gpuUtilization: gpuBusy.map { GPUUtilizationSample(busyFraction: $0) },
            dramBandwidth: dramBandwidthGBs.map {
                MemoryBandwidthSample(estimatedGBPerSecond: $0, isSaturated: false)
            },
            gpuBandwidth: gpuBandwidthGBs.map {
                MemoryBandwidthSample(estimatedGBPerSecond: $0, isSaturated: gpuBandwidthSaturated)
            },
            aneBandwidth: aneBandwidthGBs.map {
                MemoryBandwidthSample(estimatedGBPerSecond: $0, isSaturated: true)
            },
            cpuPower: nil,
            gpuPower: nil,
            anePower: anePowerWatts.map { PowerSample(watts: $0) },
            dramPower: nil,
            thermalPressure: thermal,
            memoryPressure: memory
        )
    }

    private let plainConfig = EngineGenerationConfig(
        kvBits: nil, kvGroupSize: nil, quantizedKVStart: nil, batchSize: 1)

    /// Feed a sample `times` times and return the final verdict (defaults to a
    /// single frame). Calls at least once, so no force-unwrap is needed.
    @discardableResult
    private func feed(
        _ classifier: inout BottleneckClassifier,
        _ sample: SiliconSample,
        phase: InferencePhase,
        times: Int = 1,
        config: EngineGenerationConfig? = nil
    ) -> BottleneckVerdict {
        let context = EnginePhaseContext(
            phase: phase, tokensPerSecond: nil, config: config ?? plainConfig)
        var verdict = classifier.classify(sample: sample, context: context)
        for _ in 1..<max(1, times) {
            verdict = classifier.classify(sample: sample, context: context)
        }
        return verdict
    }

    /// Establish a trustworthy calibrated ceiling of 100 GB/s: one peak reading,
    /// then one clearly below it (proving the ceiling is a real high-water mark).
    private func calibrateCeiling(_ classifier: inout BottleneckClassifier) {
        feed(&classifier, sample(gpuBusy: 1.0, gpuBandwidthGBs: 100), phase: .decode)
        feed(&classifier, sample(gpuBusy: 1.0, gpuBandwidthGBs: 50), phase: .decode)
    }

    // MARK: - Healthy phase regimes

    @Test("Decode with a pinned GPU and saturated bandwidth is bandwidth-bound")
    func decodeIsBandwidthBound() {
        var c = BottleneckClassifier()
        let v = feed(&c, sample(gpuBusy: 1.0, gpuBandwidthGBs: 100), phase: .decode, times: 3)
        #expect(v.category == .bandwidthBound)
        #expect(v.phase == .decode)
        #expect(v.restsOnEstimatedBandwidth == true)
        #expect(v.advice.contains("bandwidth-bound"))
        #expect(v.advice.contains("expected"))
    }

    @Test("Prefill with a pinned GPU is compute-bound (expected regime)")
    func prefillIsComputeBound() {
        var c = BottleneckClassifier()
        let v = feed(&c, sample(gpuBusy: 1.0, gpuBandwidthGBs: 50), phase: .prefill, times: 3)
        #expect(v.category == .computeBound)
        #expect(v.phase == .prefill)
        #expect(v.advice.contains("compute-bound"))
        #expect(v.advice.contains("expected"))
    }

    @Test("An idle GPU with no pressure is normal, not a bottleneck")
    func idleGPUIsNormal() {
        var c = BottleneckClassifier()
        let v = feed(&c, sample(gpuBusy: 0.05, gpuBandwidthGBs: 2), phase: .decode, times: 3)
        #expect(v.category == .normal)
        #expect(v.restsOnEstimatedBandwidth == false)
    }

    @Test("Bandwidth-bound decode advice adapts when the KV cache is already quantized")
    func adviceAdaptsToQuantizedKVCache() {
        var c = BottleneckClassifier()
        let quantized = EngineGenerationConfig(
            kvBits: 4, kvGroupSize: 64, quantizedKVStart: 0, batchSize: 1)
        let v = feed(
            &c, sample(gpuBusy: 1.0, gpuBandwidthGBs: 100),
            phase: .decode, times: 3, config: quantized)
        #expect(v.category == .bandwidthBound)
        #expect(v.advice.contains("already quantized"))
    }

    // MARK: - Priority ladder

    @Test("Thermal throttling outranks the compute/bandwidth profile")
    func thermalBeatsProfile() {
        var c = BottleneckClassifier()
        calibrateCeiling(&c)
        // GPU pinned and bandwidth saturated (would read bandwidth-bound), but
        // sustained serious thermal pressure must win.
        let v = feed(
            &c,
            sample(gpuBusy: 1.0, gpuBandwidthGBs: 95, thermal: .serious),
            phase: .decode, times: 3)
        #expect(v.category == .thermalThrottled)
        #expect(v.restsOnEstimatedBandwidth == false)
        #expect(v.advice.contains("Thermal"))
    }

    @Test("Memory pressure outranks thermal and the profile")
    func memoryBeatsEverything() {
        var c = BottleneckClassifier()
        calibrateCeiling(&c)
        // Everything screaming at once: critical memory, serious thermal, pinned
        // GPU, saturated bandwidth — memory must win.
        let v = feed(
            &c,
            sample(
                gpuBusy: 1.0, gpuBandwidthGBs: 95, thermal: .serious,
                memoryUsedFraction: 0.97, memoryLevel: .critical),
            phase: .decode, times: 3)
        #expect(v.category == .memoryBound)
        #expect(v.restsOnEstimatedBandwidth == false)
        #expect(v.advice.contains("memory"))
    }

    @Test("The kernel's warning level trips memory-bound even below the byte threshold")
    func kernelMemoryLevelIsAuthoritative() {
        var c = BottleneckClassifier()
        // Low used-fraction (0.40) but the kernel says warning — trust the kernel.
        let v = feed(
            &c,
            sample(gpuBusy: 1.0, gpuBandwidthGBs: 100, memoryUsedFraction: 0.40,
                   memoryLevel: .warning),
            phase: .decode, times: 3)
        #expect(v.category == .memoryBound)
    }

    // MARK: - Double-threshold hysteresis

    @Test("Saturation uses a 0.85 enter / 0.70 exit dead band, so it does not flap")
    func bandwidthSaturationHysteresisDoesNotFlap() {
        var c = BottleneckClassifier()
        calibrateCeiling(&c)  // trustworthy ceiling = 100

        // Settle clearly below saturation → compute-bound.
        let settled = feed(&c, sample(gpuBusy: 1.0, gpuBandwidthGBs: 55), phase: .decode, times: 3)
        #expect(settled.category == .computeBound)

        // 0.75 is inside the dead band, approached from BELOW → must not enter.
        let approachFromBelow = feed(
            &c, sample(gpuBusy: 1.0, gpuBandwidthGBs: 75), phase: .decode, times: 3)
        #expect(approachFromBelow.category == .computeBound)

        // 0.90 crosses the enter threshold → bandwidth-bound.
        let entered = feed(&c, sample(gpuBusy: 1.0, gpuBandwidthGBs: 90), phase: .decode, times: 3)
        #expect(entered.category == .bandwidthBound)

        // Back to 0.75 dead band, now approached from ABOVE → must hold.
        let holds = feed(&c, sample(gpuBusy: 1.0, gpuBandwidthGBs: 75), phase: .decode, times: 3)
        #expect(holds.category == .bandwidthBound)

        // Drop below the exit threshold → finally leaves.
        let exited = feed(&c, sample(gpuBusy: 1.0, gpuBandwidthGBs: 60), phase: .decode, times: 3)
        #expect(exited.category == .computeBound)
    }

    // MARK: - Self-calibrating achievable bandwidth

    @Test("The achievable ceiling ratchets up, re-reading the same bandwidth as unsaturated")
    func achievableBandwidthCalibrationClimbs() {
        var c = BottleneckClassifier()

        // With no higher reference yet, a sustained 50 GB/s decode reads as the
        // healthy bandwidth-bound regime (phase prior, ceiling not yet trusted).
        let early = feed(&c, sample(gpuBusy: 1.0, gpuBandwidthGBs: 50), phase: .decode, times: 3)
        #expect(early.category == .bandwidthBound)

        // A burst to 100 GB/s ratchets the ceiling up and, once bandwidth falls
        // clearly below it, makes the ceiling trustworthy.
        feed(&c, sample(gpuBusy: 1.0, gpuBandwidthGBs: 100), phase: .decode)

        // The SAME 50 GB/s now sits at ratio 0.5 against the risen ceiling →
        // no longer saturated → compute-bound. Calibration changed the reading.
        let late = feed(&c, sample(gpuBusy: 1.0, gpuBandwidthGBs: 50), phase: .decode, times: 3)
        #expect(late.category == .computeBound)
    }

    // MARK: - 3-frame smoothing

    @Test("A single thermal spike is smoothed away by the 3-frame mean")
    func singleThermalSpikeIsSmoothed() {
        var c = BottleneckClassifier()
        feed(&c, sample(gpuBusy: 1.0, gpuBandwidthGBs: 50, thermal: .nominal), phase: .decode)
        feed(&c, sample(gpuBusy: 1.0, gpuBandwidthGBs: 50, thermal: .nominal), phase: .decode)
        // One critical frame: window mean thermal = (0+0+3)/3 = 1.0 < serious.
        let v = feed(
            &c, sample(gpuBusy: 1.0, gpuBandwidthGBs: 50, thermal: .critical), phase: .decode)
        #expect(v.category != .thermalThrottled)
    }

    @Test("Sustained thermal pressure (not a spike) does trip the throttle verdict")
    func sustainedThermalTripsThrottle() {
        var c = BottleneckClassifier()
        let v = feed(
            &c, sample(gpuBusy: 1.0, gpuBandwidthGBs: 50, thermal: .serious),
            phase: .decode, times: 3)
        #expect(v.category == .thermalThrottled)
    }

    @Test("A transient IOReport miss frame does not drop a saturated verdict")
    func transientMissDoesNotDropVerdict() {
        var c = BottleneckClassifier()
        calibrateCeiling(&c)
        let entered = feed(&c, sample(gpuBusy: 1.0, gpuBandwidthGBs: 95), phase: .decode, times: 3)
        #expect(entered.category == .bandwidthBound)
        // Miss frame: no GPU / bandwidth reading this window (excluded from the
        // rolling means), so the verdict rides on the two surrounding frames.
        let afterMiss = feed(
            &c, sample(gpuBusy: nil, gpuBandwidthGBs: nil), phase: .decode)
        #expect(afterMiss.category == .bandwidthBound)
    }

    // MARK: - Bandwidth channel commitment (no AGX/DRAM mixing)

    @Test("Once on the GPU channel, a DRAM-aggregate frame never inflates the ceiling")
    func gpuChannelIgnoresDramAggregate() {
        var c = BottleneckClassifier()
        // Commit to the GPU (AGX) channel and calibrate a trustworthy 100 GB/s
        // ceiling, then settle at a saturated 95 → bandwidth-bound.
        calibrateCeiling(&c)
        let saturated = feed(&c, sample(gpuBusy: 1.0, gpuBandwidthGBs: 95), phase: .decode, times: 3)
        #expect(saturated.category == .bandwidthBound)

        // A frame with NO GPU reading but a huge DRAM-aggregate value (CPU+GPU+ANE
        // traffic — a structurally larger, incommensurable number). If it leaked
        // into the ceiling it would jump to ~400 and suppress saturation.
        feed(&c, sample(gpuBusy: 1.0, gpuBandwidthGBs: nil, dramBandwidthGBs: 400), phase: .decode)

        // The GPU channel is unaffected: a 95 GB/s GPU reading is still saturated
        // against the 100 ceiling. A polluted 400 ceiling would read 95/400 ≈ 0.24
        // and fall to compute-bound, so this proves the DRAM frame was ignored.
        let stillSaturated = feed(
            &c, sample(gpuBusy: 1.0, gpuBandwidthGBs: 95), phase: .decode, times: 3)
        #expect(stillSaturated.category == .bandwidthBound)
    }

    @Test("A machine that only exposes the DRAM aggregate stays on that one channel")
    func dramOnlyMachineUsesDramChannel() {
        var c = BottleneckClassifier()
        // No GPU channel ever appears; the DRAM aggregate is the only signal, so
        // it calibrates and saturates on its own without any channel switch.
        feed(&c, sample(gpuBusy: 1.0, gpuBandwidthGBs: nil, dramBandwidthGBs: 100), phase: .decode)
        feed(&c, sample(gpuBusy: 1.0, gpuBandwidthGBs: nil, dramBandwidthGBs: 50), phase: .decode)
        let v = feed(
            &c, sample(gpuBusy: 1.0, gpuBandwidthGBs: nil, dramBandwidthGBs: 95),
            phase: .decode, times: 3)
        #expect(v.category == .bandwidthBound)
    }

    // MARK: - ANE exclusion

    @Test("ANE power and bandwidth never change the verdict")
    func aneDoesNotAffectVerdict() {
        var quiet = BottleneckClassifier()
        var loud = BottleneckClassifier()
        for _ in 0..<3 {
            let quietVerdict = feed(
                &quiet,
                sample(gpuBusy: 1.0, gpuBandwidthGBs: 100, anePowerWatts: 0, aneBandwidthGBs: nil),
                phase: .decode)
            let loudVerdict = feed(
                &loud,
                sample(gpuBusy: 1.0, gpuBandwidthGBs: 100, anePowerWatts: 45, aneBandwidthGBs: 200),
                phase: .decode)
            #expect(quietVerdict == loudVerdict)
        }
    }
}
