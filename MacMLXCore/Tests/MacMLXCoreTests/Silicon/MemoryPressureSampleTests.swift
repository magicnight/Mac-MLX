// Copyright © 2026 macMLX. English comments only.

import Testing

@testable import MacMLXCore

/// Pure-logic coverage for the page-count → byte breakdown and pressure-level mapping.
struct SiliconMemoryPressureTests {

    private func input(
        free: UInt64 = 0,
        active: UInt64 = 0,
        wire: UInt64 = 0,
        compressor: UInt64 = 0,
        pageSize: UInt64 = 4096,
        total: UInt64 = 16_000_000_000,
        pressureRaw: Int32 = 1
    ) -> MemoryPressureSample.VMInput {
        MemoryPressureSample.VMInput(
            freeCount: free,
            activeCount: active,
            wireCount: wire,
            compressorPageCount: compressor,
            pageSize: pageSize,
            totalBytes: total,
            pressureLevelRaw: pressureRaw
        )
    }

    @Test
    func usedIsActivePlusWiredPlusCompressed() {
        // (100 + 50 + 10) pages × 4096 = 655_360 bytes.
        let sample = MemoryPressureSample.make(
            input: input(active: 100, wire: 50, compressor: 10)
        )
        #expect(sample.usedBytes == 160 * 4096)
        #expect(sample.wiredBytes == 50 * 4096)
        #expect(sample.compressedBytes == 10 * 4096)
    }

    @Test
    func freeBytesTracksFreeCount() {
        let sample = MemoryPressureSample.make(input: input(free: 200))
        #expect(sample.freeBytes == 200 * 4096)
    }

    @Test
    func usedFractionIsUsedOverTotal() {
        let sample = MemoryPressureSample.make(
            input: input(active: 1000, pageSize: 4096, total: 4096 * 4000)
        )
        // used = 1000 pages, total = 4000 pages → 0.25.
        #expect(sample.usedFraction == 0.25)
    }

    @Test
    func usedIsCappedAtTotal() {
        // Working set larger than total (races between sysctls) → clamp to total.
        let sample = MemoryPressureSample.make(
            input: input(active: 10_000, pageSize: 4096, total: 4096 * 1000)
        )
        #expect(sample.usedBytes == 4096 * 1000)
        #expect(sample.usedFraction == 1.0)
    }

    @Test
    func usedFractionZeroWhenTotalUnknown() {
        let sample = MemoryPressureSample.make(input: input(active: 100, total: 0))
        #expect(sample.usedFraction == 0)
    }

    @Test
    func mapsPressureLevels() {
        #expect(MemoryPressureSample.make(input: input(pressureRaw: 1)).level == .normal)
        #expect(MemoryPressureSample.make(input: input(pressureRaw: 2)).level == .warning)
        #expect(MemoryPressureSample.make(input: input(pressureRaw: 4)).level == .critical)
        #expect(MemoryPressureSample.make(input: input(pressureRaw: 3)).level == .unknown)
        #expect(MemoryPressureSample.make(input: input(pressureRaw: 0)).level == .unknown)
    }

    @Test
    func currentReadsLiveKernelWithoutCrashing() {
        // host_statistics64 + sysctl are sudoless; a healthy machine returns a sample.
        let sample = MemoryPressureSample.current()
        #expect(sample != nil)
        if let sample {
            #expect(sample.totalBytes > 0)
            #expect(sample.usedBytes <= sample.totalBytes)
        }
    }
}
