// Copyright © 2026 macMLX. English comments only.

import Testing

@testable import MacMLXCore

/// Pure-logic coverage for GPU occupancy derived from performance-state residency.
struct SiliconGPUUtilizationTests {

    @Test
    func computesBusyFractionFromResidency() {
        let sample = GPUUtilizationSample.compute(states: [
            .init(name: "OFF", residency: 40),
            .init(name: "P1", residency: 60),
        ])
        #expect(sample?.busyFraction == 0.6)
        #expect(sample?.idleFraction == 0.4)
    }

    @Test
    func fullyIdleIsZeroBusy() {
        let sample = GPUUtilizationSample.compute(states: [
            .init(name: "OFF", residency: 100),
            .init(name: "P1", residency: 0),
        ])
        #expect(sample?.busyFraction == 0)
        #expect(sample?.idleFraction == 1)
    }

    @Test
    func multipleActiveStatesAllCountAsBusy() {
        let sample = GPUUtilizationSample.compute(states: [
            .init(name: "OFF", residency: 25),
            .init(name: "P1", residency: 25),
            .init(name: "P2", residency: 25),
            .init(name: "P3", residency: 25),
        ])
        #expect(sample?.busyFraction == 0.75)
    }

    @Test
    func zeroTotalResidencyIsUnavailable() {
        // No data in the window → nil, not a fabricated 0%.
        let sample = GPUUtilizationSample.compute(states: [
            .init(name: "OFF", residency: 0),
            .init(name: "P1", residency: 0),
        ])
        #expect(sample == nil)
    }

    @Test
    func emptyOrNilStatesAreUnavailable() {
        #expect(GPUUtilizationSample.compute(states: []) == nil)
        #expect(GPUUtilizationSample.compute(states: nil) == nil)
    }

    @Test
    func idleStateNameMatchingIsRobust() {
        #expect(GPUUtilizationSample.isIdleStateName("OFF"))
        #expect(GPUUtilizationSample.isIdleStateName("off"))
        #expect(GPUUtilizationSample.isIdleStateName("IDLE_OFF"))
        #expect(GPUUtilizationSample.isIdleStateName("DOWN"))
        #expect(!GPUUtilizationSample.isIdleStateName("P1"))
        #expect(!GPUUtilizationSample.isIdleStateName("V0P4"))
    }
}
