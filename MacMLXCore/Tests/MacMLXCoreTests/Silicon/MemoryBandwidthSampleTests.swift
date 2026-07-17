// Copyright © 2026 macMLX. English comments only.

import Testing

@testable import MacMLXCore

/// Pure-logic coverage for the residency-histogram → GB/s estimator. No IOReport, no
/// hardware — these run under bare `swift test`.
struct SiliconMemoryBandwidthTests {

    // MARK: - Label parsing

    @Test
    func parsesPaddedBandwidthLabel() {
        #expect(MemoryBandwidthSample.parseBandwidthLabelGBs("  32GB/s") == 32)
        #expect(MemoryBandwidthSample.parseBandwidthLabelGBs("1GB/s") == 1)
        #expect(MemoryBandwidthSample.parseBandwidthLabelGBs("128GB/s") == 128)
    }

    @Test
    func parsesLabelCaseInsensitively() {
        #expect(MemoryBandwidthSample.parseBandwidthLabelGBs("32gb/s") == 32)
    }

    @Test
    func rejectsNonBandwidthLabels() {
        #expect(MemoryBandwidthSample.parseBandwidthLabelGBs("OFF") == nil)
        #expect(MemoryBandwidthSample.parseBandwidthLabelGBs("P1") == nil)
        #expect(MemoryBandwidthSample.parseBandwidthLabelGBs("") == nil)
    }

    // MARK: - Estimation

    @Test
    func estimatesMidpointOfSingleOccupiedBand() {
        // All residency in the "4GB/s" band → midpoint of (3, 4] = 3.5.
        let sample = MemoryBandwidthSample.estimate(
            upperEdgesGBs: [1, 2, 3, 4],
            residencies: [0, 0, 0, 10]
        )
        #expect(sample?.estimatedGBPerSecond == 3.5)
        // Top band carried all residency → flagged saturated.
        #expect(sample?.isSaturated == true)
        #expect(sample?.isEstimated == true)
    }

    @Test
    func firstBandUsesZeroLowerEdge() {
        // All residency in the first "1GB/s" band → midpoint of (0, 1] = 0.5.
        let sample = MemoryBandwidthSample.estimate(
            upperEdgesGBs: [1, 2, 3, 4],
            residencies: [10, 0, 0, 0]
        )
        #expect(sample?.estimatedGBPerSecond == 0.5)
        #expect(sample?.isSaturated == false)
    }

    @Test
    func weightsBandsByResidencyShare() {
        // Half in (0,1] (mid 0.5), half in (1,2] (mid 1.5) → 0.25 + 0.75 = 1.0.
        let sample = MemoryBandwidthSample.estimate(
            upperEdgesGBs: [1, 2],
            residencies: [1, 1]
        )
        #expect(sample?.estimatedGBPerSecond == 1.0)
    }

    @Test
    func coarseAggregateBandsResolveMidpoint() {
        // AMCC-style coarse bands: all residency in the first "32GB/s" band →
        // midpoint of (0, 32] = 16, demonstrating why we must not report the label.
        let sample = MemoryBandwidthSample.estimate(
            upperEdgesGBs: [32, 64, 96, 128],
            residencies: [10, 0, 0, 0]
        )
        #expect(sample?.estimatedGBPerSecond == 16)
    }

    @Test
    func idleRowIsZeroNotNil() {
        // A present-but-idle requestor (all-zero residency) is a real 0 GB/s reading,
        // distinct from "no data" (nil).
        let sample = MemoryBandwidthSample.estimate(
            upperEdgesGBs: [1, 2, 3],
            residencies: [0, 0, 0]
        )
        #expect(sample?.estimatedGBPerSecond == 0)
        #expect(sample?.isSaturated == false)
    }

    @Test
    func rejectsEmptyOrMismatchedArrays() {
        #expect(MemoryBandwidthSample.estimate(upperEdgesGBs: [], residencies: []) == nil)
        #expect(
            MemoryBandwidthSample.estimate(upperEdgesGBs: [1, 2], residencies: [1]) == nil
        )
    }

    // MARK: - From raw state residencies

    @Test
    func estimatesFromStatesDroppingNonBandwidthLabels() {
        let states: [IOReportChannelSample.StateResidency] = [
            .init(name: "1GB/s", residency: 0),
            .init(name: "2GB/s", residency: 4),
            .init(name: "bogus", residency: 999),  // dropped: not a bandwidth band
        ]
        // Only the two GB/s bands count: all residency in (1,2] → midpoint 1.5.
        let sample = MemoryBandwidthSample.estimate(states: states)
        #expect(sample?.estimatedGBPerSecond == 1.5)
    }

    @Test
    func estimatesNilWhenNoBandwidthStates() {
        let states: [IOReportChannelSample.StateResidency] = [
            .init(name: "OFF", residency: 10),
            .init(name: "P1", residency: 5),
        ]
        #expect(MemoryBandwidthSample.estimate(states: states) == nil)
        #expect(MemoryBandwidthSample.estimate(states: nil) == nil)
    }
}
