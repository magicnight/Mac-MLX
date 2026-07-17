// Copyright © 2026 macMLX. English comments only.

import Testing

@testable import MacMLXCore

/// Pure-logic coverage for the millijoules-over-interval → watts conversion.
struct SiliconPowerSampleTests {

    @Test
    func convertsEnergyOverIntervalToWatts() {
        // 1000 mJ = 1 J over 1 s → 1 W.
        #expect(PowerSample.from(energyMillijoules: 1000, intervalSeconds: 1.0)?.watts == 1.0)
        // 500 mJ over 0.5 s → 1 W.
        #expect(PowerSample.from(energyMillijoules: 500, intervalSeconds: 0.5)?.watts == 1.0)
        // 2000 mJ over 2 s → 1 W.
        #expect(PowerSample.from(energyMillijoules: 2000, intervalSeconds: 2.0)?.watts == 1.0)
    }

    @Test
    func nonPositiveIntervalIsUnavailable() {
        #expect(PowerSample.from(energyMillijoules: 1000, intervalSeconds: 0) == nil)
        #expect(PowerSample.from(energyMillijoules: 1000, intervalSeconds: -1) == nil)
    }

    @Test
    func negativeEnergyClampsToZero() {
        // A counter reset could yield a negative delta; report 0 W, never negative.
        #expect(PowerSample.from(energyMillijoules: -500, intervalSeconds: 1.0)?.watts == 0)
    }
}
