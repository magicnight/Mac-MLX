// Copyright © 2026 macMLX. English comments only.

import Foundation
import Testing

@testable import MacMLXCore

/// Pure-logic coverage for the `ProcessInfo.ThermalState` mapping and ordering.
struct SiliconThermalPressureTests {

    @Test
    func mapsEveryProcessInfoState() {
        #expect(ThermalPressure.from(.nominal) == .nominal)
        #expect(ThermalPressure.from(.fair) == .fair)
        #expect(ThermalPressure.from(.serious) == .serious)
        #expect(ThermalPressure.from(.critical) == .critical)
    }

    @Test
    func isOrderedBySeverity() {
        #expect(ThermalPressure.nominal < ThermalPressure.fair)
        #expect(ThermalPressure.fair < ThermalPressure.serious)
        #expect(ThermalPressure.serious < ThermalPressure.critical)
        #expect(ThermalPressure.allCases.count == 4)
    }

    @Test
    func currentReturnsAValidLevel() {
        // Reads the live public API; any of the four levels is acceptable.
        let current = ThermalPressure.current()
        #expect(ThermalPressure.allCases.contains(current))
    }
}
