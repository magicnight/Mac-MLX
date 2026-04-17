import Testing
import Foundation
@testable import MacMLXCore

// Smoke tests — we can't assert specific chip names (test host varies),
// but we can verify the shape is sane: non-empty strings and non-zero
// memory on any working macOS host.

@Test
func hardwareInfoReturnsNonEmptyChipName() {
    let info = HardwareInfo.snapshot()
    #expect(!info.chip.isEmpty, "chip name should be non-empty on a working host")
}

@Test
func hardwareInfoReturnsNonZeroMemory() {
    let info = HardwareInfo.snapshot()
    #expect(info.ramGB > 0, "ramGB should be > 0 on a working host")
}

@Test
func hardwareInfoReturnsPlausibleMacOSVersion() {
    let info = HardwareInfo.snapshot()
    #expect(info.macOSVersion.contains("."), "macOSVersion should look like X.Y or X.Y.Z")
    // Running this test presupposes macOS 14+ per project requirement,
    // so the major version number should be at least 14.
    let major = Int(info.macOSVersion.split(separator: ".").first ?? "0") ?? 0
    #expect(major >= 14, "macOS 14+ expected per project requirement; got \(info.macOSVersion)")
}
