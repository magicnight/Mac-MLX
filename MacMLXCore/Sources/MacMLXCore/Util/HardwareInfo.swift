import Foundation
import Darwin

/// Host-machine introspection for benchmark provenance.
///
/// Why a dedicated util: the benchmark runner needs chip + memory + OS
/// version at observation time, and a chat user running `macmlx.app`
/// should see the same hardware numbers as a developer running
/// `swift test` — both paths resolve against `sysctlbyname` + the
/// ProcessInfo OS version, no bundle context required.
public enum HardwareInfo {

    /// Current snapshot of `SystemInfo` for this machine.
    ///
    /// Safe to call from any actor / thread — pure reads from kernel
    /// sysctl + ProcessInfo. Never throws; unavailable pieces degrade
    /// to empty strings or 0.
    public static func snapshot() -> SystemInfo {
        SystemInfo(
            chip: chipName(),
            ramGB: totalMemoryGB(),
            macOSVersion: macOSVersion()
        )
    }

    // MARK: - Internals

    /// Chip marketing name, e.g. `"Apple M3 Pro"`.
    /// Reads `machdep.cpu.brand_string` via `sysctlbyname`. On pre-M-series
    /// Macs this returns the Intel brand; on Apple Silicon it's the chip
    /// family. Returns `""` on failure (shouldn't happen on any modern
    /// macOS install but we never want to crash).
    private static func chipName() -> String {
        var sizeNeeded = 0
        // First call: ask sysctl how big the buffer must be.
        guard sysctlbyname("machdep.cpu.brand_string", nil, &sizeNeeded, nil, 0) == 0,
              sizeNeeded > 0 else {
            return ""
        }
        var buffer = [CChar](repeating: 0, count: sizeNeeded)
        guard sysctlbyname(
            "machdep.cpu.brand_string",
            &buffer,
            &sizeNeeded,
            nil,
            0
        ) == 0 else {
            return ""
        }
        return String(cString: buffer)
    }

    /// Installed unified memory in GiB (2^30). `hw.memsize` reports raw
    /// bytes — we divide by 1 GiB and round. Returns 0 on failure.
    private static func totalMemoryGB() -> Int {
        var bytes: UInt64 = 0
        var size = MemoryLayout<UInt64>.size
        guard sysctlbyname("hw.memsize", &bytes, &size, nil, 0) == 0 else {
            return 0
        }
        return Int(bytes / 1_073_741_824)
    }

    /// macOS version as `"major.minor.patch"`, e.g. `"15.3.1"`.
    private static func macOSVersion() -> String {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        return "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
    }
}
