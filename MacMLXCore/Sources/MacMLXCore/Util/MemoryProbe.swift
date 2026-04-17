import Darwin
import Darwin.Mach

/// Reads memory sizes from the kernel.
public enum MemoryProbe {
    /// Total physical memory in gigabytes (10^9 bytes — Apple's "GB" convention).
    ///
    /// Returns 0 on failure (sysctl error). Always non-negative.
    public static func totalMemoryGB() -> Double {
        var size: UInt64 = 0
        var sizeOfSize = MemoryLayout<UInt64>.size
        let result = sysctlbyname("hw.memsize", &size, &sizeOfSize, nil, 0)
        guard result == 0 else { return 0 }
        return Double(size) / 1_000_000_000
    }

    /// Resident set size of the current Mach task, in bytes.
    ///
    /// Non-blocking `task_info` read — safe to call from any thread.
    /// Returns 0 on kernel failure. Used by:
    /// - `HummingbirdServer` to populate `/v1/status`'s `memory_used_gb`
    /// - `BenchmarkRunner`'s peak sampler to catch short-lived spikes
    public static func residentMemoryBytes() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info_data_t>.size / MemoryLayout<integer_t>.size
        )
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                task_info(
                    mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    rebound,
                    &count
                )
            }
        }
        return status == KERN_SUCCESS ? info.resident_size : 0
    }

    /// `residentMemoryBytes()` expressed in GB (10^9 bytes, Apple convention).
    public static func residentMemoryGB() -> Double {
        Double(residentMemoryBytes()) / 1_000_000_000
    }
}
