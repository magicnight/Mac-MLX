import Darwin

/// Reads physical memory size from the kernel.
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
}
