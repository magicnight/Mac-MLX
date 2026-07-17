// Copyright © 2026 macMLX. English comments only.

import Darwin

/// Unified-memory pressure for the machine, from fully public, sudoless kernel APIs.
///
/// Two independent signals, both public:
///   * `level` — the kernel's own pressure verdict, from the
///     `kern.memorystatus_vm_pressure_level` sysctl. This is the authoritative
///     "is memory tight" answer (the same signal `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE`
///     delivers) and the analogue of `ThermalPressure`.
///   * the byte breakdown — from `host_statistics64(HOST_VM_INFO64)`. macOS exposes no
///     single "memory pressure %", so rather than invent one we surface the real page
///     classes and a conservative `usedBytes` = active + wired + compressed (the
///     non-reclaimable working set). Free, inactive, speculative and purgeable pages
///     are reclaimable and excluded from "used".
///
/// For an in-process inference engine this matters because unified memory is shared
/// with the GPU: pressure here is what precedes a model-load OOM or a decode stall
/// from compression/swap — context W2 needs and an external monitor cannot correlate
/// with the engine's own allocations.
public struct MemoryPressureSample: Sendable, Equatable {

    /// The kernel's pressure verdict. Raw sysctl values are a small bitmask.
    public enum Level: Int32, Sendable, Equatable {
        case unknown = 0
        case normal = 1
        case warning = 2
        case critical = 4

        /// Map a raw `kern.memorystatus_vm_pressure_level` value.
        public static func from(raw: Int32) -> Level {
            Level(rawValue: raw) ?? .unknown
        }
    }

    /// Page-class counts and machine totals — the pure input to `make(input:)`, split
    /// out so the byte math is testable without a live kernel. Only the classes that
    /// feed the current breakdown are modelled; the rest of `vm_statistics64` is left
    /// out until a metric needs it.
    public struct VMInput: Sendable, Equatable {
        public var freeCount: UInt64
        public var activeCount: UInt64
        public var wireCount: UInt64
        public var compressorPageCount: UInt64
        public var pageSize: UInt64
        public var totalBytes: UInt64
        public var pressureLevelRaw: Int32

        public init(
            freeCount: UInt64,
            activeCount: UInt64,
            wireCount: UInt64,
            compressorPageCount: UInt64,
            pageSize: UInt64,
            totalBytes: UInt64,
            pressureLevelRaw: Int32
        ) {
            self.freeCount = freeCount
            self.activeCount = activeCount
            self.wireCount = wireCount
            self.compressorPageCount = compressorPageCount
            self.pageSize = pageSize
            self.totalBytes = totalBytes
            self.pressureLevelRaw = pressureLevelRaw
        }
    }

    public let level: Level
    /// Installed unified memory in bytes.
    public let totalBytes: UInt64
    /// Non-reclaimable working set: active + wired + compressed, in bytes.
    public let usedBytes: UInt64
    /// Truly free pages, in bytes.
    public let freeBytes: UInt64
    /// Bytes held by the memory compressor.
    public let compressedBytes: UInt64
    /// Wired (unpageable) bytes.
    public let wiredBytes: UInt64

    /// `usedBytes / totalBytes`, 0 when the total is unknown.
    public var usedFraction: Double {
        totalBytes > 0 ? Double(usedBytes) / Double(totalBytes) : 0
    }

    public init(
        level: Level,
        totalBytes: UInt64,
        usedBytes: UInt64,
        freeBytes: UInt64,
        compressedBytes: UInt64,
        wiredBytes: UInt64
    ) {
        self.level = level
        self.totalBytes = totalBytes
        self.usedBytes = usedBytes
        self.freeBytes = freeBytes
        self.compressedBytes = compressedBytes
        self.wiredBytes = wiredBytes
    }

    /// Build a sample from raw page counts — pure, so it is unit-testable without a
    /// live kernel. `usedBytes` is capped at `totalBytes` to absorb the small races
    /// between the several sysctls that feed a live read.
    public static func make(input: VMInput) -> MemoryPressureSample {
        let ps = input.pageSize
        let wired = input.wireCount &* ps
        let compressed = input.compressorPageCount &* ps
        let active = input.activeCount &* ps
        let free = input.freeCount &* ps
        let workingSet = active &+ wired &+ compressed
        let used = min(workingSet, input.totalBytes)
        return MemoryPressureSample(
            level: Level.from(raw: input.pressureLevelRaw),
            totalBytes: input.totalBytes,
            usedBytes: used,
            freeBytes: free,
            compressedBytes: compressed,
            wiredBytes: wired
        )
    }

    /// Read live memory pressure. Returns `nil` only if `host_statistics64` fails,
    /// which does not happen on a healthy system.
    public static func current() -> MemoryPressureSample? {
        var stats = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.stride / MemoryLayout<integer_t>.stride
        )
        let host = mach_host_self()
        let result = withUnsafeMutablePointer(to: &stats) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                host_statistics64(host, HOST_VM_INFO64, rebound, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }

        var pageSize: vm_size_t = 0
        guard host_page_size(host, &pageSize) == KERN_SUCCESS, pageSize > 0 else {
            return nil
        }

        var total: UInt64 = 0
        var totalSize = MemoryLayout<UInt64>.size
        sysctlbyname("hw.memsize", &total, &totalSize, nil, 0)

        var pressureRaw: Int32 = 0
        var pressureSize = MemoryLayout<Int32>.size
        sysctlbyname(
            "kern.memorystatus_vm_pressure_level", &pressureRaw, &pressureSize, nil, 0
        )

        let input = VMInput(
            freeCount: UInt64(stats.free_count),
            activeCount: UInt64(stats.active_count),
            wireCount: UInt64(stats.wire_count),
            compressorPageCount: UInt64(stats.compressor_page_count),
            pageSize: UInt64(pageSize),
            totalBytes: total,
            pressureLevelRaw: pressureRaw
        )
        return make(input: input)
    }
}
