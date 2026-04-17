import Foundation

/// One benchmark observation. Persisted as JSON in `~/.mac-mlx/benchmarks/`.
///
/// The shape aligns with `.claude/features/benchmark.md`. Fields added in
/// v0.3 (macMLX/engine versions, memory, load time, runs, notes) all have
/// sensible defaults so pre-v0.3 call sites — including the existing Stage
/// 2 test — keep compiling without change.
public struct BenchmarkResult: Codable, Hashable, Identifiable, Sendable {
    public let id: UUID
    public let modelID: String
    public let engineID: EngineID

    // Test conditions
    public let promptTokens: Int
    public let completionTokens: Int
    public let runs: Int

    // Measured metrics (medians across `runs` iterations)
    /// Tokens per second during prompt processing (prefill).
    public let promptTPS: Double
    /// Tokens per second during generation (decode).
    public let generationTPS: Double
    /// Time to first token, in milliseconds.
    public let ttftMs: Double
    /// Peak unified-memory footprint observed during the run, in GB.
    /// 0 if the runner couldn't sample (e.g. in test mocks).
    public let memoryUsedGB: Double
    /// Cold model-load time, in seconds. 0 when the model was already loaded.
    public let modelLoadTimeS: Double

    // Provenance (filled at run time)
    public let timestamp: Date
    public let system: SystemInfo
    /// Marketing version of macMLX that produced this result
    /// (`CFBundleShortVersionString`). Empty if unavailable.
    public let macMLXVersion: String
    /// Version string returned by the engine's `version` property.
    public let engineVersion: String

    // User annotations
    /// Free-form notes the user can add before sharing.
    public let notes: String

    public init(
        id: UUID = UUID(),
        modelID: String,
        engineID: EngineID,
        promptTokens: Int,
        completionTokens: Int,
        runs: Int = 3,
        promptTPS: Double,
        generationTPS: Double,
        ttftMs: Double,
        memoryUsedGB: Double = 0,
        modelLoadTimeS: Double = 0,
        timestamp: Date = Date(),
        system: SystemInfo,
        macMLXVersion: String = "",
        engineVersion: String = "",
        notes: String = ""
    ) {
        self.id = id
        self.modelID = modelID
        self.engineID = engineID
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.runs = runs
        self.promptTPS = promptTPS
        self.generationTPS = generationTPS
        self.ttftMs = ttftMs
        self.memoryUsedGB = memoryUsedGB
        self.modelLoadTimeS = modelLoadTimeS
        self.timestamp = timestamp
        self.system = system
        self.macMLXVersion = macMLXVersion
        self.engineVersion = engineVersion
        self.notes = notes
    }
}

/// Snapshot of the host machine at benchmark time.
public struct SystemInfo: Codable, Hashable, Sendable {
    /// Chip name as reported by `sysctlbyname("machdep.cpu.brand_string", …)`,
    /// e.g. `"Apple M3 Pro"`.
    public let chip: String
    /// Installed unified memory in GiB (2^30 convention — what `hw.memsize`
    /// actually reports, even though Apple marketing uses GB=10^9).
    public let ramGB: Int
    /// macOS version string, e.g. `"15.3.1"`. Empty if unavailable.
    public let macOSVersion: String

    public init(
        chip: String,
        ramGB: Int,
        macOSVersion: String = ""
    ) {
        self.chip = chip
        self.ramGB = ramGB
        self.macOSVersion = macOSVersion
    }
}
