import Foundation

/// One benchmark observation. Persisted as JSON in `~/.mac-mlx/benchmarks/`.
public struct BenchmarkResult: Codable, Hashable, Identifiable, Sendable {
    public let id: UUID
    public let modelID: String
    public let engineID: EngineID
    public let promptTokens: Int
    public let completionTokens: Int
    /// Tokens per second during prompt processing (prefill).
    public let promptTPS: Double
    /// Tokens per second during generation (decode).
    public let generationTPS: Double
    /// Time to first token, in milliseconds.
    public let ttftMs: Double
    public let timestamp: Date
    public let system: SystemInfo

    public init(
        id: UUID = UUID(),
        modelID: String,
        engineID: EngineID,
        promptTokens: Int,
        completionTokens: Int,
        promptTPS: Double,
        generationTPS: Double,
        ttftMs: Double,
        timestamp: Date = Date(),
        system: SystemInfo
    ) {
        self.id = id
        self.modelID = modelID
        self.engineID = engineID
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.promptTPS = promptTPS
        self.generationTPS = generationTPS
        self.ttftMs = ttftMs
        self.timestamp = timestamp
        self.system = system
    }
}

/// Snapshot of the host machine at benchmark time.
public struct SystemInfo: Codable, Hashable, Sendable {
    public let chip: String
    public let ramGB: Int

    public init(chip: String, ramGB: Int) {
        self.chip = chip
        self.ramGB = ramGB
    }
}
