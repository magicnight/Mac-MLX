import Foundation

/// A model file or directory present on the local filesystem.
public struct LocalModel: Codable, Hashable, Identifiable, Sendable {
    public let id: String
    public let displayName: String
    public let directory: URL
    public let sizeBytes: Int64
    public let format: ModelFormat
    public let quantization: String?
    public let parameterCount: String?
    public let architecture: String?

    public init(
        id: String,
        displayName: String,
        directory: URL,
        sizeBytes: Int64,
        format: ModelFormat,
        quantization: String?,
        parameterCount: String?,
        architecture: String?
    ) {
        self.id = id
        self.displayName = displayName
        self.directory = directory
        self.sizeBytes = sizeBytes
        self.format = format
        self.quantization = quantization
        self.parameterCount = parameterCount
        self.architecture = architecture
    }

    /// Human-readable size, e.g. "4.50 GB" or "950 MB".
    ///
    /// Uses base-10 units (Apple convention for advertised RAM/disk).
    /// Deterministic format — no locale dependence — so it's safe to assert in tests.
    public var humanSize: String {
        let bytes = Double(sizeBytes)
        if bytes >= 1_000_000_000 {
            return String(format: "%.2f GB", bytes / 1_000_000_000)
        }
        if bytes >= 1_000_000 {
            return String(format: "%.0f MB", bytes / 1_000_000)
        }
        if bytes >= 1_000 {
            return String(format: "%.0f KB", bytes / 1_000)
        }
        return "\(sizeBytes) B"
    }
}

/// Recognised on-disk model formats.
public enum ModelFormat: String, Codable, Hashable, Sendable, CaseIterable {
    case mlx
    case gguf
    case unknown

    /// Heuristic classifier from a directory's file listing.
    public static func detect(in fileNames: [String]) -> ModelFormat {
        let lower = fileNames.map { $0.lowercased() }
        if lower.contains(where: { $0.hasSuffix(".gguf") }) { return .gguf }
        let mlxSignals = ["config.json", "tokenizer.json"]
        let safetensors = lower.contains(where: { $0.hasSuffix(".safetensors") })
        let signalHits = mlxSignals.allSatisfy { lower.contains($0) }
        if safetensors && signalHits { return .mlx }
        return .unknown
    }
}
