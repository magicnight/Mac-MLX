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
    /// `true` when this entry was discovered by scanning a Hugging Face
    /// Hub cache directory (Track F HF-cache discovery) rather than the
    /// app's managed model directory. `directory` still points straight
    /// at the cache snapshot — nothing is copied — so the GUI uses this
    /// flag to hide destructive actions (deleting it would remove files
    /// from the user's shared HF cache, not an app-owned copy). Defaults
    /// to `false` so every pre-existing call site (and pre-Track-F
    /// persisted JSON, if any) keeps working unchanged.
    public let isExternalReference: Bool

    public init(
        id: String,
        displayName: String,
        directory: URL,
        sizeBytes: Int64,
        format: ModelFormat,
        quantization: String?,
        parameterCount: String?,
        architecture: String?,
        isExternalReference: Bool = false
    ) {
        self.id = id
        self.displayName = displayName
        self.directory = directory
        self.sizeBytes = sizeBytes
        self.format = format
        self.quantization = quantization
        self.parameterCount = parameterCount
        self.architecture = architecture
        self.isExternalReference = isExternalReference
    }

    private enum CodingKeys: String, CodingKey {
        case id, displayName, directory, sizeBytes, format
        case quantization, parameterCount, architecture, isExternalReference
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try c.decode(String.self, forKey: .id)
        self.displayName = try c.decode(String.self, forKey: .displayName)
        self.directory = try c.decode(URL.self, forKey: .directory)
        self.sizeBytes = try c.decode(Int64.self, forKey: .sizeBytes)
        self.format = try c.decode(ModelFormat.self, forKey: .format)
        self.quantization = try c.decodeIfPresent(String.self, forKey: .quantization)
        self.parameterCount = try c.decodeIfPresent(String.self, forKey: .parameterCount)
        self.architecture = try c.decodeIfPresent(String.self, forKey: .architecture)
        self.isExternalReference =
            try c.decodeIfPresent(Bool.self, forKey: .isExternalReference) ?? false
    }

    /// Filters `models` down to viable speculative-decoding draft-model
    /// candidates for a chat session currently targeting `currentModelID`
    /// (Track F GUI over the D1 engine plumbing).
    ///
    /// A candidate must be:
    /// 1. `.mlx` format — an obviously-wrong-shape filter for
    ///    `.mlxVLM`/`.embedder`/`.gguf`, not a full compatibility probe;
    ///    tokenizer/cache compatibility is still checked (and silently
    ///    falls back) at generation time by the engine.
    /// 2. Not the currently-loaded target model itself.
    /// 3. Not an HF-cache external reference (`isExternalReference`):
    ///    those ids are always shaped `"org/name"`, and
    ///    `MLXSwiftEngine.isValidDraftModelID`'s allowlist
    ///    (`[A-Za-z0-9._-]+`, no `/`) unconditionally rejects them —
    ///    surfacing one here would let the user pick a draft model that
    ///    hard-fails every generation round with
    ///    `EngineError.invalidDraftModelID` AND gets persisted to disk via
    ///    `ModelParametersStore`.
    ///
    /// Extracted as a pure, Core-side function (rather than left inline in
    /// `ParametersInspector`'s SwiftUI body) so this filtering logic is
    /// unit-testable without a view hierarchy.
    public static func draftCandidates(
        from models: [LocalModel], excluding currentModelID: String?
    ) -> [LocalModel] {
        models.filter {
            $0.format == .mlx && $0.id != currentModelID && !$0.isExternalReference
        }
    }

    /// `model_type` values that denote a dedicated OCR model (as opposed to a
    /// general-purpose vision-language model). Deliberately narrow — only
    /// architectures whose whole purpose is text recognition — so the "OCR" badge
    /// stays meaningful rather than tagging every VLM that can incidentally read text.
    /// `glm_ocr` is verified end-to-end (GLM-OCR runs through the stock VLM path, see
    /// `GLMOCRSmokeTests`); `dots_ocr` is unambiguously OCR and pre-listed so it earns
    /// the badge automatically once its VLM port lands — until then the `.mlxVLM` gate
    /// in `isOCR` withholds it, so the badge never appears on a model that can't load.
    /// Ambiguous general VLMs (e.g. `deepseek_vl_v2`, which also ships as a plain
    /// DeepSeek-VL) are excluded. Lowercased — compared case-insensitively.
    private static let ocrModelTypes: Set<String> = ["glm_ocr", "dots_ocr"]

    /// Whether this is a dedicated OCR model, driving the "OCR" badge (distinct from
    /// the generic "Vision" badge other VLMs get). Two conditions: the config
    /// `model_type` (carried in `architecture` by the scan) is a known OCR family, AND
    /// the model is actually vision-routable (`.mlxVLM`). The format gate keeps the
    /// badge truthful — an OCR family upstream can't yet load scans as plain `.mlx`
    /// text and shows no OCR badge until its port makes it a real VLM. Badge-only:
    /// loading is unaffected (an OCR model loads through `VLMModelFactory` like any VLM).
    public var isOCR: Bool {
        guard format == .mlxVLM, let architecture else { return false }
        return Self.ocrModelTypes.contains(architecture.lowercased())
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
    /// Vision-language model (v0.4.1+). Same on-disk shape as `.mlx`,
    /// distinguished by `model_type` in `config.json`. The library
    /// scan first runs `detect(in:)` to filter MLX / GGUF / unknown
    /// from the file listing, then upgrades `.mlx` → `.mlxVLM` if the
    /// `model_type` matches a known VLM family.
    case mlxVLM
    /// Text-embedding model (v0.5.2+). Same on-disk shape as `.mlx`
    /// (config.json + tokenizer + `.safetensors`), distinguished by an
    /// encoder `model_type` (`bert`, `roberta`, …) in `config.json`. Served
    /// by `EmbeddingEngine` via `/v1/embeddings` + `/v1/rerank`, never by the
    /// generation engine. Set by `ModelLibraryManager.upgradeFormat` after
    /// the initial `.mlx` file-listing detection.
    case embedder
    case gguf
    case unknown

    /// Heuristic classifier from a directory's file listing.
    ///
    /// File-listing inspection only — no I/O on the contents. Returns
    /// `.mlx` for any directory that looks like an MLX text model;
    /// `ModelLibraryManager.scan(_:)` is responsible for the further
    /// `.mlx` → `.mlxVLM` upgrade based on `config.json`.
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
