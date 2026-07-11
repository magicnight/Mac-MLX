import Foundation

/// Best-effort parse of a model directory's `config.json`, surfacing the
/// handful of fields the library scanner's quantization/architecture
/// inference and the GUI's model card care about.
///
/// Never throws — a missing file, malformed JSON, or absent key simply
/// leaves the corresponding property `nil`, mirroring
/// `ModelLibraryManager`'s existing best-effort config peeking
/// (`upgradeFormat(directory:)`): a scan/read must not blow up because of
/// one unparseable config.
public struct ModelConfigInfo: Sendable, Equatable {
    /// `config.json`'s `model_type` field, lowercased (e.g. `"qwen3"`,
    /// `"gemma3"`). `nil` if absent.
    public let modelType: String?
    /// `quantization.bits` from an mlx-lm-quantized `config.json`
    /// (e.g. `4`, `8`). `nil` for an unquantized (bf16/fp16) export or
    /// a config with no `quantization` block.
    public let quantizationBits: Int?
    /// `quantization.group_size`, alongside `quantizationBits`. `nil`
    /// whenever `quantizationBits` is `nil`.
    public let quantizationGroupSize: Int?
    /// Maximum context length, read from whichever of the common HF
    /// `config.json` field names is present first:
    /// `max_position_embeddings`, `max_sequence_length`, `n_positions`,
    /// `seq_length`. `nil` if none are present.
    public let contextLength: Int?

    public init(
        modelType: String?,
        quantizationBits: Int?,
        quantizationGroupSize: Int?,
        contextLength: Int?
    ) {
        self.modelType = modelType
        self.quantizationBits = quantizationBits
        self.quantizationGroupSize = quantizationGroupSize
        self.contextLength = contextLength
    }

    /// Reads and parses `<directory>/config.json`. Returns `nil` when the
    /// file is missing or isn't valid JSON — callers treat that the same
    /// as "no info available" rather than an error.
    public static func read(
        from directory: URL,
        fileManager: FileManager = .default
    ) -> ModelConfigInfo? {
        let configURL = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }

        let modelType = (json["model_type"] as? String)?.lowercased()

        var bits: Int?
        var groupSize: Int?
        if let quant = json["quantization"] as? [String: Any] {
            bits = quant["bits"] as? Int
            groupSize = quant["group_size"] as? Int
        }

        let contextLength =
            (json["max_position_embeddings"] as? Int)
            ?? (json["max_sequence_length"] as? Int)
            ?? (json["n_positions"] as? Int)
            ?? (json["seq_length"] as? Int)

        return ModelConfigInfo(
            modelType: modelType,
            quantizationBits: bits,
            quantizationGroupSize: groupSize,
            contextLength: contextLength
        )
    }
}
