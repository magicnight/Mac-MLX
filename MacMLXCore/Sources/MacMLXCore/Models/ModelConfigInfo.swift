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
    /// `config.json`'s `architectures` array (e.g.
    /// `["BertForSequenceClassification"]`, `["Qwen3ForCausalLM"]`),
    /// verbatim and case-preserved. `nil` when the key is absent.
    ///
    /// The load-bearing field for reranker detection: a cross-encoder
    /// reranker shares its `model_type` (`bert` / `xlm-roberta`) with the
    /// text embedders macMLX auto-classifies as `.embedder`, so `model_type`
    /// alone can't tell them apart — the `*ForSequenceClassification` head in
    /// `architectures` is what distinguishes a reranker (see
    /// `ModelLibraryManager.upgradeFormat`).
    public let architectures: [String]?
    /// `config.json`'s `num_labels` (a `*ForSequenceClassification` head's
    /// output width). `nil` when absent — many reranker checkpoints omit it
    /// and rely on the HF default of `1`.
    ///
    /// The second reranker-detection signal, alongside `architectures`: a
    /// GENUINE multi-class classifier (e.g. a 5-label sentiment BERT) also
    /// carries a `*ForSequenceClassification` architecture but must NOT be
    /// mistaken for a single-logit reranker — `RerankEngine` always builds
    /// `Linear(hidden, 1)`, so a checkpoint whose real `classifier.weight` is
    /// `[N, hidden]` (`N > 1`) would fail `verify: [.all]` with a cryptic
    /// load error instead of a clear "not a reranker" classification.
    public let numLabels: Int?
    /// The number of entries in `config.json`'s `id2label` map, if present.
    /// A second, independent source for the same single-vs-multi-label
    /// signal `numLabels` provides — some checkpoints populate `id2label`
    /// without an explicit `num_labels`.
    public let id2labelCount: Int?

    public init(
        modelType: String?,
        quantizationBits: Int?,
        quantizationGroupSize: Int?,
        contextLength: Int?,
        architectures: [String]? = nil,
        numLabels: Int? = nil,
        id2labelCount: Int? = nil
    ) {
        self.modelType = modelType
        self.quantizationBits = quantizationBits
        self.quantizationGroupSize = quantizationGroupSize
        self.contextLength = contextLength
        self.architectures = architectures
        self.numLabels = numLabels
        self.id2labelCount = id2labelCount
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

        // Case-preserved — `hasSuffix("ForSequenceClassification")` in the
        // reranker classifier is case-sensitive.
        let architectures = json["architectures"] as? [String]

        let numLabels = json["num_labels"] as? Int
        // `id2label` is a `{"0": "LABEL_0", ...}` map in config.json — its
        // entry count is the label count, same signal as `num_labels`.
        let id2labelCount = (json["id2label"] as? [String: Any])?.count

        return ModelConfigInfo(
            modelType: modelType,
            quantizationBits: bits,
            quantizationGroupSize: groupSize,
            contextLength: contextLength,
            architectures: architectures,
            numLabels: numLabels,
            id2labelCount: id2labelCount
        )
    }
}
