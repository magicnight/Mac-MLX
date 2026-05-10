import Foundation

/// One LoRA adapter directory present on the local filesystem.
///
/// Discovered by `AdapterStore.scan(_:)` via the presence of
/// `adapter_config.json` + `adapter_model.safetensors` (PEFT format).
/// `targetModel` is advisory — the engine layer applies the adapter
/// regardless and surfaces a clear typed error if the dimensions
/// don't fit the loaded base model.
public struct LocalAdapter: Codable, Hashable, Identifiable, Sendable {
    public var id: String { name }
    public let name: String
    public let directory: URL
    /// Base-model id from the adapter's config (e.g.
    /// `mlx-community/Qwen3-8B-4bit`). Optional — older adapters
    /// don't always carry it.
    public let targetModel: String?
    /// LoRA rank (`r` in PEFT config). Nil if absent / unparseable.
    public let rank: Int?
    /// Names of the linear layers the adapter touches (e.g.
    /// `["q_proj", "v_proj"]`). Empty array if absent.
    public let targetModules: [String]

    public init(
        name: String,
        directory: URL,
        targetModel: String?,
        rank: Int?,
        targetModules: [String]
    ) {
        self.name = name
        self.directory = directory
        self.targetModel = targetModel
        self.rank = rank
        self.targetModules = targetModules
    }

    /// On-disk PEFT `adapter_config.json` shape.
    ///
    /// Exposed publicly so `AdapterStore` and tests can decode it
    /// without re-deriving the schema. Mirrors the subset of HF PEFT
    /// fields we currently consume — extend when we start honouring
    /// `lora_dropout`, `bias`, etc.
    public struct PEFTConfig: Codable, Hashable, Sendable {
        public let baseModelNameOrPath: String?
        public let r: Int?
        public let loraAlpha: Int?
        public let targetModules: [String]?
        public let peftType: String?

        public init(
            baseModelNameOrPath: String?,
            r: Int?,
            loraAlpha: Int?,
            targetModules: [String]?,
            peftType: String?
        ) {
            self.baseModelNameOrPath = baseModelNameOrPath
            self.r = r
            self.loraAlpha = loraAlpha
            self.targetModules = targetModules
            self.peftType = peftType
        }

        private enum CodingKeys: String, CodingKey {
            case baseModelNameOrPath = "base_model_name_or_path"
            case r
            case loraAlpha = "lora_alpha"
            case targetModules = "target_modules"
            case peftType = "peft_type"
        }
    }
}
