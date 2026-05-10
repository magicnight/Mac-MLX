import Foundation

/// Scans `~/.mac-mlx/adapters/<name>/` for PEFT-format LoRA adapters.
///
/// Mirrors `ModelLibraryManager` shape but for adapters: a directory
/// is recognised when it contains both `adapter_config.json` and
/// `adapter_model.safetensors`. Bad / unreadable configs silently
/// drop — the scan must not blow up because of one malformed
/// directory.
public actor AdapterStore {
    private let fileManager: FileManager

    public init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
    }

    /// Enumerate adapters under `directory`. Sorted by `name`
    /// (case-insensitive locale compare) for stable UI rendering.
    public func scan(_ directory: URL) async throws -> [LocalAdapter] {
        // If the adapters directory hasn't been created yet, treat as
        // empty — the user simply hasn't downloaded any adapters. The
        // GUI is responsible for offering to create the directory.
        guard fileManager.fileExists(atPath: directory.path) else { return [] }

        let contents = try fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )

        var results: [LocalAdapter] = []
        for url in contents {
            guard try url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory == true,
                  !url.lastPathComponent.hasPrefix(".") else { continue }

            if let mlxAdapter = readMLXAdapter(at: url) {
                results.append(mlxAdapter)
            } else if let peftAdapter = readPEFTAdapter(at: url) {
                results.append(peftAdapter)
            }
        }
        return results.sorted { $0.name.localizedCompare($1.name) == .orderedAscending }
    }

    // MARK: - Format-specific decoders

    /// Detect mlx-native format: `adapter_config.json` (mlx schema)
    /// + `adapters.safetensors`. Returns `nil` if either file is
    /// missing or the config doesn't decode cleanly.
    private func readMLXAdapter(at url: URL) -> LocalAdapter? {
        let configURL = url.appendingPathComponent("adapter_config.json")
        let weightsURL = url.appendingPathComponent("adapters.safetensors")
        guard fileManager.fileExists(atPath: configURL.path),
              fileManager.fileExists(atPath: weightsURL.path),
              let data = try? Data(contentsOf: configURL),
              let cfg = try? JSONDecoder().decode(MLXAdapterConfig.self, from: data)
        else { return nil }
        return LocalAdapter(
            name: url.lastPathComponent,
            directory: url,
            format: .mlx,
            targetModel: nil, // mlx-native config doesn't carry base model id
            rank: cfg.loraParameters.rank,
            targetModules: cfg.loraParameters.keys ?? []
        )
    }

    /// Detect PEFT format: `adapter_config.json` (PEFT schema)
    /// + `adapter_model.safetensors`. Returns `nil` if either file is
    /// missing or the config doesn't decode cleanly.
    private func readPEFTAdapter(at url: URL) -> LocalAdapter? {
        let configURL = url.appendingPathComponent("adapter_config.json")
        let weightsURL = url.appendingPathComponent("adapter_model.safetensors")
        guard fileManager.fileExists(atPath: configURL.path),
              fileManager.fileExists(atPath: weightsURL.path),
              let data = try? Data(contentsOf: configURL),
              let cfg = try? JSONDecoder().decode(LocalAdapter.PEFTConfig.self, from: data)
        else { return nil }
        return LocalAdapter(
            name: url.lastPathComponent,
            directory: url,
            format: .peft,
            targetModel: cfg.baseModelNameOrPath,
            rank: cfg.r,
            targetModules: cfg.targetModules ?? []
        )
    }
}

/// Minimal mirror of `MLXLMCommon.LoRAConfiguration` shape used to
/// detect mlx-native adapter directories without depending on
/// MLXLMCommon at this layer (Manager file stays MLX-free).
private struct MLXAdapterConfig: Decodable {
    let numLayers: Int
    let fineTuneType: String
    let loraParameters: LoRAParameters

    struct LoRAParameters: Decodable {
        let rank: Int
        let scale: Float
        let keys: [String]?
    }

    private enum CodingKeys: String, CodingKey {
        case numLayers = "num_layers"
        case fineTuneType = "fine_tune_type"
        case loraParameters = "lora_parameters"
    }
}
