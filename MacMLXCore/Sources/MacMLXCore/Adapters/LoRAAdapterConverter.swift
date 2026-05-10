import Foundation
import MLX

/// Converts a HuggingFace **PEFT**-format LoRA adapter directory into
/// the **mlx-swift-lm** native format that
/// `MLXLMCommon.LoRAContainer.from(directory:)` expects.
///
/// The two formats differ in three ways:
///
/// 1. **Config schema.** PEFT writes `r`, `lora_alpha`, `target_modules`,
///    `peft_type`. mlx writes `lora_parameters.{rank, scale, keys}` plus
///    `num_layers` and `fine_tune_type`. `scale = lora_alpha / r`.
///
/// 2. **Weight key naming.** PEFT keys are
///    `base_model.model.<path>.lora_A.weight` and
///    `…lora_B.weight`. mlx keys drop the `base_model.model.` prefix,
///    drop the trailing `.weight`, and lowercase the `A`/`B` suffix
///    (`lora_a` / `lora_b`).
///
/// 3. **Tensor shape.** PEFT stores `lora_A` as `[rank, in]` and
///    `lora_B` as `[out, rank]` (so `forward = x @ A.T @ B.T`).
///    mlx stores `lora_a` as `[in, rank]` and `lora_b` as
///    `[rank, out]` (so `forward = x @ a @ b`). Each weight is the
///    transpose of its PEFT counterpart.
///
/// The converter writes the destination as a new directory containing
/// `adapter_config.json` (mlx schema) and `adapters.safetensors`.
/// Source files are not modified — callers that want an in-place
/// conversion should write to a sibling directory and rename.
public enum LoRAAdapterConverter {

    public enum Error: Swift.Error, CustomStringConvertible, Equatable {
        case missingPEFTConfig(URL)
        case missingPEFTWeights(URL)
        case malformedPEFTConfig(String)
        case noLoRAWeightsFound
        case unrecognisedKeyFormat(String)

        public var description: String {
            switch self {
            case .missingPEFTConfig(let url):
                return "PEFT adapter_config.json not found at \(url.path)"
            case .missingPEFTWeights(let url):
                return "PEFT adapter_model.safetensors not found at \(url.path)"
            case .malformedPEFTConfig(let reason):
                return "PEFT adapter_config.json could not be parsed: \(reason)"
            case .noLoRAWeightsFound:
                return "PEFT adapter_model.safetensors contained no recognisable LoRA weights"
            case .unrecognisedKeyFormat(let key):
                return "Unrecognised PEFT weight key shape: \(key)"
            }
        }
    }

    /// Convert one PEFT-format adapter directory into a freshly-written
    /// mlx-format adapter directory.
    ///
    /// - Parameters:
    ///   - source: directory containing `adapter_config.json` +
    ///     `adapter_model.safetensors` (PEFT).
    ///   - destination: directory to write `adapter_config.json` (mlx)
    ///     + `adapters.safetensors`. Should not equal `source` —
    ///     write to a sibling and rename if you want an in-place feel.
    ///   - numLayersOverride: explicit value for the mlx config's
    ///     `num_layers`. Pass `nil` to auto-infer from the deepest
    ///     `model.layers.<N>` index seen in PEFT weight keys.
    public static func convertPEFTAdapter(
        source: URL,
        destination: URL,
        numLayersOverride: Int? = nil
    ) throws {
        let configURL = source.appendingPathComponent("adapter_config.json")
        let weightsURL = source.appendingPathComponent("adapter_model.safetensors")

        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw Error.missingPEFTConfig(configURL)
        }
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw Error.missingPEFTWeights(weightsURL)
        }

        let peftConfig: LocalAdapter.PEFTConfig
        do {
            let data = try Data(contentsOf: configURL)
            peftConfig = try JSONDecoder().decode(LocalAdapter.PEFTConfig.self, from: data)
        } catch {
            throw Error.malformedPEFTConfig(error.localizedDescription)
        }

        let peftArrays = try MLX.loadArrays(url: weightsURL)
        let (mlxArrays, deepestLayer) = try translateWeights(peftArrays)
        guard !mlxArrays.isEmpty else { throw Error.noLoRAWeightsFound }

        let inferredNumLayers = (deepestLayer + 1)
        let mlxConfig = mlxConfiguration(
            from: peftConfig,
            numLayers: numLayersOverride ?? inferredNumLayers
        )

        try FileManager.default.createDirectory(
            at: destination, withIntermediateDirectories: true)

        // Write mlx config.
        let outConfig = destination.appendingPathComponent("adapter_config.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(mlxConfig).write(to: outConfig, options: .atomic)

        // Write mlx safetensors. MLX.save flushes the lazy compute
        // graph internally before serialising — we don't need to
        // manually realise the transposed arrays.
        let outWeights = destination.appendingPathComponent("adapters.safetensors")
        try MLX.save(arrays: mlxArrays, url: outWeights)
    }

    // MARK: - Schema translation

    /// Internal mirror of `MLXLMCommon.LoRAConfiguration` so the
    /// converter compiles in targets that don't import MLXLMCommon.
    /// JSON shape is identical so the file mlx-swift-lm reads back
    /// is byte-equivalent to one it would write itself.
    struct MLXAdapterConfig: Codable, Equatable {
        let numLayers: Int
        let fineTuneType: String
        let loraParameters: LoRAParameters

        struct LoRAParameters: Codable, Equatable {
            let rank: Int
            let scale: Float
            let keys: [String]?

            private enum CodingKeys: String, CodingKey {
                case rank, scale, keys
            }
        }

        private enum CodingKeys: String, CodingKey {
            case numLayers = "num_layers"
            case fineTuneType = "fine_tune_type"
            case loraParameters = "lora_parameters"
        }
    }

    static func mlxConfiguration(
        from peft: LocalAdapter.PEFTConfig,
        numLayers: Int
    ) -> MLXAdapterConfig {
        let rank = peft.r ?? 8
        let alpha = Float(peft.loraAlpha ?? rank)
        let scale = alpha / Float(rank)
        return MLXAdapterConfig(
            numLayers: numLayers,
            fineTuneType: "lora",
            loraParameters: .init(
                rank: rank,
                scale: scale,
                keys: peft.targetModules
            )
        )
    }

    // MARK: - Weight translation

    /// Translate PEFT-shaped weights into mlx-shaped weights.
    ///
    /// Returns the new dictionary plus the deepest `model.layers.<N>`
    /// index seen in the input keys (used to auto-infer `num_layers`
    /// when the caller doesn't override it).
    static func translateWeights(
        _ peftArrays: [String: MLXArray]
    ) throws -> (arrays: [String: MLXArray], deepestLayer: Int) {
        var out: [String: MLXArray] = [:]
        var deepest = -1

        for (peftKey, peftArray) in peftArrays {
            // Only translate keys ending in `.lora_A.weight` or
            // `.lora_B.weight`. Other keys (e.g. PEFT's
            // `…modules_to_save…`) are silently dropped — mlx-swift-lm's
            // runtime cares only about the LoRA pair.
            guard peftKey.hasSuffix(".lora_A.weight") || peftKey.hasSuffix(".lora_B.weight") else {
                continue
            }

            let mlxKey = try mlxKey(forPEFTKey: peftKey)
            // PEFT stores transposed wrt mlx; `.T` materialises lazily,
            // MLX.save flushes the graph below.
            out[mlxKey] = peftArray.T

            if let layerIdx = layerIndex(in: mlxKey) {
                deepest = max(deepest, layerIdx)
            }
        }

        return (out, deepest)
    }

    /// Map one PEFT key to the mlx-equivalent key.
    static func mlxKey(forPEFTKey peftKey: String) throws -> String {
        // Drop the `base_model.` prefix(es). PEFT can wrap the model
        // once (`base_model.model.<…>`) or twice for some causal-LM
        // setups (`base_model.model.model.<…>`).
        var key = peftKey
        while key.hasPrefix("base_model.") {
            key = String(key.dropFirst("base_model.".count))
        }
        // Collapse adjacent `model.` runs so paths like
        // `model.model.layers.0.…` become `model.layers.0.…` to match
        // mlx's module hierarchy.
        while key.hasPrefix("model.model.") {
            key = String(key.dropFirst("model.".count))
        }

        // Suffix rewrite — drop `.weight`, lowercase the A/B side.
        if key.hasSuffix(".lora_A.weight") {
            key = key.dropLast(".lora_A.weight".count) + ".lora_a"
        } else if key.hasSuffix(".lora_B.weight") {
            key = key.dropLast(".lora_B.weight".count) + ".lora_b"
        } else {
            throw Error.unrecognisedKeyFormat(peftKey)
        }
        return key
    }

    /// Extract the integer layer index from `…model.layers.<N>.…`
    /// keys. Returns nil for keys outside the per-layer hierarchy
    /// (embedding adapters etc.).
    static func layerIndex(in mlxKey: String) -> Int? {
        guard let range = mlxKey.range(of: ".layers.") else { return nil }
        let tail = mlxKey[range.upperBound...]
        let segment = tail.prefix { $0.isNumber }
        return Int(segment)
    }
}

// SubSequence + String concat helper used in `mlxKey(forPEFTKey:)`
// to keep the suffix-rewrite line readable.
private func + (lhs: String.SubSequence, rhs: String) -> String {
    String(lhs) + rhs
}
