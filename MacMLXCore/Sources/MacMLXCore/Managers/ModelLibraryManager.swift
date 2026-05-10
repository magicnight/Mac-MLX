import Foundation

/// Scans the local filesystem for MLX model directories and maintains a cached list.
public actor ModelLibraryManager {

    // MARK: - Properties

    private let fileManager: FileManager

    /// Most recent scan result. `nil` if `scan(_:)` has not been called yet.
    public private(set) var lastScan: [LocalModel] = []

    // MARK: - Init

    public init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
    }

    // MARK: - Public API

    /// Scan `directory` for subdirectories that look like MLX models.
    ///
    /// - Each top-level subdirectory is inspected by calling `ModelFormat.detect(in:)`.
    /// - MLX models are returned as `LocalModel` values sorted by `displayName`.
    /// - GGUF and unknown directories are silently skipped (with a `print` notice for GGUF).
    /// - Hidden directories (names starting with `.`) are always skipped.
    ///
    /// - Parameter directory: The root directory to scan.
    /// - Returns: Sorted array of discovered `LocalModel` values.
    @discardableResult
    public func scan(_ directory: URL) async throws -> [LocalModel] {
        let contents = try fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )

        var results: [LocalModel] = []

        for itemURL in contents {
            guard try isDirectory(itemURL) else { continue }

            let dirName = itemURL.lastPathComponent
            guard !dirName.hasPrefix(".") else { continue }

            let fileURLs = (try? fileManager.contentsOfDirectory(
                at: itemURL,
                includingPropertiesForKeys: [.fileSizeKey],
                options: [.skipsHiddenFiles]
            )) ?? []

            let fileNames = fileURLs.map { $0.lastPathComponent }
            let format = ModelFormat.detect(in: fileNames)

            switch format {
            case .mlx:
                // Peek `config.json` `model_type` — upgrade to .mlxVLM
                // when the directory contains a vision-language model.
                let upgradedFormat = upgradeFormatIfVLM(directory: itemURL)
                let model = buildLocalModel(
                    dirName: dirName,
                    dirURL: itemURL,
                    fileURLs: fileURLs,
                    format: upgradedFormat
                )
                results.append(model)

            case .mlxVLM:
                // `ModelFormat.detect(in:)` never returns this directly
                // — it's set by `upgradeFormatIfVLM` above. Reachable
                // only via tests that hand-craft a format. Fall through
                // to the same path as `.mlx`.
                let model = buildLocalModel(
                    dirName: dirName,
                    dirURL: itemURL,
                    fileURLs: fileURLs,
                    format: .mlxVLM
                )
                results.append(model)

            case .gguf:
                print("[ModelLibraryManager] Skipping GGUF directory: \(dirName)")

            case .unknown:
                break
            }
        }

        let sorted = results.sorted { $0.displayName.localizedCompare($1.displayName) == .orderedAscending }
        lastScan = sorted
        return sorted
    }

    // MARK: - Private Helpers

    private func isDirectory(_ url: URL) throws -> Bool {
        let values = try url.resourceValues(forKeys: [.isDirectoryKey])
        return values.isDirectory == true
    }

    private func buildLocalModel(
        dirName: String,
        dirURL: URL,
        fileURLs: [URL],
        format: ModelFormat = .mlx
    ) -> LocalModel {
        // Sum all .safetensors files for reported size
        let sizeBytes: Int64 = fileURLs
            .filter { $0.pathExtension.lowercased() == "safetensors" }
            .compactMap { url -> Int64? in
                guard let values = try? url.resourceValues(forKeys: [.fileSizeKey]),
                      let size = values.fileSize else { return nil }
                return Int64(size)
            }
            .reduce(0, +)

        // Extract quantization suffix, e.g. "Qwen3-8B-4bit" → "4bit"
        let quantization = extractQuantization(from: dirName)

        return LocalModel(
            id: dirName,
            displayName: dirName,
            directory: dirURL,
            sizeBytes: sizeBytes,
            format: format,
            quantization: quantization,
            parameterCount: nil, // Deferred — requires config.json parser (v0.3+)
            architecture: nil    // Deferred — requires config.json parser (v0.3+)
        )
    }

    /// `model_type` values mlx-swift-lm's `MLXVLM` library supports.
    ///
    /// Source of truth: `Libraries/MLXVLM/Models/*.swift` registry in
    /// the mlx-swift-lm checkout. Refresh this set when bumping the
    /// SPM dependency. Stored lowercased — comparisons are
    /// case-insensitive against `config.json`.
    private static let knownVLMTypes: Set<String> = [
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3_vl",
        "qwen3_5_vl",
        "gemma3",
        "smolvlm",
        "smolvlm2",
        "paligemma",
        "pixtral",
        "idefics3",
        "fast_vlm",
        "lfm2_vl",
        "glm_ocr",
        "mistral3",
    ]

    /// Peek `config.json`'s `model_type`. Returns `.mlxVLM` if the
    /// type matches a known VLM family; otherwise `.mlx`.
    ///
    /// Best-effort: any read or parse failure (missing file, malformed
    /// JSON, missing `model_type` key) falls back to `.mlx` — the scan
    /// must not blow up because of one unparseable config.
    private func upgradeFormatIfVLM(directory: URL) -> ModelFormat {
        let configURL = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let modelType = json["model_type"] as? String,
              Self.knownVLMTypes.contains(modelType.lowercased())
        else {
            return .mlx
        }
        return .mlxVLM
    }

    /// Extracts a quantization string from a directory name.
    ///
    /// Matches a trailing `-(\d+bit)` suffix, e.g. `Qwen3-8B-4bit` → `"4bit"`.
    private func extractQuantization(from name: String) -> String? {
        let pattern = #"-(\d+bit)$"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) else {
            return nil
        }
        let range = NSRange(name.startIndex..., in: name)
        guard let match = regex.firstMatch(in: name, options: [], range: range),
              let captureRange = Range(match.range(at: 1), in: name) else {
            return nil
        }
        return String(name[captureRange])
    }
}
