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
    /// Recurses into subdirectories up to `maxDepth` levels so that
    /// HuggingFace-style nested layouts (`models/<org>/<repo>/...`) are
    /// discovered as well as flat layouts (`models/<repo>/...`). Any
    /// directory that itself looks like a model (per
    /// `ModelFormat.detect(in:)`) is treated as a leaf — the scan does
    /// not recurse into it.
    ///
    /// `LocalModel.id` is the path relative to `directory`, so nested
    /// repos with identical leaf names (e.g.
    /// `nightmedia/gemma-...-q8-mlx` vs `mlx-community/gemma-...-q8-mlx`)
    /// remain distinguishable. `displayName` is just the leaf for UX.
    ///
    /// - GGUF directories are silently skipped (with a `print` notice
    ///   that includes the relative path).
    /// - Hidden directories (names starting with `.`) are always skipped.
    ///
    /// - Parameters:
    ///   - directory: The root directory to scan.
    ///   - maxDepth: Maximum directory depth to traverse. `1` matches
    ///     pre-v0.5.1 behaviour (top-level only); the default `2`
    ///     covers HF-style `<root>/<org>/<repo>/` layouts. Increase to
    ///     `3` if you nest by `<root>/<author>/<org>/<repo>/`.
    /// - Returns: Sorted array of discovered `LocalModel` values.
    @discardableResult
    public func scan(_ directory: URL, maxDepth: Int = 2) async throws -> [LocalModel] {
        var results: [LocalModel] = []
        try scanRecursive(at: directory, root: directory, depth: 1, maxDepth: maxDepth, into: &results)

        let sorted = results.sorted { $0.displayName.localizedCompare($1.displayName) == .orderedAscending }
        lastScan = sorted
        return sorted
    }

    // MARK: - Recursive scanner

    /// Walks `dir` for one level, registering any model it finds and
    /// recursing into non-model subdirectories until `depth` reaches
    /// `maxDepth`. Mutates `results` in place — caller sorts.
    ///
    /// `root` stays constant across the recursion so each leaf can
    /// compute its `LocalModel.id` as a path relative to the original
    /// scan root rather than its immediate parent.
    private func scanRecursive(
        at dir: URL,
        root: URL,
        depth: Int,
        maxDepth: Int,
        into results: inout [LocalModel]
    ) throws {
        let contents = try fileManager.contentsOfDirectory(
            at: dir,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )

        for itemURL in contents {
            guard try isDirectory(itemURL) else { continue }

            let leafName = itemURL.lastPathComponent
            guard !leafName.hasPrefix(".") else { continue }

            let fileURLs = (try? fileManager.contentsOfDirectory(
                at: itemURL,
                includingPropertiesForKeys: [.fileSizeKey],
                options: [.skipsHiddenFiles]
            )) ?? []

            let fileNames = fileURLs.map { $0.lastPathComponent }
            let format = ModelFormat.detect(in: fileNames)
            let relativeID = relativePath(of: itemURL, from: root) ?? leafName

            switch format {
            case .mlx:
                // Peek `config.json` `model_type` — upgrade to .mlxVLM
                // when the directory contains a vision-language model.
                let upgradedFormat = upgradeFormatIfVLM(directory: itemURL)
                let model = buildLocalModel(
                    id: relativeID,
                    displayName: leafName,
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
                    id: relativeID,
                    displayName: leafName,
                    dirURL: itemURL,
                    fileURLs: fileURLs,
                    format: .mlxVLM
                )
                results.append(model)

            case .gguf:
                print("[ModelLibraryManager] Skipping GGUF directory: \(relativeID)")

            case .unknown:
                // Not a model — try recursing one level deeper for
                // nested layouts like `<root>/<org>/<repo>/`. Bail out
                // at `maxDepth` so a typo'd `modelDirectory` pointing
                // at `~` doesn't walk the whole home tree.
                if depth < maxDepth {
                    try scanRecursive(
                        at: itemURL,
                        root: root,
                        depth: depth + 1,
                        maxDepth: maxDepth,
                        into: &results
                    )
                }
            }
        }
    }

    /// Path of `url` expressed relative to `root`, with the leading
    /// separator stripped. Returns `nil` if `url` is not inside `root`
    /// (shouldn't happen during normal scan; defensive only).
    private func relativePath(of url: URL, from root: URL) -> String? {
        let rootPath = root.standardizedFileURL.path
        let urlPath = url.standardizedFileURL.path
        guard urlPath.hasPrefix(rootPath) else { return nil }
        let dropped = String(urlPath.dropFirst(rootPath.count))
        return dropped.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    }

    // MARK: - Private Helpers

    private func isDirectory(_ url: URL) throws -> Bool {
        let values = try url.resourceValues(forKeys: [.isDirectoryKey])
        return values.isDirectory == true
    }

    private func buildLocalModel(
        id: String,
        displayName: String,
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

        // Extract quantization suffix from the leaf name (parent
        // directories like "nightmedia/" wouldn't carry a quant tag).
        let quantization = extractQuantization(from: displayName)

        return LocalModel(
            id: id,
            displayName: displayName,
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
