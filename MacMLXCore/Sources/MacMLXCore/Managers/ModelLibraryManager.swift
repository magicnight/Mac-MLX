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
                let model = buildLocalModel(
                    dirName: dirName,
                    dirURL: itemURL,
                    fileURLs: fileURLs
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
        fileURLs: [URL]
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
            format: .mlx,
            quantization: quantization,
            parameterCount: nil, // TODO: v0.2 — enrich from config.json
            architecture: nil    // TODO: v0.2 — enrich from config.json
        )
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
