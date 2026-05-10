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

            let configURL = url.appendingPathComponent("adapter_config.json")
            let weightsURL = url.appendingPathComponent("adapter_model.safetensors")
            guard fileManager.fileExists(atPath: configURL.path),
                  fileManager.fileExists(atPath: weightsURL.path),
                  let data = try? Data(contentsOf: configURL),
                  let cfg = try? JSONDecoder().decode(LocalAdapter.PEFTConfig.self, from: data)
            else { continue }

            results.append(LocalAdapter(
                name: url.lastPathComponent,
                directory: url,
                targetModel: cfg.baseModelNameOrPath,
                rank: cfg.r,
                targetModules: cfg.targetModules ?? []
            ))
        }
        return results.sorted { $0.name.localizedCompare($1.name) == .orderedAscending }
    }
}
