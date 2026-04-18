import Foundation

/// Metadata sidecar stored next to a downloaded model at
/// `<modelDir>/.macmlx-meta.json`. Persisted on first successful
/// download so we can later detect when the Hub repo has advanced
/// past this snapshot.
public struct DownloadedModelMeta: Codable, Sendable {
    /// Full HF model ID (e.g. `mlx-community/Qwen3-8B-4bit`).
    public let modelID: String
    /// Commit SHA of the `main` branch at download time, if HF
    /// exposed it. Nil for older downloads predating this field.
    public let commitSHA: String?
    /// `lastModified` timestamp reported by `/api/models/{id}` at
    /// download time.
    public let lastModifiedAtDownload: Date?
    /// Wall-clock time of the download event (may lag behind
    /// `lastModifiedAtDownload` by minutes).
    public let downloadedAt: Date

    public init(
        modelID: String,
        commitSHA: String?,
        lastModifiedAtDownload: Date?,
        downloadedAt: Date = Date()
    ) {
        self.modelID = modelID
        self.commitSHA = commitSHA
        self.lastModifiedAtDownload = lastModifiedAtDownload
        self.downloadedAt = downloadedAt
    }

    public static let filename = ".macmlx-meta.json"

    public static func url(inside modelDir: URL) -> URL {
        modelDir.appending(path: filename, directoryHint: .notDirectory)
    }

    public static func load(from modelDir: URL) -> DownloadedModelMeta? {
        let fileURL = url(inside: modelDir)
        guard let data = try? Data(contentsOf: fileURL) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(DownloadedModelMeta.self, from: data)
    }

    public func save(to modelDir: URL) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted]
        let data = try encoder.encode(self)
        try data.write(to: Self.url(inside: modelDir), options: .atomic)
    }
}
