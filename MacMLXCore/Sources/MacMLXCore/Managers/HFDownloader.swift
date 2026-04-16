import Foundation

// MARK: - DownloadError

/// Errors specific to the Hugging Face download pipeline.
public enum DownloadError: LocalizedError, Sendable {
    case badStatusCode(Int, url: URL)
    case writeFailed(String)
    case modelNotFound(String)

    public var errorDescription: String? {
        switch self {
        case .badStatusCode(let code, let url):
            return "HTTP \(code) from \(url.absoluteString)"
        case .writeFailed(let detail):
            return "Failed to write file: \(detail)"
        case .modelNotFound(let modelID):
            return "Model not found on Hugging Face Hub: \(modelID)"
        }
    }
}

// MARK: - HFRemoteFile

/// A file entry from the Hugging Face model-details endpoint (`siblings` array).
public struct HFRemoteFile: Codable, Hashable, Sendable {
    /// Relative path within the model repo, e.g. `"config.json"` or `"model.safetensors"`.
    public let path: String
    /// File size in bytes (may be nil if Hub does not expose it).
    public let size: Int64?
    /// `true` if the file is served via Git LFS (large files).
    public let lfs: Bool

    public init(path: String, size: Int64? = nil, lfs: Bool = false) {
        self.path = path
        self.size = size
        self.lfs = lfs
    }
}

// MARK: - HFDownloader

/// Downloads models from the Hugging Face Hub using the Hub REST API.
///
/// All network I/O is performed via `URLSession`. Inject a custom session
/// (e.g. one backed by `MockURLProtocol`) for unit testing.
public actor HFDownloader {

    // MARK: - Dependencies

    private let urlSession: URLSession

    // MARK: - Init

    public init(urlSession: URLSession = .shared) {
        self.urlSession = urlSession
    }

    // MARK: - Public API

    /// Search Hugging Face Hub for `mlx-community` models matching `query`.
    ///
    /// - Parameters:
    ///   - query: Free-text search term.
    ///   - limit: Maximum number of results (default 20).
    /// - Returns: Array of `HFModel` values sorted by the Hub's default ranking.
    public func search(query: String, limit: Int = 20) async throws -> [HFModel] {
        var components = URLComponents(string: "https://huggingface.co/api/models")!
        components.queryItems = [
            URLQueryItem(name: "author", value: "mlx-community"),
            URLQueryItem(name: "search", value: query),
            URLQueryItem(name: "limit", value: String(limit)),
            URLQueryItem(name: "sort", value: "downloads"),
            URLQueryItem(name: "direction", value: "-1"),
        ]
        guard let url = components.url else {
            throw DownloadError.writeFailed("Could not construct search URL")
        }
        let data = try await fetchData(from: url)
        return try JSONDecoder.huggingFace.decode([HFModel].self, from: data)
    }

    /// Fetch the list of files for a given model ID (e.g. `"mlx-community/Qwen3-8B-4bit"`).
    ///
    /// - Parameter modelID: Full `author/repoName` identifier.
    /// - Returns: Array of `HFRemoteFile` values from the model's `siblings` field.
    public func files(for modelID: String) async throws -> [HFRemoteFile] {
        let urlString = "https://huggingface.co/api/models/\(modelID)"
        guard let url = URL(string: urlString) else {
            throw DownloadError.modelNotFound(modelID)
        }
        let data = try await fetchData(from: url)

        // Decode only the `siblings` envelope
        let envelope = try JSONDecoder.huggingFace.decode(ModelDetailsEnvelope.self, from: data)
        return envelope.siblings.map { sibling in
            HFRemoteFile(path: sibling.rfilename, size: sibling.size, lfs: false)
        }
    }

    /// Download all files of a model into `directory/<modelName>/`.
    ///
    /// Progress is reported as a `Double` in `[0, 1]` after each file completes.
    ///
    /// - Parameters:
    ///   - modelID: Full `author/repoName` identifier.
    ///   - directory: Parent directory; a subdirectory named after the model is created here.
    ///   - onProgress: Optional closure called after each file with the fraction completed.
    /// - Returns: URL of the created model directory.
    public func download(
        modelID: String,
        to directory: URL,
        onProgress: ((Double) -> Void)? = nil
    ) async throws -> URL {
        let modelName = modelID.split(separator: "/").last.map(String.init) ?? modelID
        let modelDir = directory.appending(path: modelName, directoryHint: .isDirectory)

        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        let remoteFiles = try await files(for: modelID)
        let total = remoteFiles.count
        guard total > 0 else { return modelDir }

        for (index, remoteFile) in remoteFiles.enumerated() {
            let resolveURLString = "https://huggingface.co/\(modelID)/resolve/main/\(remoteFile.path)"
            guard let resolveURL = URL(string: resolveURLString) else {
                throw DownloadError.writeFailed("Could not construct resolve URL for \(remoteFile.path)")
            }

            let destination = modelDir.appending(path: remoteFile.path, directoryHint: .notDirectory)

            // Create any intermediate subdirectories (e.g. for nested paths)
            let parentDir = destination.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)

            let (tempURL, response) = try await urlSession.download(from: resolveURL)
            if let httpResponse = response as? HTTPURLResponse,
               httpResponse.statusCode != 200 {
                throw DownloadError.badStatusCode(httpResponse.statusCode, url: resolveURL)
            }

            // Move from temp location to final destination
            if FileManager.default.fileExists(atPath: destination.path) {
                try FileManager.default.removeItem(at: destination)
            }
            do {
                try FileManager.default.moveItem(at: tempURL, to: destination)
            } catch {
                throw DownloadError.writeFailed("Could not move \(remoteFile.path): \(error.localizedDescription)")
            }

            let progress = Double(index + 1) / Double(total)
            onProgress?(progress)
        }

        return modelDir
    }

    // MARK: - Private Helpers

    /// Perform a GET request and return the body data, throwing on non-200 status codes.
    private func fetchData(from url: URL) async throws -> Data {
        let (data, response) = try await urlSession.data(from: url)
        // Only check status when the response is actually HTTP — in unit tests backed
        // by a URLProtocol mock the cast always succeeds, so this is always enforced.
        if let httpResponse = response as? HTTPURLResponse {
            let code = httpResponse.statusCode
            guard code == 200 else {
                throw DownloadError.badStatusCode(code, url: url)
            }
        }
        return data
    }
}

// MARK: - Private Decoding Helpers

/// Envelope for the `/api/models/{id}` response — only `siblings` is needed.
private struct ModelDetailsEnvelope: Decodable {
    let siblings: [SiblingEntry]

    struct SiblingEntry: Decodable {
        let rfilename: String
        let size: Int64?
    }
}
