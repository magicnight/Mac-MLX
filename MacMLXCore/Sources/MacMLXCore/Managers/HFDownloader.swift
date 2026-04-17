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

// MARK: - DownloadProgress

/// Snapshot of an in-flight model download. Sent on every chunk write
/// so SwiftUI views can render a progress bar without rounding gaps.
public struct DownloadProgress: Sendable, Hashable {
    public let modelID: String
    public let bytesDownloaded: Int64
    public let totalBytes: Int64
    public let completedFiles: Int
    public let totalFiles: Int
    public let currentFileName: String?

    public init(
        modelID: String,
        bytesDownloaded: Int64,
        totalBytes: Int64,
        completedFiles: Int,
        totalFiles: Int,
        currentFileName: String?
    ) {
        self.modelID = modelID
        self.bytesDownloaded = bytesDownloaded
        self.totalBytes = totalBytes
        self.completedFiles = completedFiles
        self.totalFiles = totalFiles
        self.currentFileName = currentFileName
    }

    /// 0.0 - 1.0. `0` if `totalBytes <= 0`. Clamped to `1.0` upper bound.
    public var fractionCompleted: Double {
        guard totalBytes > 0 else { return 0 }
        return min(1.0, Double(bytesDownloaded) / Double(totalBytes))
    }

    /// `"2.10 GB / 4.50 GB"`-style label. Same base-10 convention as
    /// `LocalModel.humanSize`.
    public var humanProgress: String {
        "\(formatBytesBase10(bytesDownloaded)) / \(formatBytesBase10(totalBytes))"
    }

    /// `"45%"`. Useful for compact rows.
    public var humanPercent: String {
        let pct = Int((fractionCompleted * 100).rounded())
        return "\(pct)%"
    }
}

/// Base-10 byte formatter shared between `LocalModel.humanSize` and
/// `DownloadProgress.humanProgress`. Apple's "GB" convention (10^9, not 2^30).
internal func formatBytesBase10(_ bytes: Int64) -> String {
    let b = Double(bytes)
    if b >= 1_000_000_000 {
        return String(format: "%.2f GB", b / 1_000_000_000)
    }
    if b >= 1_000_000 {
        return String(format: "%.0f MB", b / 1_000_000)
    }
    if b >= 1_000 {
        return String(format: "%.0f KB", b / 1_000)
    }
    return "\(bytes) B"
}

// MARK: - HFDownloader

/// Downloads models from the Hugging Face Hub using the Hub REST API.
///
/// All network I/O is performed via `URLSession`. Inject a custom session
/// (e.g. one backed by `MockURLProtocol`) for unit testing.
public actor HFDownloader {

    // MARK: - Types

    /// `@Sendable` callback invoked on every chunk write across all the
    /// model's files. Always called from a background URLSession queue —
    /// callers that touch UI state should hop to `@MainActor` themselves.
    public typealias ProgressHandler = @Sendable (DownloadProgress) -> Void

    // MARK: - Dependencies

    private let urlSession: URLSession

    // MARK: - Init

    public init(urlSession: URLSession = .shared) {
        self.urlSession = urlSession
    }

    // MARK: - Public API

    /// Search Hugging Face Hub for `mlx-community` models matching `query`.
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

    /// Fetch the list of files for a given model ID.
    public func files(for modelID: String) async throws -> [HFRemoteFile] {
        let urlString = "https://huggingface.co/api/models/\(modelID)"
        guard let url = URL(string: urlString) else {
            throw DownloadError.modelNotFound(modelID)
        }
        let data = try await fetchData(from: url)
        let envelope = try JSONDecoder.huggingFace.decode(ModelDetailsEnvelope.self, from: data)
        return envelope.siblings.map { sibling in
            HFRemoteFile(path: sibling.rfilename, size: sibling.size, lfs: false)
        }
    }

    /// Download all files of a model into `directory/<modelName>/`.
    ///
    /// - Parameters:
    ///   - modelID: Full `author/repoName` identifier.
    ///   - directory: Parent directory; a subdirectory named after the model
    ///     is created here.
    ///   - progress: Optional `@Sendable` closure called on every chunk
    ///     write with an aggregated `DownloadProgress` snapshot. Fires from
    ///     the URLSession delegate queue; UI consumers should bridge to
    ///     `@MainActor`.
    /// - Returns: URL of the created model directory.
    public func download(
        modelID: String,
        to directory: URL,
        progress: ProgressHandler? = nil
    ) async throws -> URL {
        let modelName = modelID.split(separator: "/").last.map(String.init) ?? modelID
        let modelDir = directory.appending(path: modelName, directoryHint: .isDirectory)

        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        let remoteFiles = try await files(for: modelID)
        let fileCount = remoteFiles.count
        guard fileCount > 0 else { return modelDir }

        // Pre-compute total bytes so the progress bar has a stable denominator.
        // Files without a known size contribute 0 and don't show in the bar
        // (rare for HF — large weights always have sizes; tiny configs may not).
        let totalKnownBytes: Int64 = remoteFiles.reduce(0) { $0 + ($1.size ?? 0) }

        var aggregatedBytes: Int64 = 0

        for (index, remoteFile) in remoteFiles.enumerated() {
            let resolveURLString = "https://huggingface.co/\(modelID)/resolve/main/\(remoteFile.path)"
            guard let resolveURL = URL(string: resolveURLString) else {
                throw DownloadError.writeFailed("Could not construct resolve URL for \(remoteFile.path)")
            }

            let destination = modelDir.appending(path: remoteFile.path, directoryHint: .notDirectory)
            let parentDir = destination.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)

            // Per-file progress bridge — captured by value to be Sendable.
            let baseBytes = aggregatedBytes
            let totalDenominator = totalKnownBytes
            let localModelID = modelID
            let completedSoFar = index
            let totalFilesSnapshot = fileCount
            let currentFileNameSnapshot = remoteFile.path

            let delegate = ProgressDelegate { writtenForThisFile, _ in
                progress?(DownloadProgress(
                    modelID: localModelID,
                    bytesDownloaded: baseBytes + writtenForThisFile,
                    totalBytes: totalDenominator,
                    completedFiles: completedSoFar,
                    totalFiles: totalFilesSnapshot,
                    currentFileName: currentFileNameSnapshot
                ))
            }

            let (tempURL, response) = try await urlSession.download(
                for: URLRequest(url: resolveURL),
                delegate: delegate
            )

            if let httpResponse = response as? HTTPURLResponse,
               httpResponse.statusCode != 200 {
                throw DownloadError.badStatusCode(httpResponse.statusCode, url: resolveURL)
            }

            if FileManager.default.fileExists(atPath: destination.path) {
                try FileManager.default.removeItem(at: destination)
            }
            do {
                try FileManager.default.moveItem(at: tempURL, to: destination)
            } catch {
                throw DownloadError.writeFailed("Could not move \(remoteFile.path): \(error.localizedDescription)")
            }

            // Bump aggregate by the actually downloaded size — fall back to
            // the manifest size if we didn't observe a write callback (rare).
            let actualSize: Int64 = (try? FileManager.default.attributesOfItem(atPath: destination.path)[.size] as? Int64) ?? remoteFile.size ?? 0
            aggregatedBytes += actualSize

            // Final per-file tick so UI doesn't stall between files.
            progress?(DownloadProgress(
                modelID: modelID,
                bytesDownloaded: aggregatedBytes,
                totalBytes: max(totalKnownBytes, aggregatedBytes),
                completedFiles: index + 1,
                totalFiles: fileCount,
                currentFileName: index + 1 == fileCount ? nil : remoteFiles[index + 1].path
            ))
        }

        return modelDir
    }

    // MARK: - Private Helpers

    private func fetchData(from url: URL) async throws -> Data {
        let (data, response) = try await urlSession.data(from: url)
        if let httpResponse = response as? HTTPURLResponse {
            let code = httpResponse.statusCode
            guard code == 200 else {
                throw DownloadError.badStatusCode(code, url: url)
            }
        }
        return data
    }
}

// MARK: - URLSessionDownloadDelegate bridge

/// Tiny wrapper that translates Foundation's Obj-C delegate callbacks into
/// our Swift `@Sendable` progress closure. Marked `@unchecked Sendable`
/// because `NSObject` inheritance prevents Swift from inferring it
/// automatically; the closure itself is `@Sendable` and the only stored
/// property, so this is safe.
private final class ProgressDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {

    private let onProgress: @Sendable (Int64, Int64) -> Void

    init(onProgress: @escaping @Sendable (Int64, Int64) -> Void) {
        self.onProgress = onProgress
        super.init()
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        onProgress(totalBytesWritten, totalBytesExpectedToWrite)
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        // Required by URLSessionDownloadDelegate but the async API
        // (URLSession.download(for:delegate:)) handles the temp-URL handoff
        // for us, so we don't move the file here.
    }
}

// MARK: - Private Decoding Helpers

private struct ModelDetailsEnvelope: Decodable {
    let siblings: [SiblingEntry]

    struct SiblingEntry: Decodable {
        let rfilename: String
        let size: Int64?
    }
}
