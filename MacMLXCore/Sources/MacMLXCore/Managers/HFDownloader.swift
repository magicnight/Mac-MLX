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

/// Snapshot of an in-flight model download.
///
/// Hugging Face's model manifest doesn't report file sizes for LFS-backed
/// blobs (which is where the multi-GB weights live), so any "overall bytes"
/// denominator would be grossly wrong during the big file's download. We
/// surface **current-file** progress (always accurate — comes straight from
/// `URLSession`'s `totalBytesExpectedToWrite`) and **file-count** progress
/// (always accurate) separately, and let the UI decide which to emphasise.
public struct DownloadProgress: Sendable, Hashable {
    public let modelID: String

    // File-count axis — always reliable.
    public let completedFiles: Int
    public let totalFiles: Int

    // Current-file axis — bytes come from URLSessionDownloadDelegate and
    // match the real Content-Length of the current HTTP response. Zero
    // values mean "not yet started" or "size unknown".
    public let currentFileName: String?
    public let currentFileBytesDownloaded: Int64
    public let currentFileTotalBytes: Int64

    /// Exponentially-smoothed throughput on the current file, bytes/sec.
    /// 0 until the second chunk arrives (need two samples for a rate).
    public let currentFileBytesPerSecond: Double

    public init(
        modelID: String,
        completedFiles: Int,
        totalFiles: Int,
        currentFileName: String?,
        currentFileBytesDownloaded: Int64,
        currentFileTotalBytes: Int64,
        currentFileBytesPerSecond: Double = 0
    ) {
        self.modelID = modelID
        self.completedFiles = completedFiles
        self.totalFiles = totalFiles
        self.currentFileName = currentFileName
        self.currentFileBytesDownloaded = currentFileBytesDownloaded
        self.currentFileTotalBytes = currentFileTotalBytes
        self.currentFileBytesPerSecond = currentFileBytesPerSecond
    }

    /// Current-file fraction in [0, 1], or 0 if the file total isn't known yet.
    public var currentFileFraction: Double {
        guard currentFileTotalBytes > 0 else { return 0 }
        return min(1.0, Double(currentFileBytesDownloaded) / Double(currentFileTotalBytes))
    }

    /// `"2.10 GB / 4.50 GB"` for the current file. Empty string if unknown.
    public var currentFileHuman: String {
        guard currentFileTotalBytes > 0 else { return "" }
        return "\(formatBytesBase10(currentFileBytesDownloaded)) / \(formatBytesBase10(currentFileTotalBytes))"
    }

    /// `"47%"` for the current file. Empty if total unknown.
    public var currentFilePercent: String {
        guard currentFileTotalBytes > 0 else { return "" }
        return "\(Int((currentFileFraction * 100).rounded()))%"
    }

    /// File-count fraction in [0, 1]. Useful as a coarse "overall" bar.
    public var filesFraction: Double {
        guard totalFiles > 0 else { return 0 }
        return Double(completedFiles) / Double(totalFiles)
    }

    /// `"2 of 5 files"`.
    public var filesHuman: String {
        "\(completedFiles) of \(totalFiles) files"
    }

    /// `"12.5 MB/s"` — decimal MB per second. Empty string before the
    /// second chunk arrives (rate needs two samples).
    public var currentFileSpeedHuman: String {
        guard currentFileBytesPerSecond > 0 else { return "" }
        let bps = currentFileBytesPerSecond
        if bps >= 1_000_000 {
            return String(format: "%.1f MB/s", bps / 1_000_000)
        }
        if bps >= 1_000 {
            return String(format: "%.0f KB/s", bps / 1_000)
        }
        return String(format: "%.0f B/s", bps)
    }

    /// Estimated remaining seconds for the current file, based on the
    /// current throughput EMA. Nil if speed or total size is unknown.
    public var currentFileETASeconds: Double? {
        guard currentFileTotalBytes > 0,
              currentFileBytesPerSecond > 0 else { return nil }
        let remaining = Double(currentFileTotalBytes - currentFileBytesDownloaded)
        guard remaining > 0 else { return 0 }
        return remaining / currentFileBytesPerSecond
    }

    /// `"2m 13s"`, `"45s"`, or `"—"` if ETA is unknown.
    public var currentFileETAHuman: String {
        guard let secs = currentFileETASeconds, secs.isFinite else { return "—" }
        let total = Int(secs.rounded())
        if total < 1 { return "<1s" }
        if total < 60 { return "\(total)s" }
        if total < 3600 { return "\(total / 60)m \(total % 60)s" }
        let hours = total / 3600
        let minutes = (total % 3600) / 60
        return "\(hours)h \(minutes)m"
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

    // MARK: - Constants

    /// Canonical Hugging Face Hub origin.
    public static let defaultEndpoint = URL(string: "https://huggingface.co")!

    // MARK: - Dependencies

    private let urlSession: URLSession
    /// Base URL for all Hub API + resolve requests. Configurable via
    /// `setBaseURL(_:)` so users in regions where huggingface.co is slow
    /// or blocked can route through a mirror like hf-mirror.com (#21).
    private var baseURL: URL

    // MARK: - Init

    public init(urlSession: URLSession = .shared, baseURL: URL = HFDownloader.defaultEndpoint) {
        self.urlSession = urlSession
        self.baseURL = baseURL
    }

    // MARK: - Configuration

    /// Update the Hub endpoint in-place. Safe to call while downloads
    /// are in flight — only new requests pick up the change.
    public func setBaseURL(_ url: URL) {
        self.baseURL = url
    }

    /// Current Hub endpoint. Exposed for UI ("your endpoint is …").
    public func currentBaseURL() -> URL { baseURL }

    // MARK: - Public API

    /// Search Hugging Face Hub for `mlx-community` models matching `query`.
    public func search(query: String, limit: Int = 20) async throws -> [HFModel] {
        guard var components = URLComponents(
            url: baseURL.appending(path: "api/models"),
            resolvingAgainstBaseURL: false
        ) else {
            throw DownloadError.writeFailed("Could not construct search URL")
        }
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
        let url = baseURL.appending(path: "api/models/\(modelID)")
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
    ///
    /// Files that already exist at the destination are skipped (treated as
    /// complete — we don't re-verify sizes). Files cancelled mid-download
    /// leave a resume record at `~/.mac-mlx/downloads/{modelID}/resume.dat`
    /// that this method picks up on the next call to continue from the
    /// last byte (#6).
    ///
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

        // Prior cancel may have left a resume record — if the file it
        // points at is the next incomplete one, we'll pick up from the
        // last saved byte instead of restarting.
        let resumeRecord = loadResumeRecord(for: modelID)

        for (index, remoteFile) in remoteFiles.enumerated() {
            let destination = modelDir.appending(path: remoteFile.path, directoryHint: .notDirectory)
            let parentDir = destination.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)

            // Skip if already on disk — optimistic completeness check.
            // A mid-download cancel leaves nothing at destination (the move
            // from URLSession's temp location happens only after success),
            // so a present file is safe to treat as complete.
            if FileManager.default.fileExists(atPath: destination.path) {
                progress?(DownloadProgress(
                    modelID: modelID,
                    completedFiles: index + 1,
                    totalFiles: fileCount,
                    currentFileName: index + 1 < fileCount ? remoteFiles[index + 1].path : nil,
                    currentFileBytesDownloaded: 0,
                    currentFileTotalBytes: 0
                ))
                continue
            }

            // Resume URL: <endpoint>/<modelID>/resolve/main/<path>
            let resolveURL = baseURL.appending(
                path: "\(modelID)/resolve/main/\(remoteFile.path)"
            )

            // Snapshot context — captured by value into the @Sendable closure.
            let localModelID = modelID
            let completedSoFar = index
            let totalFilesSnapshot = fileCount
            let currentFileNameSnapshot = remoteFile.path

            // Emit an initial "starting file" tick so the UI can flip to the
            // new filename immediately, even before the first chunk arrives.
            progress?(DownloadProgress(
                modelID: localModelID,
                completedFiles: completedSoFar,
                totalFiles: totalFilesSnapshot,
                currentFileName: currentFileNameSnapshot,
                currentFileBytesDownloaded: 0,
                currentFileTotalBytes: 0
            ))

            let sampler = SpeedSampler()
            let delegate = ProgressDelegate { written, expected in
                let bps = sampler.record(bytes: written)
                progress?(DownloadProgress(
                    modelID: localModelID,
                    completedFiles: completedSoFar,
                    totalFiles: totalFilesSnapshot,
                    currentFileName: currentFileNameSnapshot,
                    currentFileBytesDownloaded: written,
                    currentFileTotalBytes: max(0, expected),
                    currentFileBytesPerSecond: bps
                ))
            }

            // Resume if we have valid data for THIS exact file from a
            // prior cancellation. Safety: resumeData is opaque and tied
            // to a specific URL — using it for a different file would
            // fail at the URLSession layer. We match by filename.
            let resumeDataForThisFile: Data? =
                (resumeRecord?.currentFile == remoteFile.path) ? resumeRecord?.resumeData : nil

            do {
                let (tempURL, response): (URL, URLResponse)
                if let resumeData = resumeDataForThisFile {
                    (tempURL, response) = try await urlSession.download(
                        resumeFrom: resumeData,
                        delegate: delegate
                    )
                } else {
                    (tempURL, response) = try await urlSession.download(
                        for: URLRequest(url: resolveURL),
                        delegate: delegate
                    )
                }

                if let httpResponse = response as? HTTPURLResponse,
                   httpResponse.statusCode != 200 {
                    throw DownloadError.badStatusCode(httpResponse.statusCode, url: resolveURL)
                }

                if FileManager.default.fileExists(atPath: destination.path) {
                    try FileManager.default.removeItem(at: destination)
                }
                try FileManager.default.moveItem(at: tempURL, to: destination)

                // This file is done — any saved resume state for it is stale.
                clearResumeRecord(for: modelID)

            } catch let urlErr as URLError where urlErr.code == .cancelled {
                // Capture resume data if URLSession was able to produce it.
                // `URLError.downloadTaskResumeData` is Foundation's Swift-native
                // accessor for the opaque blob (tail-end byte range + ETag)
                // that the server needs for a Range continuation.
                if let data = urlErr.downloadTaskResumeData {
                    saveResumeRecord(
                        for: modelID,
                        record: ResumeRecord(
                            currentFile: remoteFile.path,
                            resumeData: data
                        )
                    )
                }
                throw urlErr
            } catch {
                // Non-cancel errors: don't bother saving resume data, just
                // bubble up. (Typical cases: HTTP 4xx/5xx, DNS, filesystem.)
                throw error
            }

            // Post-file tick: bump completedFiles, point currentFileName at
            // the next file (or nil if this was the last).
            let nextFileName = index + 1 < fileCount ? remoteFiles[index + 1].path : nil
            progress?(DownloadProgress(
                modelID: localModelID,
                completedFiles: index + 1,
                totalFiles: fileCount,
                currentFileName: nextFileName,
                currentFileBytesDownloaded: 0,
                currentFileTotalBytes: 0
            ))
        }

        // All files complete — clear any leftover record (covers the edge
        // case where user cancelled file 2, then later resumed and
        // completed everything including a pending record on file 2).
        clearResumeRecord(for: modelID)
        return modelDir
    }

    // MARK: - Resume record (#6)

    /// Saved state from a cancelled download, pointing at the specific
    /// file and byte offset to resume from.
    private struct ResumeRecord {
        let currentFile: String
        let resumeData: Data
    }

    /// `~/.mac-mlx/downloads/{encoded-modelID}/` — sibling of conversation
    /// and parameter stores. Under App Sandbox the dotfile exemption
    /// applies.
    private func resumeDirectory(for modelID: String) -> URL {
        let home: URL = {
            if let path = NSHomeDirectoryForUser(NSUserName()) {
                return URL(filePath: path, directoryHint: .isDirectory)
            }
            return URL(filePath: NSHomeDirectory(), directoryHint: .isDirectory)
        }()
        let encoded = modelID.addingPercentEncoding(
            withAllowedCharacters: .urlPathAllowed.subtracting(.init(charactersIn: "/"))
        ) ?? modelID
        return home.appending(
            path: ".mac-mlx/downloads/\(encoded)",
            directoryHint: .isDirectory
        )
    }

    private func saveResumeRecord(for modelID: String, record: ResumeRecord) {
        let dir = resumeDirectory(for: modelID)
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try record.currentFile.write(
                to: dir.appending(path: "current-file.txt"),
                atomically: true,
                encoding: .utf8
            )
            try record.resumeData.write(
                to: dir.appending(path: "resume.dat"),
                options: .atomic
            )
        } catch {
            // Best-effort — failing to save resume data isn't fatal; the
            // user just can't resume this particular cancel.
        }
    }

    private func loadResumeRecord(for modelID: String) -> ResumeRecord? {
        let dir = resumeDirectory(for: modelID)
        guard let currentFile = try? String(
            contentsOf: dir.appending(path: "current-file.txt"),
            encoding: .utf8
        ),
        let resumeData = try? Data(contentsOf: dir.appending(path: "resume.dat")) else {
            return nil
        }
        return ResumeRecord(
            currentFile: currentFile.trimmingCharacters(in: .whitespacesAndNewlines),
            resumeData: resumeData
        )
    }

    private func clearResumeRecord(for modelID: String) {
        let dir = resumeDirectory(for: modelID)
        try? FileManager.default.removeItem(at: dir)
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

// MARK: - SpeedSampler (exponential moving average throughput)

/// Computes a smoothed bytes-per-second rate over consecutive URLSession
/// didWriteData callbacks for a single file. Internally locked because the
/// delegate may fire from any URLSession worker queue; EMA state is tiny so
/// a plain `NSLock` is the right primitive.
private final class SpeedSampler: @unchecked Sendable {
    /// Smoothing factor — 0.3 weights the most recent sample, 0.7 keeps
    /// the previous average. Small enough to mask spikes, large enough to
    /// follow real throughput changes (network hiccups, LFS CDN ramps).
    private let alpha = 0.3

    private let lock = NSLock()
    private var lastWallclock: Date?
    private var lastTotalBytes: Int64 = 0
    private var ema: Double = 0

    /// Feed a new `totalBytesWritten` sample and return the current EMA
    /// throughput in bytes/sec. Returns 0 on the first call (one sample
    /// isn't enough for a rate).
    func record(bytes: Int64) -> Double {
        lock.lock(); defer { lock.unlock() }
        let now = Date()
        guard let last = lastWallclock else {
            lastWallclock = now
            lastTotalBytes = bytes
            return 0
        }
        let dt = now.timeIntervalSince(last)
        guard dt > 0 else { return ema }  // duplicate callback, no time elapsed
        let dbytes = Double(bytes - lastTotalBytes)
        guard dbytes >= 0 else { return ema }  // shouldn't happen but guard anyway
        let instantaneous = dbytes / dt
        ema = ema == 0
            ? instantaneous
            : (alpha * instantaneous + (1 - alpha) * ema)
        lastWallclock = now
        lastTotalBytes = bytes
        return ema
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
