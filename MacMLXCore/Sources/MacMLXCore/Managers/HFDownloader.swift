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

    /// URLSession used for small metadata requests (search, file-list JSON).
    /// Background sessions don't support data tasks, so this stays a regular
    /// foreground session — typically `URLSession.shared`.
    private let metadataSession: URLSession

    /// URLSession used for the large binary downloads. In production this is
    /// a `URLSessionConfiguration.background`-backed session so transfers
    /// continue through App Nap and across full app quit-and-relaunch (#8).
    /// Tests can inject a foreground session to avoid the process-wide
    /// background session identifier.
    private let downloadSession: URLSession

    /// Routes URLSession delegate callbacks (progress + finish + error) to
    /// per-task continuations and progress closures. Owned here so tasks
    /// created on `downloadSession` can find their handlers.
    private let router: DownloadSessionRouter

    /// Base URL for all Hub API + resolve requests. Configurable via
    /// `setBaseURL(_:)` so users in regions where huggingface.co is slow
    /// or blocked can route through a mirror like hf-mirror.com (#21).
    private var baseURL: URL

    // MARK: - Shared background session (process-wide)

    /// Container binding the default background `URLSession` to its router.
    /// Foundation forbids two live background sessions with the same
    /// identifier in one process (second init logs an error and delivers
    /// undefined behaviour), so we claim the identifier exactly once at
    /// first use and hand the same pair to every subsequent `HFDownloader`
    /// that doesn't inject its own `downloadSession`.
    private struct BackgroundContainer {
        let session: URLSession
        let router: DownloadSessionRouter
    }

    private static let sharedBackground: BackgroundContainer = {
        let router = DownloadSessionRouter()
        let config = URLSessionConfiguration.background(
            withIdentifier: processScopedBackgroundIdentifier()
        )
        // Don't relaunch the app to deliver completion events — user will
        // see their downloads done the next time they open macMLX.
        config.sessionSendsLaunchEvents = false
        // Don't let macOS defer for power/network preferences.
        config.isDiscretionary = false
        // Soft-pause on network drops instead of erroring.
        config.waitsForConnectivity = true
        let session = URLSession(
            configuration: config,
            delegate: router,
            delegateQueue: nil
        )
        return BackgroundContainer(session: session, router: router)
    }()

    /// Background-session identifier suffixed by process role so the
    /// GUI app (`macMLX.app`) and the CLI binary (`macmlx`) don't fight
    /// for the same identifier when both run concurrently — Foundation
    /// forbids two live sessions with the same identifier in one
    /// **process**, but two different processes *can* each own a
    /// differently-named identifier. Reviewer-flagged MEDIUM.
    private static func processScopedBackgroundIdentifier() -> String {
        let base = "com.magicnight.macmlx.downloader"
        if Bundle.main.bundlePath.hasSuffix(".app") {
            return base + ".app"
        }
        return base + ".cli"
    }

    // MARK: - Init

    /// Designated initialiser.
    ///
    /// - Parameters:
    ///   - urlSession: Session used for metadata JSON calls. Defaults to
    ///     `URLSession.shared`. Pass a `MockURLProtocol`-backed session
    ///     in tests.
    ///   - baseURL: Hub endpoint. Change via `setBaseURL(_:)` for mirrors.
    ///
    /// The download path always uses the process-wide shared background
    /// session (`sharedBackground`), which is lazily created on first use
    /// and claims the background identifier exactly once. Tests that need
    /// to exercise `download(...)` use the internal initialiser below to
    /// inject a foreground session + router.
    public init(
        urlSession: URLSession = .shared,
        baseURL: URL = HFDownloader.defaultEndpoint
    ) {
        self.metadataSession = urlSession
        self.baseURL = baseURL
        let shared = HFDownloader.sharedBackground
        self.downloadSession = shared.session
        self.router = shared.router
    }

    /// Internal initialiser for tests — lets the test construct a
    /// foreground `URLSession` whose delegate is a fresh router, so each
    /// test is isolated and no test contends for the process-wide
    /// background session identifier.
    internal init(
        metadataSession: URLSession,
        downloadSession: URLSession,
        downloadRouter: DownloadSessionRouter,
        baseURL: URL = HFDownloader.defaultEndpoint
    ) {
        self.metadataSession = metadataSession
        self.downloadSession = downloadSession
        self.router = downloadRouter
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

    /// Fetch files + commit metadata in a single Hub request.
    /// Used by `download(...)` to write the `.macmlx-meta.json`
    /// sidecar without a second round-trip.
    public func downloadMeta(for modelID: String) async throws -> (
        files: [HFRemoteFile],
        sha: String?,
        lastModified: Date?
    ) {
        let url = baseURL.appending(path: "api/models/\(modelID)")
        let data = try await fetchData(from: url)
        let envelope = try JSONDecoder.huggingFace.decode(ModelDetailsEnvelope.self, from: data)
        let files = envelope.siblings.map { HFRemoteFile(path: $0.rfilename, size: $0.size, lfs: false) }
        return (files, envelope.sha, envelope.lastModified)
    }

    /// Snapshot of how a local download compares to the Hub's current
    /// head.
    public enum UpdateStatus: Sendable, Equatable {
        case upToDate
        case updateAvailable(commitSHA: String?, lastModified: Date?)
        case unknown
    }

    public func updateStatus(for meta: DownloadedModelMeta) async -> UpdateStatus {
        do {
            let url = baseURL.appending(path: "api/models/\(meta.modelID)")
            let data = try await fetchData(from: url)
            let envelope = try JSONDecoder.huggingFace.decode(
                ModelDetailsEnvelope.self, from: data
            )
            if let localSHA = meta.commitSHA, let remoteSHA = envelope.sha {
                return localSHA == remoteSHA
                    ? .upToDate
                    : .updateAvailable(commitSHA: remoteSHA, lastModified: envelope.lastModified)
            }
            if let localTime = meta.lastModifiedAtDownload, let remoteTime = envelope.lastModified {
                return remoteTime > localTime
                    ? .updateAvailable(commitSHA: envelope.sha, lastModified: remoteTime)
                    : .upToDate
            }
            return .unknown
        } catch {
            return .unknown
        }
    }

    /// Total size of all files in the model repo, in bytes.
    ///
    /// HF's `/api/models/{id}` endpoint omits `size` for LFS-backed
    /// files (which is where all the multi-GB weights live), so summing
    /// `siblings[*].size` gives zero for most real models. We fall back
    /// to HEAD on `/{id}/resolve/main/{path}` for any sibling without a
    /// declared size; HF responds with `x-linked-size` (LFS blob size
    /// before CDN redirect) or `content-length`. Weight-file filter
    /// keeps the request count bounded — configs/tokenisers are small
    /// and sometimes declared, so their absence doesn't matter.
    public func sizeBytes(for modelID: String) async throws -> Int64 {
        let files = try await files(for: modelID)
        // Step 1: declared sizes we already have.
        var total: Int64 = files.compactMap(\.size).reduce(0, +)

        // Step 2: for any sibling without a declared size, HEAD-resolve it.
        // Cap the concurrency so we don't flood the Hub — 4 is consistent
        // with the search-enrichment TaskGroup in ModelLibraryViewModel.
        let missing = files.filter { $0.size == nil }
        guard !missing.isEmpty else { return total }

        let baseURL = self.baseURL
        let session = self.metadataSession

        let fetched: [Int64] = await withTaskGroup(of: Int64?.self) { group in
            var inflight = 0
            let maxInflight = 4
            var iterator = missing.makeIterator()

            func enqueue() {
                guard let file = iterator.next() else { return }
                inflight += 1
                group.addTask {
                    await Self.headSize(
                        session: session,
                        baseURL: baseURL,
                        modelID: modelID,
                        path: file.path
                    )
                }
            }
            while inflight < maxInflight { enqueue() }
            var collected: [Int64] = []
            while let size = await group.next() {
                inflight -= 1
                if let size { collected.append(size) }
                enqueue()
            }
            return collected
        }
        total += fetched.reduce(0, +)
        return total
    }

    /// Issue a HEAD against `{baseURL}/{modelID}/resolve/main/{path}` and
    /// return the bytes count. Tries `x-linked-size` first (LFS), falls
    /// back to `content-length`. Returns nil on any failure so the
    /// caller can silently proceed.
    private static func headSize(
        session: URLSession,
        baseURL: URL,
        modelID: String,
        path: String
    ) async -> Int64? {
        let url = baseURL.appending(path: "\(modelID)/resolve/main/\(path)")
        var request = URLRequest(url: url)
        request.httpMethod = "HEAD"
        // Ask HF not to follow the LFS redirect — we want the header,
        // not the payload. HF honours this via the Accept header; most
        // public-CDN blobs set x-linked-size on the 302 itself.
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        do {
            let (_, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse else { return nil }
            if let linkedSize = http.value(forHTTPHeaderField: "x-linked-size"),
               let parsed = Int64(linkedSize) {
                return parsed
            }
            if let contentLength = http.value(forHTTPHeaderField: "content-length"),
               let parsed = Int64(contentLength),
               parsed > 0 {
                return parsed
            }
            return nil
        } catch {
            return nil
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

        let (remoteFiles, remoteSHA, remoteLastModified) = try await downloadMeta(for: modelID)
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
            let onProgress: @Sendable (Int64, Int64) -> Void = { written, expected in
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
            var pendingResumeData: Data? =
                (resumeRecord?.currentFile == remoteFile.path) ? resumeRecord?.resumeData : nil

            // Inner retry loop — we get at most one retry, used exclusively
            // to recover from a stale resumeData blob (server rotated its
            // ETag, Hub cycled the object on the CDN, etc.). Without this,
            // a bad blob wedges the download in a permanent fail-loop.
            var attempt = 0
            retry: while true {
                do {
                    let (tempURL, response) = try await downloadFile(
                        request: URLRequest(url: resolveURL),
                        resumeData: pendingResumeData,
                        onProgress: onProgress
                    )

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
                    break retry

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

                } catch let urlErr as URLError
                    where attempt == 0 && pendingResumeData != nil &&
                          Self.isStaleResumeError(urlErr) {
                    // The prior resume blob is unusable — server changed,
                    // blob expired, or we got a 416. Wipe the record and
                    // retry once from scratch before giving up. This
                    // prevents the download from being permanently wedged.
                    clearResumeRecord(for: modelID)
                    pendingResumeData = nil
                    attempt += 1
                    continue retry

                } catch {
                    // Any other error: don't save resume data, just bubble
                    // up. Typical cases: HTTP 4xx/5xx, DNS, filesystem.
                    throw error
                }
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

        // Persist the sidecar so a later `updateStatus(for:)` call can
        // compare the Hub's head against what we snapshotted at download
        // time. Best-effort: a write failure shouldn't fail the download
        // itself — the user just won't get an "Update available" badge
        // for this model until they re-download.
        let meta = DownloadedModelMeta(
            modelID: modelID,
            commitSHA: remoteSHA,
            lastModifiedAtDownload: remoteLastModified
        )
        try? meta.save(to: modelDir)

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
    /// applies (see `DataRoot.macMLX`).
    private func resumeDirectory(for modelID: String) -> URL {
        let encoded = modelID.addingPercentEncoding(
            withAllowedCharacters: .urlPathAllowed.subtracting(.init(charactersIn: "/"))
        ) ?? modelID
        return DataRoot.macMLX("downloads/\(encoded)")
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

    /// Is this `URLError` the kind that suggests our resumeData blob has
    /// gone stale on the server side (ETag rotated, blob expired, range
    /// not satisfiable)?  Used by the retry-without-resume path so we
    /// don't wedge permanently on a bad cached blob.
    ///
    /// Foundation doesn't expose a specific `cannotResumeDownload` code;
    /// the symptoms of a stale blob surface as one of these server-shape
    /// errors. We intentionally cast a wider net than strictly necessary
    /// — a false positive just costs one extra fresh download; a false
    /// negative wedges the file permanently.
    private static func isStaleResumeError(_ error: URLError) -> Bool {
        switch error.code {
        case .badServerResponse,           // ETag mismatch → 416 typically
             .resourceUnavailable,         // object evicted from CDN
             .fileDoesNotExist,            // URL 404 after our blob captured
             .zeroByteResource,            // server now serves empty
             .dataLengthExceedsMaximum:    // range-end past new size
            return true
        default:
            return false
        }
    }

    // MARK: - Private Helpers

    private func fetchData(from url: URL) async throws -> Data {
        let (data, response) = try await metadataSession.data(from: url)
        if let httpResponse = response as? HTTPURLResponse {
            let code = httpResponse.statusCode
            guard code == 200 else {
                throw DownloadError.badStatusCode(code, url: url)
            }
        }
        return data
    }

    /// Kick off a single-file download through `downloadSession` (which may
    /// be a background session) and bridge its delegate callbacks to
    /// async/await + structured-concurrency cancellation.
    ///
    /// - Task cancellation is honoured: if the enclosing Swift Task is
    ///   cancelled, we call `URLSessionDownloadTask.cancel()`, which triggers
    ///   `didCompleteWithError(URLError.cancelled)` — the caller's catch
    ///   block then extracts resumeData from `URLError.downloadTaskResumeData`.
    /// - The delegate's `didFinishDownloadingTo` callback hands us a temp URL
    ///   that Foundation deletes the moment the callback returns, so the
    ///   router moves it to a stable scratch path synchronously before
    ///   resuming the continuation. Caller is responsible for moving the
    ///   returned URL to its final destination.
    private func downloadFile(
        request: URLRequest,
        resumeData: Data?,
        onProgress: @escaping @Sendable (Int64, Int64) -> Void
    ) async throws -> (URL, URLResponse) {
        let task: URLSessionDownloadTask
        if let resumeData {
            task = downloadSession.downloadTask(withResumeData: resumeData)
        } else {
            task = downloadSession.downloadTask(with: request)
        }

        let router = self.router
        let taskID = task.taskIdentifier

        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<(URL, URLResponse), Error>) in
                router.register(
                    taskID: taskID,
                    handlers: DownloadSessionRouter.Handlers(
                        onProgress: onProgress,
                        onComplete: { result in
                            continuation.resume(with: result)
                        }
                    )
                )
                task.resume()
            }
        } onCancel: {
            // Calls delegate didCompleteWithError(URLError.cancelled) —
            // resumeData (if any) ends up in the error's userInfo and
            // eventually reaches the caller's catch block.
            task.cancel()
        }
    }
}

// MARK: - SpeedSampler (exponential moving average throughput)

/// Computes a smoothed bytes-per-second rate over URLSession
/// didWriteData callbacks for a single file. Internally locked because the
/// delegate may fire from any URLSession worker queue; EMA state is tiny so
/// a plain `NSLock` is the right primitive.
internal final class SpeedSampler: @unchecked Sendable {
    /// Smoothing factor — weights the most recent sample against the
    /// previous average. Small enough to mask single-window spikes while
    /// still following real throughput changes within a handful of
    /// windows (network hiccups, LFS CDN ramps).
    private let alpha: Double
    /// Minimum elapsed time between EMA updates. URLSession fires progress
    /// every few ms during LFS downloads; we only need ~2 Hz cadence for
    /// a stable ETA display, so hold the previous value in between.
    private let minSampleInterval: TimeInterval

    private let clock: @Sendable () -> Date
    private let lock = NSLock()
    private var windowStart: Date?
    private var windowStartBytes: Int64 = 0
    private var ema: Double = 0

    init(
        alpha: Double = 0.15,
        minSampleInterval: TimeInterval = 0.5,
        clock: @escaping @Sendable () -> Date = { Date() }
    ) {
        self.alpha = alpha
        self.minSampleInterval = minSampleInterval
        self.clock = clock
    }

    /// Feed a new `totalBytesWritten` sample. Returns the smoothed EMA
    /// throughput in bytes/sec, or the previous value if we're still
    /// inside the throttle window. Returns 0 on the first call (one
    /// sample isn't enough for a rate).
    func record(bytes: Int64) -> Double {
        lock.lock(); defer { lock.unlock() }
        let now = clock()
        guard let start = windowStart else {
            windowStart = now
            windowStartBytes = bytes
            return 0
        }
        let dt = now.timeIntervalSince(start)
        // Throttle: hold the previous EMA until at least minSampleInterval
        // has elapsed. Caller gets a stable number for the whole window.
        guard dt >= minSampleInterval else { return ema }
        let dbytes = Double(bytes - windowStartBytes)
        guard dbytes >= 0 else { return ema }  // shouldn't happen but guard anyway
        let instantaneous = dbytes / dt
        ema = ema == 0
            ? instantaneous
            : alpha * instantaneous + (1 - alpha) * ema
        windowStart = now
        windowStartBytes = bytes
        return ema
    }
}

// MARK: - DownloadSessionRouter

/// Session-level `URLSessionDownloadDelegate` that dispatches callbacks to
/// per-task handlers registered by `HFDownloader.downloadFile(...)`.
///
/// Why a router instead of per-task delegates:
/// - `URLSessionConfiguration.background` sessions require a session-level
///   delegate (set at construction). Per-task delegates are not durable
///   across app relaunch, whereas the session-level delegate is called
///   again on the newly created session for any transfers the system
///   completed in the background.
/// - Having one object route to many in-flight transfers lets us keep the
///   existing continuation-per-file bridge without spawning a fresh
///   delegate object for every file.
///
/// Thread-safety: URLSession delegate callbacks arrive on a private OS
/// operation queue. The router's state (`handlers`, `savedLocations`) is
/// mutated from that queue and read from the actor; a plain `NSLock`
/// protects it.
/// Session-level delegate (see `HFDownloader.sharedBackground`). Marked
/// `internal` so the test-only initialiser on `HFDownloader` can reference
/// it; callers should never interact with it directly.
internal final class DownloadSessionRouter: NSObject,
    URLSessionDownloadDelegate, @unchecked Sendable {

    struct Handlers {
        let onProgress: @Sendable (Int64, Int64) -> Void
        /// Called exactly once. Success delivers `(movedTempURL, response)`;
        /// failure delivers the completion error (or a synthetic
        /// `URLError(.unknown)` if the response vanished).
        let onComplete: @Sendable (Result<(URL, URLResponse), Error>) -> Void
    }

    private let lock = NSLock()
    private var handlers: [Int: Handlers] = [:]
    /// Temp URL we moved the finished download to in
    /// `didFinishDownloadingTo` — read back by `didCompleteWithError`.
    private var savedLocations: [Int: URL] = [:]
    private var responses: [Int: URLResponse] = [:]
    /// Filesystem errors captured during the synchronous scratch-file move
    /// inside `didFinishDownloadingTo` — read back in
    /// `didCompleteWithError` so the caller sees the real cause (sandbox
    /// denial, disk full, etc.) instead of a mysterious `URLError(.unknown)`.
    private var moveErrors: [Int: Error] = [:]

    func register(taskID: Int, handlers: Handlers) {
        lock.lock(); defer { lock.unlock() }
        self.handlers[taskID] = handlers
    }

    // MARK: - URLSessionDownloadDelegate

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        let tid = downloadTask.taskIdentifier
        lock.lock()
        let h = handlers[tid]
        lock.unlock()
        h?.onProgress(totalBytesWritten, totalBytesExpectedToWrite)
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        // Foundation will delete `location` the instant this method returns,
        // so we MUST move the file out synchronously. The destination is a
        // unique path inside the process's temp directory; the HFDownloader
        // actor then moves it to its final location in model directory.
        let tid = downloadTask.taskIdentifier
        let scratch = URL(filePath: NSTemporaryDirectory(), directoryHint: .isDirectory)
            .appending(
                path: "macmlx-dl-\(tid)-\(UUID().uuidString)",
                directoryHint: .notDirectory
            )
        do {
            try FileManager.default.moveItem(at: location, to: scratch)
            lock.lock()
            savedLocations[tid] = scratch
            if let response = downloadTask.response {
                responses[tid] = response
            }
            lock.unlock()
        } catch {
            // Couldn't move the file (sandbox denial, disk full, etc.).
            // Stash the real cause so didCompleteWithError surfaces it
            // instead of falling into the generic `URLError(.unknown)` path.
            lock.lock()
            moveErrors[tid] = error
            lock.unlock()
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        let tid = task.taskIdentifier
        lock.lock()
        let h = handlers.removeValue(forKey: tid)
        let location = savedLocations.removeValue(forKey: tid)
        let response = responses.removeValue(forKey: tid) ?? task.response
        let moveError = moveErrors.removeValue(forKey: tid)
        lock.unlock()

        // Orphaned task (handler gone — e.g. background-session replay on
        // app relaunch, or register() failed): clean up the scratch file
        // so we don't leak into /tmp, then drop.
        guard let h else {
            if let location {
                try? FileManager.default.removeItem(at: location)
            }
            return
        }

        // Prefer a move error over a URLSession error — the move failure
        // is usually the root cause (e.g. sandbox blocked the move, so
        // URLSession "succeeds" but we can't actually deliver the file).
        if let failure = moveError ?? error {
            if let location {
                try? FileManager.default.removeItem(at: location)
            }
            h.onComplete(.failure(failure))
        } else if let location, let response {
            h.onComplete(.success((location, response)))
        } else {
            // Finished without error but the finish delegate didn't land —
            // shouldn't happen, but don't hang the continuation.
            if let location {
                try? FileManager.default.removeItem(at: location)
            }
            h.onComplete(.failure(URLError(.unknown)))
        }
    }
}

// MARK: - Private Decoding Helpers

private struct ModelDetailsEnvelope: Decodable {
    let siblings: [SiblingEntry]
    let sha: String?
    let lastModified: Date?

    struct SiblingEntry: Decodable {
        let rfilename: String
        let size: Int64?
    }
}
