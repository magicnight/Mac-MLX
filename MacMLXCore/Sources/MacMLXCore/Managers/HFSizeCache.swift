import Foundation

/// Persistent disk cache for `HFDownloader.sizeBytes(for:)` results.
///
/// HF's model-metadata endpoint plus the per-file HEAD fallback can
/// take 1-3 seconds per model on slow networks. Caching the result
/// for 7 days means a re-search hits the cache and populates the
/// size badge instantly — HF model repo sizes rarely change week
/// to week, and a stale size is harmless (shown as a hint, not a
/// contract).
///
/// Persistence format: a single JSON at
/// `~/.mac-mlx/hf-cache/sizes.json` mapping `modelID → (size, ts)`.
/// Concurrent actor access; writes debounced on each mutation but
/// we keep it simple and write-through for correctness.
public actor HFSizeCache {

    // MARK: - Types

    private struct Entry: Codable {
        let sizeBytes: Int64
        let fetchedAt: Date
    }

    private struct Payload: Codable {
        var entries: [String: Entry]
    }

    // MARK: - Constants

    /// Entries older than this are treated as missing. Popular models
    /// update maybe monthly; a week keeps the cache warm while still
    /// letting fresh data trickle in.
    public static let ttl: TimeInterval = 7 * 24 * 60 * 60  // 7 days

    // MARK: - State

    private let fileURL: URL
    private var payload: Payload
    private var loaded = false

    // MARK: - Init

    /// Production initialiser — persists to `~/.mac-mlx/hf-cache/sizes.json`.
    public init() {
        self.fileURL = DataRoot.macMLX("hf-cache/sizes.json")
        self.payload = Payload(entries: [:])
    }

    /// Test initialiser — caller picks the backing file URL.
    public init(fileURL: URL) {
        self.fileURL = fileURL
        self.payload = Payload(entries: [:])
    }

    // MARK: - Public API

    /// Return the cached size for `modelID`, or nil if absent or stale.
    public func get(_ modelID: String) async -> Int64? {
        await ensureLoaded()
        guard let entry = payload.entries[modelID] else { return nil }
        if Date().timeIntervalSince(entry.fetchedAt) > Self.ttl { return nil }
        return entry.sizeBytes
    }

    /// Store a freshly-fetched size for `modelID`.
    public func put(_ modelID: String, size: Int64) async {
        await ensureLoaded()
        payload.entries[modelID] = Entry(sizeBytes: size, fetchedAt: Date())
        await flush()
    }

    /// Remove all cached entries. Exposed for tests and for a future
    /// "Clear HF cache" setting.
    public func clear() async {
        payload.entries = [:]
        await flush()
    }

    // MARK: - Private

    private func ensureLoaded() async {
        guard !loaded else { return }
        loaded = true
        guard let data = try? Data(contentsOf: fileURL) else { return }
        if let decoded = try? JSONDecoder().decode(Payload.self, from: data) {
            payload = decoded
        }
    }

    private func flush() async {
        let parent = fileURL.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        guard let data = try? encoder.encode(payload) else { return }
        try? data.write(to: fileURL, options: .atomic)
    }
}
