import Foundation

/// A model entry returned by the Hugging Face Hub `/api/models` endpoint.
///
/// Only the fields macMLX cares about are decoded. Use `JSONDecoder.huggingFace`
/// for any decoding work — it pre-configures a date strategy compatible with
/// the Hub's ISO-8601 timestamps.
public struct HFModel: Codable, Hashable, Identifiable, Sendable {
    public let id: String
    public let author: String?
    public let downloads: Int?
    public let likes: Int?
    public let tags: [String]
    public let lastModified: Date?
    /// Total repo size in bytes (sum of `siblings[*].size`). `nil` until
    /// the size-enrichment fetch completes, since the `/api/models` search
    /// endpoint doesn't return it. Marked `var` so the UI layer can fill
    /// it in after an out-of-band `HFDownloader.sizeBytes(for:)` call.
    public var sizeBytes: Int64?

    public init(
        id: String,
        author: String? = nil,
        downloads: Int? = nil,
        likes: Int? = nil,
        tags: [String] = [],
        lastModified: Date? = nil,
        sizeBytes: Int64? = nil
    ) {
        self.id = id
        self.author = author
        self.downloads = downloads
        self.likes = likes
        self.tags = tags
        self.lastModified = lastModified
        self.sizeBytes = sizeBytes
    }

    /// `"2.5 GB"`, `"820 MB"`, `"—"` when size is unknown. Uses the
    /// same base-10 convention as `DownloadProgress.currentFileHuman`.
    public var sizeHuman: String {
        guard let bytes = sizeBytes, bytes > 0 else { return "—" }
        let b = Double(bytes)
        if b >= 1_000_000_000 { return String(format: "%.1f GB", b / 1_000_000_000) }
        if b >= 1_000_000     { return String(format: "%.0f MB", b / 1_000_000) }
        if b >= 1_000         { return String(format: "%.0f KB", b / 1_000) }
        return "\(bytes) B"
    }
}

extension JSONDecoder {
    /// JSON decoder pre-configured for Hugging Face Hub responses.
    public static var huggingFace: JSONDecoder {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }
}
