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

    public init(
        id: String,
        author: String? = nil,
        downloads: Int? = nil,
        likes: Int? = nil,
        tags: [String] = [],
        lastModified: Date? = nil
    ) {
        self.id = id
        self.author = author
        self.downloads = downloads
        self.likes = likes
        self.tags = tags
        self.lastModified = lastModified
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
