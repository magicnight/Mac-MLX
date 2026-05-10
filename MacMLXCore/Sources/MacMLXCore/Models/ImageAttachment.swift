import Foundation

/// One image attached to a `ChatMessage`.
///
/// Sendable + Codable so the attachment round-trips cleanly through
/// `ConversationStore` JSON, `GenerateRequest` payloads, and the
/// OpenAI-multimodal HTTP wire format. The carrier lives in Core
/// (not the GUI target) because the CLI surfaces, the HTTP server,
/// and the GUI all need to talk about it.
public struct ImageAttachment: Codable, Hashable, Sendable, Equatable {
    /// Local file URL where the image bytes live.
    ///
    /// Conversation save / load is responsible for copying user-picked
    /// files into `~/.mac-mlx/conversations/<uuid>/images/` so the URL
    /// stays stable across app restarts and survives the user moving
    /// the original file. Until the persistence step lands (v0.4.1
    /// UI+HTTP PR), the URL points at the user's pick site directly.
    public let fileURL: URL

    /// IANA MIME type, e.g. `image/jpeg` or `image/png`.
    ///
    /// Required for the OpenAI multimodal `image_url` data-URL payload
    /// shape (`data:<mime>;base64,…`). We carry it explicitly rather
    /// than re-deriving from `fileURL.pathExtension` at every send
    /// because some pickers return URLs with empty extensions
    /// (Photos library picks, paste-from-clipboard temp files).
    public let mimeType: String

    public init(fileURL: URL, mimeType: String) {
        self.fileURL = fileURL
        self.mimeType = mimeType
    }

    /// Best-effort MIME-type lookup from a path extension.
    ///
    /// Returns `nil` for extensions we don't recognise — callers
    /// should treat that as "this isn't an image we know how to
    /// attach" and reject the file. Case-insensitive.
    public static func mimeType(forPathExtension ext: String) -> String? {
        switch ext.lowercased() {
        case "jpg", "jpeg": return "image/jpeg"
        case "png":         return "image/png"
        case "webp":        return "image/webp"
        case "gif":         return "image/gif"
        case "heic":        return "image/heic"
        case "bmp":         return "image/bmp"
        default:            return nil
        }
    }
}
