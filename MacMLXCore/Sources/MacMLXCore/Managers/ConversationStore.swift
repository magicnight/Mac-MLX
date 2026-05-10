// ConversationStore.swift
// MacMLXCore
//
// Persistent chat history store.
//
// v0.1.1 scope (issue #9):
// - One JSON file per conversation at `<directory>/{uuid}.json`.
// - `save(_:)` writes atomically.
// - `loadLatest()` returns the most-recently-updated conversation (by
//   `updatedAt`), or nil on an empty store.
// - UI side auto-saves after each message and loads the latest on launch,
//   giving the user-visible guarantee "my chat survives app restart".
// - Multi-conversation sidebar UI (rename, delete, list) is deferred —
//   `list()` is already exposed here so the UI can ship later without a
//   storage-layer change.

import Foundation

// MARK: - Message

/// A chat message as persisted to disk. Mirrors the UI-layer UIChatMessage
/// minus transient fields (`isGenerating`).
public struct StoredMessage: Codable, Hashable, Identifiable, Sendable {
    public let id: UUID
    public let role: MessageRole
    public var content: String
    public let timestamp: Date
    public var tokenCount: Int?
    /// Image attachments tied to this turn. Empty for text-only — the
    /// common case. URLs point into
    /// `<conversations>/<conv-uuid>/images/...` once the conversation
    /// has been saved (see `ConversationStore.save(_:)`).
    /// Backwards-compatible: pre-v0.4.1 JSON without an `images` key
    /// decodes with an empty array.
    public var images: [ImageAttachment]

    public init(
        id: UUID = UUID(),
        role: MessageRole,
        content: String,
        timestamp: Date = Date(),
        tokenCount: Int? = nil,
        images: [ImageAttachment] = []
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.timestamp = timestamp
        self.tokenCount = tokenCount
        self.images = images
    }

    private enum CodingKeys: String, CodingKey {
        case id, role, content, timestamp, tokenCount, images
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try c.decode(UUID.self, forKey: .id)
        self.role = try c.decode(MessageRole.self, forKey: .role)
        self.content = try c.decode(String.self, forKey: .content)
        self.timestamp = try c.decode(Date.self, forKey: .timestamp)
        self.tokenCount = try c.decodeIfPresent(Int.self, forKey: .tokenCount)
        self.images = try c.decodeIfPresent([ImageAttachment].self, forKey: .images) ?? []
    }
}

// MARK: - Conversation

/// A single chat session with its metadata and full message history.
public struct Conversation: Codable, Hashable, Identifiable, Sendable {
    public let id: UUID
    public var title: String
    public var messages: [StoredMessage]
    public let createdAt: Date
    public var updatedAt: Date
    /// Which model produced the assistant messages (best-effort; nil if
    /// no model was loaded when the conversation started).
    public var modelID: String?
    public var systemPrompt: String

    public init(
        id: UUID = UUID(),
        title: String = "New Chat",
        messages: [StoredMessage] = [],
        createdAt: Date = Date(),
        updatedAt: Date = Date(),
        modelID: String? = nil,
        systemPrompt: String = ""
    ) {
        self.id = id
        self.title = title
        self.messages = messages
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.modelID = modelID
        self.systemPrompt = systemPrompt
    }

    /// Derive a short title from the first user message (up to ~40 chars).
    /// Falls back to "New Chat" if no user message exists yet.
    public var derivedTitle: String {
        guard let first = messages.first(where: { $0.role == .user })?.content,
              !first.isEmpty else { return "New Chat" }
        let trimmed = first.trimmingCharacters(in: .whitespacesAndNewlines)
        let prefix = trimmed.prefix(40)
        return prefix.count < trimmed.count ? String(prefix) + "…" : String(prefix)
    }
}

// MARK: - Store

/// Filesystem-backed conversation store. Every method is `async` because
/// all I/O is serialised on the actor.
public actor ConversationStore {

    private let directory: URL
    private let fileManager: FileManager

    /// Create a store backed by `<directory>/{uuid}.json`.
    ///
    /// Default directory is `~/.mac-mlx/conversations/` (real home — takes
    /// advantage of macOS App Sandbox's dotfile exemption, same as the
    /// rest of the `.mac-mlx` data root).
    public init(directory: URL? = nil, fileManager: FileManager = .default) {
        self.directory = directory ?? DataRoot.macMLX("conversations")
        self.fileManager = fileManager
    }

    // MARK: - Public API

    /// Persist `conversation` to disk atomically. Creates the directory if
    /// missing. Bumps `updatedAt` to "now" before writing.
    ///
    /// Image attachments referenced by any message get copied (best-
    /// effort) into `<directory>/<conv-uuid>/images/<image-uuid>.<ext>`
    /// the first time we see them, so the saved JSON URLs are stable
    /// across user moves of the picked file. Already-internal URLs
    /// (already pointing at the conversation's images dir) are left
    /// in place. A copy failure logs to stderr and falls through —
    /// the conversation still saves with the original URL, which the
    /// reader will tolerate (image just won't load if the source
    /// disappears).
    ///
    /// Uses `JSONCoding.precisionEncoder` so rapid saves produce distinct
    /// `updatedAt` values for `list()` sort stability. Decoder accepts
    /// pre-v0.3 ISO-8601-string files for backward compat.
    public func save(_ conversation: Conversation) async throws {
        try ensureDirectory()
        var copy = conversation
        copy.updatedAt = Date()
        // Auto-derive title if it's still the default.
        if copy.title == "New Chat" {
            copy.title = copy.derivedTitle
        }
        copy.messages = copy.messages.map { internaliseImages(of: $0, conversationID: copy.id) }

        let data = try JSONCoding.precisionEncoder().encode(copy)
        let url = fileURL(for: copy.id)
        try data.write(to: url, options: .atomic)
    }

    /// Copy any image attachments that live outside the conversation's
    /// own `images/` directory into it, and rewrite the URLs. Idempotent:
    /// images already inside the conversation directory are kept verbatim.
    private func internaliseImages(
        of message: StoredMessage,
        conversationID: UUID
    ) -> StoredMessage {
        guard !message.images.isEmpty else { return message }
        let imagesDir = imagesDirectory(for: conversationID)

        let updated: [ImageAttachment] = message.images.map { att in
            // Already internal? — leave it alone.
            if att.fileURL.path.hasPrefix(imagesDir.path) {
                return att
            }
            // Try to copy. On any error, fall through to the original
            // attachment so the save still succeeds.
            do {
                try fileManager.createDirectory(at: imagesDir, withIntermediateDirectories: true)
                let ext = att.fileURL.pathExtension.isEmpty ? "img" : att.fileURL.pathExtension
                let dest = imagesDir.appending(
                    path: "\(UUID().uuidString).\(ext)", directoryHint: .notDirectory)
                try fileManager.copyItem(at: att.fileURL, to: dest)
                return ImageAttachment(fileURL: dest, mimeType: att.mimeType)
            } catch {
                FileHandle.standardError.write(Data(
                    "[ConversationStore] image copy failed for \(att.fileURL.path): \(error)\n".utf8
                ))
                return att
            }
        }
        var out = message
        out.images = updated
        return out
    }

    /// Return the most-recently-updated conversation, or nil if the store
    /// is empty. Corrupt files are skipped (they don't block other loads).
    public func loadLatest() async throws -> Conversation? {
        try ensureDirectory()
        let urls = try fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.contentModificationDateKey]
        )
        .filter { $0.pathExtension == "json" }
        guard !urls.isEmpty else { return nil }

        let decoder = JSONCoding.tolerantDecoder()
        var best: Conversation?
        for url in urls {
            guard let data = try? Data(contentsOf: url),
                  let conv = try? decoder.decode(Conversation.self, from: data) else {
                continue
            }
            if let current = best {
                if conv.updatedAt > current.updatedAt { best = conv }
            } else {
                best = conv
            }
        }
        return best
    }

    /// Metadata for all conversations, sorted by `updatedAt` descending.
    /// Used by the (future) sidebar list view.
    public func list() async throws -> [Conversation] {
        try ensureDirectory()
        let urls = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" }

        let decoder = JSONCoding.tolerantDecoder()
        var loaded: [Conversation] = []
        for url in urls {
            if let data = try? Data(contentsOf: url),
               let conv = try? decoder.decode(Conversation.self, from: data) {
                loaded.append(conv)
            }
        }
        return loaded.sorted { $0.updatedAt > $1.updatedAt }
    }

    /// Remove a conversation from disk along with any internalised
    /// image attachments. Idempotent — no error if missing.
    public func delete(id: UUID) async throws {
        let url = fileURL(for: id)
        try? fileManager.removeItem(at: url)
        // Also remove the per-conversation directory if it exists
        // (images live under it; pre-v0.4.1 conversations didn't
        // create one and this no-ops cleanly).
        let convDir = conversationDirectory(for: id)
        try? fileManager.removeItem(at: convDir)
    }

    // MARK: - Private

    private func fileURL(for id: UUID) -> URL {
        directory.appending(path: "\(id.uuidString).json", directoryHint: .notDirectory)
    }

    /// Per-conversation directory holding sidecar resources (images,
    /// future audio attachments). Created on demand in
    /// `internaliseImages(of:conversationID:)` and torn down by
    /// `delete(id:)`.
    private func conversationDirectory(for id: UUID) -> URL {
        directory.appending(path: id.uuidString, directoryHint: .isDirectory)
    }

    /// Path used for image attachments of a given conversation.
    private func imagesDirectory(for id: UUID) -> URL {
        conversationDirectory(for: id)
            .appending(path: "images", directoryHint: .isDirectory)
    }

    private func ensureDirectory() throws {
        if !fileManager.fileExists(atPath: directory.path) {
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        }
    }
}
