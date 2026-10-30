// Copyright © 2026 macMLX. English comments only.

import Foundation

/// Carries a turn's reasoning through the opaque fields clients replay
/// verbatim: an Anthropic `thinking` block's `signature` and a Responses
/// `reasoning` item's `encrypted_content`.
///
/// A client that hides reasoning (Claude Code sends `display: "omitted"`) or
/// replays only what it cannot read still sends these fields back on the next
/// turn, so the server can recover the text and give it to the chat template.
///
/// This is NOT encryption, and not a signature: the text is only
/// base64-encoded behind a version prefix. Nothing needs protecting — the
/// server is local, the client already holds the plaintext, and any client
/// can send whatever reasoning it likes through `reasoning_content` anyway.
/// The prefix only tells macMLX's own blobs apart from a real provider's
/// signature or encrypted content, which decode to nil.
enum ReasoningCarrier {

    /// What every carried blob starts with.
    static let prefix = "macmlx.r1."

    /// `text` as a blob `decode(_:)` turns back into it.
    static func encode(_ text: String) -> String {
        prefix + Data(text.utf8).base64EncodedString()
    }

    /// The reasoning `blob` carries, or nil when it is not one of macMLX's
    /// (another provider's blob, junk), is not valid base64 or UTF-8, or
    /// carries no text.
    static func decode(_ blob: String?) -> String? {
        guard let blob, blob.hasPrefix(prefix),
              let data = Data(base64Encoded: String(blob.dropFirst(prefix.count))),
              let text = String(data: data, encoding: .utf8),
              !text.isEmpty
        else { return nil }
        return text
    }
}
