// Copyright © 2026 macMLX. English comments only.

import Foundation

/// One persisted cold-tier entry.
///
/// The `tokens` array is the load-bearing field: the raw prompt tokens live
/// NOWHERE else on disk. A cold file is named by `SHA256(modelID, tokens)`
/// (``PromptCacheKey``) and its embedded safetensors metadata carries only the
/// token *count*, so a cross-session longest-common-prefix rebuild — the whole
/// point of Wave 2b — can only recover the tokens from this manifest. Everything
/// else here is bookkeeping the rebuild / cold-trie fetch needs without touching
/// the (potentially many-hundred-MB) safetensors payload:
/// - `hashString` locates the file and integrity-checks the record,
/// - `modelFingerprint` is the Wave 2a weight-identity stamp (always present —
///   ``PromptCacheStore/demoteToCold`` never spills a fingerprint-less entry),
/// - `isTrimmable` lets the cold *longer* fetch pre-gate a trim without a load,
/// - `nbytes` / `mtime` are recorded for completeness and possible future
///   manifest-level accounting (the byte budget itself is enforced against the
///   filesystem, not this field).
struct ColdIndexEntry: Codable, Sendable, Equatable {
    let hashString: String
    let modelID: String
    let tokens: [Int]
    let tokenCount: Int
    let modelFingerprint: String
    let nbytes: Int
    let mtime: Date
    let isTrimmable: Bool
}

/// The on-disk cold-tier manifest: a version-stamped list of ``ColdIndexEntry``.
/// Persisted as `index.json` at the cold root and reloaded at startup to rebuild
/// the in-memory cold trie so cross-session LCP survives a restart.
struct ColdIndexManifest: Codable, Sendable, Equatable {
    let formatVersion: Int
    let entries: [ColdIndexEntry]
}

/// Load / store helpers and the integrity check for the cold-tier manifest.
/// MLX-free — only Foundation and ``PromptCacheKey`` — so it is exhaustively
/// unit-testable under bare `swift test`.
enum ColdIndex {

    /// Stamps the cold-tier on-disk LAYOUT schema. It is the UNION of three
    /// things that must all stay compatible for a persisted cold entry to be
    /// safely reusable across a restart:
    ///   1. mlx-swift-lm's `savePromptCache` / `loadPromptCache` state +
    ///      metaState serialisation (the safetensors payload schema),
    ///   2. ``PromptCacheKey``'s hash construction (how a file is named), and
    ///   3. this manifest's own JSON schema (``ColdIndexEntry`` fields).
    /// Bump on ANY of those changing. ``PromptCacheStore`` version-gates the
    /// manifest on load: a mismatch discards it wholesale and starts the cold
    /// trie empty (degraded — exact-hash re-hits still work, never wrong output).
    static let coldFormatVersion = 1

    /// Decode the manifest at `url`, or `nil` on any failure — an absent file,
    /// a truncated/garbage payload, or a decode error. The read is atomic via
    /// `Data(contentsOf:)`. The VERSION GATE is deliberately NOT applied here:
    /// the caller compares `formatVersion` against ``coldFormatVersion`` so the
    /// gate lives next to the degraded-mode handling it drives.
    static func load(from url: URL) -> ColdIndexManifest? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(ColdIndexManifest.self, from: data)
    }

    /// Encode and atomically write `manifest` to `url`. Best-effort: a failed
    /// encode or write is swallowed, matching the cold tier's leniency
    /// elsewhere — the manifest is a re-derivable hint (the next demote rewrites
    /// it), never a source of truth whose loss can corrupt output.
    static func write(_ manifest: ColdIndexManifest, to url: URL) {
        guard let data = try? JSONEncoder().encode(manifest) else { return }
        try? data.write(to: url, options: .atomic)
    }

    /// An entry is trustworthy only when its stored `hashString` equals the hash
    /// the CURRENT ``PromptCacheKey`` derives from its `(modelID, tokens)`. A
    /// mismatch means the manifest and the file-naming scheme disagree — a
    /// corrupted record, or a hash-scheme change that slipped past the version
    /// gate — so the rebuild drops it rather than trust a record that can't name
    /// its own backing file.
    static func isConsistent(_ entry: ColdIndexEntry) -> Bool {
        PromptCacheKey(modelID: entry.modelID, tokens: entry.tokens).hashString
            == entry.hashString
    }
}
