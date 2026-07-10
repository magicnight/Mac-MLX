// Copyright © 2026 macMLX. English comments only.

import Foundation

/// Process-wide cache of ``TokenVocabularyTable`` keyed by model id, so the
/// one-time cost of decoding an entire vocabulary is paid at most once per
/// resident model rather than once per constrained request.
///
/// Thread-safe via an internal lock (`@unchecked Sendable`): concurrent first
/// requests for the same model serialize on the build and then share the
/// result. Keyed additionally on vocabulary size so a model reloaded with a
/// different logit dimension rebuilds instead of returning a stale table.
public final class TokenVocabularyCache: @unchecked Sendable {

    private struct Entry {
        let vocabularySize: Int
        let table: TokenVocabularyTable
    }

    private let lock = NSLock()
    private var store: [String: Entry] = [:]

    public init() {}

    /// Return the cached table for `modelID`, building and caching it on a miss
    /// (or on a vocabulary-size change).
    ///
    /// - Parameters:
    ///   - modelID: resident model identity — the cache key.
    ///   - vocabularySize: authoritative vocabulary size (the model's logit
    ///     dimension). A change from a cached entry forces a rebuild.
    ///   - stopTokenIDs: stop/EOS ids for classification.
    ///   - decode: standalone decode of one token id (`skipSpecialTokens:false`).
    ///     Invoked only on a cache miss.
    public func table(
        modelID: String,
        vocabularySize: Int,
        stopTokenIDs: Set<Int>,
        decode: (Int) -> String?
    ) -> TokenVocabularyTable {
        lock.lock()
        defer { lock.unlock() }
        if let entry = store[modelID], entry.vocabularySize == vocabularySize {
            return entry.table
        }
        let table = TokenVocabularyTable(
            vocabularySize: vocabularySize,
            stopTokenIDs: stopTokenIDs,
            decode: decode
        )
        store[modelID] = Entry(vocabularySize: vocabularySize, table: table)
        return table
    }

    /// Drop any cached table for `modelID` (e.g. when the model is unloaded).
    public func invalidate(modelID: String) {
        lock.lock()
        defer { lock.unlock() }
        store[modelID] = nil
    }
}
