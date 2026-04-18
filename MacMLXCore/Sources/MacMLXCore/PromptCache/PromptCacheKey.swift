import CryptoKit
import Foundation

/// Deterministic hash key identifying a cached KV-cache snapshot.
///
/// MVP hashes the entire token prefix. v0.4.1+ will switch to a
/// vLLM-style chained block hash (256 tokens per block + parent
/// hash) to enable longest-common-prefix matching across siblings;
/// today two requests have to share the EXACT same prefix to
/// benefit from the cache.
public struct PromptCacheKey: Hashable, Sendable {
    public let modelID: String
    public let tokenCount: Int
    public let hashString: String

    public init(modelID: String, tokens: [Int]) {
        self.modelID = modelID
        self.tokenCount = tokens.count
        self.hashString = Self.hash(modelID: modelID, tokens: tokens)
    }

    /// SHA-256 over `(modelID, tokens)`. Tokens encoded as
    /// little-endian Int32 for cross-platform stability.
    private static func hash(modelID: String, tokens: [Int]) -> String {
        var hasher = SHA256()
        if let modelBytes = modelID.data(using: .utf8) {
            hasher.update(data: modelBytes)
        }
        hasher.update(data: Data([0x00]))  // separator
        var buf = Data(capacity: tokens.count * 4)
        for tok in tokens {
            var v = Int32(tok).littleEndian
            withUnsafeBytes(of: &v) { buf.append(contentsOf: $0) }
        }
        hasher.update(data: buf)
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// `<root>/<shardChar>/<fullHash>.safetensors`. 16-way fanout
    /// keeps any single directory from getting huge when the cold
    /// store grows. `shardChar` is the first hex char of the hash.
    public func shardedFileURL(under root: URL) -> URL {
        let shard = String(hashString.prefix(1))
        return root
            .appending(path: shard, directoryHint: .isDirectory)
            .appending(path: "\(hashString).safetensors", directoryHint: .notDirectory)
    }
}
