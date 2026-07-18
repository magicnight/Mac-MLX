import CryptoKit
import Foundation

/// Weight-identity fingerprint of a local model directory, used by the cold
/// (SSD) prompt-cache tier to guarantee a restored KV snapshot is only ever
/// reused against the exact weights it was computed from.
///
/// ``PromptCacheStore`` content-addresses cold entries by `(modelID, tokens)`,
/// where `modelID` is a DIRECTORY NAME — not weight identity. If the same path
/// later holds re-downloaded, re-quantized, or otherwise swapped weights, a
/// restored KV cache would produce silently wrong output. Stamping each cold
/// entry with this fingerprint (checked on restore, reject-and-delete on
/// mismatch) closes that gap.
public enum ModelFingerprint {

    /// SHA-256 over the on-disk bytes that determine a model's inference
    /// identity:
    ///
    /// ```
    /// SHA256(
    ///   <config.json raw bytes>                     // layout: model_type, head_dim,
    ///                                               //   layers, rope, scale,
    ///                                               //   quantization.bits/group
    ///   ‖ 0x00
    ///   ‖ for each *.safetensors in <dir>, sorted by filename:   // weight-value changes
    ///        resolvedBasename ‖ "\0" ‖ String(fileSize)
    ///                         ‖ "\0" ‖ String(mtimeNanoseconds) ‖ "\n"
    ///   ‖ <model.safetensors.index.json raw bytes, when present> // opportunistic
    /// )
    /// ```
    ///
    /// Design choices, all deliberate:
    /// - **Config is hashed as RAW bytes**, never canonicalized. A reformatted
    ///   `config.json` that hashes differently is a safe false-mismatch: it
    ///   costs one needless cold miss (a fresh prefill), never a wrong reuse.
    /// - **Shards are sorted by filename** so the digest is independent of the
    ///   directory-enumeration order the filesystem happens to hand back.
    /// - **Shard size + mtime stand in for the (many-GB) weight bytes.** Hashing
    ///   the weights themselves on every load would be prohibitively slow; a
    ///   re-download / re-quantize / swap rewrites at least one shard's size or
    ///   mtime, which this catches cheaply (a handful of `stat`s, no reads).
    ///   `mtime` is folded in as `Int64((timeIntervalSince1970 * 1e9).rounded())`.
    ///   `Date` is `Double`-backed; at the current epoch (~1.77e9 s) `t*1e9` lands
    ///   in `[2^60, 2^61)`, so the ULP is ~256 ns — mtimes inside one ~256 ns
    ///   bucket collapse to the same value. Harmless: a false match would need a
    ///   weight rewrite that keeps identical size AND lands mtime in that same
    ///   256 ns window, which no real filesystem write does (wall-clock ≫ 256 ns).
    ///   The mapping is deterministic for a given `Date`, which is all this needs.
    /// - **Symlinks are resolved before taking the basename.** HF-cache layouts
    ///   store `snapshots/<rev>/model.safetensors` as symlinks into
    ///   content-addressed `blobs/<sha>`; the resolved basename is the stable
    ///   weight identity, so a re-download that repoints the link to a new blob
    ///   changes the fingerprint even if size and mtime coincide.
    ///
    /// - Parameters:
    ///   - directory: the model's local directory (holding `config.json` and
    ///     `*.safetensors` shards at the top level).
    ///   - fileManager: injection seam for tests; defaults to `.default`.
    /// - Returns: the lowercase hex digest, or `nil` when there is no readable
    ///   `config.json`. The caller MUST treat `nil` as "never reuse cold" — a
    ///   model that can't be fingerprinted must never match, never a wildcard.
    public static func compute(
        directory: URL,
        fileManager: FileManager = .default
    ) -> String? {
        let configURL = directory.appending(
            path: "config.json", directoryHint: .notDirectory)
        guard let configData = try? Data(contentsOf: configURL) else {
            return nil
        }

        var hasher = SHA256()
        hasher.update(data: configData)
        hasher.update(data: Data([0x00]))  // separator between config and shard list

        // Enumerate `*.safetensors` RECURSIVELY. mlx-swift-lm's `loadWeights` reads
        // weights with a deep `FileManager.enumerator(at:)`, so any shard the loader
        // loads — including one nested in a subdirectory — must be in this digest;
        // hashing only the top level would let a nested-only weight change go
        // undetected and a stale KV cache be restored (a false match, the exact
        // failure this fingerprint exists to prevent). Each shard is identified by
        // its path RELATIVE to `directory` so two subdirectories holding a
        // same-named shard don't alias, plus the resolved (symlink-followed)
        // basename so an HF-cache re-download that repoints the link to a new
        // content-addressed blob is caught even if size and mtime coincide.
        let base = directory.resolvingSymlinksInPath().standardizedFileURL.path
        var shardLines: [String] = []
        if let enumerator = fileManager.enumerator(
            at: directory,
            includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ) {
            for case let shard as URL in enumerator where shard.pathExtension == "safetensors" {
                // Relative path from the model dir (unresolved — distinguishes
                // subdirectories); resolved size/mtime/basename reflect the real file.
                let relative = shard.standardizedFileURL.path.hasPrefix(base + "/")
                    ? String(shard.standardizedFileURL.path.dropFirst(base.count + 1))
                    : shard.lastPathComponent
                let resolved = shard.resolvingSymlinksInPath()
                let values = try? resolved.resourceValues(
                    forKeys: [.fileSizeKey, .contentModificationDateKey])
                let size = values?.fileSize ?? 0
                let mtimeNanos: Int64 = values?.contentModificationDate.map {
                    Int64(($0.timeIntervalSince1970 * 1_000_000_000).rounded())
                } ?? 0
                shardLines.append("\(relative)\u{0}\(resolved.lastPathComponent)\u{0}\(size)\u{0}\(mtimeNanos)\n")
            }
        }
        // Sorted so the digest is independent of enumeration order.
        for line in shardLines.sorted() {
            if let lineBytes = line.data(using: .utf8) {
                hasher.update(data: lineBytes)
            }
        }

        // Opportunistic: fold in the shard index when present. Never required —
        // its absence simply means one fewer input to the digest.
        let indexURL = directory.appending(
            path: "model.safetensors.index.json", directoryHint: .notDirectory)
        if let indexData = try? Data(contentsOf: indexURL) {
            hasher.update(data: indexData)
        }

        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
