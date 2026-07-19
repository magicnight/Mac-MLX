import Foundation
import Testing

@testable import MacMLXCore

/// MLX-free tests for ``ModelFingerprint`` — the cold prompt-cache tier's
/// weight-identity stamp. Everything here is pure filesystem + CryptoKit, so it
/// runs under bare `swift test` (no metallib needed).

// MARK: - Helpers

/// A fixed, integer-seconds epoch so pinned mtimes round-trip exactly through
/// `Date`'s `Double` backing and the nanosecond fold is fully deterministic.
private let baseDate = Date(timeIntervalSince1970: 1_700_000_000)

private func makeTestDir() -> URL {
    FileManager.default.temporaryDirectory
        .appending(path: "fingerprint-test-\(UUID().uuidString)", directoryHint: .isDirectory)
}

private func writeConfig(_ json: String, in dir: URL) throws {
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    try Data(json.utf8).write(
        to: dir.appending(path: "config.json", directoryHint: .notDirectory))
}

/// Write a `byteCount`-byte (zero-filled) `*.safetensors` shard with an explicit
/// modification date. Only the shard's name, size, and mtime feed the
/// fingerprint — never its contents — so zero-fill is sufficient.
@discardableResult
private func writeShard(
    _ name: String, byteCount: Int, mtime: Date, in dir: URL
) throws -> URL {
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let url = dir.appending(path: name, directoryHint: .notDirectory)
    try Data(count: byteCount).write(to: url)
    try FileManager.default.setAttributes(
        [.modificationDate: mtime], ofItemAtPath: url.path)
    return url
}

// MARK: - Tests

/// (a) Compute is stable across repeated calls on an unchanged directory.
@Test
func fingerprintIsStableAcrossCalls() throws {
    let dir = makeTestDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    try writeConfig(#"{"model_type":"qwen3"}"#, in: dir)
    try writeShard("model.safetensors", byteCount: 128, mtime: baseDate, in: dir)

    let first = ModelFingerprint.compute(directory: dir)
    let second = ModelFingerprint.compute(directory: dir)
    #expect(first != nil)
    #expect(first == second)
}

/// (b) A change to `config.json`'s bytes changes the fingerprint (layout
/// identity: model_type, head_dim, rope, quantization, …).
@Test
func fingerprintChangesWhenConfigBytesChange() throws {
    let dir = makeTestDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    try writeConfig(#"{"model_type":"qwen3"}"#, in: dir)
    try writeShard("model.safetensors", byteCount: 128, mtime: baseDate, in: dir)
    let before = ModelFingerprint.compute(directory: dir)

    // Rewrite config with different bytes; the shard is untouched.
    try writeConfig(#"{"model_type":"qwen3","tie_word_embeddings":false}"#, in: dir)
    let after = ModelFingerprint.compute(directory: dir)

    #expect(before != nil)
    #expect(before != after)
}

/// (c) A change to a shard's SIZE changes the fingerprint, with mtime pinned
/// equal so the difference is attributable to size alone.
@Test
func fingerprintChangesWhenShardSizeChanges() throws {
    let dir = makeTestDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    try writeConfig(#"{"model_type":"qwen3"}"#, in: dir)
    try writeShard("model.safetensors", byteCount: 128, mtime: baseDate, in: dir)
    let before = ModelFingerprint.compute(directory: dir)

    // Same name, same mtime, DIFFERENT size.
    try writeShard("model.safetensors", byteCount: 256, mtime: baseDate, in: dir)
    let after = ModelFingerprint.compute(directory: dir)

    #expect(before != nil)
    #expect(before != after)
}

/// (d) A change to a shard's MTIME changes the fingerprint, with bytes/size held
/// constant so the difference is attributable to mtime alone.
@Test
func fingerprintChangesWhenShardMtimeChanges() throws {
    let dir = makeTestDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    try writeConfig(#"{"model_type":"qwen3"}"#, in: dir)
    let shard = try writeShard("model.safetensors", byteCount: 128, mtime: baseDate, in: dir)
    let before = ModelFingerprint.compute(directory: dir)

    // Same bytes/size, DIFFERENT mtime (one hour later).
    try FileManager.default.setAttributes(
        [.modificationDate: baseDate.addingTimeInterval(3600)], ofItemAtPath: shard.path)
    let after = ModelFingerprint.compute(directory: dir)

    #expect(before != nil)
    #expect(before != after)
}

/// (e) No readable `config.json` ⇒ `nil` (the caller treats nil as "never reuse
/// cold", never a wildcard match).
@Test
func fingerprintIsNilWhenConfigAbsent() throws {
    let dir = makeTestDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    // Shards present, but NO config.json.
    try writeShard("model.safetensors", byteCount: 128, mtime: baseDate, in: dir)

    #expect(ModelFingerprint.compute(directory: dir) == nil)
}

/// (f) The fingerprint is independent of directory-enumeration order: two
/// directories with identical inputs written in OPPOSITE creation order produce
/// the same digest, because shards are sorted by filename before hashing.
@Test
func fingerprintIsOrderIndependentOfEnumeration() throws {
    let dirA = makeTestDir()
    let dirB = makeTestDir()
    defer { try? FileManager.default.removeItem(at: dirA) }
    defer { try? FileManager.default.removeItem(at: dirB) }

    let config = #"{"model_type":"qwen3"}"#
    let shard1 = "model-00001-of-00002.safetensors"
    let shard2 = "model-00002-of-00002.safetensors"

    // Dir A: config, then shard1, then shard2.
    try writeConfig(config, in: dirA)
    try writeShard(shard1, byteCount: 128, mtime: baseDate, in: dirA)
    try writeShard(shard2, byteCount: 256, mtime: baseDate.addingTimeInterval(10), in: dirA)

    // Dir B: identical inputs, created in REVERSE order.
    try writeShard(shard2, byteCount: 256, mtime: baseDate.addingTimeInterval(10), in: dirB)
    try writeShard(shard1, byteCount: 128, mtime: baseDate, in: dirB)
    try writeConfig(config, in: dirB)

    let fa = ModelFingerprint.compute(directory: dirA)
    let fb = ModelFingerprint.compute(directory: dirB)
    #expect(fa != nil)
    #expect(fa == fb)
}

/// (g) A shard NESTED in a subdirectory is covered. mlx-swift-lm's `loadWeights`
/// reads weights recursively, so a nested-only change must move the fingerprint;
/// a shallow (top-level-only) scan would miss it and let a stale KV cache be
/// restored against changed weights. Fails against a non-recursive `compute`.
@Test
func fingerprintCoversNestedShards() throws {
    let dir = makeTestDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    try writeConfig(#"{"model_type":"qwen3"}"#, in: dir)
    let sub = dir.appending(path: "shards", directoryHint: .isDirectory)
    try writeShard("model.safetensors", byteCount: 128, mtime: baseDate, in: sub)
    let before = ModelFingerprint.compute(directory: dir)

    // Change ONLY the nested shard's size; config + top level are untouched.
    try writeShard("model.safetensors", byteCount: 256, mtime: baseDate, in: sub)
    let after = ModelFingerprint.compute(directory: dir)

    #expect(before != nil)
    #expect(before != after, "a nested shard change must move the fingerprint")
}
