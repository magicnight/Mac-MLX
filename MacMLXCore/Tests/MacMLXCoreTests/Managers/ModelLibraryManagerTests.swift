import Testing
import Foundation
@testable import MacMLXCore

// MARK: - Helpers

private func makeTestRoot() -> URL {
    FileManager.default.temporaryDirectory
        .appending(path: "lib-test-\(UUID().uuidString)", directoryHint: .isDirectory)
}

/// Creates a directory at `url` and writes empty files for each name.
private func populateDir(_ url: URL, files: [String]) throws {
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    for fileName in files {
        let fileURL = url.appending(path: fileName, directoryHint: .notDirectory)
        try Data().write(to: fileURL)
    }
}

// MARK: - Tests

@Test func scanFindsMLXLikeDirectory() async throws {
    let root = makeTestRoot()
    defer { try? FileManager.default.removeItem(at: root) }

    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let modelDir = root.appending(path: "MyModel", directoryHint: .isDirectory)
    try populateDir(modelDir, files: ["config.json", "tokenizer.json", "model.safetensors"])

    let manager = ModelLibraryManager()
    let results = try await manager.scan(root)

    #expect(results.count == 1)
    #expect(results[0].id == "MyModel")
    #expect(results[0].format == .mlx)
}

@Test func scanIgnoresGGUFAndUnknown() async throws {
    let root = makeTestRoot()
    defer { try? FileManager.default.removeItem(at: root) }

    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

    // GGUF model — should be skipped
    let ggufDir = root.appending(path: "gguf-model", directoryHint: .isDirectory)
    try populateDir(ggufDir, files: ["model.gguf"])

    // Unknown model — should be skipped
    let unknownDir = root.appending(path: "unknown-stuff", directoryHint: .isDirectory)
    try populateDir(unknownDir, files: ["random.bin", "data.dat"])

    let manager = ModelLibraryManager()
    let results = try await manager.scan(root)

    #expect(results.isEmpty)
}

@Test func scanExtractsQuantizationFromName() async throws {
    let root = makeTestRoot()
    defer { try? FileManager.default.removeItem(at: root) }

    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let modelDir = root.appending(path: "Qwen3-8B-4bit", directoryHint: .isDirectory)
    try populateDir(modelDir, files: ["config.json", "tokenizer.json", "model.safetensors"])

    let manager = ModelLibraryManager()
    let results = try await manager.scan(root)

    #expect(results.count == 1)
    #expect(results[0].quantization == "4bit")
}

@Test func scanReturnsEmptyForEmptyDirectory() async throws {
    let root = makeTestRoot()
    defer { try? FileManager.default.removeItem(at: root) }

    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

    let manager = ModelLibraryManager()
    let results = try await manager.scan(root)

    #expect(results.isEmpty)
}

@Test func scanCachesResultInLastScan() async throws {
    let root = makeTestRoot()
    defer { try? FileManager.default.removeItem(at: root) }

    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let modelDir = root.appending(path: "MyModel2", directoryHint: .isDirectory)
    try populateDir(modelDir, files: ["config.json", "tokenizer.json", "model.safetensors"])

    let manager = ModelLibraryManager()
    let results = try await manager.scan(root)
    let cached = await manager.lastScan

    #expect(results.count == cached.count)
    #expect(results.first?.id == cached.first?.id)
}

@Test func scanFindsModelInNestedSubdirectory() async throws {
    // HuggingFace-style layout: `<root>/<org>/<repo>/<weights>`. The
    // <org> dir doesn't itself contain model files, so the old
    // single-level scan missed it entirely.
    let root = makeTestRoot()
    defer { try? FileManager.default.removeItem(at: root) }

    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let orgDir = root.appending(path: "nightmedia", directoryHint: .isDirectory)
    let modelDir = orgDir.appending(path: "my-fine-tune-q8-mlx", directoryHint: .isDirectory)
    try populateDir(modelDir, files: ["config.json", "tokenizer.json", "model.safetensors"])

    let manager = ModelLibraryManager()
    let results = try await manager.scan(root)

    #expect(results.count == 1)
    #expect(results[0].id == "nightmedia/my-fine-tune-q8-mlx")
    #expect(results[0].displayName == "my-fine-tune-q8-mlx")
    #expect(results[0].format == .mlx)
    // Quantization parser still works against the leaf name only.
    #expect(results[0].quantization == nil)
}

@Test func scanFindsBothFlatAndNestedInSameRoot() async throws {
    // Mixed layout: one model at top level, one nested. Both should
    // be discovered; their ids should be distinguishable (leaf vs
    // org/leaf) so downstream lookups don't collide.
    let root = makeTestRoot()
    defer { try? FileManager.default.removeItem(at: root) }

    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let flat = root.appending(path: "FlatModel", directoryHint: .isDirectory)
    try populateDir(flat, files: ["config.json", "tokenizer.json", "model.safetensors"])

    let nested = root.appending(path: "org-x/NestedModel", directoryHint: .isDirectory)
    try populateDir(nested, files: ["config.json", "tokenizer.json", "model.safetensors"])

    let manager = ModelLibraryManager()
    let results = try await manager.scan(root)

    #expect(results.count == 2)
    let ids = Set(results.map(\.id))
    #expect(ids == ["FlatModel", "org-x/NestedModel"])
}

@Test func scanDoesNotRecurseIntoModelDirectories() async throws {
    // A real model dir often has subdirectories (e.g. checkpoint
    // shards, tokenizer assets). Once a dir is recognised as a model,
    // the scan must stop descending so nested fragments aren't
    // mis-registered as separate models.
    let root = makeTestRoot()
    defer { try? FileManager.default.removeItem(at: root) }

    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let modelDir = root.appending(path: "OuterModel", directoryHint: .isDirectory)
    try populateDir(modelDir, files: ["config.json", "tokenizer.json", "model.safetensors"])

    // A sub-subdir that *would* qualify as a model on its own.
    let bogusInner = modelDir.appending(path: "tokenizer-files", directoryHint: .isDirectory)
    try populateDir(bogusInner, files: ["config.json", "tokenizer.json", "model.safetensors"])

    let manager = ModelLibraryManager()
    let results = try await manager.scan(root)

    #expect(results.count == 1)
    #expect(results[0].id == "OuterModel")
}

@Test func scanRespectsMaxDepthOne() async throws {
    // maxDepth=1 = pre-v0.5.1 behaviour (top-level only). A model
    // hidden under an <org>/ subdir should NOT be discovered.
    let root = makeTestRoot()
    defer { try? FileManager.default.removeItem(at: root) }

    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let nested = root.appending(path: "org/Hidden", directoryHint: .isDirectory)
    try populateDir(nested, files: ["config.json", "tokenizer.json", "model.safetensors"])

    let manager = ModelLibraryManager()
    let results = try await manager.scan(root, maxDepth: 1)

    #expect(results.isEmpty)
}

@Test func scanHonoursDeeperMaxDepth() async throws {
    // Caller-controlled depth bound. With maxDepth=3, a model at
    // `<root>/a/b/<model>/` is discoverable.
    let root = makeTestRoot()
    defer { try? FileManager.default.removeItem(at: root) }

    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let deep = root.appending(path: "a/b/DeepModel", directoryHint: .isDirectory)
    try populateDir(deep, files: ["config.json", "tokenizer.json", "model.safetensors"])

    let manager = ModelLibraryManager()
    let resultsShallow = try await manager.scan(root, maxDepth: 2)
    #expect(resultsShallow.isEmpty)

    let resultsDeep = try await manager.scan(root, maxDepth: 3)
    #expect(resultsDeep.count == 1)
    #expect(resultsDeep[0].id == "a/b/DeepModel")
}

@Test func scanSumsSafetensorsSize() async throws {
    let root = makeTestRoot()
    defer { try? FileManager.default.removeItem(at: root) }

    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let modelDir = root.appending(path: "SizedModel", directoryHint: .isDirectory)
    try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

    // Write config/tokenizer (zero-size, not counted)
    try Data().write(to: modelDir.appending(path: "config.json"))
    try Data().write(to: modelDir.appending(path: "tokenizer.json"))

    // Write a .safetensors file with 100 bytes
    let payload = Data(repeating: 0xAB, count: 100)
    try payload.write(to: modelDir.appending(path: "model.safetensors"))

    let manager = ModelLibraryManager()
    let results = try await manager.scan(root)

    #expect(results.count == 1)
    #expect(results[0].sizeBytes == 100)
}
