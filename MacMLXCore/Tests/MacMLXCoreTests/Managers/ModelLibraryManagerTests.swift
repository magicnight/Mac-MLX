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
