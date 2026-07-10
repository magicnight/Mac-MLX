import Testing
import Foundation
@testable import MacMLXCore

// MARK: - Helpers

private func makeTestDir() -> URL {
    FileManager.default.temporaryDirectory
        .appending(path: "config-info-test-\(UUID().uuidString)", directoryHint: .isDirectory)
}

private func writeConfig(_ json: String, in dir: URL) throws {
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    try Data(json.utf8).write(to: dir.appending(path: "config.json", directoryHint: .notDirectory))
}

// MARK: - Tests

@Test
func configInfoReadsQuantizationAndContextLength() throws {
    let dir = makeTestDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    try writeConfig("""
    {
        "model_type": "qwen3",
        "max_position_embeddings": 32768,
        "quantization": { "bits": 4, "group_size": 64 }
    }
    """, in: dir)

    let info = ModelConfigInfo.read(from: dir)
    #expect(info?.modelType == "qwen3")
    #expect(info?.quantizationBits == 4)
    #expect(info?.quantizationGroupSize == 64)
    #expect(info?.contextLength == 32768)
}

@Test
func configInfoModelTypeIsLowercased() throws {
    let dir = makeTestDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    try writeConfig(#"{"model_type": "Gemma3"}"#, in: dir)

    let info = ModelConfigInfo.read(from: dir)
    #expect(info?.modelType == "gemma3")
}

@Test
func configInfoFallsBackThroughContextLengthFieldNames() throws {
    let dir = makeTestDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    try writeConfig(#"{"n_positions": 2048}"#, in: dir)

    let info = ModelConfigInfo.read(from: dir)
    #expect(info?.contextLength == 2048)
}

@Test
func configInfoReturnsNilQuantizationWhenNoBlockPresent() throws {
    let dir = makeTestDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    try writeConfig(#"{"model_type": "llama"}"#, in: dir)

    let info = ModelConfigInfo.read(from: dir)
    #expect(info?.quantizationBits == nil)
    #expect(info?.quantizationGroupSize == nil)
}

@Test
func configInfoReturnsNilForMissingFile() {
    let dir = makeTestDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    // Directory exists but has no config.json at all.
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

    #expect(ModelConfigInfo.read(from: dir) == nil)
}

@Test
func configInfoReturnsNilForMalformedJSON() throws {
    let dir = makeTestDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    try writeConfig("{not valid json", in: dir)

    #expect(ModelConfigInfo.read(from: dir) == nil)
}
