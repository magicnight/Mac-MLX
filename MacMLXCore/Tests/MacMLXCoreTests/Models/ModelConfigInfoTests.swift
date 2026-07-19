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

@Test
func configInfoParsesArchitecturesCasePreserved() throws {
    let dir = makeTestDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    try writeConfig("""
    {
        "model_type": "bert",
        "architectures": ["BertForSequenceClassification"]
    }
    """, in: dir)

    let info = ModelConfigInfo.read(from: dir)
    // Case-preserved (unlike model_type) — the reranker suffix check is
    // case-sensitive.
    #expect(info?.architectures == ["BertForSequenceClassification"])
}

@Test
func configInfoArchitecturesIsNilWhenAbsent() throws {
    let dir = makeTestDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    try writeConfig(#"{"model_type": "bert"}"#, in: dir)

    #expect(ModelConfigInfo.read(from: dir)?.architectures == nil)
}

@Test
func configInfoParsesNumLabels() throws {
    let dir = makeTestDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    try writeConfig(#"{"model_type": "bert", "num_labels": 5}"#, in: dir)

    #expect(ModelConfigInfo.read(from: dir)?.numLabels == 5)
}

@Test
func configInfoNumLabelsIsNilWhenAbsent() throws {
    let dir = makeTestDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    try writeConfig(#"{"model_type": "bert"}"#, in: dir)

    #expect(ModelConfigInfo.read(from: dir)?.numLabels == nil)
}

@Test
func configInfoCountsId2LabelEntries() throws {
    let dir = makeTestDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    try writeConfig("""
    {
        "model_type": "bert",
        "id2label": {"0": "negative", "1": "neutral", "2": "positive"}
    }
    """, in: dir)

    #expect(ModelConfigInfo.read(from: dir)?.id2labelCount == 3)
}

@Test
func configInfoId2LabelCountIsNilWhenAbsent() throws {
    let dir = makeTestDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    try writeConfig(#"{"model_type": "bert"}"#, in: dir)

    #expect(ModelConfigInfo.read(from: dir)?.id2labelCount == nil)
}
