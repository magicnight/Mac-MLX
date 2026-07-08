import Testing
import Foundation
@testable import MacMLXCore

/// Filesystem-backed tests run serialised for the same reason as the VLM
/// suite — parallel tmpdir thrash + actor scans trip a Swift-stdlib flake.
@Suite("ModelLibraryManager embedder detection", .serialized)
struct ModelLibraryManagerEmbedderTests {

    @Test
    func detectsBertAsEmbedder() async throws {
        let temp = try TemporaryDirectory()
        try writeModel(in: temp.url, name: "bert-embed", modelType: "bert")
        let mgr = ModelLibraryManager()
        let models = try await mgr.scan(temp.url)
        #expect(models.count == 1)
        #expect(models[0].format == .embedder)
    }

    @Test
    func detectsXLMRobertaAsEmbedder() async throws {
        let temp = try TemporaryDirectory()
        try writeModel(in: temp.url, name: "xlmr-embed", modelType: "xlm-roberta")
        let mgr = ModelLibraryManager()
        let models = try await mgr.scan(temp.url)
        #expect(models[0].format == .embedder)
    }

    @Test
    func detectsDistilbertAsEmbedder() async throws {
        let temp = try TemporaryDirectory()
        try writeModel(in: temp.url, name: "distil-embed", modelType: "distilbert")
        let mgr = ModelLibraryManager()
        let models = try await mgr.scan(temp.url)
        #expect(models[0].format == .embedder)
    }

    @Test
    func gemma3StaysVLMNotEmbedder() async throws {
        // gemma3 is in BOTH the VLM and embedder registries — VLM wins.
        let temp = try TemporaryDirectory()
        try writeModel(in: temp.url, name: "gemma3-shared", modelType: "gemma3")
        let mgr = ModelLibraryManager()
        let models = try await mgr.scan(temp.url)
        #expect(models[0].format == .mlxVLM)
    }

    @Test
    func qwen3StaysMLXNotEmbedder() async throws {
        // Decoder `model_type`s (qwen3, lfm2, gemma3*) are deliberately
        // excluded from embedder detection — they share their type with
        // generative chat models, so they must stay `.mlx` to keep
        // generation working.
        let temp = try TemporaryDirectory()
        try writeModel(in: temp.url, name: "qwen3-text", modelType: "qwen3")
        let mgr = ModelLibraryManager()
        let models = try await mgr.scan(temp.url)
        #expect(models[0].format == .mlx)
    }

    @Test
    func mixedDirectoryDetectsMLXVLMAndEmbedder() async throws {
        let temp = try TemporaryDirectory()
        try writeModel(in: temp.url, name: "aaa-text", modelType: "qwen3")
        try writeModel(in: temp.url, name: "bbb-vision", modelType: "qwen2_5_vl")
        try writeModel(in: temp.url, name: "ccc-embed", modelType: "bert")
        let mgr = ModelLibraryManager()
        let models = try await mgr.scan(temp.url)
        #expect(models.count == 3)
        let byID = Dictionary(uniqueKeysWithValues: models.map { ($0.id, $0.format) })
        #expect(byID["aaa-text"] == .mlx)
        #expect(byID["bbb-vision"] == .mlxVLM)
        #expect(byID["ccc-embed"] == .embedder)
    }

    // MARK: - Helpers

    /// Lay down a directory whose file listing makes
    /// `ModelFormat.detect(in:)` return `.mlx` (tokenizer.json +
    /// `.safetensors` + config.json), with a `config.json` carrying the
    /// given `model_type` so the format upgrade has something to inspect.
    private func writeModel(in root: URL, name: String, modelType: String) throws {
        let dir = root.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        try Data("{}".utf8).write(to: dir.appendingPathComponent("tokenizer.json"))
        try Data("\u{00}".utf8).write(to: dir.appendingPathComponent("model.safetensors"))

        let json = "{\"model_type\":\"\(modelType)\"}"
        try Data(json.utf8).write(to: dir.appendingPathComponent("config.json"))
    }
}

/// Auto-cleaning temp directory for filesystem-backed tests.
private struct TemporaryDirectory {
    let url: URL

    init() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("macmlx-embedder-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        self.url = base
    }
}
