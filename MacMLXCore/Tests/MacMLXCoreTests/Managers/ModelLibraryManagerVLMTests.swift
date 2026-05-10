import Testing
import Foundation
@testable import MacMLXCore

/// Filesystem-backed tests run serialised — swift-testing's default
/// parallelism + tmpdir thrash + actor scans interact badly enough to
/// trip a Swift-stdlib `Index out of range` on the test harness side
/// when running interleaved with the existing ModelLibraryManager
/// suite. Serialising costs ~30ms total and removes the flake.
@Suite("ModelLibraryManager VLM detection", .serialized)
struct ModelLibraryManagerVLMTests {

    @Test
    func detectsQwen2_5VLAsVLM() async throws {
        let temp = try TemporaryDirectory()
        try writeModel(in: temp.url, name: "qwen-vl-test", modelType: "qwen2_5_vl")
        let mgr = ModelLibraryManager()
        let models = try await mgr.scan(temp.url)
        #expect(models.count == 1)
        #expect(models[0].format == .mlxVLM)
    }

    @Test
    func detectsGemma3AsVLM() async throws {
        let temp = try TemporaryDirectory()
        try writeModel(in: temp.url, name: "gemma-3-test", modelType: "gemma3")
        let mgr = ModelLibraryManager()
        let models = try await mgr.scan(temp.url)
        #expect(models[0].format == .mlxVLM)
    }

    @Test
    func detectsSmolVLMAsVLM() async throws {
        let temp = try TemporaryDirectory()
        try writeModel(in: temp.url, name: "smolvlm-test", modelType: "smolvlm")
        let mgr = ModelLibraryManager()
        let models = try await mgr.scan(temp.url)
        #expect(models[0].format == .mlxVLM)
    }

    @Test
    func qwen3LLMStaysMLX() async throws {
        // qwen3 (text) is .mlx, qwen3_vl is .mlxVLM
        let temp = try TemporaryDirectory()
        try writeModel(in: temp.url, name: "qwen3-text", modelType: "qwen3")
        let mgr = ModelLibraryManager()
        let models = try await mgr.scan(temp.url)
        #expect(models[0].format == .mlx)
    }

    @Test
    func configJSONWithoutModelTypeFallsBackToMLX() async throws {
        // Real-world LLM configs without `model_type` (rare but
        // observed in some user-converted checkpoints) must not be
        // mis-tagged as VLM.
        let temp = try TemporaryDirectory()
        let dir = temp.url.appendingPathComponent("no-model-type")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: dir.appendingPathComponent("tokenizer.json"))
        try Data("\u{00}".utf8).write(to: dir.appendingPathComponent("model.safetensors"))
        try Data(#"{"vocab_size":32000}"#.utf8)
            .write(to: dir.appendingPathComponent("config.json"))

        let mgr = ModelLibraryManager()
        let models = try await mgr.scan(temp.url)
        #expect(models.count == 1)
        #expect(models[0].format == .mlx)
    }

    @Test
    func malformedConfigJSONFallsBackToMLX() async throws {
        let temp = try TemporaryDirectory()
        let dir = temp.url.appendingPathComponent("malformed")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: dir.appendingPathComponent("tokenizer.json"))
        try Data("\u{00}".utf8).write(to: dir.appendingPathComponent("model.safetensors"))
        // Truncated/invalid JSON
        try Data("{not json".utf8).write(to: dir.appendingPathComponent("config.json"))

        let mgr = ModelLibraryManager()
        let models = try await mgr.scan(temp.url)
        #expect(models[0].format == .mlx, "malformed JSON must not crash the scan")
    }

    @Test
    func mixedDirectoryDetectsBothLLMAndVLM() async throws {
        let temp = try TemporaryDirectory()
        try writeModel(in: temp.url, name: "qwen3-text", modelType: "qwen3")
        try writeModel(in: temp.url, name: "qwen-vl", modelType: "qwen2_5_vl")
        let mgr = ModelLibraryManager()
        let models = try await mgr.scan(temp.url)
        // displayName-sorted: qwen-vl < qwen3-text
        #expect(models.count == 2)
        let byID = Dictionary(uniqueKeysWithValues: models.map { ($0.id, $0.format) })
        #expect(byID["qwen3-text"] == .mlx)
        #expect(byID["qwen-vl"] == .mlxVLM)
    }

    // MARK: - Helpers

    /// Lay down a directory whose file listing makes
    /// `ModelFormat.detect(in:)` return `.mlx` (tokenizer.json +
    /// .safetensors + config.json are the required signals), with a
    /// `config.json` containing the given `model_type` so the VLM
    /// upgrade has something to inspect.
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
            .appendingPathComponent("macmlx-vlm-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        self.url = base
    }
}
