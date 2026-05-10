import Testing
import Foundation
@testable import MacMLXCore

/// Filesystem-backed: serialised so swift-testing's parallel executor
/// doesn't thrash on the temp directory (same rationale as the v0.4.1
/// VLM detection suite).
@Suite("AdapterStore", .serialized)
struct AdapterStoreTests {

    @Test
    func scanFindsAdapterWithPEFTConfig() async throws {
        let temp = try TempDir()
        try writeAdapter(
            in: temp.url,
            name: "qwen3-medical",
            targetModel: "mlx-community/Qwen3-8B-4bit",
            r: 8
        )
        let store = AdapterStore()
        let found = try await store.scan(temp.url)
        #expect(found.count == 1)
        #expect(found[0].name == "qwen3-medical")
        #expect(found[0].rank == 8)
        #expect(found[0].targetModel == "mlx-community/Qwen3-8B-4bit")
        #expect(found[0].targetModules == ["q_proj", "v_proj"])
    }

    @Test
    func scanIgnoresDirsWithoutAdapterConfig() async throws {
        let temp = try TempDir()
        let stray = temp.url.appendingPathComponent("not-an-adapter")
        try FileManager.default.createDirectory(at: stray, withIntermediateDirectories: true)
        let store = AdapterStore()
        let found = try await store.scan(temp.url)
        #expect(found.isEmpty)
    }

    @Test
    func scanRequiresAdapterModelSafetensors() async throws {
        // Has config but no safetensors → not a usable adapter.
        let temp = try TempDir()
        let dir = temp.url.appendingPathComponent("config-only")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data(#"{"r":8,"target_modules":["q_proj"]}"#.utf8)
            .write(to: dir.appendingPathComponent("adapter_config.json"))
        let store = AdapterStore()
        let found = try await store.scan(temp.url)
        #expect(found.isEmpty)
    }

    @Test
    func scanIgnoresMalformedConfigJSON() async throws {
        let temp = try TempDir()
        let dir = temp.url.appendingPathComponent("malformed")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("{not json".utf8)
            .write(to: dir.appendingPathComponent("adapter_config.json"))
        try Data().write(to: dir.appendingPathComponent("adapter_model.safetensors"))
        let store = AdapterStore()
        let found = try await store.scan(temp.url)
        #expect(found.isEmpty)
    }

    @Test
    func scanReturnsEmptyForMissingDirectory() async throws {
        let temp = try TempDir()
        let neverExisted = temp.url.appendingPathComponent("does-not-exist")
        let store = AdapterStore()
        let found = try await store.scan(neverExisted)
        #expect(found.isEmpty)
    }

    @Test
    func scanDetectsMLXNativeFormat() async throws {
        let temp = try TempDir()
        try writeMLXAdapter(in: temp.url, name: "mlx-cached", rank: 8, scale: 2.0, keys: ["q_proj"])
        let store = AdapterStore()
        let found = try await store.scan(temp.url)
        #expect(found.count == 1)
        #expect(found[0].format == .mlx)
        #expect(found[0].rank == 8)
        #expect(found[0].targetModel == nil)  // mlx-native doesn't carry base id
        #expect(found[0].targetModules == ["q_proj"])
    }

    @Test
    func scanPrefersMLXOverPEFTWhenBothFilesPresent() async throws {
        // A directory that has both PEFT and mlx-native files (e.g. a
        // user kept the converter output side-by-side with the source)
        // should be reported as .mlx — that's the format the engine
        // can load directly without re-converting.
        let temp = try TempDir()
        let dir = temp.url.appendingPathComponent("dual-format")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        // Both weights files…
        try Data().write(to: dir.appendingPathComponent("adapter_model.safetensors"))
        try Data().write(to: dir.appendingPathComponent("adapters.safetensors"))
        // …but only the mlx config (the readMLXAdapter path is tried
        // first, so this is what the scan should latch onto).
        let mlxCfg = """
        {
          "num_layers": 1,
          "fine_tune_type": "lora",
          "lora_parameters": { "rank": 4, "scale": 2.0, "keys": null }
        }
        """
        try Data(mlxCfg.utf8).write(to: dir.appendingPathComponent("adapter_config.json"))

        let store = AdapterStore()
        let found = try await store.scan(temp.url)
        #expect(found.count == 1)
        #expect(found[0].format == .mlx)
    }

    @Test
    func scanSortsAdaptersByName() async throws {
        let temp = try TempDir()
        try writeAdapter(in: temp.url, name: "zeta", targetModel: nil, r: 4)
        try writeAdapter(in: temp.url, name: "alpha", targetModel: nil, r: 4)
        try writeAdapter(in: temp.url, name: "mu", targetModel: nil, r: 4)
        let store = AdapterStore()
        let found = try await store.scan(temp.url)
        #expect(found.map(\.name) == ["alpha", "mu", "zeta"])
    }

    /// Lay down a directory in mlx-native format (mlx schema config +
    /// `adapters.safetensors`). Same shape as the converter writes.
    private func writeMLXAdapter(in root: URL, name: String, rank: Int, scale: Float, keys: [String]) throws {
        let dir = root.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let keysJSON = "[\(keys.map { "\"\($0)\"" }.joined(separator: ","))]"
        let cfg = """
        {
          "num_layers": 1,
          "fine_tune_type": "lora",
          "lora_parameters": { "rank": \(rank), "scale": \(scale), "keys": \(keysJSON) }
        }
        """
        try Data(cfg.utf8).write(to: dir.appendingPathComponent("adapter_config.json"))
        try Data().write(to: dir.appendingPathComponent("adapters.safetensors"))
    }

    private func writeAdapter(in root: URL, name: String, targetModel: String?, r: Int) throws {
        let dir = root.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let cfg: String
        if let targetModel {
            cfg = """
            {
              "base_model_name_or_path": "\(targetModel)",
              "r": \(r),
              "lora_alpha": \(r * 2),
              "target_modules": ["q_proj", "v_proj"],
              "peft_type": "LORA"
            }
            """
        } else {
            cfg = """
            {
              "r": \(r),
              "target_modules": ["q_proj", "v_proj"],
              "peft_type": "LORA"
            }
            """
        }
        try Data(cfg.utf8).write(to: dir.appendingPathComponent("adapter_config.json"))
        try Data().write(to: dir.appendingPathComponent("adapter_model.safetensors"))
    }
}

private struct TempDir {
    let url: URL
    init() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("macmlx-adapter-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        self.url = base
    }
}
