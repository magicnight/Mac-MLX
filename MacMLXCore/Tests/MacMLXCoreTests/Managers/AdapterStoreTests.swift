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
    func scanSortsAdaptersByName() async throws {
        let temp = try TempDir()
        try writeAdapter(in: temp.url, name: "zeta", targetModel: nil, r: 4)
        try writeAdapter(in: temp.url, name: "alpha", targetModel: nil, r: 4)
        try writeAdapter(in: temp.url, name: "mu", targetModel: nil, r: 4)
        let store = AdapterStore()
        let found = try await store.scan(temp.url)
        #expect(found.map(\.name) == ["alpha", "mu", "zeta"])
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
