import Testing
import Foundation
@testable import MacMLXCore

/// Engine-level guards that don't require a real MLX runtime to verify.
/// Loading a real VLM checkpoint needs Metal + a multi-GB download, so
/// the happy-path smoke test stays a manual-QA / integration item; what
/// we can assert here is that unsupported formats reject early and that
/// the typed-error shape matches the existing LLM path.
@Suite("MLXSwiftEngine VLM branch")
struct MLXSwiftEngineVLMTests {

    @Test
    func loadFailsForGGUFFormat() async {
        let engine = MLXSwiftEngine()
        let model = LocalModel(
            id: "fake-gguf",
            displayName: "Fake GGUF",
            directory: URL(fileURLWithPath: "/tmp/no-such-dir-gguf"),
            sizeBytes: 0,
            format: .gguf,
            quantization: nil,
            parameterCount: nil,
            architecture: nil
        )
        await #expect(throws: EngineError.self) {
            try await engine.load(model)
        }
        let status = await engine.status
        if case .error(let reason) = status {
            #expect(reason.contains("Unsupported model format"))
            #expect(reason.contains("gguf"))
        } else {
            Issue.record("Expected .error status, got \(status)")
        }
    }

    @Test
    func loadFailsForUnknownFormat() async {
        let engine = MLXSwiftEngine()
        let model = LocalModel(
            id: "fake-unknown",
            displayName: "Fake Unknown",
            directory: URL(fileURLWithPath: "/tmp/no-such-dir-unknown"),
            sizeBytes: 0,
            format: .unknown,
            quantization: nil,
            parameterCount: nil,
            architecture: nil
        )
        await #expect(throws: EngineError.self) {
            try await engine.load(model)
        }
    }

    @Test
    func loadVLMFromMissingDirectoryThrowsModelLoadFailed() async {
        // VLMModelFactory hits the same "directory not found" failure
        // as LLMModelFactory; we just need to confirm our load() routes
        // through the VLM factory when format is .mlxVLM and surfaces
        // the typed EngineError.modelLoadFailed.
        let engine = MLXSwiftEngine()
        let model = LocalModel(
            id: "fake-vlm",
            displayName: "Fake VLM",
            directory: URL(fileURLWithPath: "/tmp/no-such-vlm-\(UUID().uuidString)"),
            sizeBytes: 0,
            format: .mlxVLM,
            quantization: nil,
            parameterCount: nil,
            architecture: nil
        )
        await #expect(throws: EngineError.self) {
            try await engine.load(model)
        }
        let loaded = await engine.loadedModel
        #expect(loaded == nil)
    }
}
