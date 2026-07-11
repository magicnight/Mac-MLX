import Testing
import Foundation
@testable import MacMLXCore

@Test
func localModelHumanSizeFormatsBytes() {
    let m = LocalModel(
        id: "Qwen3-8B-4bit",
        displayName: "Qwen3 8B (4-bit)",
        directory: URL(filePath: "/tmp/Qwen3-8B-4bit"),
        sizeBytes: 4_500_000_000,
        format: .mlx,
        quantization: "4bit",
        parameterCount: "8B",
        architecture: "qwen3"
    )
    #expect(m.humanSize == "4.50 GB")
}

@Test
func localModelRoundTripsThroughJSON() throws {
    let original = LocalModel(
        id: "test-1",
        displayName: "Test",
        directory: URL(filePath: "/tmp/test"),
        sizeBytes: 1024,
        format: .mlx,
        quantization: nil,
        parameterCount: nil,
        architecture: nil
    )
    let data = try JSONEncoder().encode(original)
    let decoded = try JSONDecoder().decode(LocalModel.self, from: data)
    #expect(decoded == original)
}

@Test
func modelFormatRecognizesMLXAndGGUFExtensions() {
    #expect(ModelFormat.detect(in: ["model.safetensors", "config.json", "tokenizer.json"]) == .mlx)
    #expect(ModelFormat.detect(in: ["llama-2-7b.Q4_K_M.gguf"]) == .gguf)
    #expect(ModelFormat.detect(in: ["random.bin"]) == .unknown)
}

// MARK: - draftCandidates(from:excluding:)

private func makeLocalModel(
    id: String, format: ModelFormat = .mlx, isExternalReference: Bool = false
) -> LocalModel {
    LocalModel(
        id: id,
        displayName: id,
        directory: URL(filePath: "/tmp/\(id)"),
        sizeBytes: 0,
        format: format,
        quantization: nil,
        parameterCount: nil,
        architecture: nil,
        isExternalReference: isExternalReference
    )
}

@Test
func draftCandidatesExcludesExternalReferences() {
    // HF-cache ids are always "org/name" — `MLXSwiftEngine.isValidDraftModelID`'s
    // allowlist unconditionally rejects any `/`, so an external reference
    // must never surface as a pickable draft candidate.
    let models = [
        makeLocalModel(id: "Qwen3-8B-4bit"),
        makeLocalModel(id: "mlx-community/Qwen3-0.6B-4bit", isExternalReference: true),
    ]
    let candidates = LocalModel.draftCandidates(from: models, excluding: nil)
    #expect(candidates.map(\.id) == ["Qwen3-8B-4bit"])
}

@Test
func draftCandidatesExcludesCurrentTargetModel() {
    let models = [
        makeLocalModel(id: "Qwen3-8B-4bit"),
        makeLocalModel(id: "Qwen3-0.6B-4bit"),
    ]
    let candidates = LocalModel.draftCandidates(from: models, excluding: "Qwen3-8B-4bit")
    #expect(candidates.map(\.id) == ["Qwen3-0.6B-4bit"])
}

@Test
func draftCandidatesExcludesNonMLXFormats() {
    let models = [
        makeLocalModel(id: "Qwen3-8B-4bit"),
        makeLocalModel(id: "Qwen2-VL-2B", format: .mlxVLM),
        makeLocalModel(id: "bge-small", format: .embedder),
    ]
    let candidates = LocalModel.draftCandidates(from: models, excluding: nil)
    #expect(candidates.map(\.id) == ["Qwen3-8B-4bit"])
}
