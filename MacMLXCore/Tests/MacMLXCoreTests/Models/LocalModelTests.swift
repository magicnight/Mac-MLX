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
