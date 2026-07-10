import Testing
import Foundation
@testable import MacMLXCore

/// Track F: quantization-from-config, parameter-count-from-name, and
/// architecture-from-config inference layered onto `ModelLibraryManager.scan(_:)`.
///
/// Serialised for the same reason as the VLM/Embedder suites — filesystem-
/// backed scans interact badly with swift-testing's default parallelism.
@Suite("ModelLibraryManager quantization/parameter/architecture inference", .serialized)
struct ModelLibraryManagerInferenceTests {

    @Test
    func quantizationPrefersConfigBitsOverNameSuffix() async throws {
        let temp = try TemporaryDirectory()
        // Directory name suggests 4bit, but config.json — the more
        // accurate source — says 8. Config must win.
        try writeModel(
            in: temp.url,
            name: "Conflicting-Model-4bit",
            config: #"{"quantization": {"bits": 8, "group_size": 32}}"#
        )

        let mgr = ModelLibraryManager()
        let results = try await mgr.scan(temp.url)

        #expect(results.count == 1)
        #expect(results[0].quantization == "8bit")
    }

    @Test
    func quantizationFallsBackToNameWhenConfigHasNoQuantizationBlock() async throws {
        let temp = try TemporaryDirectory()
        try writeModel(
            in: temp.url,
            name: "Qwen3-8B-4bit",
            config: #"{"model_type": "qwen3"}"#
        )

        let mgr = ModelLibraryManager()
        let results = try await mgr.scan(temp.url)

        #expect(results[0].quantization == "4bit")
    }

    @Test
    func quantizationInfersUnquantizedDtypeFromNameSuffix() async throws {
        let temp = try TemporaryDirectory()
        try writeModel(
            in: temp.url,
            name: "Qwen3-8B-bf16",
            config: #"{"model_type": "qwen3"}"#
        )

        let mgr = ModelLibraryManager()
        let results = try await mgr.scan(temp.url)

        #expect(results[0].quantization == "bf16")
    }

    @Test
    func architectureIsPopulatedFromConfigModelType() async throws {
        let temp = try TemporaryDirectory()
        try writeModel(
            in: temp.url,
            name: "SomeModel",
            config: #"{"model_type": "llama"}"#
        )

        let mgr = ModelLibraryManager()
        let results = try await mgr.scan(temp.url)

        #expect(results[0].architecture == "llama")
    }

    @Test
    func parameterCountInferredForPlainIntegerSize() async throws {
        try await expectParameterCount("Qwen3-8B-4bit", equals: "8B")
    }

    @Test
    func parameterCountInferredForDecimalSize() async throws {
        try await expectParameterCount("Qwen2.5-0.5B-Instruct-bf16", equals: "0.5B")
    }

    @Test
    func parameterCountInferredForLowercaseBSuffix() async throws {
        try await expectParameterCount("gemma-2-9b-it-4bit", equals: "9B")
    }

    @Test
    func parameterCountInferredForThreeDigitSize() async throws {
        try await expectParameterCount("DeepSeek-V3-671B-4bit", equals: "671B")
    }

    @Test
    func parameterCountIsNilForUnconventionalName() async throws {
        try await expectParameterCount("some-unconventional-name", equals: nil)
    }

    private func expectParameterCount(_ name: String, equals expected: String?) async throws {
        let temp = try TemporaryDirectory()
        try writeModel(in: temp.url, name: name, config: "{}")

        let mgr = ModelLibraryManager()
        let results = try await mgr.scan(temp.url)

        #expect(results[0].parameterCount == expected)
    }

    // MARK: - Helpers

    private func writeModel(in root: URL, name: String, config: String) throws {
        let dir = root.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: dir.appendingPathComponent("tokenizer.json"))
        try Data("\u{00}".utf8).write(to: dir.appendingPathComponent("model.safetensors"))
        try Data(config.utf8).write(to: dir.appendingPathComponent("config.json"))
    }
}

/// Auto-cleaning temp directory for filesystem-backed tests (mirrors the
/// private helper duplicated in the VLM/Embedder suites).
private struct TemporaryDirectory {
    let url: URL

    init() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("macmlx-inference-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        self.url = base
    }
}
