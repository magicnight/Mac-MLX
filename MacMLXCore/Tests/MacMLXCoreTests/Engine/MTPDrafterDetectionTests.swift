import Testing
import Foundation
import MLXLMCommon
@testable import MacMLXCore

// MARK: - MTP drafter detection unit tests
//
// Groundwork for routing the existing `draft_model` surface to
// `MTPSpeculativeTokenIterator` when the named drafter is an MTP (Multi-Token
// Prediction) block drafter. The engine does not consume this yet — see the
// status note on `MTPDrafterDetection` for why MTP cannot fire for any
// checkpoint macMLX can currently load.
//
// Everything below is pure filesystem/logic — a temp dir with a hand-crafted
// `config.json`. No Metal, no model download, no weights, mirroring
// `ModelLibraryManagerRerankerTests`' detection style.

/// `.serialized` because these await `ModelOverlay.registerAll()`, which
/// mutates the process-global `MTPDrafterTypeRegistry.shared`. The assertions
/// are post-conditions that hold regardless of cross-suite ordering.
@Suite("MTP drafter type registration", .serialized)
struct MTPDrafterRegistrationTests {

    @Test("registerAll registers the gemma4 assistant drafter type, and is idempotent")
    func registerAllRegistersGemma4Assistant() async {
        await ModelOverlay.registerAll()
        await ModelOverlay.registerAll()  // double-call must not throw or corrupt state
        #expect(await MTPDrafterTypeRegistry.shared.contains("gemma4_assistant") == true)
    }

    @Test("a model_type that is not a drafter stays unregistered in the MTP registry")
    func unrelatedTypeIsNotAnMTPDrafter() async {
        await ModelOverlay.registerAll()
        // `qwen3` is a perfectly good LLM type — it must not leak into the
        // drafter registry, or every classic draft model would misclassify.
        #expect(await MTPDrafterTypeRegistry.shared.contains("qwen3") == false)
    }
}

@Suite("MTP drafter detection from config.json", .serialized)
struct MTPDrafterDetectionTests {

    @Test("a config.json whose model_type is a registered MTP drafter is detected")
    func registeredDrafterTypeDetected() async throws {
        await ModelOverlay.registerAll()
        let temp = try MTPDetectionTempDir()
        let directory = try temp.writeModel(
            name: "gemma-4-assistant", modelType: "gemma4_assistant")
        #expect(await MTPDrafterDetection.isMTPDrafter(directory: directory))
    }

    @Test("a plain LLM draft model's model_type is NOT detected as an MTP drafter")
    func classicDraftModelNotDetected() async throws {
        await ModelOverlay.registerAll()
        let temp = try MTPDetectionTempDir()
        let directory = try temp.writeModel(name: "qwen3-small", modelType: "qwen3")
        #expect(await MTPDrafterDetection.isMTPDrafter(directory: directory) == false)
    }

    @Test("an unregistered drafter-shaped model_type is not detected")
    func unregisteredDrafterTypeNotDetected() async throws {
        await ModelOverlay.registerAll()
        let temp = try MTPDetectionTempDir()
        // Shaped like a drafter but with no creator registered: detection must
        // say "no", so a request naming it stays on the classic loader (which
        // reports the unsupported type loudly) rather than reaching a factory
        // that has nothing to instantiate.
        let directory = try temp.writeModel(
            name: "future-assistant", modelType: "some_future_assistant")
        #expect(await MTPDrafterDetection.isMTPDrafter(directory: directory) == false)
    }

    @Test("a missing config.json degrades gracefully to false")
    func missingConfigIsNotAnMTPDrafter() async throws {
        await ModelOverlay.registerAll()
        let temp = try MTPDetectionTempDir()
        let directory = temp.url.appendingPathComponent("no-config")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        #expect(await MTPDrafterDetection.isMTPDrafter(directory: directory) == false)
    }

    @Test("a directory that does not exist at all degrades gracefully to false")
    func absentDirectoryIsNotAnMTPDrafter() async throws {
        await ModelOverlay.registerAll()
        let temp = try MTPDetectionTempDir()
        #expect(
            await MTPDrafterDetection.isMTPDrafter(
                directory: temp.url.appendingPathComponent("nothing-here")) == false)
    }

    @Test("a config.json with no model_type key degrades gracefully to false")
    func configWithoutModelTypeIsNotAnMTPDrafter() async throws {
        await ModelOverlay.registerAll()
        let temp = try MTPDetectionTempDir()
        let directory = temp.url.appendingPathComponent("typeless")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(#"{"hidden_size": 16}"#.utf8)
            .write(to: directory.appendingPathComponent("config.json"))
        #expect(MTPDrafterDetection.modelType(inModelDirectory: directory) == nil)
        #expect(await MTPDrafterDetection.isMTPDrafter(directory: directory) == false)
    }

    @Test("an id rejected by the draft-model path-traversal guard is not an MTP drafter")
    func unsafeModelIDIsNotAnMTPDrafter() async {
        await ModelOverlay.registerAll()
        // `draftModelDirectory(id:)` throws for these; detection must swallow
        // that as "not MTP" and let the classic loader report the id error.
        #expect(await MTPDrafterDetection.isMTPDrafter(modelID: "../escape") == false)
        #expect(await MTPDrafterDetection.isMTPDrafter(modelID: "a/b") == false)
        #expect(await MTPDrafterDetection.isMTPDrafter(modelID: "") == false)
    }

    @Test("modelType reads the top-level model_type, matching BaseConfiguration")
    func modelTypeReadsTopLevelKey() throws {
        let temp = try MTPDetectionTempDir()
        let directory = try temp.writeModel(name: "typed", modelType: "gemma4_assistant")
        #expect(MTPDrafterDetection.modelType(inModelDirectory: directory) == "gemma4_assistant")
    }
}

// MARK: - Helpers

/// Temp directory that can lay down a minimal model directory carrying just
/// the `config.json` `model_type` detection reads. A `final class` so `deinit`
/// can actually remove the tree — a `struct` would leave one behind per test.
private final class MTPDetectionTempDir {
    let url: URL

    init() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("macmlx-mtp-detection-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        self.url = base
    }

    deinit {
        try? FileManager.default.removeItem(at: url)
    }

    @discardableResult
    func writeModel(name: String, modelType: String) throws -> URL {
        let directory = url.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let config: [String: Any] = ["model_type": modelType]
        let data = try JSONSerialization.data(withJSONObject: config)
        try data.write(to: directory.appendingPathComponent("config.json"))
        return directory
    }
}
