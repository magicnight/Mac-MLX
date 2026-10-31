import Testing
import Foundation
import MLXLLM
import MLXVLM
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

    /// The keys of mlx-swift-lm's `VLMTypeRegistry.shared` at the pinned
    /// revision (3.32.3), as `Libraries/MLXVLM/VLMModelFactory.swift` lists
    /// them. `vlmRegistryMirrorMatchesTheResolvedSource` fails when the
    /// resolved mlx-swift-lm lists other keys, so a bump that adds one is
    /// caught there; the tests after it then say which side it belongs on.
    private static let upstreamVLMRegistryKeys: Set<String> = [
        "paligemma", "qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe",
        "qwen3_5", "qwen3_5_moe", "idefics3", "gemma3", "gemma4", "gemma4_unified",
        "smolvlm", "fastvlm", "llava_qwen2", "pixtral", "mistral3", "lfm2_vl",
        "lfm2-vl", "glm_ocr", "muse_glimmer",
    ]

    /// Every registry key is classified exactly once: loaded as a VLM, or on
    /// purpose as text. A stale list (it once held `fast_vlm` and
    /// `qwen3_5_vl`, which upstream never registered) fails here.
    @Test
    func everyVLMRegistryKeyIsClassifiedOnce() async {
        let vlm = ModelLibraryManager.knownVLMTypes
        let text = ModelLibraryManager.vlmTypesServedAsText
        let fullPath = ModelLibraryManager.fullPathVLMTypes
        #expect(vlm.isDisjoint(with: text))
        #expect(vlm.isDisjoint(with: fullPath))
        #expect(text.isDisjoint(with: fullPath))
        #expect(vlm.union(text).union(fullPath) == Self.upstreamVLMRegistryKeys)
        #expect(fullPath == ["gemma4"])
        for key in Self.upstreamVLMRegistryKeys {
            #expect(await VLMTypeRegistry.shared.contains(key), "\(key) is not in the VLM registry")
        }
    }

    /// The mirror above against the registry the package actually resolved:
    /// its keys read from the checked-out `VLMModelFactory.swift`, since
    /// `ModelTypeRegistry` can only answer `contains`, not list. Every line
    /// of the registry's dictionary that opens with a quoted key counts,
    /// whatever creates the model; a type registered elsewhere through
    /// `registerModelType` would not be seen (there is none at 3.32.3).
    ///
    /// Runs under `swift test`, where the checkout sits in the scratch
    /// directory next to the test bundle; under xcodebuild there is none and
    /// it is skipped. The spm job, in CI and ci-local, sets
    /// `MACMLX_REQUIRE_RESOLVED_SOURCE=1`, which turns a missing checkout
    /// into a failure there, so the gate cannot pass by skipping.
    @Test(.enabled(if: Self.resolvedVLMFactorySource() != nil || Self.requiresResolvedSource,
                   "no SwiftPM checkout of mlx-swift-lm next to the test bundle"))
    func vlmRegistryMirrorMatchesTheResolvedSource() throws {
        let source = try #require(Self.resolvedVLMFactorySource(),
                                  "MACMLX_REQUIRE_RESOLVED_SOURCE is set but no checkout was found")
        let keys = try #require(Self.vlmRegistryKeys(in: source), "no VLMTypeRegistry in VLMModelFactory.swift")
        #expect(!keys.isEmpty, "found no registry entries; has VLMModelFactory.swift changed shape?")
        #expect(keys == Self.upstreamVLMRegistryKeys,
                "mirror out of date: added \(keys.subtracting(Self.upstreamVLMRegistryKeys).sorted()), removed \(Self.upstreamVLMRegistryKeys.subtracting(keys).sorted())")
    }

    /// The keys of the `VLMTypeRegistry` dictionary in `source`: every line
    /// between it and `VLMProcessorTypeRegistry` that opens with a quoted key
    /// and a colon, whatever value follows. Nil without the registry.
    static func vlmRegistryKeys(in source: String) -> Set<String>? {
        guard let start = source.range(of: "public enum VLMTypeRegistry") else { return nil }
        let end = source.range(of: "public enum VLMProcessorTypeRegistry", range: start.upperBound..<source.endIndex)
        let block = source[start.upperBound..<(end?.lowerBound ?? source.endIndex)]
        let entry = /^\s*"([^"]+)"\s*:/.anchorsMatchLineEndings()
        return Set(block.matches(of: entry).map { String($0.1) })
    }

    /// A key is read whatever creates its model, as `LLMModelFactory` already
    /// registers one with a plain function; the processor registry after it
    /// and comments are not read.
    @Test
    func registryKeysAreReadWhateverCreatesTheModel() {
        let source = """
            /// Registry of model type, e.g "llama": not an entry.
            public enum VLMTypeRegistry {
                public static let shared: ModelTypeRegistry<LanguageModel> = .init(creators: [
                    "paligemma": create(PaliGemmaConfiguration.self, PaliGemma.init),
                    "qwen3": createQwen3CompatibleModel,
                    "closure_vl" : { data in try make(data) },
                ])
            }
            public enum VLMProcessorTypeRegistry {
                    "PaliGemmaProcessor": create(PaliGemmaProcessorConfiguration.self, PaliGemmaProcessor.init),
            }
            """
        #expect(Self.vlmRegistryKeys(in: source) == ["paligemma", "qwen3", "closure_vl"])
        #expect(Self.vlmRegistryKeys(in: "no registry here") == nil)
    }

    static var requiresResolvedSource: Bool { ResolvedUpstreamSource.isRequired }

    /// `VLMModelFactory.swift` in the mlx-swift-lm checkout SwiftPM resolved
    /// for this build (see ``ResolvedUpstreamSource``).
    static func resolvedVLMFactorySource() -> String? {
        ResolvedUpstreamSource.mlxSwiftLM("Libraries/MLXVLM/VLMModelFactory.swift")
    }

    /// A type kept on the text path must be one the LLM factory loads, or
    /// its checkpoints would not load at all. So must a full-path type: its
    /// checkpoints without a vision tower stay `.mlx`.
    @Test
    func typesServedAsTextLoadThroughTheLLMFactory() async {
        for key in ModelLibraryManager.vlmTypesServedAsText.union(ModelLibraryManager.fullPathVLMTypes) {
            #expect(await LLMTypeRegistry.shared.contains(key), "\(key) is not in the LLM registry")
        }
    }

    @Test(arguments: ["fastvlm", "llava_qwen2", "qwen3_vl_moe", "muse_glimmer", "lfm2-vl"])
    func detectsTheVLMTypesTheListUsedToMiss(modelType: String) async throws {
        let temp = try TemporaryDirectory()
        try writeModel(in: temp.url, name: "vlm-\(modelType)", modelType: modelType)
        let models = try await ModelLibraryManager().scan(temp.url)
        #expect(models.first?.format == .mlxVLM)
    }

    /// With no vision tower in sight (the config names no `vision_config`),
    /// a Gemma 4 checkpoint is text, and the types not served on the full path
    /// are text whatever they carry.
    @Test(arguments: ["gemma4", "gemma4_unified", "qwen3_5", "qwen3_5_moe"])
    func multimodalAgentModelsStayOnTheTextPath(modelType: String) async throws {
        let temp = try TemporaryDirectory()
        try writeModel(in: temp.url, name: "text-\(modelType)", modelType: modelType)
        let models = try await ModelLibraryManager().scan(temp.url)
        #expect(models.first?.format == .mlx)
    }

    // MARK: - Gemma 4 on the full path

    /// A Gemma 4 checkpoint as mlx-community converts it: a `vision_config`,
    /// the encoder and its projection in the index, and processor metadata
    /// naming `Gemma4Processor`.
    @Test
    func aGemma4CheckpointWithItsVisionTowerIsAVLM() async throws {
        let temp = try TemporaryDirectory()
        try writeGemma4(in: temp.url, name: "gemma-4-vision")
        #expect(try await scanFormat(temp.url) == .mlxVLM)
    }

    /// Each requirement on its own: without it the checkpoint loads as text,
    /// as it did before Gemma 4 flipped, rather than going to the VLM
    /// factory and failing there.
    @Test(arguments: Gemma4Gap.allCases)
    func aGemma4CheckpointMissingAnyPieceStaysText(_ gap: Gemma4Gap) async throws {
        let temp = try TemporaryDirectory()
        try writeGemma4(in: temp.url, name: "gemma-4-\(gap)", gap: gap)
        #expect(try await scanFormat(temp.url) == .mlx)
    }

    /// Upstream reads `preprocessor_config.json` first and builds whatever it
    /// names, so a `processor_config.json` naming the right class beside it
    /// does not help.
    @Test
    func thePreprocessorConfigIsTheOneRead() async throws {
        let temp = try TemporaryDirectory()
        let dir = try writeGemma4(in: temp.url, name: "gemma-4-two-configs")
        try Data(#"{"processor_class": "Gemma3Processor"}"#.utf8)
            .write(to: dir.appending(path: "preprocessor_config.json"))
        #expect(try await scanFormat(temp.url) == .mlx)
        try Data(#"{"processor_class": "Gemma4Processor"}"#.utf8)
            .write(to: dir.appending(path: "preprocessor_config.json"))
        #expect(try await scanFormat(temp.url) == .mlxVLM)
    }

    /// Without an index, the tensor names come from the `.safetensors`
    /// headers. A raw Hugging Face layout's leading `model.` is dropped, as
    /// upstream's sanitizer drops it.
    @Test(arguments: [false, true])
    func withoutAnIndexTheHeadersAreRead(rawLayout: Bool) async throws {
        let temp = try TemporaryDirectory()
        let dir = try writeGemma4(in: temp.url, name: "gemma-4-headers", withIndex: false)
        let prefix = rawLayout ? "model." : ""
        try writeSafetensors(
            at: dir.appending(path: "model.safetensors"),
            names: ["\(prefix)language_model.model.embed_tokens.weight",
                    "\(prefix)vision_tower.encoder.layers.0.mlp.down_proj.weight",
                    "\(prefix)embed_vision.embedding_projection.weight"])
        #expect(try await scanFormat(temp.url) == .mlxVLM)

        try writeSafetensors(
            at: dir.appending(path: "model.safetensors"),
            names: ["\(prefix)language_model.model.embed_tokens.weight",
                    "\(prefix)vision_tower.encoder.layers.0.mlp.down_proj.weight"])
        #expect(try await scanFormat(temp.url) == .mlx, "no embed_vision")
    }

    /// A header too short to read or claiming more bytes than the file has
    /// gives no names, and the checkpoint stays text.
    @Test
    func anUnreadableHeaderGivesNoNames() throws {
        let temp = try TemporaryDirectory()
        let short = temp.url.appending(path: "short.safetensors")
        try Data([1, 2, 3]).write(to: short)
        #expect(VisionCheckpoint.safetensorsHeaderKeys(short) == nil)
        let truncated = temp.url.appending(path: "truncated.safetensors")
        try (withUnsafeBytes(of: UInt64(4096).littleEndian) { Data($0) } + Data("{}".utf8)).write(to: truncated)
        #expect(VisionCheckpoint.safetensorsHeaderKeys(truncated) == nil)
        let huge = temp.url.appending(path: "huge.safetensors")
        try withUnsafeBytes(of: (VisionCheckpoint.maxHeaderBytes + 1).littleEndian) { Data($0) }.write(to: huge)
        #expect(VisionCheckpoint.safetensorsHeaderKeys(huge) == nil)
    }

    /// Qwen3.5 has not flipped, so a complete-looking Qwen3.5 checkpoint is
    /// still text.
    @Test
    func aCompleteQwen35CheckpointIsStillText() async throws {
        let temp = try TemporaryDirectory()
        let dir = try writeGemma4(in: temp.url, name: "qwen-3-5", modelType: "qwen3_5")
        try Data(#"{"processor_class": "Qwen3VLProcessor"}"#.utf8)
            .write(to: dir.appending(path: "processor_config.json"))
        #expect(try await scanFormat(temp.url) == .mlx)
    }

    enum Gemma4Gap: String, CaseIterable, CustomStringConvertible {
        case visionConfig, visionTower, visionEmbedding, processorConfig, processorClass, wrongProcessor, unreadableIndex
        var description: String { rawValue }
    }

    private func scanFormat(_ root: URL) async throws -> ModelFormat? {
        try await ModelLibraryManager().scan(root).first?.format
    }

    /// A Gemma 4 checkpoint in mlx-community's layout, less `gap`.
    @discardableResult
    private func writeGemma4(
        in root: URL, name: String, modelType: String = "gemma4", gap: Gemma4Gap? = nil,
        withIndex: Bool = true
    ) throws -> URL {
        let dir = root.appending(path: name)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: dir.appending(path: "tokenizer.json"))
        try Data("\u{00}".utf8).write(to: dir.appending(path: "model.safetensors"))
        let vision = gap == .visionConfig ? "" : #", "vision_config": {"model_type": "gemma4_vision"}"#
        try Data(#"{"model_type": "\#(modelType)", "text_config": {}\#(vision)}"#.utf8)
            .write(to: dir.appending(path: "config.json"))
        if withIndex {
            var names = ["language_model.model.embed_tokens.weight", "audio_tower.layers.0.weight"]
            if gap != .visionTower { names.append("vision_tower.encoder.layers.0.mlp.down_proj.weight") }
            if gap != .visionEmbedding { names.append("embed_vision.embedding_projection.weight") }
            let map = Dictionary(uniqueKeysWithValues: names.map { ($0, "model.safetensors") })
            let index = gap == .unreadableIndex
                ? Data("not json".utf8)
                : try JSONSerialization.data(withJSONObject: ["metadata": [:], "weight_map": map])
            try index.write(to: dir.appending(path: "model.safetensors.index.json"))
        }
        switch gap {
        case .processorConfig: break
        case .processorClass:
            try Data(#"{"image_seq_length": 280}"#.utf8).write(to: dir.appending(path: "processor_config.json"))
        case .wrongProcessor:
            try Data(#"{"processor_class": "Gemma4UnifiedProcessor"}"#.utf8)
                .write(to: dir.appending(path: "processor_config.json"))
        default:
            try Data(#"{"processor_class": "Gemma4Processor"}"#.utf8)
                .write(to: dir.appending(path: "processor_config.json"))
        }
        return dir
    }

    /// A `.safetensors` file holding only a header that lists `names`.
    private func writeSafetensors(at url: URL, names: [String]) throws {
        var header: [String: Any] = ["__metadata__": ["format": "mlx"]]
        for name in names { header[name] = ["dtype": "F32", "shape": [1], "data_offsets": [0, 4]] }
        let body = try JSONSerialization.data(withJSONObject: header)
        let length = withUnsafeBytes(of: UInt64(body.count).littleEndian) { Data($0) }
        try (length + body + Data(count: 4)).write(to: url)
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
