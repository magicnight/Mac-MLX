// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import Testing

@testable import MacMLXCore

/// The engine seam that serves a VLM on the text generation path
/// (``FullPathVLM``): which checkpoints it applies to, what it renders with,
/// the state a warm continuation starts from, and the input shape the text
/// path slices.
@Suite("FullPathVLM")
struct FullPathVLMTests {

    private func config(_ json: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appending(path: "full-path-vlm-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appending(path: "config.json", directoryHint: .notDirectory)
        try Data(json.utf8).write(to: url)
        return url
    }

    @Test(arguments: FullPathVLM.Family.allCases)
    func aKnownFamilyIsServedWhenForcedOrOnceItIsInTheSet(_ family: FullPathVLM.Family) throws {
        let url = try config(#"{"model_type": "\#(family.rawValue)", "vision_config": {}}"#)
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        #expect(FullPathVLM.resolve(configURL: url, forced: true) == FullPathVLM(family: family))
        #expect(
            (FullPathVLM.resolve(configURL: url, forced: false) != nil)
                == FullPathVLM.servedFamilies.contains(family),
            "unforced, only a served family takes the full path")
    }

    /// No family has passed its parity gate yet, so no checkpoint changes
    /// path in this build. A family joins with its own flip.
    @Test
    func noFamilyIsServedYet() {
        #expect(FullPathVLM.servedFamilies.isEmpty)
    }

    /// The forcing seam reaches only the families this path knows how to
    /// drive; the siblings that share a generator but not a class, a text
    /// type and an unrelated VLM stay where they are.
    @Test(arguments: ["gemma4_unified", "qwen3_5_moe", "gemma4_text", "qwen3_5_text", "qwen3_vl", "llama"])
    func otherTypesNeverTakeTheFullPath(_ modelType: String) throws {
        let url = try config(#"{"model_type": "\#(modelType)"}"#)
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        #expect(FullPathVLM.resolve(configURL: url, forced: true) == nil)
    }

    @Test
    func theModelTypeIsReadCaseInsensitively() throws {
        let url = try config(#"{"model_type": "Gemma4"}"#)
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        #expect(FullPathVLM.resolve(configURL: url, forced: true)?.family == .gemma4)
    }

    @Test
    func anUnreadableConfigStaysOnTheStockPath() {
        let missing = URL(fileURLWithPath: "/nonexistent/\(UUID().uuidString)/config.json")
        #expect(FullPathVLM.resolve(configURL: missing, forced: true) == nil)
    }

    /// Every family is a key the VLM factory registers and macMLX knows
    /// about, so a family can never name a type nothing loads.
    @Test(arguments: FullPathVLM.Family.allCases)
    func everyFamilyIsAVLMRegistryKey(_ family: FullPathVLM.Family) {
        #expect(
            ModelLibraryManager.vlmTypesServedAsText.contains(family.rawValue)
                || ModelLibraryManager.knownVLMTypes.contains(family.rawValue))
    }

    /// The generator each family's processor builds for itself
    /// (`Gemma4Processor.prepare`, `Qwen3VLProcessor.prepare`), so the
    /// engine's own renders (replayed reasoning, the history boundary) match
    /// the prompt.
    @Test
    func eachFamilyRendersWithItsProcessorsGenerator() {
        #expect(FullPathVLM(family: .gemma4).processorMessageGenerator is Gemma4MessageGenerator)
        #expect(FullPathVLM(family: .qwen35).processorMessageGenerator is Qwen3VLMessageGenerator)
    }

    /// The key id is rebuilt from a string because upstream keeps the key
    /// private; this pins the string to the source this build resolved, as
    /// `ModelLibraryManagerVLMTests` does for the registry. A wrong id would
    /// leave every warm Qwen3.5 continuation without its anchor.
    @Test(.enabled(if: ResolvedUpstreamSource.mlxSwiftLM(Self.qwen35Source) != nil
                   || ResolvedUpstreamSource.isRequired,
                   "no SwiftPM checkout of mlx-swift-lm next to the test bundle"))
    func theQwen35RopeDeltasKeyIsUpstreamsKey() throws {
        let source = try #require(ResolvedUpstreamSource.mlxSwiftLM(Self.qwen35Source),
                                  "MACMLX_REQUIRE_RESOLVED_SOURCE is set but no checkout was found")
        let declaration = /let ropeDeltasKey = LMOutput\.Key<MLXArray>\(\s*"([^"]+)"\s*\)/
        let match = try #require(source.firstMatch(of: declaration), "no ropeDeltasKey in Qwen35.swift")
        #expect(String(match.1) == FullPathVLM.qwen35RopeDeltasKeyID)
    }

    private static let qwen35Source = "Libraries/MLXVLM/Models/Qwen35.swift"
}

/// The MLX-backed half: the state a warm continuation starts from and the
/// shape the text path gets.
@Suite("FullPathVLM inputs", .enabled(if: mlxMetallibIsAvailable, "needs the MLX Metal library"))
struct FullPathVLMInputTests {

    @Test
    func qwen35StartsFromAZeroRopeDeltaAndGemma4FromNothing() throws {
        let state = try #require(FullPathVLM(family: .qwen35).iteratorState())
        let delta = try #require(state[LMOutput.Key<MLXArray>(FullPathVLM.qwen35RopeDeltasKeyID)])
        #expect(delta.ndim == 0)
        #expect(delta.item(Int32.self) == 0)
        #expect(FullPathVLM(family: .gemma4).iteratorState() == nil)
    }

    @Test
    func aBatchShapedTextPromptBecomesFlatWithNoMask() {
        let input = LMInput(text: .init(
            tokens: MLXArray([Int32(5), 6, 7]).reshaped(1, 3),
            mask: MLXArray.ones([1, 3], dtype: .int8)))
        let flat = FullPathVLM.textOnlyInput(input)
        #expect(flat.text.tokens.shape == [3])
        #expect(flat.text.tokens.asArray(Int32.self) == [5, 6, 7])
        #expect(flat.text.mask == nil)
    }

    @Test
    func aFlatPromptLosesOnlyItsMask() {
        let masked = FullPathVLM.textOnlyInput(LMInput(text: .init(
            tokens: MLXArray([Int32(5), 6]), mask: MLXArray.ones([2], dtype: .int8))))
        #expect(masked.text.tokens.asArray(Int32.self) == [5, 6])
        #expect(masked.text.mask == nil)
        let plain = FullPathVLM.textOnlyInput(LMInput(tokens: MLXArray([Int32(5), 6])))
        #expect(plain.text.tokens.asArray(Int32.self) == [5, 6])
    }

    @Test
    func aPromptWithAnImageKeepsTheProcessorsShape() {
        let input = LMInput(
            text: .init(tokens: MLXArray([Int32(5), 6, 7]).reshaped(1, 3), mask: MLXArray.ones([1, 3], dtype: .int8)),
            image: .init(pixels: MLXArray.zeros([1, 3, 2, 2])))
        #expect(FullPathVLM.carriesMedia(input))
        let kept = FullPathVLM.textOnlyInput(input)
        #expect(kept.text.tokens.shape == [1, 3])
        #expect(kept.text.mask != nil)
        #expect(kept.image != nil)
    }
}
