// Copyright © 2026 macMLX. English comments only.

import Foundation
import Testing

@testable import MacMLXCore

// Pure config.json inspection — no MLX/Metal — so it runs under bare
// `swift test`. Guards the macMLX-side backfill that makes the streaming
// `ToolCallProcessor` fire for tool-capable families upstream
// `ToolCallFormat.infer` leaves nil (notably plain Qwen3). `.json` is the
// hermes `<tool_call>{JSON}</tool_call>` format; its rawValue is "json".
@Suite("ToolCallFormatFallback")
struct ToolCallFormatFallbackTests {

    /// Write a throwaway `config.json` carrying (optionally) a `model_type`.
    private func writeConfig(modelType: String?) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("macmlx-tcf-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("config.json")
        let json: [String: Any] = modelType.map { ["model_type": $0] } ?? [:]
        try JSONSerialization.data(withJSONObject: json).write(to: url)
        return url
    }

    @Test("plain qwen3 backfills to .json (hermes tool_call)")
    func qwen3MapsToJSON() throws {
        let url = try writeConfig(modelType: "qwen3")
        #expect(MLXSwiftEngine.inferToolCallFormatFallback(configURL: url)?.rawValue == "json")
    }

    @Test("qwen2 / qwen2.5 / qwen3_moe backfill to .json")
    func qwen2FamilyMapsToJSON() throws {
        #expect(try MLXSwiftEngine.inferToolCallFormatFallback(
            configURL: writeConfig(modelType: "qwen2"))?.rawValue == "json")
        #expect(try MLXSwiftEngine.inferToolCallFormatFallback(
            configURL: writeConfig(modelType: "qwen2_5"))?.rawValue == "json")
        #expect(try MLXSwiftEngine.inferToolCallFormatFallback(
            configURL: writeConfig(modelType: "qwen3_moe"))?.rawValue == "json")
    }

    @Test("qwen3_5 / qwen3_next defer to upstream (nil — upstream sets xml_function)")
    func qwenNextFamilyDefersToUpstream() throws {
        #expect(try MLXSwiftEngine.inferToolCallFormatFallback(
            configURL: writeConfig(modelType: "qwen3_5")) == nil)
        #expect(try MLXSwiftEngine.inferToolCallFormatFallback(
            configURL: writeConfig(modelType: "qwen3_next")) == nil)
    }

    @Test("non-qwen, absent model_type, and missing file yield no fallback")
    func othersYieldNil() throws {
        #expect(try MLXSwiftEngine.inferToolCallFormatFallback(
            configURL: writeConfig(modelType: "llama")) == nil)
        #expect(try MLXSwiftEngine.inferToolCallFormatFallback(
            configURL: writeConfig(modelType: nil)) == nil)
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("macmlx-missing-\(UUID().uuidString)/config.json")
        #expect(MLXSwiftEngine.inferToolCallFormatFallback(configURL: missing) == nil)
    }
}
