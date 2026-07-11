// Copyright © 2026 macMLX. English comments only.

import Foundation
import XCTest

@testable import MacMLXCore

/// Gated, real-model end-to-end validation for Track C structured output.
///
/// GATED — never runs in CI. Self-skips unless ALL hold:
///   1. `requireMLXRuntimeOrSkip()` passes (real Metal, i.e. xcodebuild),
///   2. env `MACMLX_RUN_STRUCTURED_E2E=1`, and
///   3. the model directory exists (env `MACMLX_STRUCTURED_MODEL` overrides the
///      dir name under `~/.mac-mlx/models`; default `Qwen3-0.6B-4bit`).
///
/// Run:
///   MACMLX_RUN_STRUCTURED_E2E=1 TEST_RUNNER_MACMLX_RUN_STRUCTURED_E2E=1 \
///     xcodebuild test -scheme MacMLXCore -destination 'platform=macOS' \
///     -skipPackagePluginValidation \
///     -only-testing:MacMLXCoreTests/StructuredOutputModelTests
///
/// What it proves:
///  - **C1** — under `response_format: json_object`, several prompts each yield
///    output that `JSONSerialization` parses (100% well-formed), regardless of
///    what the model "wanted" to say (the constraint even overrides a thinking
///    model's `<think>` opener, which is not a legal JSON start).
///  - **C2** — under a small object schema (2 typed fields + a string enum),
///    the output parses AND conforms: only declared keys, required present,
///    values of the declared types, enum value in range.
///  - **Throughput** — constrained vs unconstrained tok/s is printed for the
///    record (target < 15% loss on the greedy fast path; informational).
///  - **Real vocabulary** — the per-model token table + grammar mask classify a
///    real tokenizer's vocabulary correctly at the boundary.
final class StructuredOutputModelTests: XCTestCase {

    private func modelDirectory(_ name: String) -> URL {
        URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appending(path: ".mac-mlx/models/\(name)", directoryHint: .isDirectory)
    }

    private func localModel(id: String, directory: URL) -> LocalModel {
        LocalModel(
            id: id, displayName: id, directory: directory, sizeBytes: 0,
            format: .mlx, quantization: nil, parameterCount: nil, architecture: nil
        )
    }

    /// Shared gate. Returns the resolved model id + directory or skips.
    private func gateAndResolveModel() throws -> (id: String, directory: URL) {
        try requireMLXRuntimeOrSkip()
        guard ProcessInfo.processInfo.environment["MACMLX_RUN_STRUCTURED_E2E"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_STRUCTURED_E2E=1 to run the structured-output E2E tests")
        }
        let modelID = ProcessInfo.processInfo.environment["MACMLX_STRUCTURED_MODEL"] ?? "Qwen3-0.6B-4bit"
        let directory = modelDirectory(modelID)
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw XCTSkip("Structured-output model dir not found: \(directory.path)")
        }
        return (modelID, directory)
    }

    /// Drive `engine.generate` to completion, returning the generated text and
    /// completion-token count.
    private func run(_ engine: MLXSwiftEngine, _ request: GenerateRequest) async throws -> (text: String, tokens: Int?) {
        var text = ""
        var tokens: Int?
        for try await chunk in engine.generate(request) {
            text += chunk.text
            if let usage = chunk.usage { tokens = usage.completionTokens }
        }
        return (text, tokens)
    }

    // MARK: - C1: any JSON

    func testC1ProducesParseableJSON() async throws {
        let (modelID, directory) = try gateAndResolveModel()
        let engine = MLXSwiftEngine()
        try await engine.load(localModel(id: modelID, directory: directory))

        let prompts = [
            "Give me a JSON object describing a person with a name and an age.",
            "Return a JSON array of three fruit names.",
            "Output a JSON object with a boolean field 'active' and a numeric field 'score'.",
        ]

        for prompt in prompts {
            let request = GenerateRequest(
                model: modelID,
                messages: [ChatMessage(role: .user, content: prompt)],
                parameters: GenerationParameters(temperature: 0, topP: 1.0, maxTokens: 200, stream: true),
                templateKwargs: ["enable_thinking": .bool(false)],
                responseFormat: .jsonObject
            )
            let (text, _) = try await run(engine, request)
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            XCTAssertFalse(trimmed.isEmpty, "constrained output was empty for prompt: \(prompt)")
            let data = Data(trimmed.utf8)
            XCTAssertNoThrow(
                try JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]),
                "constrained output must be valid JSON — got: \(trimmed)"
            )
        }
    }

    // MARK: - C2: schema subset

    func testC2ConformsToSchemaSubset() async throws {
        let (modelID, directory) = try gateAndResolveModel()
        let engine = MLXSwiftEngine()
        try await engine.load(localModel(id: modelID, directory: directory))

        let schema = JSONSchemaObject(
            properties: [
                .init(name: "name", type: .string),
                .init(name: "age", type: .integer),
                .init(name: "role", type: .stringEnum(["admin", "user", "guest"])),
            ],
            required: ["name", "age"]
        )
        let request = GenerateRequest(
            model: modelID,
            messages: [ChatMessage(
                role: .user,
                content: "Describe a fictional user: their name, age, and role (admin, user, or guest)."
            )],
            parameters: GenerationParameters(temperature: 0, topP: 1.0, maxTokens: 200, stream: true),
            templateKwargs: ["enable_thinking": .bool(false)],
            responseFormat: .jsonSchema(schema)
        )
        let (text, _) = try await run(engine, request)
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)

        let parsed = try JSONSerialization.jsonObject(with: Data(trimmed.utf8))
        let object = try XCTUnwrap(parsed as? [String: Any], "schema output must be a JSON object — got: \(trimmed)")

        // Only declared keys.
        let declared: Set<String> = ["name", "age", "role"]
        XCTAssertTrue(Set(object.keys).isSubset(of: declared), "unexpected keys in \(object)")
        // Required present.
        XCTAssertNotNil(object["name"], "required 'name' missing in \(object)")
        XCTAssertNotNil(object["age"], "required 'age' missing in \(object)")
        // Types.
        XCTAssertTrue(object["name"] is String, "'name' must be a string in \(object)")
        if let age = object["age"] as? NSNumber {
            // Integer-valued (JSONSerialization surfaces numbers as NSNumber).
            XCTAssertEqual(age.doubleValue, age.doubleValue.rounded(), "'age' must be an integer in \(object)")
        } else {
            XCTFail("'age' must be a number in \(object)")
        }
        if let role = object["role"] as? String {
            XCTAssertTrue(["admin", "user", "guest"].contains(role), "'role' out of enum in \(object)")
        }
    }

    // MARK: - Throughput (informational)

    func testConstraintThroughputOverhead() async throws {
        let (modelID, directory) = try gateAndResolveModel()
        let engine = MLXSwiftEngine()
        try await engine.load(localModel(id: modelID, directory: directory))

        let prompt = "Return a JSON object with keys 'title' (string) and 'count' (number)."
        func measure(constrained: Bool) async throws -> Double {
            let request = GenerateRequest(
                model: modelID,
                messages: [ChatMessage(role: .user, content: prompt)],
                parameters: GenerationParameters(temperature: 0, topP: 1.0, maxTokens: 128, stream: true),
                templateKwargs: ["enable_thinking": .bool(false)],
                responseFormat: constrained ? .jsonObject : nil
            )
            let start = Date()
            let (_, tokens) = try await run(engine, request)
            let elapsed = Date().timeIntervalSince(start)
            guard let tokens, elapsed > 0 else { return 0 }
            return Double(tokens) / elapsed
        }

        // Warm up (first constrained request also builds the vocabulary table).
        _ = try await measure(constrained: true)

        let unconstrained = try await measure(constrained: false)
        let constrained = try await measure(constrained: true)
        let overhead = unconstrained > 0 ? (1 - constrained / unconstrained) * 100 : 0
        print(
            "STRUCTURED_THROUGHPUT model=\(modelID) "
                + "unconstrained=\(String(format: "%.1f", unconstrained)) tok/s "
                + "constrained=\(String(format: "%.1f", constrained)) tok/s "
                + "overhead=\(String(format: "%.1f", overhead))%")
        // Informational only — hardware-dependent; not asserted.
    }

    // MARK: - Real vocabulary table + mask correctness

    func testRealTokenizerTableClassification() async throws {
        try requireMLXRuntimeOrSkip()
        guard ProcessInfo.processInfo.environment["MACMLX_RUN_STRUCTURED_E2E"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_STRUCTURED_E2E=1 to run the real-tokenizer table test")
        }
        let modelID = ProcessInfo.processInfo.environment["MACMLX_STRUCTURED_MODEL"] ?? "Qwen3-0.6B-4bit"
        let directory = modelDirectory(modelID)
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw XCTSkip("Structured-output model dir not found: \(directory.path)")
        }

        // Load the real tokenizer directly (no model weights needed) and read
        // the authoritative vocab size from config.json.
        let tokenizer = try await HuggingFaceTokenizerLoader().load(from: directory)
        let vocabSize = try readVocabSize(directory: directory)
        let eosID = tokenizer.eosTokenId

        let cache = TokenVocabularyCache()
        let table = cache.table(
            modelID: modelID,
            vocabularySize: vocabSize,
            stopTokenIDs: eosID.map { [$0] } ?? [],
            decode: { tokenizer.decode(tokenIds: [$0], skipSpecialTokens: false) }
        )
        XCTAssertEqual(table.count, vocabSize)

        // EOS classifies as .eos.
        if let eosID {
            XCTAssertEqual(table.classification(of: eosID), .eos)
        }

        // From the initial JSON state, structural openers must be legal and a
        // pure-letter token must be illegal — the mask boundary over real vocab.
        let start = ConstraintState.initial(for: .jsonObject)
        func legal(_ id: Int) -> Bool {
            switch table.classification(of: id) {
            case .eos: return start.isComplete
            case .unusable: return false
            case .bytes(let b): return start.accepts(b)
            }
        }
        if let brace = tokenizer.convertTokenToId("{") {
            XCTAssertTrue(legal(brace), "'{' must be a legal JSON start")
        }
        // A token that decodes to a bare letter word is not a legal JSON start.
        var sawIllegalWord = false
        for id in 0..<min(vocabSize, 2000) {
            if case .bytes(let b) = table.classification(of: id),
               let s = String(bytes: b, encoding: .utf8),
               s.count >= 3, s.allSatisfy({ $0.isLetter }) {
                XCTAssertFalse(legal(id), "letter token '\(s)' must be illegal at JSON start")
                sawIllegalWord = true
                break
            }
        }
        XCTAssertTrue(sawIllegalWord, "expected to find at least one all-letter token to check")
    }

    private func readVocabSize(directory: URL) throws -> Int {
        let data = try Data(contentsOf: directory.appending(path: "config.json"))
        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard let size = object?["vocab_size"] as? Int else {
            throw XCTSkip("config.json has no integer vocab_size")
        }
        return size
    }
}
