// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLXLLM
import MLXLMCommon
import XCTest

@testable import MacMLXCore

/// End-to-end proof of the A2d-2 engine ``BatchGenerationServing`` seam: four
/// concurrent streaming HTTP clients hit a REAL `HummingbirdServer` backed by a REAL
/// `MLXSwiftEngine`, and each gets its OWN complete, correct response with no
/// cross-talk — the Track A acceptance criterion. The concurrent burst forms a real
/// cohort under the M1 "batch only under concurrency" heuristic, so this exercises
/// `submit` → coordinator → `container.perform` drive loop → per-slot SSE fan-out,
/// the whole stack the MLX-free `BatchServingCoordinatorTests` /
/// `BatchDecodeSessionTests` stub out.
///
/// The oracle each concurrent response is checked against comes from a SEPARATE
/// no-seam server on the engine's legacy single-stream path (M3) — an independent
/// reference that shares no code with the batched cohort, so a systematic batch-path
/// defect cannot hide by corrupting oracle and measurement identically.
///
/// GATED — never runs in CI. Self-skips unless ALL hold:
///   1. `requireMLXRuntimeOrSkip()` (real Metal, i.e. xcodebuild),
///   2. env `MACMLX_RUN_BATCH_SPIKE=1`,
///   3. a DENSE, allowlisted model is on disk (env `MACMLX_BATCH_RAGGED_MODEL`
///      names a dir under `~/.mac-mlx/models`, else built-in candidates such as
///      `Qwen3-4B-4bit`), and
///   4. that model passes the batch coverage gate (`engine.batchServingCoverage`);
///      an uncovered model skips (it would silently serve every request on the
///      legacy path, which this test is not measuring).
final class BatchServingE2ETests: XCTestCase {

    private func denseModelDirectory() -> URL? {
        let root = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appending(path: ".mac-mlx/models", directoryHint: .isDirectory)
        var candidates: [String] = []
        if let override = ProcessInfo.processInfo.environment["MACMLX_BATCH_RAGGED_MODEL"] {
            candidates.append(override)
        }
        candidates.append(contentsOf: ["Qwen3-4B-4bit", "Qwen3-4B-8bit"])
        for name in candidates {
            let dir = root.appending(path: name, directoryHint: .isDirectory)
            if FileManager.default.fileExists(atPath: dir.path) { return dir }
        }
        return nil
    }

    private static func chatBody(_ prompt: String, model: String) -> [String: Any] {
        [
            "model": model,
            "messages": [["role": "user", "content": prompt]],
            "stream": true,
            "max_tokens": 40,
            "temperature": 0,
        ]
    }

    /// Reassemble the streamed text from an SSE body, and report whether a
    /// terminal `finish_reason` was seen. Aggregates BOTH `delta.content` and
    /// `delta.reasoning_content`: thinking models (Qwen3 by default) spend the
    /// whole short budget inside a `<think>` block, which the server's
    /// reasoning splitter routes to `reasoning_content` — `content` alone
    /// would be legitimately empty. Greedy determinism holds for the merged
    /// text, so the oracle comparison stays exact.
    private static func streamedContent(_ data: Data) -> (text: String, finished: Bool) {
        let raw = String(decoding: data, as: UTF8.self)
        var content = ""
        var finished = false
        for line in raw.split(separator: "\n") {
            guard line.hasPrefix("data: ") else { continue }
            let payload = line.dropFirst("data: ".count)
            if payload == "[DONE]" { continue }
            guard let d = payload.data(using: .utf8),
                let json = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
                let choices = json["choices"] as? [[String: Any]],
                let choice = choices.first
            else { continue }
            if let delta = choice["delta"] as? [String: Any] {
                if let piece = delta["reasoning_content"] as? String {
                    content += piece
                }
                if let piece = delta["content"] as? String {
                    content += piece
                }
            }
            if choice["finish_reason"] is String { finished = true }
        }
        return (content, finished)
    }

    /// Character length of the common prefix of two strings.
    private func commonPrefixLength(_ a: String, _ b: String) -> Int {
        a.commonPrefix(with: b).count
    }

    private static func post(_ url: URL, body: [String: Any]) async throws -> (String, Bool) {
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        request.timeoutInterval = 120
        let (data, _) = try await URLSession.shared.data(for: request)
        return streamedContent(data)
    }

    func testFourConcurrentClientsEachStreamOwnResponse() async throws {
        try requireMLXRuntimeOrSkip()
        guard ProcessInfo.processInfo.environment["MACMLX_RUN_BATCH_SPIKE"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_BATCH_SPIKE=1 to run the A2d-2 concurrent E2E")
        }
        guard let modelDir = denseModelDirectory() else {
            throw XCTSkip("No dense model found (set MACMLX_BATCH_RAGGED_MODEL to a dir name)")
        }

        let modelID = modelDir.lastPathComponent
        let model = LocalModel(
            id: modelID, displayName: modelID, directory: modelDir,
            sizeBytes: 0, format: .mlx, quantization: nil,
            parameterCount: nil, architecture: nil)

        let engine = MLXSwiftEngine()
        try await engine.load(model)
        guard await engine.batchServingCoverage else {
            throw XCTSkip("Model failed the batch coverage gate (non-dense or unlisted arch)")
        }

        let prompts = [
            "The capital of France is",
            "Two plus two equals",
            "The opposite of hot is",
            "Once upon a time there was a",
        ]

        // LEGACY ORACLE (M3): each prompt through a NO-SEAM server, so it runs on the
        // engine's legacy single-stream path — an INDEPENDENT reference that shares no
        // code with the batched cohort. The old oracle was a B=1 batched request, so a
        // SYSTEMATIC batch-path defect would corrupt oracle and measurement identically
        // and hide. A legacy oracle cannot: it never touches the cohort machinery.
        //
        // (Under the M1 "batch only under concurrency" heuristic these serial requests
        // would ALSO take the legacy path even against the seam server — each is idle
        // and uncontended, so it goes solo — but we use an explicit no-seam server so
        // the oracle's independence does not rely on the very heuristic under test.)
        let legacyServer = HummingbirdServer(engineProvider: { engine })
        let legacyPort = try await legacyServer.start(preferredPort: 19_911)
        let legacyURL = try XCTUnwrap(
            URL(string: "http://127.0.0.1:\(legacyPort)/v1/chat/completions"))
        var oracle: [String] = []
        let serialStart = Date()
        for prompt in prompts {
            let (text, finished) = try await Self.post(
                legacyURL, body: Self.chatBody(prompt, model: modelID))
            XCTAssertTrue(finished, "legacy oracle for \(prompt.debugDescription) must finish")
            XCTAssertFalse(text.isEmpty, "legacy oracle must produce content")
            oracle.append(text)
        }
        let serialElapsed = Date().timeIntervalSince(serialStart)
        await legacyServer.stop()

        // BATCHED MEASUREMENT: four concurrent streaming clients on the SEAM server.
        // A concurrent burst overlaps submit windows, so the M1 heuristic forms a real
        // cohort — this exercises submit → coordinator → `container.perform` drive loop
        // → per-slot SSE fan-out, the whole stack the MLX-free tests stub out.
        let server = HummingbirdServer(
            engineProvider: { engine },
            batchServing: engine as (any BatchGenerationServing))
        let port = try await server.start(preferredPort: 19_910)
        let url = try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)/v1/chat/completions"))

        let concurrentStart = Date()
        let concurrent: [(String, Bool)] = try await withThrowingTaskGroup(
            of: (Int, String, Bool).self
        ) { group in
            for (index, prompt) in prompts.enumerated() {
                group.addTask {
                    let (text, finished) = try await Self.post(url, body: Self.chatBody(prompt, model: modelID))
                    return (index, text, finished)
                }
            }
            var collected = Array(repeating: ("", false), count: prompts.count)
            for try await (index, text, finished) in group {
                collected[index] = (text, finished)
            }
            return collected
        }
        let concurrentElapsed = Date().timeIntervalSince(concurrentStart)
        await server.stop()

        // Each concurrent client got a complete, non-empty response…
        for (index, result) in concurrent.enumerated() {
            XCTAssertTrue(result.1, "concurrent client \(index) must see a terminal finish_reason")
            XCTAssertFalse(result.0.isEmpty, "concurrent client \(index) must receive content")
        }
        // …and each matches its own LEGACY oracle up to a LONG common prefix.
        // Byte-for-byte equality is deliberately NOT asserted: legacy is B=1 while the
        // live cohort's batch size changes step-by-step (admission timing is
        // nondeterministic), and different B takes different matmul tiling paths — the
        // legal batch-size kernel non-invariance documented on
        // `BatchPositionedCacheWrapper` (same as vLLM/TGI). Greedy then flips at a
        // near-tie token and both continuations stay coherent. What MUST hold instead,
        // and what actually catches cross-talk:
        //  1. a long shared prefix with the row's OWN oracle (mixed-up rows
        //     diverge at token one, not after dozens of tokens), and
        //  2. topic anchoring — each response talks about its own prompt.
        let topicAnchors = ["France", "Two plus two", "hot", "Once upon a time"]
        for index in prompts.indices {
            let prefix = commonPrefixLength(concurrent[index].0, oracle[index])
            XCTAssertGreaterThanOrEqual(
                prefix, 40,
                "concurrent client \(index) must share a long prefix with its own "
                    + "legacy oracle (got \(prefix) chars) — a short prefix means "
                    + "cross-talk or bookkeeping corruption, not FP non-invariance")
            XCTAssertTrue(
                concurrent[index].0.contains(topicAnchors[index]),
                "concurrent client \(index) must be answering ITS OWN prompt "
                    + "(anchor \(topicAnchors[index].debugDescription) missing)")
        }

        // Informational throughput signal (not asserted — depends on hardware/model).
        // Note: `serialElapsed` is the LEGACY single-stream baseline, so this compares
        // legacy-serial against batched-concurrent end to end.
        let speedup = serialElapsed / max(concurrentElapsed, 0.0001)
        print(
            "[A2d-2 E2E] legacy-serial=\(String(format: "%.2f", serialElapsed))s "
                + "batched-concurrent=\(String(format: "%.2f", concurrentElapsed))s "
                + "speedup=\(String(format: "%.2fx", speedup)) (model=\(modelID))")
    }
}
