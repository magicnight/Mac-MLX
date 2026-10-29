// Copyright © 2026 macMLX. English comments only.

import Foundation
import XCTest

@testable import MacMLXCore

/// Gated, real-model end-to-end validation of one tool round trip over the
/// Responses API (`POST /v1/responses`), driven through the real
/// `HummingbirdServer` over HTTP:
///   • Round 1 (non-streaming) — one flat `get_weather` function tool and a
///     prompt that demands it ⇒ the output carries a `function_call` item.
///   • Round 2 (streaming) — the call item is replayed with a
///     `function_call_output` ("22°C and sunny") ⇒ the stream ends in
///     `response.completed` and the answer references the result.
///
/// GATED — never runs in CI. Self-skips unless ALL hold:
///   1. `requireMLXRuntimeOrSkip()` passes (real Metal, i.e. xcodebuild),
///   2. env `MACMLX_RUN_RESPONSES_E2E=1`, and
///   3. the model directory exists (env `MACMLX_RESPONSES_MODEL` overrides the
///      dir name under `~/.mac-mlx/models`; default `Qwen3.6-27B-4bit`).
///
/// Run:
///   MACMLX_RUN_RESPONSES_E2E=1 TEST_RUNNER_MACMLX_RUN_RESPONSES_E2E=1 \
///     xcodebuild test -scheme MacMLXCore -destination 'platform=macOS' \
///     -skipPackagePluginValidation \
///     -only-testing:MacMLXCoreTests/ResponsesAgentLoopE2ETests
///
/// The model thinks before it answers; the reasoning arrives as its own
/// output item, so the anchor check reads the answer and the reasoning. No
/// `max_output_tokens` is sent, as Codex sends none, so the endpoint's default
/// budget has to cover the thinking and the call.
final class ResponsesAgentLoopE2ETests: XCTestCase {

    private func gateAndResolveModel() throws -> (id: String, directory: URL) {
        try requireMLXRuntimeOrSkip()
        guard ProcessInfo.processInfo.environment["MACMLX_RUN_RESPONSES_E2E"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_RESPONSES_E2E=1 to run the Responses tool-loop E2E test")
        }
        let modelID = ProcessInfo.processInfo.environment["MACMLX_RESPONSES_MODEL"] ?? "Qwen3.6-27B-4bit"
        let directory = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appending(path: ".mac-mlx/models/\(modelID)", directoryHint: .isDirectory)
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw XCTSkip("Responses tool-loop model dir not found: \(directory.path)")
        }
        return (modelID, directory)
    }

    private func post(port: Int, body: [String: Any]) async throws -> Data {
        let url = try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)/v1/responses"))
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 900
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await URLSession.shared.data(for: request)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(http.statusCode, 200, "HTTP \(http.statusCode): \(String(decoding: data, as: UTF8.self))")
        return data
    }

    private let weatherTool: [String: Any] = [
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": [
            "type": "object",
            "properties": ["city": ["type": "string", "description": "City name"]],
            "required": ["city"],
        ],
    ]

    func testResponsesToolLoopTwoRounds() async throws {
        let (modelID, directory) = try gateAndResolveModel()
        let engine = MLXSwiftEngine()
        try await engine.load(LocalModel(
            id: modelID, displayName: modelID, directory: directory, sizeBytes: 0,
            format: .mlx, quantization: nil, parameterCount: nil, architecture: nil))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 21_600)
        defer { Task { await server.stop() } }

        let userItem: [String: Any] = [
            "type": "message", "role": "user",
            "content": [["type": "input_text",
                         "text": "What is the weather in Paris right now? You must call the get_weather tool to find out."]],
        ]

        // Round 1 — expect a get_weather function_call item.
        let round1Data = try await post(port: port, body: [
            "model": modelID, "input": [userItem], "tools": [weatherTool],
            "temperature": 0,
        ])
        let round1 = try XCTUnwrap(try JSONSerialization.jsonObject(with: round1Data) as? [String: Any])
        XCTAssertEqual(round1["status"] as? String, "completed", "round 1: \(round1)")
        let output1 = try XCTUnwrap(round1["output"] as? [[String: Any]])
        let calls = output1.filter { $0["type"] as? String == "function_call" }
        XCTAssertFalse(calls.isEmpty, "round 1 produced no function_call: \(output1)")
        XCTAssertTrue(calls.contains { $0["name"] as? String == "get_weather" }, "round 1 calls: \(calls)")
        print("RESPONSES_TOOL_LOOP round1 items=\(output1.compactMap { $0["type"] as? String }) "
              + "calls=\(calls.compactMap { $0["arguments"] as? String })")

        // Round 2 — replay the reasoning and call items as Codex does, plus one
        // output per call, and stream the answer.
        var input: [[String: Any]] = [userItem]
        input.append(contentsOf: output1.filter { $0["type"] as? String != "message" })
        for call in calls {
            input.append([
                "type": "function_call_output",
                "call_id": call["call_id"] as? String ?? "",
                "output": "22°C and sunny",
            ])
        }
        let round2Data = try await post(port: port, body: [
            "model": modelID, "input": input, "tools": [weatherTool], "stream": true,
            "temperature": 0,
        ])
        let streamed = String(decoding: round2Data, as: UTF8.self)
        let payloads = try streamed.components(separatedBy: "\n\n").filter { !$0.isEmpty }.map { frame in
            let data = try XCTUnwrap(frame.components(separatedBy: "\n").last?.dropFirst("data: ".count))
            return try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(data.utf8)) as? [String: Any])
        }
        XCTAssertEqual(payloads.last?["type"] as? String, "response.completed", "round 2 ended in: \(payloads.last ?? [:])")
        let deltas = payloads.filter {
            ["response.output_text.delta", "response.reasoning_text.delta"].contains($0["type"] as? String)
        }
        let aggregate = deltas.compactMap { $0["delta"] as? String }.joined()
        let answer = payloads.filter { $0["type"] as? String == "response.output_text.delta" }
            .compactMap { $0["delta"] as? String }.joined()
        print("RESPONSES_TOOL_LOOP round2 answer=\(answer.prefix(200))")
        let lower = aggregate.lowercased()
        XCTAssertTrue(lower.contains("22") || lower.contains("sunny"),
                      "round 2 did not reference the tool result (22 / sunny): \(aggregate)")
    }
}
