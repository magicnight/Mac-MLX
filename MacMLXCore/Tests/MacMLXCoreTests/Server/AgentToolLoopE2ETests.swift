// Copyright © 2026 macMLX. English comments only.

import Foundation
import XCTest

@testable import MacMLXCore

/// Gated, real-model end-to-end validation for the agent tool-call loop across
/// BOTH protocol surfaces (OpenAI `/v1/chat/completions` and Anthropic
/// `/v1/messages`), driven through the real `HummingbirdServer` over HTTP.
///
/// Proves a two-round loop on each protocol:
///   • Round 1 — one `get_weather` tool + a prompt that demands it ⇒ the model's
///     response carries a tool call named `get_weather`.
///   • Round 2 — the assistant tool-call turn and a tool result ("22°C and
///     sunny") are appended and replayed ⇒ the model's final answer references
///     the result (anchor "22" or "sunny"). This round exercises exactly the
///     request-decode paths this wave wired: OpenAI `role:"tool"` + assistant
///     `tool_calls`, and Anthropic `tool_result` + `tool_use` blocks.
///
/// GATED — never runs in CI. Self-skips unless ALL hold:
///   1. `requireMLXRuntimeOrSkip()` passes (real Metal, i.e. xcodebuild),
///   2. env `MACMLX_RUN_TOOLS_E2E=1`, and
///   3. the model directory exists (env `MACMLX_TOOLS_MODEL` overrides the dir
///      name under `~/.mac-mlx/models`; default `Qwen3-4B-4bit`, whose template
///      is hermes-style tool calling).
///
/// Run:
///   MACMLX_RUN_TOOLS_E2E=1 TEST_RUNNER_MACMLX_RUN_TOOLS_E2E=1 \
///     xcodebuild test -scheme MacMLXCore -destination 'platform=macOS' \
///     -skipPackagePluginValidation \
///     -only-testing:MacMLXCoreTests/AgentToolLoopE2ETests
///
/// Thinking-model discipline: Qwen3 may emit a `<think>` block. The server
/// routes that to `reasoning_content` (OpenAI) / drops it (Anthropic), so the
/// anchor assertions aggregate answer + reasoning and use a generous
/// `max_tokens` budget.
final class AgentToolLoopE2ETests: XCTestCase {

    // MARK: - Gate + fixtures

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

    private func gateAndResolveModel() throws -> (id: String, directory: URL) {
        try requireMLXRuntimeOrSkip()
        guard ProcessInfo.processInfo.environment["MACMLX_RUN_TOOLS_E2E"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_TOOLS_E2E=1 to run the agent tool-loop E2E tests")
        }
        let modelID = ProcessInfo.processInfo.environment["MACMLX_TOOLS_MODEL"] ?? "Qwen3-4B-4bit"
        let directory = modelDirectory(modelID)
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw XCTSkip("Tool-loop model dir not found: \(directory.path)")
        }
        return (modelID, directory)
    }

    /// Boot a real server with the model already loaded, returning it + its port.
    private func startServer(modelID: String, directory: URL) async throws -> (HummingbirdServer, Int) {
        let engine = MLXSwiftEngine()
        try await engine.load(localModel(id: modelID, directory: directory))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 20_500)
        return (server, port)
    }

    /// POST a JSON body and return the decoded top-level object.
    private func postJSON(port: Int, path: String, body: [String: Any]) async throws -> [String: Any] {
        let url = URL(string: "http://127.0.0.1:\(port)\(path)")!
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.timeoutInterval = 300
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await URLSession.shared.data(for: req)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(http.statusCode, 200, "HTTP \(http.statusCode): \(String(decoding: data, as: UTF8.self))")
        return try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    /// The one tool both protocols offer, in each protocol's own shape.
    private var openAIWeatherTool: [String: Any] {
        [
            "type": "function",
            "function": [
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": [
                    "type": "object",
                    "properties": ["city": ["type": "string", "description": "City name"]],
                    "required": ["city"],
                ],
            ],
        ]
    }

    private var anthropicWeatherTool: [String: Any] {
        [
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "input_schema": [
                "type": "object",
                "properties": ["city": ["type": "string", "description": "City name"]],
                "required": ["city"],
            ],
        ]
    }

    private let userPrompt =
        "What is the weather in Paris right now? You must call the get_weather tool to find out."
    private let toolResultText = "22°C and sunny"

    private func referencesResult(_ text: String) -> Bool {
        let lower = text.lowercased()
        return lower.contains("22") || lower.contains("sunny")
    }

    // MARK: - OpenAI /v1/chat/completions

    func testOpenAIToolLoopTwoRounds() async throws {
        let (modelID, directory) = try gateAndResolveModel()
        let (server, port) = try await startServer(modelID: modelID, directory: directory)
        defer { Task { await server.stop() } }

        let userMessage: [String: Any] = ["role": "user", "content": userPrompt]

        // Round 1 — expect a get_weather tool call.
        let round1 = try await postJSON(port: port, path: "/v1/chat/completions", body: [
            "model": modelID,
            "messages": [userMessage],
            "tools": [openAIWeatherTool],
            "stream": false,
            "temperature": 0,
            "max_tokens": 512,
        ])
        let choices1 = try XCTUnwrap(round1["choices"] as? [[String: Any]])
        let message1 = try XCTUnwrap(choices1.first?["message"] as? [String: Any])
        let toolCalls = try XCTUnwrap(
            message1["tool_calls"] as? [[String: Any]],
            "round 1 produced no tool_calls: \(message1)")
        XCTAssertFalse(toolCalls.isEmpty)
        let names = toolCalls.compactMap { ($0["function"] as? [String: Any])?["name"] as? String }
        XCTAssertTrue(names.contains("get_weather"), "expected a get_weather call, saw \(names)")
        print("OPENAI_TOOL_LOOP round1 finish_reason=\(choices1.first?["finish_reason"] as? String ?? "?") calls=\(names)")

        // Round 2 — replay the assistant tool-call turn + a tool result per call.
        var round2Messages: [[String: Any]] = [userMessage]
        round2Messages.append([
            "role": "assistant",
            "content": message1["content"] ?? NSNull(),
            "tool_calls": toolCalls,
        ])
        for call in toolCalls {
            round2Messages.append([
                "role": "tool",
                "tool_call_id": call["id"] as? String ?? "",
                "content": toolResultText,
            ])
        }
        let round2 = try await postJSON(port: port, path: "/v1/chat/completions", body: [
            "model": modelID,
            "messages": round2Messages,
            "tools": [openAIWeatherTool],
            "stream": false,
            "temperature": 0,
            "max_tokens": 512,
        ])
        let message2 = try XCTUnwrap((round2["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any])
        let answer = (message2["content"] as? String ?? "")
        let reasoning = (message2["reasoning_content"] as? String ?? "")
        let aggregate = answer + "\n" + reasoning
        print("OPENAI_TOOL_LOOP round2 answer=\(answer.prefix(200))")
        XCTAssertTrue(
            referencesResult(aggregate),
            "round 2 answer did not reference the tool result (22 / sunny): \(aggregate)")
    }

    // MARK: - Anthropic /v1/messages

    func testAnthropicToolLoopTwoRounds() async throws {
        let (modelID, directory) = try gateAndResolveModel()
        let (server, port) = try await startServer(modelID: modelID, directory: directory)
        defer { Task { await server.stop() } }

        let userMessage: [String: Any] = ["role": "user", "content": userPrompt]

        // Round 1 — expect a tool_use block for get_weather.
        let round1 = try await postJSON(port: port, path: "/v1/messages", body: [
            "model": modelID,
            "max_tokens": 512,
            "temperature": 0,
            "messages": [userMessage],
            "tools": [anthropicWeatherTool],
        ])
        XCTAssertEqual(round1["stop_reason"] as? String, "tool_use", "round 1 did not stop on tool_use: \(round1)")
        let content1 = try XCTUnwrap(round1["content"] as? [[String: Any]])
        let toolUses = content1.filter { $0["type"] as? String == "tool_use" }
        XCTAssertFalse(toolUses.isEmpty, "round 1 produced no tool_use block: \(content1)")
        let names = toolUses.compactMap { $0["name"] as? String }
        XCTAssertTrue(names.contains("get_weather"), "expected a get_weather tool_use, saw \(names)")
        print("ANTHROPIC_TOOL_LOOP round1 stop_reason=tool_use calls=\(names)")

        // Round 2 — replay the assistant content (incl. tool_use) + a user turn
        // of tool_result blocks (one per tool_use).
        var round2Messages: [[String: Any]] = [userMessage]
        round2Messages.append(["role": "assistant", "content": content1])
        let results: [[String: Any]] = toolUses.map { use in
            [
                "type": "tool_result",
                "tool_use_id": use["id"] as? String ?? "",
                "content": toolResultText,
            ]
        }
        round2Messages.append(["role": "user", "content": results])
        let round2 = try await postJSON(port: port, path: "/v1/messages", body: [
            "model": modelID,
            "max_tokens": 512,
            "temperature": 0,
            "messages": round2Messages,
            "tools": [anthropicWeatherTool],
        ])
        let content2 = try XCTUnwrap(round2["content"] as? [[String: Any]])
        let aggregate = content2.compactMap { $0["text"] as? String }.joined(separator: "\n")
        print("ANTHROPIC_TOOL_LOOP round2 answer=\(aggregate.prefix(200))")
        XCTAssertTrue(
            referencesResult(aggregate),
            "round 2 answer did not reference the tool result (22 / sunny): \(aggregate)")
    }
}
