// Copyright © 2026 macMLX. English comments only.

import Foundation
import XCTest

@testable import MacMLXCore

/// Gated, real-model check that `POST /v1/messages/count_tokens` counts the
/// prompt `/v1/messages` generates from, driven through the real
/// `HummingbirdServer` over HTTP with a Claude Code-shaped body: a top-level
/// system prompt, a user turn followed by a `role:"system"` turn, and a tool.
/// The count must equal the prompt length the generation of the same body
/// reports — `input_tokens + cache_read_input_tokens`, which is
/// `TokenUsage.promptTokens` whatever the prompt cache served (a cache hit
/// moves tokens from one to the other). A per-run nonce opens the system
/// prompt, so the cache cannot serve more than the template's header.
///
/// GATED — never runs in CI. Self-skips unless ALL hold:
///   1. `requireMLXRuntimeOrSkip()` passes (real Metal, i.e. xcodebuild),
///   2. env `MACMLX_RUN_ANTHROPIC_COUNT_E2E=1`, and
///   3. the model directory exists (env `MACMLX_ANTHROPIC_COUNT_MODEL`
///      overrides the dir name under `~/.mac-mlx/models`; default
///      `Qwen3.6-27B-4bit`).
///
/// Run:
///   MACMLX_RUN_ANTHROPIC_COUNT_E2E=1 TEST_RUNNER_MACMLX_RUN_ANTHROPIC_COUNT_E2E=1 \
///     xcodebuild test -scheme MacMLXCore -destination 'platform=macOS' \
///     -skipPackagePluginValidation \
///     -only-testing:MacMLXCoreTests/AnthropicCountTokensE2ETests
final class AnthropicCountTokensE2ETests: XCTestCase {

    private func gateAndResolveModel() throws -> (id: String, directory: URL) {
        try requireMLXRuntimeOrSkip()
        guard ProcessInfo.processInfo.environment["MACMLX_RUN_ANTHROPIC_COUNT_E2E"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_ANTHROPIC_COUNT_E2E=1 to run the count_tokens E2E test")
        }
        let modelID = ProcessInfo.processInfo.environment["MACMLX_ANTHROPIC_COUNT_MODEL"] ?? "Qwen3.6-27B-4bit"
        let directory = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appending(path: ".mac-mlx/models/\(modelID)", directoryHint: .isDirectory)
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw XCTSkip("count_tokens model dir not found: \(directory.path)")
        }
        return (modelID, directory)
    }

    private func post(port: Int, path: String, body: [String: Any]) async throws -> [String: Any] {
        let url = try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)\(path)"))
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 900
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await URLSession.shared.data(for: request)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(http.statusCode, 200, "HTTP \(http.statusCode): \(String(decoding: data, as: UTF8.self))")
        return try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    func testCountEqualsGenerationPromptTokens() async throws {
        let (modelID, directory) = try gateAndResolveModel()
        let engine = MLXSwiftEngine()
        try await engine.load(LocalModel(
            id: modelID, displayName: modelID, directory: directory, sizeBytes: 0,
            format: .mlx, quantization: nil, parameterCount: nil, architecture: nil))
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 22_600)
        defer { Task { await server.stop() } }

        let body: [String: Any] = [
            "model": modelID,
            "max_tokens": 1,
            "system": [
                ["type": "text", "text": "Run \(UUID().uuidString)."],
                ["type": "text", "text": "You are a coding agent. Use the tools when they help."],
            ],
            "tools": [[
                "name": "Bash",
                "description": "Run a shell command.",
                "input_schema": [
                    "type": "object",
                    "properties": ["command": ["type": "string", "description": "The command"]],
                    "required": ["command"],
                ],
            ]],
            "messages": [
                ["role": "user", "content": [["type": "text", "text": "Print hi with echo."]]],
                ["role": "system", "content": [["type": "text", "text": "<system-reminder>Be brief.</system-reminder>"]]],
            ],
        ]

        let counted = try await post(port: port, path: "/v1/messages/count_tokens", body: body)
        let inputTokens = try XCTUnwrap(counted["input_tokens"] as? Int)
        let message = try await post(port: port, path: "/v1/messages", body: body)
        let usage = try XCTUnwrap(message["usage"] as? [String: Any])
        let uncached = try XCTUnwrap(usage["input_tokens"] as? Int)
        let cached = usage["cache_read_input_tokens"] as? Int ?? 0
        print("ANTHROPIC_COUNT_TOKENS count=\(inputTokens) generation=\(uncached)+\(cached)")
        XCTAssertGreaterThan(inputTokens, 0)
        XCTAssertEqual(inputTokens, uncached + cached)
    }
}
