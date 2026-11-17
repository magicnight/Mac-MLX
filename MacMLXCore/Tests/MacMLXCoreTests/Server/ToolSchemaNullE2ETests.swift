// Copyright © 2026 macMLX. English comments only.

import Foundation
import XCTest

@testable import MacMLXCore

/// Gated, real-model check that a JSON null in a request reaches the model's
/// own chat template, driven through the real `HummingbirdServer` over HTTP:
/// the tool of issue #175 (`"default": null` in a parameter schema, which
/// Zed's agent sends), and a replayed tool call whose arguments hold a null.
/// Both answered HTTP 500, "Cannot convert value of type NSNull to Jinja
/// Value".
///
/// GATED — never runs in CI. Self-skips unless ALL hold:
///   1. `requireMLXRuntimeOrSkip()` passes (real Metal, i.e. xcodebuild),
///   2. env `MACMLX_RUN_TOOL_NULL_E2E=1`, and
///   3. the model directory exists (env `MACMLX_TOOL_NULL_MODEL` overrides
///      the dir name under `~/.mac-mlx/models`; default
///      `gemma-4-e4b-it-8bit`).
///
/// Run:
///   MACMLX_RUN_TOOL_NULL_E2E=1 TEST_RUNNER_MACMLX_RUN_TOOL_NULL_E2E=1 \
///     xcodebuild test -scheme MacMLXCore -destination 'platform=macOS' \
///     -skipPackagePluginValidation \
///     -only-testing:MacMLXCoreTests/ToolSchemaNullE2ETests
final class ToolSchemaNullE2ETests: XCTestCase {

    private func gateAndResolveModel() throws -> (id: String, directory: URL) {
        try requireMLXRuntimeOrSkip()
        guard ProcessInfo.processInfo.environment["MACMLX_RUN_TOOL_NULL_E2E"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_TOOL_NULL_E2E=1 to run the tool null E2E test")
        }
        let modelID = ProcessInfo.processInfo.environment["MACMLX_TOOL_NULL_MODEL"] ?? "gemma-4-e4b-it-8bit"
        let directory = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appending(path: ".mac-mlx/models/\(modelID)", directoryHint: .isDirectory)
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw XCTSkip("tool null model dir not found: \(directory.path)")
        }
        return (modelID, directory)
    }

    private func post(port: Int, body: [String: Any]) async throws -> (status: Int, text: String) {
        let url = try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)/v1/chat/completions"))
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 900
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await URLSession.shared.data(for: request)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        return (http.statusCode, String(decoding: data, as: UTF8.self))
    }

    private func promptTokens(_ text: String) throws -> Int {
        let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
        let usage = try XCTUnwrap(object["usage"] as? [String: Any])
        return try XCTUnwrap(usage["prompt_tokens"] as? Int)
    }

    func testNullsReachTheTemplate() async throws {
        let (modelID, directory) = try gateAndResolveModel()
        let engine = MLXSwiftEngine()
        try await GatedLoad.load(engine, id: modelID, directory: directory)
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 22_640)
        defer { Task { await server.stop() } }

        func tool(property: [String: Any]) -> [String: Any] {
            [
                "type": "function",
                "function": [
                    "name": "x",
                    "parameters": ["type": "object", "properties": ["a": property]],
                ],
            ]
        }
        func body(tools: [[String: Any]], messages: [[String: Any]]) -> [String: Any] {
            ["model": modelID, "max_tokens": 5, "messages": messages, "tools": tools]
        }
        let hi: [[String: Any]] = [["role": "user", "content": "hi"]]

        // Issue #175's two requests: the schema without the null, then with it.
        let plain = try await post(port: port, body: body(tools: [tool(property: ["type": "string"])], messages: hi))
        XCTAssertEqual(plain.status, 200, plain.text)
        let withNull = try await post(
            port: port,
            body: body(tools: [tool(property: ["type": "string", "default": NSNull()])], messages: hi))
        XCTAssertEqual(withNull.status, 200, withNull.text)
        // The null is rendered, not dropped: the prompt is longer by it.
        let plainTokens = try promptTokens(plain.text)
        let nullTokens = try promptTokens(withNull.text)
        print("TOOL_NULL \(modelID) schema prompt_tokens plain=\(plainTokens) null=\(nullTokens)")
        XCTAssertGreaterThan(nullTokens, plainTokens)

        // A replayed tool call with a null argument.
        func replay(arguments: String) -> [[String: Any]] {
            [
                ["role": "user", "content": "Call x."],
                [
                    "role": "assistant", "content": "",
                    "tool_calls": [[
                        "id": "c1", "type": "function",
                        "function": ["name": "x", "arguments": arguments],
                    ]],
                ],
                ["role": "tool", "tool_call_id": "c1", "content": "done"],
            ]
        }
        let tools = [tool(property: ["type": "string"])]
        let replayed = try await post(port: port, body: body(tools: tools, messages: replay(arguments: #"{"a":"v"}"#)))
        XCTAssertEqual(replayed.status, 200, replayed.text)
        let replayedNull = try await post(
            port: port, body: body(tools: tools, messages: replay(arguments: #"{"a":null,"b":[1,null]}"#)))
        XCTAssertEqual(replayedNull.status, 200, replayedNull.text)
        print(
            "TOOL_NULL \(modelID) replay prompt_tokens plain=\(try promptTokens(replayed.text)) "
                + "null=\(try promptTokens(replayedNull.text))")
    }
}
