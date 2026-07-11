// Copyright © 2026 macMLX. English comments only.

import Foundation
import Jinja
import XCTest

@testable import MacMLXCore

/// Render-parity gate proving the MiniCPM3 checkpoint's OWN `chat_template` renders
/// correctly under swift-jinja on the STANDARD conversation path, so NO built-in
/// chat-template override is needed — the Hunyuan V1 Dense precedent, NOT the
/// Cohere2 / Seed-OSS one.
///
/// The checkpoint ships a heavy tool-use `chat_template` (recursive Jinja macros
/// with `{% call %}`/`caller()`, `|items`, `|tojson`, `|title`, `is iterable`), but
/// on the standard path (`tools = null`) those macros are DEFINED yet never
/// invoked, and swift-jinja 2.3.6 parses and renders the whole template cleanly.
/// This test renders the SAME representative message sets through swift-jinja (the
/// exact engine swift-transformers drives in production, configured identically
/// with `lstripBlocks: true, trimBlocks: true`) and asserts byte-for-byte equality
/// against a fixture captured from that ORIGINAL template by
/// `docs/reference/capture_minicpm3_chat_template.py` (jinja2, cross-checked ==
/// transformers `apply_chat_template`).
///
/// SCOPE: only the standard conversation path (no tools / tool_calls / thought).
/// The tool-use branches would require the macro machinery this test does not
/// exercise; if a future need arises to render them, a built-in override (or a
/// per-model `macmlx.chat_template.jinja`) would be the route — today it is
/// intentionally out of scope.
///
/// UNGATED — needs no model weights and no Metal, so it runs under both bare
/// `swift test` and xcodebuild.
final class MiniCPM3ChatTemplateParityTests: XCTestCase {

    private struct Fixture: Decodable {
        let template: String
        let cases: [Case]

        struct Case: Decodable {
            let name: String
            let messages: [[String: String]]
            let addGenerationPrompt: Bool
            let expected: String

            enum CodingKeys: String, CodingKey {
                case name, messages, expected
                case addGenerationPrompt = "add_generation_prompt"
            }
        }
    }

    private func loadFixture() throws -> Fixture {
        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "minicpm3_chat_template_fixture",
                withExtension: "json",
                subdirectory: "Fixtures"),
            "minicpm3_chat_template_fixture.json missing from the test bundle")
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }

    /// Build the Jinja render context the way swift-transformers does: `messages`
    /// as an array of objects, `add_generation_prompt` as a boolean, `tools` as
    /// `.null` (transformers' `None` default) so the tool branch is skipped.
    private func makeContext(_ testCase: Fixture.Case) throws -> [String: Jinja.Value] {
        let anyMessages: [Any?] = testCase.messages.map { message in
            var dict: [String: Any?] = [:]
            for (key, value) in message { dict[key] = value }
            return dict
        }
        return [
            "messages": try Jinja.Value(any: anyMessages),
            "add_generation_prompt": .boolean(testCase.addGenerationPrompt),
            "tools": .null,
        ]
    }

    func testRendersIdenticallyToOriginal() throws {
        let fixture = try loadFixture()
        XCTAssertFalse(fixture.cases.isEmpty, "fixture must carry at least one case")

        // Render the checkpoint's OWN template (native — no override needed), with
        // the same options swift-transformers uses (Tokenizer.compiledTemplate).
        let template = try Jinja.Template(
            fixture.template,
            with: .init(lstripBlocks: true, trimBlocks: true))

        for testCase in fixture.cases {
            let context = try makeContext(testCase)
            let rendered = try template.render(context)
            XCTAssertEqual(
                rendered, testCase.expected,
                "render diverged from the original checkpoint template for "
                    + "case '\(testCase.name)'")
        }
    }
}
