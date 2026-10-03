// Copyright © 2026 macMLX. English comments only.

import Foundation
import Jinja
import XCTest

@testable import MacMLXCore

/// Render-parity gate proving Cohere2 / Command R7B needs NO built-in
/// chat-template override as of swift-jinja 2.4.0: the checkpoint's OWN `default`
/// named template — including the off-path tool/RAG branch — parses and renders
/// byte-for-byte under swift-jinja on the standard conversation path.
///
/// The checkpoint ships its `chat_template` as a LIST of named templates; the
/// `default` one (selected when no tools/documents) embeds a tool/RAG branch that
/// swift-jinja 2.3.6 could not PARSE ("Unexpected token type: closeExpression", a
/// literal `}}` misparse — huggingface/swift-jinja #63, reported by macMLX), so
/// the whole template failed to compile even though that branch is off the
/// standard path. That forced a built-in override; 2.4.0 fixes #63, so the full
/// `default` template compiles and the standard path renders natively — this test
/// is the standing proof, matching the Hunyuan V1 Dense / MiniCPM3 native-render
/// precedent rather than the former override-parity shape.
///
/// It renders the UNMODIFIED `default` template (loaded from the fixture, exactly
/// as it ships) via swift-jinja — the same engine swift-transformers drives in
/// production, configured identically with `lstripBlocks: true, trimBlocks: true`
/// — and asserts byte-for-byte equality against a fixture captured from that
/// ORIGINAL `default` template by
/// `docs/reference/capture_cohere2_chat_template.py` (jinja2 with `loopcontrols`,
/// cross-checked == transformers `apply_chat_template`).
///
/// SCOPE: only the standard conversation path is covered (all cases pass
/// `documents = null`). The tool/RAG branch — which transformers routes to the
/// SEPARATE `tool_use` / `rag` named templates, never the `default` one — is out
/// of scope for this test.
///
/// UNGATED — needs no model weights and no Metal, so it runs under both bare
/// `swift test` and xcodebuild. The real-weights `Cohere2SmokeTests` proves the
/// end-to-end tokenizer path.
///
/// FIDELITY CAVEAT (same as `SeedOssChatTemplateParityTests`): these tests render
/// via a BARE `Jinja.Template`/`Value` context (messages, add_generation_prompt,
/// documents, tools) rather than the swift-transformers `Tokenizer.
/// applyChatTemplate` wrapper production uses. That is faithful because the Command
/// R7B `default` template reads only those context keys (it references neither
/// `bos_token` nor `eos_token`).
final class Cohere2ChatTemplateParityTests: XCTestCase {

    private struct Fixture: Decodable {
        /// The checkpoint's OWN `default` named template, rendered directly by
        /// swift-jinja (no macMLX override) — the exact string the tokenizer sees.
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

        enum CodingKeys: String, CodingKey {
            case template, cases
        }
    }

    private func loadFixture() throws -> Fixture {
        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "cohere2_chat_template_fixture",
                withExtension: "json",
                subdirectory: "Fixtures"),
            "cohere2_chat_template_fixture.json missing from the test bundle")
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }

    /// Build the Jinja render context the way swift-transformers does: `messages`
    /// as an array of objects and `add_generation_prompt` as a boolean.
    /// `documents`/`tools` are `.null` (transformers' `None` default) so the
    /// tool/RAG branch is skipped identically.
    private func makeContext(_ testCase: Fixture.Case) throws -> [String: Jinja.Value] {
        let anyMessages: [Any?] = testCase.messages.map { message in
            var dict: [String: Any?] = [:]
            for (key, value) in message { dict[key] = value }
            return dict
        }
        return [
            "messages": try Jinja.Value(any: anyMessages),
            "add_generation_prompt": .boolean(testCase.addGenerationPrompt),
            "documents": .null,
            "tools": .null,
        ]
    }

    func testCheckpointTemplateRendersNativelyUnderSwiftJinja() throws {
        let fixture = try loadFixture()
        XCTAssertFalse(fixture.cases.isEmpty, "fixture must carry at least one case")

        // Same options swift-transformers uses (Tokenizer.compiledTemplate):
        // `Template(_, with: .init(lstripBlocks: true, trimBlocks: true))`.
        let template = try Jinja.Template(
            fixture.template,
            with: .init(lstripBlocks: true, trimBlocks: true))

        for testCase in fixture.cases {
            let context = try makeContext(testCase)
            let rendered = try template.render(context)
            XCTAssertEqual(
                rendered, testCase.expected,
                "swift-jinja render of the checkpoint `default` template diverged "
                    + "from the transformers reference for case '\(testCase.name)'")
        }
    }
}
