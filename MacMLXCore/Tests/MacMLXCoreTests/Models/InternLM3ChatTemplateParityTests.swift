// Copyright © 2026 macMLX. English comments only.

import Foundation
import Jinja
import XCTest

@testable import MacMLXCore

/// Render-parity gate proving InternLM3 needs NO built-in chat-template override: the
/// checkpoint's OWN `chat_template` (shipped in `tokenizer_config.json`) renders
/// byte-for-byte under swift-jinja — the Hunyuan V1 Dense / MiniCPM3 precedent, NOT
/// the Seed-OSS / Cohere2 one.
///
/// InternLM3's template is plain ChatML — `{{ bos_token }}` then, per message,
/// `<|im_start|>role\ncontent<|im_end|>\n`, then an optional `<|im_start|>assistant\n`
/// generation prompt. It uses only constructs swift-jinja fully supports (the message
/// loop, string concatenation, `add_generation_prompt`, and the tokenizer-injected
/// `bos_token`), so it renders identically on every path. This test renders the
/// UNMODIFIED template (loaded from the fixture, exactly as it ships) via swift-jinja
/// — the same engine swift-transformers drives in production, configured identically
/// with `lstripBlocks: true, trimBlocks: true` — and asserts equality against renders
/// captured from the ORIGINAL template by
/// `docs/reference/capture_internlm3_chat_template.py` (jinja2, trim_blocks +
/// lstrip_blocks).
///
/// UNGATED — needs no model weights and no Metal, so it runs under both bare
/// `swift test` and xcodebuild.
///
/// FIDELITY CAVEAT (same as the Hunyuan / MiniCPM3 chat-template tests): these render
/// via a BARE `Jinja.Template`/`Value` context (messages, add_generation_prompt,
/// bos_token, eos_token) rather than the swift-transformers `Tokenizer.applyChatTemplate`
/// wrapper production uses. That is faithful because the InternLM3 template reads only
/// those context keys; `bos_token`/`eos_token` are supplied here exactly as the
/// tokenizer injects them (captured into the fixture). The real-weights
/// `InternLM3SmokeTests` proves the end-to-end tokenizer path.
final class InternLM3ChatTemplateParityTests: XCTestCase {

    private struct Fixture: Decodable {
        let template: String
        let bosToken: String
        let eosToken: String
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
            case bosToken = "bos_token"
            case eosToken = "eos_token"
        }
    }

    private func loadFixture() throws -> Fixture {
        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "internlm3_chat_template_fixture",
                withExtension: "json",
                subdirectory: "Fixtures"),
            "internlm3_chat_template_fixture.json missing from the test bundle")
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }

    /// Build the Jinja render context the way swift-transformers does: `messages` as
    /// an array of objects, `add_generation_prompt` as a boolean, and the
    /// tokenizer-injected `bos_token`/`eos_token` (the template reads `bos_token`).
    private func makeContext(_ testCase: Fixture.Case, fixture: Fixture) throws
        -> [String: Jinja.Value]
    {
        let anyMessages: [Any?] = testCase.messages.map { message in
            var dict: [String: Any?] = [:]
            for (key, value) in message { dict[key] = value }
            return dict
        }
        return [
            "messages": try Jinja.Value(any: anyMessages),
            "add_generation_prompt": .boolean(testCase.addGenerationPrompt),
            "bos_token": .string(fixture.bosToken),
            "eos_token": .string(fixture.eosToken),
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
            let context = try makeContext(testCase, fixture: fixture)
            let rendered = try template.render(context)
            XCTAssertEqual(
                rendered, testCase.expected,
                "swift-jinja render of the checkpoint template diverged from the "
                    + "jinja2 reference for case '\(testCase.name)'")
        }
    }
}
