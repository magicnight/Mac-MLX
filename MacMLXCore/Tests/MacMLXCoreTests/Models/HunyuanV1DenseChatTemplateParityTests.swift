// Copyright © 2026 macMLX. English comments only.

import Foundation
import Jinja
import XCTest

@testable import MacMLXCore

/// Render-parity gate proving Hunyuan V1 Dense needs NO built-in chat-template
/// override: the checkpoint's OWN `chat_template.jinja` renders byte-for-byte
/// under swift-jinja, on both the standard conversation path AND the historical
/// `<answer>`-stripping branch.
///
/// The Hunyuan template uses `namespace`, the message loop, string concatenation,
/// `in`, `loop.last`, and tokenizer-injected `bos_token`/`eos_token` — all
/// swift-jinja constructs. The one construct that once rendered differently,
/// `content.split('<answer>')[-1].strip('</answer>').strip()` behind `'<answer>'
/// in content and not loop.last` (a HISTORICAL assistant turn embedding `<answer>`
/// tags), was blocked by swift-jinja 2.3.6's argument-ignoring `.strip()`;
/// swift-jinja 2.4.0 fixes `strip(arg)` (huggingface/swift-jinja #64, reported by
/// macMLX), so it now renders identically too. This test renders the UNMODIFIED
/// template (loaded from the fixture, exactly as it ships) via swift-jinja — the
/// same engine swift-transformers drives in production, configured identically
/// with `lstripBlocks: true, trimBlocks: true` — and asserts equality against
/// renders captured from the ORIGINAL template by
/// `docs/reference/capture_hunyuan_v1_dense_chat_template.py` (jinja2,
/// cross-checked == transformers `apply_chat_template`).
///
/// COVERAGE: the standard conversation path plus the `answer_history` case, which
/// replays a non-last `<answer>`-tagged assistant turn to exercise the
/// `strip('</answer>')` branch. (Seed-OSS's integer-keyed dict — huggingface/
/// swift-jinja #62 — and Command R7B's literal `}}` — #63 — are likewise fixed in
/// 2.4.0 and rendered natively by their own parity tests, so no model in the
/// matrix needs a built-in override.)
///
/// UNGATED — needs no model weights and no Metal, so it runs under both bare
/// `swift test` and xcodebuild.
///
/// FIDELITY CAVEAT (same as `SeedOssChatTemplateParityTests`): these tests render
/// via a BARE `Jinja.Template`/`Value` context (messages, add_generation_prompt,
/// bos_token, eos_token, tools) rather than the swift-transformers
/// `Tokenizer.applyChatTemplate` wrapper production uses. That is faithful because
/// the Hunyuan template reads only those context keys; `bos_token`/`eos_token` are
/// supplied here exactly as the tokenizer injects them (captured into the
/// fixture). The real-weights `HunyuanV1DenseSmokeTests` proves the end-to-end
/// tokenizer path.
final class HunyuanV1DenseChatTemplateParityTests: XCTestCase {

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
                forResource: "hunyuan_v1_dense_chat_template_fixture",
                withExtension: "json",
                subdirectory: "Fixtures"),
            "hunyuan_v1_dense_chat_template_fixture.json missing from the test bundle")
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }

    /// Build the Jinja render context the way swift-transformers does: `messages`
    /// as an array of objects, `add_generation_prompt` as a boolean, and the
    /// tokenizer-injected `bos_token`/`eos_token` the Hunyuan template reads.
    /// `tools` is `.null` (transformers' `tools=None` default) so the tool block —
    /// which the standard path never emits — is skipped identically.
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
            let context = try makeContext(testCase, fixture: fixture)
            let rendered = try template.render(context)
            XCTAssertEqual(
                rendered, testCase.expected,
                "swift-jinja render of the checkpoint template diverged from the "
                    + "transformers reference for case '\(testCase.name)'")
        }
    }
}
