// Copyright © 2026 macMLX. English comments only.

import Foundation
import Jinja
import XCTest

@testable import MacMLXCore

/// Render-parity gate proving Seed-OSS needs NO built-in chat-template override as
/// of swift-jinja 2.4.0: the checkpoint's OWN `chat_template.jinja` renders
/// byte-for-byte under swift-jinja on every path these cases exercise.
///
/// The checkpoint builds its thinking-budget reflection-interval table as a Jinja
/// object literal with INTEGER keys (`{0: 0, 512: 128, ...}`) consumed by a
/// `dictsort` threshold search. swift-jinja 2.3.6 could not parse numeric object
/// keys (huggingface/swift-jinja #62, reported by macMLX), which forced a built-in
/// override; 2.4.0 fixes #62, so the stock template parses and renders natively —
/// this test is the standing proof, matching the Hunyuan V1 Dense / MiniCPM3
/// native-render precedent rather than the former override-parity shape.
///
/// It renders the UNMODIFIED template (loaded from the fixture, exactly as it
/// ships) via swift-jinja — the same engine swift-transformers drives in
/// production, configured identically with `lstripBlocks: true, trimBlocks: true`
/// — and asserts byte-for-byte equality against a fixture captured from that
/// ORIGINAL template by `docs/reference/capture_seed_oss_chat_template.py`
/// (jinja2, cross-checked == transformers `apply_chat_template`).
///
/// UNGATED — needs no model weights and no Metal, so it runs under both bare
/// `swift test` and xcodebuild. The real-weights `SeedOssSmokeTests` proves the
/// end-to-end tokenizer path.
///
/// FIDELITY CAVEAT: these tests render via a BARE `Jinja.Template`/`Value` context
/// built directly from the fixture (messages, `add_generation_prompt`,
/// `thinking_budget`) — not through the swift-transformers `Tokenizer.
/// applyChatTemplate` wrapper that production uses (see
/// `HuggingFaceTokenizerLoader`). That is faithful here because the Seed-OSS
/// template is fully self-contained: it `{%- set -%}`s all its own special tokens
/// (`bos_token`, `eos_token`, …) and never reads a tokenizer-injected global. A
/// future template that relies on swift-transformers-injected context (e.g.
/// tokenizer special-token attributes merged into the render context — see
/// `Tokenizer.applyChatTemplate`'s `tokenizerConfig.dictionary(or:)` loop) would
/// need its parity test to go through that wrapper instead, or this
/// bare-environment render would silently diverge from what the tokenizer actually
/// produces.
final class SeedOssChatTemplateParityTests: XCTestCase {

    private struct Fixture: Decodable {
        /// The checkpoint's OWN `chat_template.jinja`, rendered directly by
        /// swift-jinja (no macMLX override) — the exact string the tokenizer sees.
        let template: String
        let cases: [Case]
        let boundaryCases: [Case]
        struct Case: Decodable {
            let name: String
            let messages: [[String: String]]
            let addGenerationPrompt: Bool
            let thinkingBudget: Int?
            let expected: String

            enum CodingKeys: String, CodingKey {
                case name, messages, expected
                case addGenerationPrompt = "add_generation_prompt"
                case thinkingBudget = "thinking_budget"
            }
        }

        enum CodingKeys: String, CodingKey {
            case template, cases
            case boundaryCases = "boundary_cases"
        }
    }

    private func loadFixture() throws -> Fixture {
        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "seed_oss_chat_template_fixture",
                withExtension: "json",
                subdirectory: "Fixtures"),
            "seed_oss_chat_template_fixture.json missing from the test bundle")
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }

    /// Build the Jinja render context the same way swift-transformers does:
    /// `messages` as an array of objects, `add_generation_prompt` as a boolean,
    /// and `thinking_budget` (when set) as an integer. The Seed-OSS template
    /// sets all its own special tokens, so no other context is needed.
    private func makeContext(_ testCase: Fixture.Case) throws -> [String: Jinja.Value] {
        let anyMessages: [Any?] = testCase.messages.map { message in
            var dict: [String: Any?] = [:]
            for (key, value) in message { dict[key] = value }
            return dict
        }
        var context: [String: Jinja.Value] = [
            "messages": try Jinja.Value(any: anyMessages),
            "add_generation_prompt": .boolean(testCase.addGenerationPrompt),
        ]
        if let budget = testCase.thinkingBudget {
            context["thinking_budget"] = .int(budget)
        }
        return context
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
                "swift-jinja render of the checkpoint template diverged from the "
                    + "transformers reference for case '\(testCase.name)'")
        }
    }

    /// PRIMARY proof for the thinking-budget reflection-interval lookup: renders
    /// the checkpoint template via swift-jinja for every tier boundary (±1) — plus
    /// 0, a negative budget, and one past-top value — and asserts FULL-STRING
    /// equality against `boundary_cases` in the fixture. Those fixture strings are
    /// the ORIGINAL template's own render (its integer-keyed dict + `dictsort`
    /// search) for the identical budgets, captured by
    /// `docs/reference/capture_seed_oss_chat_template.py` and cross-checked there
    /// against `transformers.apply_chat_template`. This proves swift-jinja 2.4.0
    /// renders the integer-keyed dict + `dictsort` lookup (huggingface/swift-jinja
    /// #62) natively and identically for every budget.
    func testThinkingBudgetBoundariesMatchCheckpointTemplate() throws {
        let fixture = try loadFixture()
        XCTAssertFalse(
            fixture.boundaryCases.isEmpty, "fixture must carry boundary_cases")

        let template = try Jinja.Template(
            fixture.template,
            with: .init(lstripBlocks: true, trimBlocks: true))

        for testCase in fixture.boundaryCases {
            let context = try makeContext(testCase)
            let rendered = try template.render(context)
            XCTAssertEqual(
                rendered, testCase.expected,
                "swift-jinja render of the checkpoint template diverged from the "
                    + "reference at thinking_budget="
                    + "\(testCase.thinkingBudget.map(String.init) ?? "nil") "
                    + "(case '\(testCase.name)')")
        }
    }

    /// SECONDARY sanity, redundant with `testThinkingBudgetBoundariesMatchCheckpointTemplate`
    /// above (which is the real proof, against the checkpoint template's own render
    /// via the fixture). This checks the rendered reflection interval against a
    /// hand-maintained ladder transcribed separately from both the template and the
    /// fixture generator, so it does not share a mistranscription with either — an
    /// independent trip-wire, not the parity proof itself.
    func testThinkingBudgetIntervalLadderBoundaries() throws {
        let fixture = try loadFixture()
        let template = try Jinja.Template(
            fixture.template,
            with: .init(lstripBlocks: true, trimBlocks: true))

        // (thinking_budget, expected reflection interval) — mirrors the upstream
        // {0:0, 512:128, 1024:256, 2048:512, 4096:512, 8192:1024, 16384:1024}
        // sorted-threshold search (first tier whose key >= budget), with the
        // >16384 fallback to 1024. budget 0 and -1 take other template branches
        // (no interval emitted) and are covered by the render-parity fixture.
        let expectations: [(budget: Int, interval: Int)] = [
            (1, 128), (512, 128),
            (513, 256), (1024, 256),
            (1025, 512), (2048, 512),
            (2049, 512), (4096, 512),
            (4097, 1024), (8192, 1024),
            (8193, 1024), (16384, 1024),
            (16385, 1024), (100000, 1024),
        ]

        let messages: [Any?] = [
            ["role": "user", "content": "hi"] as [String: Any?]
        ]
        for (budget, interval) in expectations {
            let context: [String: Jinja.Value] = [
                "messages": try Jinja.Value(any: messages),
                "add_generation_prompt": .boolean(true),
                "thinking_budget": .int(budget),
            ]
            let rendered = try template.render(context)
            XCTAssertTrue(
                rendered.contains("every \(interval) tokens"),
                "thinking_budget=\(budget) must emit reflection interval \(interval); "
                    + "rendered: \(rendered)")
        }
    }
}
