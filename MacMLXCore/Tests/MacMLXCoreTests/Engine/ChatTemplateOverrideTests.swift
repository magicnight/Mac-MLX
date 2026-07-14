// Copyright © 2026 macMLX. English comments only.

import Foundation
import XCTest

@testable import MacMLXCore

/// Resolution-order tests for `ChatTemplateOverride`.
///
/// The load-time seam (`HuggingFaceTokenizerLoader`) applies an override ONLY
/// when `resolve` returns non-nil; a `nil` result means the checkpoint's own
/// template reaches the tokenizer unchanged. These tests pin that logic — in
/// particular the NO-OVERRIDE regression (a model whose `model_type` has no
/// built-in and which carries no user file is byte-for-byte unaffected) — plus
/// the built-in and user-file precedence rules. Pure and ungated (no weights,
/// no Metal).
///
/// The SHIPPING `builtIns` map is EMPTY as of swift-jinja 2.4.0 (Seed-OSS and
/// Command R7B render natively, so both built-ins were removed). To keep the
/// built-in resolution branch covered, the built-in tests inject a synthetic
/// `testBuiltIns` map through the `builtIns:` test seam — the same "injectable for
/// tests" idiom as `fileManager`/`modelType`; `testShippingBuiltInsAreEmptyForFormerlyOverriddenType`
/// pins that the default map ships empty.
final class ChatTemplateOverrideTests: XCTestCase {

    private var tempDir: URL!

    /// Synthetic built-in map for exercising the built-in resolution branch, which
    /// ships EMPTY (`ChatTemplateOverride.builtIns == [:]`) now that swift-jinja
    /// 2.4.0 renders every previously-overridden checkpoint natively.
    private let testBuiltIns: [String: String] = [
        "seed_oss": "{{ bos_token }}TEST BUILT-IN seed_oss{{ eos_token }}"
    ]

    override func setUpWithError() throws {
        tempDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("chat-template-override-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let tempDir { try? FileManager.default.removeItem(at: tempDir) }
    }

    private func writeConfig(modelType: String) throws {
        let json = "{\"model_type\": \"\(modelType)\"}"
        try json.write(
            to: tempDir.appendingPathComponent("config.json"),
            atomically: true, encoding: .utf8)
    }

    private func writeUserOverride(_ text: String) throws {
        try text.write(
            to: tempDir.appendingPathComponent(ChatTemplateOverride.userOverrideFilename),
            atomically: true, encoding: .utf8)
    }

    /// NO-OVERRIDE REGRESSION: a model whose `model_type` has no registered
    /// built-in, and which carries no user file, resolves to `nil` — so the
    /// checkpoint's own chat template is used unchanged, exactly as before this
    /// feature.
    func testNoOverrideForUnregisteredModelType() throws {
        try writeConfig(modelType: "llama")
        XCTAssertNil(ChatTemplateOverride.resolve(modelDirectory: tempDir))
    }

    /// A directory with neither `config.json` nor a user file also resolves to
    /// `nil` (the loader still hands the checkpoint template through untouched).
    func testNoOverrideWhenNothingPresent() throws {
        XCTAssertNil(ChatTemplateOverride.resolve(modelDirectory: tempDir))
    }

    /// SHIPPING built-ins are empty: `seed_oss` and `cohere2` — whose built-ins
    /// were removed once swift-jinja 2.4.0 rendered their checkpoint templates
    /// natively — resolve to `nil` under the DEFAULT `builtIns`, so each
    /// checkpoint's own template is used unchanged. Guards the removal itself.
    func testShippingBuiltInsAreEmptyForFormerlyOverriddenType() throws {
        try writeConfig(modelType: "seed_oss")
        XCTAssertNil(ChatTemplateOverride.resolve(modelDirectory: tempDir))
        try writeConfig(modelType: "cohere2")
        XCTAssertNil(ChatTemplateOverride.resolve(modelDirectory: tempDir))
    }

    /// A `seed_oss` checkpoint (no user file) resolves to the INJECTED built-in
    /// override. The shipping `builtIns` is empty (see
    /// `testShippingBuiltInsAreEmptyForFormerlyOverriddenType`); this exercises the
    /// resolution branch via the `builtIns:` test seam.
    func testBuiltInOverrideForSeedOss() throws {
        try writeConfig(modelType: "seed_oss")
        let resolved = try XCTUnwrap(
            ChatTemplateOverride.resolve(modelDirectory: tempDir, builtIns: testBuiltIns))
        XCTAssertEqual(resolved.template, testBuiltIns["seed_oss"])
        XCTAssertEqual(resolved.source, "built-in seed_oss")
    }

    /// `model_type` matching is case-insensitive (`ModelConfigInfo` lowercases).
    func testBuiltInOverrideMatchesCaseInsensitively() throws {
        try writeConfig(modelType: "SEED_OSS")
        let resolved = try XCTUnwrap(
            ChatTemplateOverride.resolve(modelDirectory: tempDir, builtIns: testBuiltIns))
        XCTAssertEqual(resolved.template, testBuiltIns["seed_oss"])
    }

    /// A user file `macmlx.chat_template.jinja` takes precedence over a built-in
    /// override for the same `model_type` (built-in injected via the test seam).
    func testUserFileOverridesBuiltIn() throws {
        try writeConfig(modelType: "seed_oss")
        let custom = "{{ bos_token }}custom user template{{ eos_token }}"
        try writeUserOverride(custom)
        let resolved = try XCTUnwrap(
            ChatTemplateOverride.resolve(modelDirectory: tempDir, builtIns: testBuiltIns))
        XCTAssertEqual(resolved.template, custom)
        XCTAssertEqual(
            resolved.source, "user file \(ChatTemplateOverride.userOverrideFilename)")
    }

    /// A user file wins even when no built-in exists for the model_type.
    func testUserFileOverrideForUnregisteredModelType() throws {
        try writeConfig(modelType: "llama")
        let custom = "custom template body"
        try writeUserOverride(custom)
        let resolved = try XCTUnwrap(ChatTemplateOverride.resolve(modelDirectory: tempDir))
        XCTAssertEqual(resolved.template, custom)
        XCTAssertTrue(resolved.source.hasPrefix("user file "))
    }

    /// An empty user file is ignored (falls through to the built-in / nil),
    /// rather than silently forcing an empty template. Here it falls through to
    /// the injected built-in.
    func testEmptyUserFileIsIgnored() throws {
        try writeConfig(modelType: "seed_oss")
        try writeUserOverride("")
        let resolved = try XCTUnwrap(
            ChatTemplateOverride.resolve(modelDirectory: tempDir, builtIns: testBuiltIns))
        XCTAssertEqual(resolved.template, testBuiltIns["seed_oss"])
        XCTAssertEqual(resolved.source, "built-in seed_oss")
    }

    // MARK: - resolveDetailed: "never silent" diagnostics for a broken user file
    //
    // `HuggingFaceTokenizerLoader.load` uses `resolveDetailed` (not `resolve`) so
    // it can log a warning when the user override file is PRESENT but unusable,
    // rather than silently falling through the way an ABSENT file does. These
    // tests assert the diagnostic as data (`skippedUserFileReason`) — the seam
    // that's actually testable without standing up a `LogManager` + `LoggerStore`
    // harness for this one call site. The warning log line itself
    // (`LogManager.shared.warning(...)` in `HuggingFaceTokenizerLoader.swift`) is
    // manual-verified by inspection: it fires exactly when `skippedUserFileReason`
    // is non-nil, using the same `warning(_:category:)` API already exercised by
    // `LogManagerTests`. The fall-back target here is the INJECTED built-in (via
    // the `builtIns:` test seam), since the shipping map is empty.

    /// A user file that exists but is not valid UTF-8 is skipped (not silently —
    /// `skippedUserFileReason` reports why) and resolution falls back to the
    /// built-in for this `model_type`.
    func testUserFileNonUTF8IsSkippedWithDiagnosis() throws {
        try writeConfig(modelType: "seed_oss")
        // 0xFF and 0xFE are never valid UTF-8 lead bytes, so this cannot decode.
        let invalidUTF8 = Data([0xFF, 0xFE, 0x00])
        try invalidUTF8.write(to: tempDir.appendingPathComponent(ChatTemplateOverride.userOverrideFilename))

        let resolution = ChatTemplateOverride.resolveDetailed(
            modelDirectory: tempDir, builtIns: testBuiltIns)
        XCTAssertEqual(resolution.skippedUserFileReason, "not valid UTF-8")
        XCTAssertEqual(resolution.resolved?.template, testBuiltIns["seed_oss"])
        XCTAssertEqual(resolution.resolved?.source, "built-in seed_oss")
    }

    /// A user file that exists but has no read permission is skipped and
    /// diagnosed as `"unreadable"`, falling back to the built-in.
    func testUserFileUnreadablePermsIsSkippedWithDiagnosis() throws {
        try writeConfig(modelType: "seed_oss")
        let userFile = tempDir.appendingPathComponent(ChatTemplateOverride.userOverrideFilename)
        try "{{ bos_token }}unreadable{{ eos_token }}".write(
            to: userFile, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o000], ofItemAtPath: userFile.path)
        // Restore read permission so `tearDownWithError`'s `removeItem` can
        // delete the temp directory regardless of test outcome.
        defer {
            try? FileManager.default.setAttributes(
                [.posixPermissions: 0o644], ofItemAtPath: userFile.path)
        }
        guard getuid() != 0 else {
            throw XCTSkip(
                "Running as root — permission bits do not restrict reads, so the "
                    + "unreadable-file path cannot be exercised")
        }

        let resolution = ChatTemplateOverride.resolveDetailed(
            modelDirectory: tempDir, builtIns: testBuiltIns)
        XCTAssertEqual(resolution.skippedUserFileReason, "unreadable")
        XCTAssertEqual(resolution.resolved?.template, testBuiltIns["seed_oss"])
        XCTAssertEqual(resolution.resolved?.source, "built-in seed_oss")
    }

    /// `resolveDetailed` reports `"empty"` for an empty user file (the counterpart
    /// to `testEmptyUserFileIsIgnored`, which only checks the `resolve` value).
    func testUserFileEmptyIsSkippedWithDiagnosis() throws {
        try writeConfig(modelType: "seed_oss")
        try writeUserOverride("")

        let resolution = ChatTemplateOverride.resolveDetailed(
            modelDirectory: tempDir, builtIns: testBuiltIns)
        XCTAssertEqual(resolution.skippedUserFileReason, "empty")
        XCTAssertEqual(resolution.resolved?.template, testBuiltIns["seed_oss"])
    }

    /// A whitespace-only file is as unusable as a zero-byte one: accepting it
    /// would render a blank prompt with no diagnostic. It must take the same
    /// "empty" skip path as a truly empty file.
    func testUserFileWhitespaceOnlyIsSkippedWithDiagnosis() throws {
        try writeConfig(modelType: "seed_oss")
        try writeUserOverride("  \n\t\n  ")

        let resolution = ChatTemplateOverride.resolveDetailed(
            modelDirectory: tempDir, builtIns: testBuiltIns)
        XCTAssertEqual(resolution.skippedUserFileReason, "empty")
        XCTAssertEqual(resolution.resolved?.template, testBuiltIns["seed_oss"])
    }

    /// When a broken user file's `model_type` has no built-in either, resolution
    /// still reports the skip reason (for logging) while `resolved` is `nil` —
    /// the "falling back to the checkpoint's own template" branch of the log
    /// message in `HuggingFaceTokenizerLoader`.
    func testUserFileIssueFallsBackToNilWhenNoBuiltIn() throws {
        try writeConfig(modelType: "llama")
        try writeUserOverride("")

        let resolution = ChatTemplateOverride.resolveDetailed(modelDirectory: tempDir)
        XCTAssertEqual(resolution.skippedUserFileReason, "empty")
        XCTAssertNil(resolution.resolved)
    }
}
