// Copyright © 2026 macMLX. English comments only.

import XCTest

@testable import MacMLXCore

/// Membership of ``BatchModelAllowlist`` on real model instances: the match is
/// on the concrete type's name, so a renamed or wrapped type would silently
/// drop out (or in). MLX-backed, since building a model allocates its weights.
final class BatchModelAllowlistTests: XCTestCase {

    func testGemma2IsNotAllowlisted() throws {
        try requireMLXRuntimeOrSkip()
        // Its mask is built without the cache, so left padding is unmasked;
        // see BatchRaggedDecodeParityTests.testGemma2RaggedDecodeAttendsToLeftPadding.
        XCTAssertFalse(BatchModelAllowlist.contains(try TinyRandomLanguageModel.gemma2.make()))
    }

    /// `gemma2` checkpoints load as macMLX's corrected Gemma 2, which takes its
    /// mask from the cache; see Gemma2OverlayTests and the ragged parity suite.
    func testGemma2OverlayIsAllowlisted() throws {
        try requireMLXRuntimeOrSkip()
        XCTAssertTrue(
            BatchModelAllowlist.contains(try TinyRandomLanguageModel.gemma2Overlay.make()))
    }

    func testEveryOtherTinyArchitectureIsAllowlisted() throws {
        try requireMLXRuntimeOrSkip()
        for architecture in TinyRandomLanguageModel.allCases where architecture != .gemma2 {
            XCTAssertTrue(
                BatchModelAllowlist.contains(try architecture.make()),
                "\(architecture) should be allowlisted")
        }
    }
}
