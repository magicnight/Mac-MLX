import Testing
import Foundation
import MLXLMCommon
@testable import MacMLXCore

// MARK: - MLXSwiftEngine D1 (speculative decoding) unit tests
//
// These cover the draft-container keep/swap/unload state machine, the draft
// directory convention, and the tokenizer-compatibility guard — all PURE
// functions with no model loading and no Metal, per the same "stub/mock,
// don't touch Metal" style as `MLXSwiftEngineTests`. Real draft-model
// loading + on/off token-identity parity is covered by the gated real-model
// test in `SpeculativeDecoding/SpeculativeDecodingModelTests.swift`.

@Suite("MLXSwiftEngine draft container state machine (D1)")
struct MLXSwiftEngineDraftContainerActionTests {

    @Test("no draft requested and none resident: keep")
    func noneRequestedNoneResidentKeeps() {
        let action = MLXSwiftEngine.DraftContainerAction.decide(
            currentDraftModelID: nil, requestedDraftModelID: nil
        )
        #expect(action == .keep)
    }

    @Test("requested draft matches the resident one: keep")
    func matchingRequestKeeps() {
        let action = MLXSwiftEngine.DraftContainerAction.decide(
            currentDraftModelID: "small-model", requestedDraftModelID: "small-model"
        )
        #expect(action == .keep)
    }

    @Test("request sets draftModelID to nil while one is resident: unload")
    func nilRequestWithResidentUnloads() {
        let action = MLXSwiftEngine.DraftContainerAction.decide(
            currentDraftModelID: "small-model", requestedDraftModelID: nil
        )
        #expect(action == .unload)
    }

    @Test("request names a draft model while none is resident: load")
    func newRequestWithNoneResidentLoads() {
        let action = MLXSwiftEngine.DraftContainerAction.decide(
            currentDraftModelID: nil, requestedDraftModelID: "small-model"
        )
        #expect(action == .load(id: "small-model"))
    }

    @Test("request names a DIFFERENT draft model than what's resident: load (swap)")
    func differentRequestSwaps() {
        let action = MLXSwiftEngine.DraftContainerAction.decide(
            currentDraftModelID: "model-a", requestedDraftModelID: "model-b"
        )
        #expect(action == .load(id: "model-b"))
    }
}

@Suite("MLXSwiftEngine speculative-decoding cache-trimmability precheck (D1 fallback fix)")
struct MLXSwiftEngineSpeculativeCachePrecheckTests {

    // `canUseSpeculativeDecoding` is the stubbable seam of the fallback
    // decision: the actual `canTrimPromptCache` check against a real
    // `[KVCache]` has to run inside `ModelContainer.perform` against a
    // loaded model (not stubbable cheaply), but the AND-decision that
    // follows from its two booleans is pure and — per this bug's real E2E
    // repro (a hybrid/linear-attention target model's `MambaCache` is
    // non-trimmable) — is exactly the logic that must route to a fallback
    // instead of letting `SpeculativeTokenIterator.init` throw.

    @Test("both target and draft caches trimmable: speculation is allowed")
    func bothTrimmableAllowsSpeculation() {
        #expect(MLXSwiftEngine.canUseSpeculativeDecoding(
            targetCacheIsTrimmable: true, draftCacheIsTrimmable: true
        ))
    }

    @Test("target cache not trimmable (e.g. hybrid architecture's MambaCache): falls back")
    func targetNotTrimmableFallsBack() {
        #expect(!MLXSwiftEngine.canUseSpeculativeDecoding(
            targetCacheIsTrimmable: false, draftCacheIsTrimmable: true
        ))
    }

    @Test("draft cache not trimmable: falls back")
    func draftNotTrimmableFallsBack() {
        #expect(!MLXSwiftEngine.canUseSpeculativeDecoding(
            targetCacheIsTrimmable: true, draftCacheIsTrimmable: false
        ))
    }

    @Test("neither cache trimmable: falls back")
    func neitherTrimmableFallsBack() {
        #expect(!MLXSwiftEngine.canUseSpeculativeDecoding(
            targetCacheIsTrimmable: false, draftCacheIsTrimmable: false
        ))
    }
}

@Suite("MLXSwiftEngine draft model directory resolution (D1)")
struct MLXSwiftEngineDraftModelDirectoryTests {

    @Test("draftModelDirectory(id:) resolves under ~/.mac-mlx/models, mirroring SettingsManager's default")
    func resolvesUnderDefaultModelsRoot() throws {
        let directory = try MLXSwiftEngine.draftModelDirectory(id: "Qwen3.5-1.5B-4bit")
        let expected = DataRoot.macMLX("models").appending(
            path: "Qwen3.5-1.5B-4bit", directoryHint: .isDirectory)
        #expect(directory == expected)
        #expect(directory.path.hasSuffix(".mac-mlx/models/Qwen3.5-1.5B-4bit"))
    }
}

// MARK: - MLXSwiftEngine draft model id validation (D1 / M1 hardening)
//
// `draftModelID` arrives verbatim from the wire (client-controlled) and is
// used to build a filesystem path — these tests lock in the path-traversal
// gate in `draftModelDirectory(id:)` / `isValidDraftModelID(_:)`.

@Suite("MLXSwiftEngine draft model id validation (D1)")
struct MLXSwiftEngineDraftModelIDValidationTests {

    @Test("a plain alphanumeric id (with dots/dashes/underscores) is valid")
    func validIDIsAccepted() {
        #expect(MLXSwiftEngine.isValidDraftModelID("Qwen3.5-1.5B-4bit"))
        #expect(MLXSwiftEngine.isValidDraftModelID("small_model-v2"))
    }

    @Test("draftModelDirectory(id:) succeeds and resolves under the models root for a valid id")
    func validIDResolvesUnderModelsRoot() throws {
        let directory = try MLXSwiftEngine.draftModelDirectory(id: "small_model-v2")
        #expect(directory.path.hasSuffix(".mac-mlx/models/small_model-v2"))
    }

    @Test("a parent-directory escape id is rejected")
    func parentDirectoryEscapeIsRejected() {
        #expect(!MLXSwiftEngine.isValidDraftModelID("../escape"))
        do {
            _ = try MLXSwiftEngine.draftModelDirectory(id: "../escape")
            Issue.record("Expected draftModelDirectory(id:) to throw for a traversal id")
        } catch EngineError.invalidDraftModelID(let id) {
            #expect(id == "../escape")
        } catch {
            Issue.record("Expected EngineError.invalidDraftModelID, got \(type(of: error)): \(error)")
        }
    }

    @Test("an id containing an embedded path separator is rejected")
    func embeddedPathSeparatorIsRejected() {
        #expect(!MLXSwiftEngine.isValidDraftModelID("a/b"))
        do {
            _ = try MLXSwiftEngine.draftModelDirectory(id: "a/b")
            Issue.record("Expected draftModelDirectory(id:) to throw for an embedded separator")
        } catch EngineError.invalidDraftModelID(let id) {
            #expect(id == "a/b")
        } catch {
            Issue.record("Expected EngineError.invalidDraftModelID, got \(type(of: error)): \(error)")
        }
    }

    @Test("a leading-dot (dotfile-style) id is rejected")
    func leadingDotIsRejected() {
        #expect(!MLXSwiftEngine.isValidDraftModelID(".hidden"))
        do {
            _ = try MLXSwiftEngine.draftModelDirectory(id: ".hidden")
            Issue.record("Expected draftModelDirectory(id:) to throw for a leading-dot id")
        } catch EngineError.invalidDraftModelID(let id) {
            #expect(id == ".hidden")
        } catch {
            Issue.record("Expected EngineError.invalidDraftModelID, got \(type(of: error)): \(error)")
        }
    }

    @Test("a bare '..' id is rejected")
    func bareParentComponentIsRejected() {
        #expect(!MLXSwiftEngine.isValidDraftModelID(".."))
    }

    @Test("an id containing a backslash is rejected")
    func backslashIsRejected() {
        #expect(!MLXSwiftEngine.isValidDraftModelID("a\\b"))
    }

    @Test("an id containing a NUL byte is rejected")
    func nulByteIsRejected() {
        #expect(!MLXSwiftEngine.isValidDraftModelID("a\0b"))
    }

    @Test("an empty id is rejected")
    func emptyIDIsRejected() {
        #expect(!MLXSwiftEngine.isValidDraftModelID(""))
    }
}

@Suite("MLXSwiftEngine draft/target tokenizer compatibility guard (D1)")
struct MLXSwiftEngineTokenizerCompatibilityTests {

    /// Minimal `Tokenizer` conformance for exercising
    /// `tokenizersAreCompatible` without loading any real model —
    /// `encode`/`decode`/`applyChatTemplate` are never called by the
    /// function under test, so they're stubbed trivially. `convertTokenToId`
    /// looks up `tokenIDs` (default empty ⇒ always nil) so tests can
    /// simulate "same token STRING, different underlying id" — the exact
    /// case string-only comparison can't catch. `MLXLMCommon.Tokenizer`'s
    /// protocol extension derives `eosTokenId`/`unknownTokenId` from this
    /// same `convertTokenToId`, so populating `tokenIDs` drives those too.
    private struct MockTokenizer: Tokenizer {
        let bosToken: String?
        let eosToken: String?
        let unknownToken: String?
        var tokenIDs: [String: Int] = [:]

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { tokenIDs[token] }
        func convertIdToToken(_ id: Int) -> String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    @Test("identical bos/eos/unknown tokens are compatible")
    func identicalTokenizersAreCompatible() {
        let a = MockTokenizer(bosToken: "<s>", eosToken: "</s>", unknownToken: "<unk>")
        let b = MockTokenizer(bosToken: "<s>", eosToken: "</s>", unknownToken: "<unk>")
        #expect(MLXSwiftEngine.tokenizersAreCompatible(target: a, draft: b))
    }

    @Test("a differing eosToken is reported incompatible")
    func differingEOSTokenIsIncompatible() {
        let target = MockTokenizer(bosToken: "<s>", eosToken: "</s>", unknownToken: "<unk>")
        let draft = MockTokenizer(bosToken: "<s>", eosToken: "<|endoftext|>", unknownToken: "<unk>")
        #expect(!MLXSwiftEngine.tokenizersAreCompatible(target: target, draft: draft))
    }

    @Test("a differing bosToken is reported incompatible")
    func differingBOSTokenIsIncompatible() {
        let target = MockTokenizer(bosToken: "<s>", eosToken: "</s>", unknownToken: "<unk>")
        let draft = MockTokenizer(bosToken: "<|startoftext|>", eosToken: "</s>", unknownToken: "<unk>")
        #expect(!MLXSwiftEngine.tokenizersAreCompatible(target: target, draft: draft))
    }

    @Test("a differing unknownToken is reported incompatible")
    func differingUnknownTokenIsIncompatible() {
        let target = MockTokenizer(bosToken: "<s>", eosToken: "</s>", unknownToken: "<unk>")
        let draft = MockTokenizer(bosToken: "<s>", eosToken: "</s>", unknownToken: nil)
        #expect(!MLXSwiftEngine.tokenizersAreCompatible(target: target, draft: draft))
    }

    // MARK: M3 — id-level checks (same special-token STRINGS, different ids)

    @Test("identical strings AND identical underlying ids are compatible")
    func identicalStringsAndIdsAreCompatible() {
        let ids = ["<s>": 1, "</s>": 2, "<unk>": 3]
        let target = MockTokenizer(bosToken: "<s>", eosToken: "</s>", unknownToken: "<unk>", tokenIDs: ids)
        let draft = MockTokenizer(bosToken: "<s>", eosToken: "</s>", unknownToken: "<unk>", tokenIDs: ids)
        #expect(MLXSwiftEngine.tokenizersAreCompatible(target: target, draft: draft))
    }

    @Test("identical bosToken STRING but a differing underlying id is reported incompatible")
    func differingBOSTokenIdWithSameStringIsIncompatible() {
        let target = MockTokenizer(
            bosToken: "<s>", eosToken: "</s>", unknownToken: "<unk>",
            tokenIDs: ["<s>": 1, "</s>": 2, "<unk>": 3])
        let draft = MockTokenizer(
            bosToken: "<s>", eosToken: "</s>", unknownToken: "<unk>",
            tokenIDs: ["<s>": 999, "</s>": 2, "<unk>": 3])
        #expect(!MLXSwiftEngine.tokenizersAreCompatible(target: target, draft: draft))
    }

    @Test("identical eosToken STRING but a differing underlying id is reported incompatible")
    func differingEOSTokenIdWithSameStringIsIncompatible() {
        let target = MockTokenizer(
            bosToken: "<s>", eosToken: "</s>", unknownToken: "<unk>",
            tokenIDs: ["<s>": 1, "</s>": 2, "<unk>": 3])
        let draft = MockTokenizer(
            bosToken: "<s>", eosToken: "</s>", unknownToken: "<unk>",
            tokenIDs: ["<s>": 1, "</s>": 999, "<unk>": 3])
        #expect(!MLXSwiftEngine.tokenizersAreCompatible(target: target, draft: draft))
    }

    @Test("identical unknownToken STRING but a differing underlying id is reported incompatible")
    func differingUnknownTokenIdWithSameStringIsIncompatible() {
        let target = MockTokenizer(
            bosToken: "<s>", eosToken: "</s>", unknownToken: "<unk>",
            tokenIDs: ["<s>": 1, "</s>": 2, "<unk>": 3])
        let draft = MockTokenizer(
            bosToken: "<s>", eosToken: "</s>", unknownToken: "<unk>",
            tokenIDs: ["<s>": 1, "</s>": 2, "<unk>": 999])
        #expect(!MLXSwiftEngine.tokenizersAreCompatible(target: target, draft: draft))
    }
}

// MARK: - MLXSwiftEngine draft state lifecycle (D1)

@Suite("MLXSwiftEngine unload() resets D1 draft state")
struct MLXSwiftEngineDraftUnloadResetTests {

    @Test("unload() leaves no draft container resident")
    func unloadLeavesNoDraftContainerResident() async throws {
        // Stub-test environment (no Metal, no real model): `draftContainer`
        // can only become non-nil via `ensureDraftContainer`'s `.load`
        // branch, which requires a real `LLMModelFactory.loadContainer`
        // call — so this locks in the POST-unload invariant (never
        // resident) rather than a true before/after flip. The flip itself
        // (load a draft, then unload, then assert it's gone) is covered by
        // the gated real-model suite
        // (`SpeculativeDecoding/SpeculativeDecodingModelTests.swift`).
        let engine = MLXSwiftEngine()
        #expect(await engine.hasDraftContainer == false)
        try await engine.unload()
        #expect(await engine.hasDraftContainer == false)
    }
}
