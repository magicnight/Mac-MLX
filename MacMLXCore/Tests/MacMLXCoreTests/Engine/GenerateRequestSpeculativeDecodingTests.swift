import Testing
import Foundation
@testable import MacMLXCore

/// D1 — classic per-request speculative decoding fields on `GenerateRequest`.
/// Mirrors `GenerateRequestToolsTests`'s structure for the `tools` field.
@Suite("GenerateRequest speculative decoding (D1)")
struct GenerateRequestSpeculativeDecodingTests {

    // MARK: Memberwise init clamp

    @Test("numDraftTokens above 8 clamps to 8 via the memberwise init")
    func numDraftTokensClampsToUpperBoundViaInit() {
        let req = GenerateRequest(
            model: "m",
            messages: [ChatMessage(role: .user, content: "hi")],
            draftModelID: "small-model",
            numDraftTokens: 20
        )
        #expect(req.numDraftTokens == 8)
    }

    @Test("numDraftTokens below 1 clamps to 1 via the memberwise init")
    func numDraftTokensClampsToLowerBoundViaInit() {
        let req = GenerateRequest(
            model: "m",
            messages: [ChatMessage(role: .user, content: "hi")],
            draftModelID: "small-model",
            numDraftTokens: 0
        )
        #expect(req.numDraftTokens == 1)

        let negative = GenerateRequest(
            model: "m",
            messages: [ChatMessage(role: .user, content: "hi")],
            draftModelID: "small-model",
            numDraftTokens: -5
        )
        #expect(negative.numDraftTokens == 1)
    }

    @Test("numDraftTokens within 1...8 passes through unchanged")
    func numDraftTokensWithinRangeUnchanged() {
        let req = GenerateRequest(
            model: "m",
            messages: [ChatMessage(role: .user, content: "hi")],
            draftModelID: "small-model",
            numDraftTokens: 4
        )
        #expect(req.numDraftTokens == 4)
    }

    @Test("numDraftTokens defaults to nil — defers to mlx-swift-lm's own default")
    func numDraftTokensDefaultsToNil() {
        let req = GenerateRequest(
            model: "m",
            messages: [ChatMessage(role: .user, content: "hi")],
            draftModelID: "small-model"
        )
        #expect(req.numDraftTokens == nil)
    }

    @Test("draftModelID defaults to nil — speculative decoding disabled")
    func draftModelIDDefaultsToNil() {
        let req = GenerateRequest(
            model: "m",
            messages: [ChatMessage(role: .user, content: "hi")]
        )
        #expect(req.draftModelID == nil)
    }

    // MARK: JSON round trip

    @Test("draftModelID and numDraftTokens round-trip through Codable")
    func speculativeFieldsRoundTrip() throws {
        let req = GenerateRequest(
            model: "m",
            messages: [ChatMessage(role: .user, content: "hi")],
            draftModelID: "Qwen3.5-1.5B-4bit",
            numDraftTokens: 3
        )
        let data = try JSONEncoder().encode(req)
        let back = try JSONDecoder().decode(GenerateRequest.self, from: data)
        #expect(back == req)
        #expect(back.draftModelID == "Qwen3.5-1.5B-4bit")
        #expect(back.numDraftTokens == 3)
    }

    /// The clamp must hold on EVERY decode path, not just the memberwise
    /// init — a raw `JSONDecoder` decode (e.g. a persisted/replayed request)
    /// must not bypass it.
    @Test("numDraftTokens clamps to 8 when decoded directly from raw JSON")
    func numDraftTokensClampsWhenDecodedFromRawJSON() throws {
        let json = """
        {"model":"m",\
        "messages":[{"id":"1FAA0000-0000-0000-0000-000000000001","role":"user","content":"hi"}],\
        "parameters":{"temperature":0.7,"topP":0.95,"maxTokens":2048,"stream":true},\
        "draftModelID":"small-model","numDraftTokens":99}
        """
        let decoded = try JSONDecoder().decode(GenerateRequest.self, from: Data(json.utf8))
        #expect(decoded.draftModelID == "small-model")
        #expect(decoded.numDraftTokens == 8)
    }

    @Test("an explicit JSON null for draftModelID decodes as nil (unload semantics)")
    func draftModelIDExplicitNullDecodesAsNil() throws {
        let json = """
        {"model":"m",\
        "messages":[{"id":"1FAA0000-0000-0000-0000-000000000001","role":"user","content":"hi"}],\
        "parameters":{"temperature":0.7,"topP":0.95,"maxTokens":2048,"stream":true},\
        "draftModelID":null}
        """
        let decoded = try JSONDecoder().decode(GenerateRequest.self, from: Data(json.utf8))
        #expect(decoded.draftModelID == nil)
    }

    /// Pre-D1 request JSON has no `draftModelID`/`numDraftTokens` keys at
    /// all. Both are optional, so the custom decoder must default them to
    /// nil exactly like the synthesised decoder did for `tools`.
    @Test("legacy request JSON without draft fields decodes with nil draftModelID")
    func legacyDecodesWithNilDraftFields() throws {
        let legacy = """
        {"model":"m",\
        "messages":[{"id":"1FAA0000-0000-0000-0000-000000000001","role":"user","content":"hi"}],\
        "parameters":{"temperature":0.7,"topP":0.95,"maxTokens":2048,"stream":true}}
        """
        let decoded = try JSONDecoder().decode(GenerateRequest.self, from: Data(legacy.utf8))
        #expect(decoded.draftModelID == nil)
        #expect(decoded.numDraftTokens == nil)
        #expect(decoded.tools == nil)
        #expect(decoded.templateKwargs == nil)
        #expect(decoded.messages.count == 1)
    }
}
