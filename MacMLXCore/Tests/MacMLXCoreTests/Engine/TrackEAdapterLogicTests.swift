import Testing
import Foundation
@testable import MacMLXCore

/// Track E — pure decision/normalization logic for the per-request LoRA adapter
/// hot-swap and the `adapters` request field. No Metal / no model load.
@Suite("Track E adapter logic")
struct TrackEAdapterLogicTests {

    private let msg = [ChatMessage(role: .user, content: "hi")]

    // MARK: AdapterAction.decide (keep / apply / reload state machine)

    @Test("unchanged pairing (including both nil) keeps")
    func keepWhenUnchanged() {
        #expect(MLXSwiftEngine.AdapterAction.decide(current: nil, requested: nil) == .keep)
        #expect(MLXSwiftEngine.AdapterAction.decide(current: "a", requested: "a") == .keep)
    }

    @Test("applying onto a clean base does not reload")
    func applyOnlyFromBase() {
        #expect(
            MLXSwiftEngine.AdapterAction.decide(current: nil, requested: "a")
                == .applyOnly(id: "a"))
    }

    @Test("dropping the resident adapter reloads the base")
    func reloadOnlyToUnload() {
        #expect(
            MLXSwiftEngine.AdapterAction.decide(current: "a", requested: nil) == .reloadOnly)
    }

    @Test("switching adapters reloads then applies the new one")
    func reloadThenApplyOnSwitch() {
        #expect(
            MLXSwiftEngine.AdapterAction.decide(current: "a", requested: "b")
                == .reloadThenApply(id: "b"))
    }

    // MARK: GenerateRequest.adapters normalization + Codable

    @Test("empty / whitespace adapters normalizes to nil")
    func adaptersNormalizeEmpty() {
        #expect(GenerateRequest(model: "m", messages: msg, adapters: "").adapters == nil)
        #expect(GenerateRequest(model: "m", messages: msg, adapters: "   ").adapters == nil)
        #expect(GenerateRequest(model: "m", messages: msg, adapters: nil).adapters == nil)
    }

    @Test("adapters is trimmed")
    func adaptersTrimmed() {
        #expect(GenerateRequest(model: "m", messages: msg, adapters: "  my-lora \n").adapters == "my-lora")
    }

    @Test("adapters round-trips through Codable")
    func adaptersRoundTrip() throws {
        let req = GenerateRequest(model: "m", messages: msg, adapters: "my-lora")
        let data = try JSONEncoder().encode(req)
        let back = try JSONDecoder().decode(GenerateRequest.self, from: data)
        #expect(back.adapters == "my-lora")
    }

    @Test("legacy request JSON without adapters decodes as nil")
    func legacyDecodesNilAdapters() throws {
        let legacy = """
        {"model":"m",\
        "messages":[{"id":"1FAA0000-0000-0000-0000-000000000001","role":"user","content":"hi"}],\
        "parameters":{"temperature":0.7,"topP":0.95,"maxTokens":2048,"stream":true}}
        """
        let decoded = try JSONDecoder().decode(GenerateRequest.self, from: Data(legacy.utf8))
        #expect(decoded.adapters == nil)
    }

    @Test("adapter name allowlist rejects traversal-y ids")
    func adapterNameAllowlist() {
        // Reuses the draft-model id allowlist — a bare, traversal-safe component.
        #expect(MLXSwiftEngine.isValidDraftModelID("my-lora_v2.1"))
        #expect(!MLXSwiftEngine.isValidDraftModelID("../evil"))
        #expect(!MLXSwiftEngine.isValidDraftModelID(".hidden"))
        #expect(!MLXSwiftEngine.isValidDraftModelID("a/b"))
        #expect(!MLXSwiftEngine.isValidDraftModelID(""))
    }
}
