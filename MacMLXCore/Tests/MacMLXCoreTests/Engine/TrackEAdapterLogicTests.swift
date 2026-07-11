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

    // MARK: Identity sync (applyAdapter now records loadedAdapterID)

    /// Regression for the LoRA identity desync: a directly-applied adapter (GUI
    /// auto-pin / EngineCoordinator) now records its identity, so a subsequent
    /// request routes correctly against it — `nil` sheds to base (never `.keep`),
    /// and a DIFFERENT adapter reloads-then-applies (never additive `.applyOnly`,
    /// which would stack two adapters). With the identity left nil (the bug),
    /// `decide(nil, nil) == .keep` never restored the base and `decide(nil, "b")
    /// == .applyOnly` stacked.
    @Test("a recorded resident identity sheds to base and never stacks a switch")
    func recordedIdentityRoutesCorrectly() {
        let resident = "pinned-lora"  // what applyAdapter now writes to loadedAdapterID
        #expect(
            MLXSwiftEngine.AdapterAction.decide(current: resident, requested: nil) == .reloadOnly)
        #expect(
            MLXSwiftEngine.AdapterAction.decide(current: resident, requested: "other")
                == .reloadThenApply(id: "other"))
        // Same adapter re-requested is a no-op.
        #expect(
            MLXSwiftEngine.AdapterAction.decide(current: resident, requested: resident) == .keep)
    }

    // MARK: GUI self-describing pin (BLOCKER regression fix)
    //
    // `ChatViewModel.generate()` builds its `GenerateRequest` OUT-OF-BAND from
    // the engine: `AppState.onModelLoaded` pins an adapter by calling
    // `engine.applyAdapter` directly (once, on model load), while every chat
    // turn used to build its request with NO `adapters` field. That silently
    // defaulted `request.adapters` to nil, so `reconcileAdapter`'s very first
    // call saw `decide(pinned, nil) == .reloadOnly` — a multi-second base
    // reload that silently SHED the pin while the GUI kept showing it as
    // active. The fix threads `adapters: params.adapterName` (the same
    // persisted value `onModelLoaded` reads) onto every request, so it always
    // mirrors the pin instead of omitting it.
    //
    // `ChatViewModel` itself lives in the `macMLX` GUI target, which has no
    // unit-test bundle, so this test drives the exact two REAL production
    // functions the fix touches — `GenerateRequest`'s `adapters` normalization
    // (the same initializer ChatViewModel calls) and `AdapterAction.decide`
    // (what `reconcileAdapter` switches on) — composed the way the GUI now
    // composes them. The GUI call site itself is compile-verified (`xcodebuild
    // -scheme macMLX … build`), not exercised at runtime here.

    @Test("a GUI request mirroring its own resident pin is a no-op, not a reload")
    func guiRequestMirroringPinIsKept() {
        let pinned = "pinned-lora"
        // What ChatViewModel.generate() now builds: `adapters: params.adapterName`
        // where `params.adapterName == pinned` (the value `onModelLoaded` used to
        // apply it in the first place) — normalized exactly as production does.
        let request = GenerateRequest(model: "m", messages: msg, adapters: pinned)
        #expect(request.adapters == pinned, "a non-empty pin name must survive normalization unchanged")

        // Fixed behaviour: the request mirrors the resident pin -> no reload, no shed.
        #expect(MLXSwiftEngine.AdapterAction.decide(current: pinned, requested: request.adapters) == .keep)

        // The regression this guards against: a request that still omitted
        // `adapters` (nil) would reload the base model and drop the pin out from
        // under the user on the very first chat turn.
        #expect(
            MLXSwiftEngine.AdapterAction.decide(current: pinned, requested: nil) == .reloadOnly)
    }

    @Test("a GUI request with no pin (adapterName nil/empty) matches an adapter-free base")
    func guiRequestWithNoPinKeepsBase() {
        // `ModelParameters.adapterName`'s own doc: empty string means "no adapter",
        // identical to nil — `GenerateRequest`'s normalization already collapses
        // both, so passing either straight through from `params.adapterName`
        // (without pre-checking emptiness in ChatViewModel) is safe.
        #expect(GenerateRequest(model: "m", messages: msg, adapters: nil).adapters == nil)
        #expect(GenerateRequest(model: "m", messages: msg, adapters: "").adapters == nil)
        #expect(MLXSwiftEngine.AdapterAction.decide(current: nil, requested: nil) == .keep)
    }

    // MARK: Prompt-cache bypass predicate

    @Test("prompt cache is bypassed while an adapter is resident (or KV quantizing)")
    func bypassPromptCacheWhileAdapted() {
        // A resident adapter forces bypass — base and adapted weights share the
        // base modelID cache key, so reuse would poison it.
        #expect(MLXSwiftEngine.shouldBypassPromptCache(kvBits: nil, adapterID: "a"))
        // KV quantization also forces bypass.
        #expect(MLXSwiftEngine.shouldBypassPromptCache(kvBits: 4, adapterID: nil))
        #expect(MLXSwiftEngine.shouldBypassPromptCache(kvBits: 4, adapterID: "a"))
        // Neither → the cache is used.
        #expect(!MLXSwiftEngine.shouldBypassPromptCache(kvBits: nil, adapterID: nil))
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
