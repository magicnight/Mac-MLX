# Foundation Models integration design

**Date:** 2026-09-19
**Status:** design, not yet implemented
**Target:** macOS 27 (`FoundationModels` provider APIs); macMLX's own baseline stays macOS 14

## What this is for

macOS 27 turned `FoundationModels` from "Apple's model SDK" into a model-agnostic
inference abstraction. Two new public protocols, `LanguageModel` and
`LanguageModelExecutor`, let a third party supply the model behind Apple's
`LanguageModelSession` — the same session type that carries `@Generable` guided
generation, tool calling, streaming, and transcript management. Apple refactored
its own `SystemLanguageModel` into a conformance of that protocol, so the
extension point is the same road Apple drives on, not a side door.

This document designs two integrations:

- **B — macMLX as a Foundation Models provider.** An app written against Apple's
  API runs arbitrary MLX checkpoints through a running macMLX server.
- **C — Apple's model as a macMLX engine.** An `InferenceEngine` conformance that
  exposes the built-in system model in macMLX's GUI and CLI alongside MLX models.

They are independent and can ship separately.

## Evidence this design rests on

Everything below was verified on this machine (macOS 27.0 build 26A428, Xcode
27.0, M5 Max) or against the upstream repositories, not from documentation alone.

**The extension point works.** A minimal third-party `LanguageModel` +
`LanguageModelExecutor` was compiled and run against `LanguageModelSession`:
Apple's session called `respond(to:model:streamingInto:)`, collected streamed
`.appendText` events, and returned the assembled reply to the caller. With a
`@Generable` return type the session delivered the schema to the third-party
executor as standard JSON Schema and parsed the returned JSON back into the
typed Swift value. Guided generation therefore works through a foreign backend,
but the backend has to do the constraining itself — the schema arrives as a
request field, not as a service Apple performs.

**There is no system-wide model registry.** The `.swiftinterface` has no
registration, discovery, or provider-catalog API, and `ExtensionFoundation` has
no Foundation Models extension point. A `LanguageModel` is constructed by the
consuming app. macMLX cannot install itself as a system model provider; an app
opts in by taking a dependency.

**The built-in model, measured here:** `availability == .available`, 24 supported
languages, capabilities `vision` and `guidedGeneration` and `toolCalling` true,
`reasoning` false. Streaming a one-paragraph answer took 0.83 s to first token
and produced roughly 228 characters per second.

**Apple already ships an OpenAI-compatible provider.**
`apple/foundation-models-utilities` (Apache-2.0, `platforms: [.macOS("27.0")]`)
contains `Sources/FoundationModelsUtilities/LanguageModels/ChatCompletionsLanguageModel.swift`:
a `LanguageModel` with a settable `url`, `additionalHeaders`, and
`supportsGuidedGeneration`, whose executor speaks SSE chat completions with
`tool_choice` mapping, index-keyed tool-call deltas, `response_format` of type
`json_schema`, and `reasoning_content` deltas.

**Apple also ships an MLX provider.** `ml-explore/mlx-swift-lm` main contains
`Libraries/MLXFoundationModels/MLXLanguageModel.swift` and
`Libraries/MLXGuidedGeneration`. It is **not** in a tagged release — the latest
release is 3.31.4 (2026-06-30), which is what macMLX pins, and the pinned
checkout does not contain the module.

## The prerequisite correction this forces

The first instinct was to write a bespoke `LanguageModelExecutor` that drives
`MLXSwiftEngine`. That would duplicate work Apple has already done twice over.
The design below deliberately does not do that.

It also means one project claim needs revisiting, separately from this work:
CLAUDE.md calls macMLX "the only MLX tool whose inference engine itself is native
Swift running in-process". Once `MLXFoundationModels` lands in a tagged
mlx-swift-lm release, native-Swift in-process MLX inference is a first-party
Apple capability. The defensible claim moves from the engine to the product —
GUI plus CLI/TUI plus model library plus benchmarking plus the tiered SSD KV
cache plus audio, in one dependency-free DMG. This mirrors the earlier
"only native GUI" correction made when oMLX shipped a SwiftUI shell. Handling
that wording is out of scope here and should not be bundled into this change.

---

## Direction B — macMLX as a Foundation Models provider

### Shape

Because `ChatCompletionsLanguageModel` already exists and macMLX already serves
OpenAI-compatible chat completions, the work is **not** writing an executor. It
is closing the wire gaps and removing the configuration friction.

```
Third-party app (macOS 27)
  └── LanguageModelSession(model: MacMLXLanguageModel(...))
        └── MacMLXLanguageModel        ← new, thin, this project
              └── ChatCompletionsLanguageModel   ← Apple, Apache-2.0
                    └── HTTP  →  macmlx serve  /v1/chat/completions
                                   └── MLXSwiftEngine (any MLX checkpoint)
```

`MacMLXLanguageModel` wraps rather than reimplements. It exists to supply what
an app should not have to know: the default endpoint, server discovery and a
clear error when nothing is listening, the model identifier, and the correct
capability set for the checkpoint actually loaded.

### Why wrap instead of telling people to use Apple's type directly

Three things an app cannot get right on its own:

1. **Capabilities are per-checkpoint, not per-server.** `vision` is true for a
   VLM and false for a dense text model; `toolCalling` depends on the chat
   template. `ChatCompletionsLanguageModel` has a single static
   `supportsGuidedGeneration` flag and no notion of the rest. macMLX already
   knows all of this from `ModelLibraryManager` and can answer it from
   `GET /v1/models` plus the `/x/models` extensions.
2. **A dead server should fail as a typed error, not a connection refused.**
   `LanguageModelError` has no "provider not running" case; the nearest honest
   mapping is a thrown error with a message telling the user to launch macMLX or
   run `macmlx serve`, surfaced before the first token rather than mid-stream.
3. **Guided generation must be declared truthfully.** macMLX supports a JSON
   Schema subset and rejects unsupported keywords with 400. Setting Apple's
   `supportsGuidedGeneration` to an unconditional `true` would turn a clean 400
   into an opaque stream failure.

### Compatibility audit

This section was rewritten after running the thing rather than reading it. A
protocol-only stub reproducing `HummingbirdServer`'s exact frame shapes was
served on `127.0.0.1:8123`, and Apple's `ChatCompletionsLanguageModel` was
pointed at it through a real `LanguageModelSession`. No model is involved, so
this is a wire-contract result, not a generation result.

**Both paths completed.** A plain `respond(to:)` returned the streamed text, and
a `@Generable` round trip returned a parsed typed value. Notably the stub sent
**no usage frame**, and the conversation still completed — so the missing
`stream_options` support costs usage reporting, it does not break the
conversation. Earlier drafts of this document overstated that.

**What Apple's client actually sends** (captured verbatim):

```
User-Agent: com.apple.FoundationModels
Accept: text/event-stream
{"tool_choice":"auto","stream":true,"stream_options":{"include_usage":true},
 "messages":[…],"model":"…","tools":[]}
```

`tools: []` and `tool_choice: "auto"` are sent unconditionally, even with no
tools registered. Whether an empty `tools` array changes macMLX's chat-template
path is untested and should be checked.

Static audit of the remaining fields, Apple's client against `HummingbirdServer`:

| Apple's client sends / expects | macMLX server | Status |
|---|---|---|
| `response_format: {"type":"json_schema", …}` | `ResponseFormat.jsonSchema(JSONSchemaObject)` | supported |
| SSE `data: ` frames, `[DONE]` sentinel | `doneBuf.writeString("data: [DONE]\n\n")` | supported |
| `delta.tool_calls` keyed by `index` | `openAIToolCallDelta(index:call:)` | supported |
| `delta.reasoning_content` | emitted for reasoning models | supported |
| `tool_choice` auto / required / none | accepted | supported |
| `stream_options: {"include_usage": true}` | **not implemented** | **gap** |

The gap has two halves, both confirmed by reading the server rather than by
inference. `ChatCompletionRequest` (`HummingbirdServer.swift:16`) decodes
`model`, `messages`, `stream`, `temperature`, `top_p`, `max_tokens`, `tools` and
`tool_choice`, but has no `stream_options` field, so the request option is
dropped silently. And `"usage"` is emitted only from `nonStreamingChatResponse`
(line 2492, usage at 2632); `streamingChatResponse` (line 2637) never sends a
usage frame at all. A repo-wide search for `stream_options`, `streamOptions` and
`include_usage` across `MacMLXCore` finds one hit, and it is a comment about the
Ollama request shape — not an implementation.

The gap is small and well defined, and it degrades rather than fails: without a
usage frame the executor cannot emit `.updateUsage`, so `cachedTokenCount` —
which is exactly what macMLX's tiered prompt cache is for — never reaches the
app, and neither do the `tokensPerSecond` / `timeToFirstToken` metadata values
Apple's own provider examples surface.

### Xcode 27 as a consumer

Xcode 27 ships `OpenAICompliantChatModelProvider` in
`IDEIntelligenceModelService`, alongside an `openAIBaseURL` setting, so pointing
it at macMLX is a supported configuration rather than a hack.

Its model-discovery call, captured from a real Xcode 27 against the stub:

```
GET /v1/models?
User-Agent: Xcode/25183.107.5 CFNetwork/3896.100.1.1.1 Darwin/27.0.0
Accept: application/json
Authorization: Bearer
```

Two things follow, both verified against Hummingbird 2.25.0 — the exact version
macMLX pins — using `router.get("/v1/models")`, the exact registration
`HummingbirdServer.swift:1372` uses:

| path | result |
|---|---|
| `/v1/models` | 200 |
| `/v1/models?` — what Xcode actually sends | **200**, the query string is stripped by the router |
| `/v1/v1/models?` | 404 |

So macMLX's routing accepts Xcode's discovery request unchanged. But the base
URL entered in Xcode must **not** end in `/v1`: Xcode appends `/v1/models`
itself, and a base of `http://localhost:8000/v1` produces `/v1/v1/models`, which
404s. Documentation must state the correct value is `http://localhost:8000`.

A second consequence of `/v1/models` reporting **only the currently loaded
model**: with nothing loaded macMLX returns `{"object":"list","data":[]}`, and
Xcode's provider will have an empty model list. Whether Xcode treats that as a
misconfigured provider is untested. If it does, the model-list endpoint may need
to report available-but-not-loaded models for this integration to feel working
rather than broken.

### Guided generation has a real boundary, and it is narrower than it looks

The captured `response_format` is
`{"type":"json_schema","json_schema":{"name":…,"strict":true,"schema":{…}}}`.
`ResponseFormatDecoder.compileObjectSchema` reads the top level by key lookup,
so Apple's extra `title`, `x-order` and `$defs` keys are ignored harmlessly.
Property schemas are different: `compilePropertyType` runs an allow-list gate
(`type`, `enum`, `description`, `title`, `default`) and throws
`unsupportedFeature` — a 400 — for anything else.

Measured against schemas Apple actually emitted:

| `@Generable` shape | macMLX decoder |
|---|---|
| flat struct of `String` / `Int`, with or without `@Guide(description:)` | **accepted** — verified end to end against the stub |
| array property, e.g. `[String]` | **400** — Apple emits `items`, which is not allow-listed |
| nested `@Generable` | **400** — Apple emits `$ref: "#/$defs/…"`, which is not allow-listed |
| enum-backed property | untested; `enum` is allow-listed so it should pass |
| range or pattern guides | untested — the description guide used here emitted only `description`, so it is unknown whether `.range(…)` emits `minimum`/`maximum` |

So "macMLX supports guided generation through Apple's API" is true only for flat
structs today. Widening `allowedPropertyKeys` is not the fix on its own: the
constraint automaton has to actually enforce `items` and resolve `$ref`, or the
400 would become silently wrong output — which the allow-list exists to prevent.
This is a separate work item from B1, and larger. Until it is done, the
documentation must say which shapes work.

### Work items

**B1 — `stream_options.include_usage`.** Accept the field on
`/v1/chat/completions`; when set, emit a final SSE frame with an empty `choices`
array and a populated `usage` object before `[DONE]`, per the OpenAI contract.
Wire `TokenUsage.promptTokens` / `.completionTokens`, and report the prompt-cache
hit length as the cached-token count so the number is real rather than zero.

**B2 — `MacMLXLanguageModel`.** A new SPM product, `MacMLXFoundationModels`, in
its own target so its `platforms: [.macOS("27.0")]` requirement does not raise
the floor for `MacMLXCore` or for macMLX.app.

```swift
public struct MacMLXLanguageModel: LanguageModel {
    public typealias Executor = ChatCompletionsLanguageModel.Executor

    public var capabilities: LanguageModelCapabilities   // resolved per checkpoint
    public var executorConfiguration: Executor.Configuration

    /// Resolves capabilities from the running server, so the declared set
    /// matches the checkpoint actually loaded.
    public static func resolving(
        modelID: String,
        endpoint: URL = URL(string: "http://localhost:8000/v1/chat/completions")!,
        apiKey: String? = nil
    ) async throws -> MacMLXLanguageModel
}
```

The synchronous `init` stays available for callers who already know what they
loaded and do not want a probe round-trip; it takes an explicit capability set
rather than guessing.

**B3 — capability resolution.** Map `LocalModel` facts to the four capability
values: `vision` from VLM family detection, `toolCalling` from the chat
template's tool support, `guidedGeneration` from the constraint decoder's
coverage, `reasoning` from reasoning-model detection. Where macMLX genuinely
does not know, declare the capability absent — `unsupportedCapability` is a clean
failure; a silently wrong answer is not.

**B4 — documentation.** A page under `docs/` and a site fact, both stating what
is and is not claimed: this runs arbitrary MLX checkpoints through Apple's API,
it requires a running macMLX server, and it requires macOS 27 in the consuming
app.

### Explicitly not in scope

A native in-process executor that links `MacMLXCore` directly (the original
"direction A"). It would force every consuming app to carry MLX and Metal and to
manage model downloads, and it would duplicate the batching, pooling, and cache
work the server already does. Revisit only if a concrete consumer needs
inference with no server process, and note that such a consumer is probably
better served by Apple's own `MLXLanguageModel` once it ships in a release.

---

## Direction C — Apple's model as a macMLX engine

### Shape

`InferenceEngine` is already the seam every engine goes through, and the GUI
"never knows which engine runs". An AFM conformance drops in beside the MLX
engine:

```swift
@available(macOS 27.0, *)
public actor AFMEngine: InferenceEngine {
    public let engineID: EngineID = .appleFoundationModels
    // load/unload are availability transitions, not weight loading
}
```

`EngineID` is a `String`-backed `CaseIterable` enum with three cases today
(`mlxSwift`, `swiftLM`, `pythonMLX`). A fourth case
`appleFoundationModels = "apple-foundation-models"` is additive, but every
`CaseIterable` consumer and anything that persists an engine ID needs checking —
a settings file written by a newer build must not break an older one.

### Mapping

| `InferenceEngine` | AFM |
|---|---|
| `load(_:)` | assert `SystemLanguageModel.default.availability == .available`; construct a session. No weights, no download, no memory budget. |
| `unload()` | drop the session |
| `generate(_:)` | `session.streamResponse(to:)` → `GenerateChunk` |
| `applyAdapter(_:)` | throws. The adapter toolkit is EOL on 27 and `SystemLanguageModel(adapter:)` is gone. |
| `healthCheck()` | re-read `availability` |
| `version` | the OS build, since the model ships with the OS |

### What must be reported honestly rather than papered over

- **Sampling.** `GenerationOptions` exposes greedy, top-k, and a probability
  threshold with a seed. There is no `logitBias`, no XTC, no `kvBits`. Requests
  carrying them must be rejected with an explicit error, the same way the
  existing compatibility matrix rejects unsupported combinations, rather than
  silently ignoring the parameter.
- **Context.** `SystemLanguageModel.contextSize` reports 8192 here. That is a
  fraction of what MLX checkpoints run, and the UI has to show it.
- **Prompt cache.** The tiered SSD KV cache does not apply. AFM exposes
  `cachedTokenCount` as a readout with no control surface. The cache settings UI
  must be disabled for this engine, not silently ineffective.
- **Availability.** Three failure modes — `.deviceNotEligible`,
  `.appleIntelligenceNotEnabled`, `.modelNotReady` — each needs its own message.
  "Apple Intelligence is off" is a user-actionable setting; "this device is not
  eligible" is permanent.
- **Benchmarking.** Benchmark runs against this engine measure Apple's model on
  Apple's terms. Silicon bottleneck attribution reads process-level counters and
  AFM runs partly out of process, so attribution must be reported unavailable
  rather than wrong.

### Why it is worth doing anyway

It costs little, it gives every macMLX user a zero-download model that starts in
under a second, and it puts an honest side-by-side comparison in the product —
which is the kind of thing macMLX's benchmark surface exists for.

---

## Packaging and deployment targets

macMLX targets macOS 14. The provider protocols are macOS 27. Neither direction
may raise the floor.

- **B** ships as a separate SPM product with its own `platforms` line. Apps that
  adopt it are macOS 27 apps; macMLX itself is unaffected, and `MacMLXCore` does
  not gain a `FoundationModels` dependency.
- **C** lives inside `MacMLXCore` behind `@available(macOS 27.0, *)` and a
  runtime availability check, registered in the engine list only when the OS
  supports it. The existing `InferenceEngine` protocol needs no change.

## Verification plan

macMLX has no conditions for real-checkpoint testing, so this plan deliberately
contains no claim that depends on one.

**Can be verified without a real model:**

- B1 by unit test: a request with `stream_options.include_usage` produces the
  usage frame before `[DONE]`, and one without it does not. Mutation-verify by
  breaking the frame ordering and confirming the test fails.
- B1 wire shape against Apple's parser by replaying recorded SSE through
  `ChatCompletionsLanguageModel`'s decoding path with a stub server.
- B2/B3 by unit test over capability resolution: each `LocalModel` shape maps to
  the expected capability set, and an unknown shape declares the capability
  absent rather than present.
- C's error mapping by unit test with a stubbed availability value, including all
  three unavailable reasons.
- C's parameter rejection by unit test: a request carrying `logitBias` or
  `kvBits` fails with an explicit error.

**Cannot be verified here, and must be labelled as such:** end-to-end generation
quality through either path, throughput comparisons, and anything requiring a
downloaded checkpoint. The changelog and site wording must say "not validated
against real checkpoints", consistent with how the audio and reranker paths were
described in v0.9.0.

## Open questions

1. **Does Apple's `ChatCompletionsLanguageModel` actually drive macMLX's server
   end to end?** The audit above is static. The first implementation step should
   be an integration probe, not more design.
2. **Should `MacMLXFoundationModels` live in this repo or its own?** A separate
   repo keeps macMLX's release cadence free of a macOS 27 dependency; the same
   repo keeps the compatibility tests next to the server they test. Leaning
   toward the same repo, separate target.
3. **When `MLXFoundationModels` reaches a tagged mlx-swift-lm release, does B
   still earn its place?** It should: the server path gives an app model
   management, continuous batching, the tiered cache, and a model library it
   would otherwise build itself. But the positioning has to be stated against
   that alternative, not against a vacuum.

## Consequence to handle separately

`MacMLXCore/Package.swift:106` defers ml-explore/mlx#3963 on the stated grounds
that "CI, release and this machine are all pinned at Xcode 26.x". That premise
has expired: this machine now has only Xcode 27.0. CI and release still pin
26.4.1, so shipped builds are unaffected, but local builds and CI now use
different Metal compilers. Whether the Xcode 27 compiler actually rejects the
kernels has not been tested. This is not part of this design; it needs its own
investigation.
