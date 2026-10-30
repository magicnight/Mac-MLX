# Xcode 27 and Foundation Models

macMLX's server can be the model behind Xcode 27's coding assistant and behind
an app that uses Apple's Foundation Models framework through
`ChatCompletionsLanguageModel` from
[apple/foundation-models-utilities](https://github.com/apple/foundation-models-utilities).
Either one talks to a running macMLX server over HTTP, on
`/v1/chat/completions`; nothing from macMLX is linked into the app. So:

- macMLX's server must be running: start it from the menu bar or in
  Settings (turn on "Auto-start server on launch" to have it start with the
  app), or run `macmlx serve`.
- An app using the Foundation Models API needs macOS 27, which is where a
  third-party `LanguageModel` can drive a `LanguageModelSession`.
- Nothing has to be loaded first: `GET /v1/models` lists every chat model in
  macMLX's model directory, and the first request naming one loads it. Models
  the app finds only in the Hugging Face cache are not listed.

What follows was checked against a real `macmlx serve`: the Foundation Models
API on Qwen3.6-27B-4bit and Gemma 4 E4B (8-bit) with foundation-models-utilities
at `cc3820d`, and Xcode 27.1 on Qwen3.6-27B-4bit. Other MLX checkpoints go
through the same server path and should behave the same, but were not run.

## Xcode 27

In Xcode's Intelligence settings, add a locally hosted model provider on
macMLX's port (8000 unless you changed it). If Xcode asks for a base URL, give
`http://localhost:8000`, without `/v1`: Xcode appends `/v1/models` itself. (The
base URL macMLX's Server settings show ends in `/v1`; leave that part off.)

Xcode lists the downloaded models and loads the one you pick when you send the
first message, so that first answer waits for the load. Follow-up messages in
the same conversation work. Xcode sends no `max_tokens`, so a reply may run to
32,768 tokens, reasoning included; it does not show the reasoning, so a thinking
model looks idle until its answer starts. On Qwen3.6-27B-4bit the first reply
of a conversation took 902 prompt tokens (mostly Xcode's own system prompt) and
1,869 completion tokens.

## Foundation Models

```swift
import FoundationModels
import FoundationModelsUtilities

let model = ChatCompletionsLanguageModel(
    name: "Qwen3.6-27B-4bit",                  // an id from GET /v1/models
    url: URL(string: "http://127.0.0.1:8000")!)
let session = LanguageModelSession(model: model)
let answer = try await session.respond(to: "Why is the sky blue?")
```

The client appends `/v1/chat/completions` to the URL, or only
`/chat/completions` when the URL already has a version segment (`/v1`, `/v2`,
…), so pass the server's address, never the full endpoint. With bearer auth on,
pass `additionalHeaders: ["Authorization": "Bearer <key>"]`. The default URL session
gives up after 60 seconds without a byte; loading a large model and prefilling a
long prompt can take longer than that, so pass a `urlSessionConfiguration` with
a larger `timeoutIntervalForRequest` if the model may not be loaded yet.

| Checked | Qwen3.6-27B-4bit | Gemma 4 E4B |
|---|---|---|
| `respond` and `streamResponse` | works | works |
| A session over several turns, with instructions | works; the reasoning the session records is sent back and reaches the template | works |
| A tool, a tool loop, and a second turn after it | works | works |
| `.greedy` sampling | runs, but the model can reason in a loop until the limit (see below) | works; three runs gave the same answer |
| `maximumResponseTokens` | honored; 32 left no answer after the reasoning (see below) | honored |
| Guided generation: a flat struct, a nested `@Generable`, `[String]`, `.count(…)`, an enum, `.range(…)`, an optional, `Double` | works (`Double` not run) | works; an unbounded `Int` is often a poor value (see below) |
| Streamed guided generation | works | works |
| `@Guide(.pattern(…))` | 400 | 400 |

Inside a tool loop the prompt cache serves the conversation so far: on
Qwen3.6-27B-4bit the turn after the tool output reused 402 of its 429 prompt
tokens, and with 60 lines of instructions 2,342 of 2,572.

## Limits

- **Schemas.** Guided generation uses macMLX's JSON Schema subset (see the
  changelog for what it covers). `pattern`, `multipleOf`, `null` and unions are
  not in it, and a schema using them is answered with a 400, which the client
  throws as an HTTP error before anything is generated. The model never writes
  a value the constraint does not allow, but within it the value is the
  model's: Gemma 4 E4B often wrote Lyon's population as 5180000000000000000, as
  though it wanted a thousands separator after 518 and, refused one, carried on
  with zeros. An unbounded `Int` is held to the range of a 64-bit integer, so
  it always parses; a `.range(…)` guide bounds it where that matters. An
  unbounded `Double` has no such limit: the constraint lets its digits run on,
  so a model could write one until the token limit, or past what a `Double`
  holds. It did not happen in these runs; a `.range(…)` guide closes it.
- **Thinking models.** `maximumResponseTokens` counts the reasoning too. A
  small limit can be spent before the answer starts, and the client then
  throws "The model produced an empty response". With `.greedy`, Qwen3.6
  sometimes reasons in a loop until the limit, a known trait of greedy
  decoding on thinking models; sample instead, or use a model that does not
  think.
- **Sampling.** The client itself refuses top-k sampling and a fixed seed, and
  sends `.greedy` as `top_p: 0`, which macMLX samples greedily.
- **Capabilities.** `ChatCompletionsLanguageModel` declares vision, tool
  calling and reasoning for every model, and guided generation too unless it
  is created with `supportsGuidedGeneration: false`. macMLX does not yet tell
  the app which of these the loaded checkpoint really has.
- **Browsers.** A request that carries an `Origin` header, as a web page's
  does, sees only the loaded model in the list, so a page cannot learn which
  models are downloaded.

Not checked yet: images, a model other than these two, what the app sees when
no server is running, and the Ollama endpoints.
