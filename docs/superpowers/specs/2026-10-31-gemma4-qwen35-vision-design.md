# Images for Gemma 4 and Qwen3.5/3.6 on the full serving path — design

Status: draft for review, 2026-10-31. Nothing here is built yet.

## Why

Gemma 4 and Qwen3.6 are the two families people run here as local agents
(Xcode 27, Claude Code, the chat window). Their published checkpoints ship a
vision tower, but macMLX loads them text-only (`vlmTypesServedAsText` in
`ModelLibraryManager.swift`), so:

- the GUI hides the image button (`ChatViewModel.swift:323` gates it on
  `.mlxVLM`);
- an API request that carries an image gets an answer that never saw it. The
  only trace is a debug-level log line (`MLXSwiftEngine.userInput`).

Loading them through the VLM factory today would make images work but take
away what text requests have. This design gives them images without that
loss.

## What the VLM path lacks today

`runVLMGeneration` (`MLXSwiftEngine.swift`) is the stock
`generateTokensTask`. Against `runLLMGeneration` it has none of:

- prompt cache (hot or cold tier) and history-boundary snapshots;
- the streaming `ToolCallProcessor`, so tool calls come back as plain text;
- structured output, `logit_bias`, XTC, `logprobs` (logged and ignored);
- speculative decoding with a draft model;
- replayed reasoning (`userInput` passes `messageGenerator: nil` for a VLM).

The tool-call gap alone rules the VLM path out for an agent model.

Batching is not on the list. The text path loads these checkpoints as
`Gemma4Model` / `Qwen35Model` (the multimodal wrappers in MLXLLM), and neither
type is on `BatchModelAllowlist`, so they do not batch today either. PR #163's
description said otherwise. That was wrong, and it changes nothing here.

## Upstream facts this relies on (mlx-swift-lm 3.32.3, 3b339ad)

- `VLMModelFactory` registers `gemma4`, `gemma4_unified`, `qwen3_5`, and
  `qwen3_5_moe`. `gemma4` uses `Gemma4Processor`, `qwen3_5` uses
  `Qwen3VLProcessor`. Both render through a message generator
  (`Gemma4MessageGenerator`, `Qwen3VLMessageGenerator`), and both pass a
  `.messages` prompt through, with images taken from `UserInput.images`.
- `gemma4_unified` is a different class (`Gemma4Unified`) with a different
  processor (`Gemma4UnifiedProcessor`). It shares `Gemma4MessageGenerator`,
  but its vision weights sit under `vision_embedder` / `embed_vision` and it
  keeps `embed_audio` resident. `qwen3_5_moe` is a subclass of `Qwen35`.
- VLM `Gemma4.prepare`, text-only input: `gemma4PrepareTextOnly`, with
  chunked prefill and any cache offset. VLM Gemma4 also has an MTP emit path,
  which the text path's `Gemma4Model` lacks. The `gemma4` VLM sanitizer
  drops the audio tower. With an image, `prepare` prefills in one shot, with
  no chunking.
- VLM `Qwen35.prepare` continues a warm cache only when `state` carries
  `qwen35.ropeDeltas`. Otherwise it throws
  `ContinuationStateError.missingState` (`QwenVL.continuationAnchor`). For an
  image-free prefix the delta is zero. `TokenIterator` has an `init` that
  takes `state:`. `LMOutput.Key` is public, so the key can be built from its
  id. The upstream constant is private.
- `VLMModel` refines `LoRAModel`. VLM `Gemma4`, `Gemma4Unified` and `Qwen35`
  all expose `loraLayers` as `languageModel.model.layers`, the same module
  path as the text-path classes.
- Draft-model speculative decoding (`SpeculativeTokenIterator`) needs
  trimmable caches. macMLX gates on `canTrimPromptCache` in
  `runLLMGeneration`. Gemma 4's caches pass, so Gemma 4 speculates today.
  Qwen3.5's `MambaCache` never trims, so Qwen3.5 falls back to plain
  decoding today. `SpeculativeCacheRewindModel` matters only to the MTP
  iterator.

## Measured cost

Extra resident weights when these load through the VLM factory, measured from
the local checkpoints' safetensors headers:

| Checkpoint | Total | Vision tower (loaded) |
|---|---|---|
| gemma-4-e4b-it-8bit | 8.35 GiB | 0.31 GiB (the `gemma4` sanitizer drops the 0.57 GiB audio tower) |
| Qwen3.6-27B-4bit | 14.95 GiB | 0.86 GiB |

## Options considered

1. **Two containers, LLM for text and VLM for images.** This doubles weight
   memory (about 30 GiB for the 27B). Sharing the arrays between the two
   would mean rebuilding one factory's load by hand: quantization, processor
   config, sanitize, EOS and tool-format resolution. That breaks on every
   upstream bump. Rejected.
2. **Swap the load per request.** A mixed chat would reload a 27B model on
   every switch. Rejected.
3. **Today's VLM path behind a per-model "vision" switch.** Cheap, but it
   trades away tool calls and the cache. Rejected.
4. **Load through the VLM factory and run every request through the LLM
   generation path.** Chosen. `runLLMGeneration` takes `any LanguageModel`,
   but it was written against the LLM processor's output and the
   `LLMModel` type, so the engine work below is what makes that true.
   - It slices 1-D token arrays with no mask.
   - It renders history through `(context.model as? LLMModel)`.
   - It puts the speculative and history-boundary branches on every
     request.

## Design

### Classification

A type moves out of `vlmTypesServedAsText` into a new set, the "full-path
VLM families", only after its own parity run (the gate below). This work
moves `gemma4` and `qwen3_5`, the two types with a local checkpoint.

- `gemma4_unified` has a different class, processor, weight names and
  resident audio. It stays on text until a checkpoint of it passes the gate.
- `qwen3_5_moe` has never been run here. It stays on text on the same
  terms.

The #163 guard tests must still classify every registry key exactly once.

A checkpoint of a full-path type is `.mlxVLM` only when all three hold:

- `config.json` has `vision_config`;
- the weights hold vision-tower tensors, as listed in
  `model.safetensors.index.json` or else in the safetensors headers:
  - gemma4: `vision_tower.*` or `embed_vision.*`;
  - qwen3_5: `vision_tower.*`, or `model.visual.*` in raw HF layout;
- the processor metadata the VLM factory reads is present
  (`preprocessor_config.json` or `processor_config.json`).

Everything else stays `.mlx`, so a checkpoint missing any of them loads as
text, exactly as today. `Qwen3.6-27B-AEON…-8Bit` has none of them and is the
guard case. `gemma4_text` and `qwen3_5_text` are untouched.

The engine decides "full path" again at load, from `config.json`, and does
not trust `LocalModel.architecture`: hand-built models and tests leave it
`nil`.

If a VLM load of a full-path checkpoint fails, it is not retried through the
LLM factory. The error surfaces, so a broken vision load is never masked.

### Engine

**Prepared input.** The VLM processors return `[1, N]` tokens and an int8
mask, even for text-only input:

- `Gemma4Processor.prepare` (`Gemma4.swift:2853-2856`);
- `Qwen3VLProcessor` (`Qwen3VL.swift:119-121`).

`LLMUserInputProcessor` returns `[N]` with no mask. `runLLMGeneration`
slices axis 0, at `MLXSwiftEngine.swift:1922` and `:2013-2022`. On `[1, N]`
that is an empty array, or the whole prompt treated as history.

So a text-only prepared input on a full-path load is normalized to `[N]`
tokens with `mask = nil` before any cache, boundary or speculative logic. An
image-bearing input keeps the processor's shape, and nothing slices it
except through the image rule below.

A unit test drives a stub processor that returns `[1, N]` plus a mask
through the hit path and the boundary path.

**Message generator.** `LoadedSupport.vlm` gains the generator its
processor renders with:

- `Gemma4MessageGenerator` for `gemma4`;
- `Qwen3VLMessageGenerator` for `qwen3_5`.

Other VLMs get `nil`, as today. A test pins two inputs to the same tokens
and the same image count: `.chat` through the processor, and generator plus
`.messages` through the processor. The same generator reaches the history
render (`MLXSwiftEngine+HistoryBoundary.swift:52-55`), which today asks
`as? LLMModel` and falls back to `DefaultMessageGenerator`.

**Reasoning with images.** The reasoning branch of `userInput` builds
`UserInput(messages:tools:additionalContext:)`, whose `images` defaults to
empty (`UserInput.swift:504`). The template would still emit image
placeholders, but the processor would see no images, and the model would
answer without the picture. So that branch passes `images:` and `videos:`
through from `chatInput`.

**Tool-call format.** The `inferToolCallFormatFallback` backfill runs on
these containers too.

**Dispatch.**

- Full-path family → `runLLMGeneration`.
- Every other VLM → `runVLMGeneration`, unchanged.

**Images and the prompt cache.** Each image expands to the same placeholder
ids whatever its pixels. A key made only of token ids would let one image's
KV answer for another, so image requests are handled in two steps.

- **Step 1 (engine PR).** An image-bearing request may reuse a cached prefix
  that ends before its first image block:
  - reused length = `min(longest common prefix, start of the first image
    block)`;
  - the block boundary is never split, since Gemma 4 attends
    bidirectionally inside an image block;
  - before the first image, KV and positions do not depend on pixels;
  - the request records no entry and takes no history-boundary snapshot.
  - Qwen3.5 continues warm through an image-bearing remainder upstream
    (`Qwen35.swift:1287-1385`). Gemma 4's warm image `prepare` must be shown
    to work, by gate 4, before Gemma 4 uses this step. If it fails, Gemma 4
    image requests run cold.
- **Step 2 (its own PR, in scope).** Image-digest keys:
  - each image block's placeholder ids are replaced in the cache key, and
    only there, by ids derived from the image file's SHA-256, out of the
    vocabulary's range, so different images never match;
  - with that, image requests record entries and later turns of an image
    session reuse the whole history;
  - Qwen3.5 needs its rope delta stored with the entry, in both hot and cold
    tiers, because after an image it is no longer zero;
  - Gemma 4 has no such state;
  - until step 2 lands for a family, a session that once had an image
    re-prefills from its first image on every turn, and gate 5 measures what
    that costs.

**Qwen3.5 warm continuation.** Every `TokenIterator` that starts on a
non-empty cache gets `state` carrying `qwen35.ropeDeltas`:

- zero for an image-free prefix (all of step 1, and every text request);
- the stored delta for a step-2 entry.

The call sites are `prefillHistory` (which throws its iterator's state
away, so the generation leg after it starts warm even on a cache miss),
`customTokenStream`, and the stock `generateTokensTask` call.

A missing seed is not a thrown error. Decode trips the `precondition` at
`Qwen35.swift:1390-1392` and kills the process. The tests drive a real warm
continuation, and a unit test pins the key id against the resolved upstream
source, as #163 does for the registry.

**Speculative decoding.**

- Text requests keep today's `canTrimPromptCache` gate unchanged: Gemma 4
  keeps speculating, Qwen3.5 keeps falling back.
- Image-bearing requests never speculate. An explicit check skips the draft
  and logs it. The old log line lives in the `if isVLM` branch, which these
  requests no longer take.
- The draft is fed the normalized `[N]` input.

**Draft models.** `LocalModel.draftCandidates` keeps only `.mlx`, so a
flipped small checkpoint would fall out of the draft picker.

- The filter also admits `.mlxVLM` models of a full-path family.
- `ensureDraftContainer` keeps loading drafts through `LLMModelFactory`,
  text-only, as now.

**LoRA adapters.** These loads keep accepting adapters. `VLMModel` refines
`LoRAModel` with the same layer paths. A unit test pins that the VLM class's
`loraLayers` key paths match the text class's. The gate runs an adapter only
if a local one exists, and none does today.

**Cold tier across the flip.** `ModelFingerprint` hashes `config.json` and
shard stats, so after the flip an entry written by the text-path class
would be restored into the VLM class. The fingerprint is salted with the
loader path (`llm` / `vlm`), so pre-flip entries miss rather than load into
a different implementation.

**Batching.** No change. VLM loads keep `clearBatchServingState()`.

**MTP.** No change in this work. `MTPDrafterDetection`'s comments assume
`gemma4` loads as `Gemma4Model`, and they get updated.

### GUI and server

There is nothing new to build. The image button and the "Vision" badge follow
`.mlxVLM`. The server already decodes images on chat completions, Anthropic
messages and responses. Ollama chat does not, and stays that way.

## Parity gate (each family, before it flips)

Gated real-model tests run through xcodebuild with `TEST_RUNNER_` env, on
`gemma-4-e4b-it-8bit` and `Qwen3.6-27B-4bit`. The engine PR ships with the
family set empty, so the tests reach the full path through a test-only seam
that forces it.

**Existing suites on the new load.** The gated suites below hardcode
`format: .mlx` on these exact checkpoints, so after the flip they would keep
testing the text path:

- `HistoryBoundaryPromptCacheSmokeTests`
- `ReasoningHistorySmokeTests`
- `Gemma4PromptCacheSmokeTests`
- `AnthropicCountTokensE2ETests`
- `Qwen36ModelTests`

They run on both loads, and a family flips only when they pass on the
full-path load.

**New gates.**

1. **Same tokens.** For each fixed agent-shaped prompt, the two loads produce
   identical prompt token ids. The prompts are over 256 tokens, so the
   boundary snapshot runs, and they carry a system turn, tools, tool calls,
   tool results and replayed reasoning.
2. **Same distribution.** On those prompts, the two loads stay close:
   - first-step logits within max |Δlogit| ≤ 0.05;
   - KL(text ‖ vision) ≤ 1e-3;
   - a greedy 64-token continuation token-identical up to the first position
     where the text load's top-2 margin falls below 0.05.

   The two paths are separate implementations (norm classes, prefill
   chunking, M-RoPE vs 1-D RoPE), so bit-identity is not expected. These
   bounds are the starting point, and the builder may tighten them after the
   first run, never loosen them.
3. **Cache.** On the same agent-shaped two-turn conversation, the full-path
   load reuses the same number of tokens as the text-path load, and the
   Qwen3.5 warm path runs.
4. **Image.** All on the API paths:
   - a fixture image is described correctly;
   - an image request with tools returns a parsed tool call;
   - an image with replayed reasoning and tool-result history still sees the
     image;
   - a text request after an image request never gets a hit from the image
     request's KV;
   - step 1: an image request after a long text prefix reuses that prefix
     and answers about its own image.
5. **Image session.** Five turns of a 16k-token agent conversation with an
   image in turn 1, recording the prefill time and peak memory of each turn.
   Gemma 4 prefills an image remainder in one shot. If a turn does not fit
   where the same conversation without the image fits, a guard refuses image
   requests above a measured size with a clear error, before Metal fails.
   The gate states what the guard does to a session mid-way. After step 2,
   turns 2–5 reuse the history.
6. **Memory.** Resident growth after load is the vision-tower size, within
   the noise.
7. **Unchanged speculation.** Gemma 4 E4B, with itself as a draft loaded
   through `LLMModelFactory`, still speculates on a text request on the
   full-path load, with the same output as on the text-path load.

Any failed gate, or any existing suite failing on the full-path load, keeps
that family on text. A finding that is upstream's goes there as a report
only, not a code PR.

## Rollout, one small PR each

1. **Engine seam.** All of the engine work above:
   - input normalization;
   - the generator on `.vlm` and in the history render;
   - images through the reasoning branch;
   - the dispatch;
   - image cache step 1;
   - the Qwen3.5 state seed;
   - the speculation check;
   - the loader-path fingerprint salt.

   Include unit tests and the gated suites behind the forcing seam. The
   family set is empty, so nothing changes for users.
2. **Gemma 4 flips.** Classification with the weight and processor checks,
   the draft filter, and CHANGELOG and README.
3. **Qwen3.5/3.6 flips**, the same way.
4. **Image cache step 2** (digest keys, with Qwen's stored rope delta).

Qwen3.5 MTP (v0.11, waiting on mlx-swift-lm 3.33) is wired after Qwen
flips. MLXVLM has its own `Qwen35MTP.swift`, so it targets the VLM `Qwen35`
for a checkpoint with a vision tower.

Later, and not committed: `gemma4_unified` and `qwen3_5_moe` flip after
their own parity runs, and other VLMs move to the full path after theirs.

## Goals, non-goals, anti-goals

- **Goal:** images work on these two families with tool calls, constraints,
  reasoning replay and the text-request cache intact, and a long agent
  session with an image stays usable.
- **Non-goals:** batching, video and audio input, moving other VLM families,
  MTP.
- **Anti-goals**, each a failure even if images work:
  - any change to text-only output, cache reuse, speculation or tool calls
    for these families;
  - resident memory beyond the vision tower;
  - an image answer built on another image's cache;
  - an image silently dropped on any path.

## Calls made without asking

Two reviews raised these, and I settled them from decisions already made.
Each can be overruled.

- Only `gemma4` and `qwen3_5` flip in this work.
- Speculation keeps today's gate for text and is off for images.
- Adapters keep working rather than being refused.
- Image-digest cache keys are in scope as their own PR, because without them
  an agent session that once had an image re-prefills every turn.
- Pre-flip cold-tier entries are dropped (fingerprint salt) rather than
  restored into the other implementation.
- A failed VLM load is not retried as text.
- Qwen3.5 MTP comes after the flip and targets the VLM class.
- There is no upstream fix: no code PRs to mlx-swift-lm.

## Left to the builder

- The shape of the `LoadedSupport` change (an extra associated value or a
  new case).
- The prompt set for the gates, within the shape stated above.
- How the weight check reads the index.
- Where the full-path set lives (`ModelLibraryManager` or the engine).
- The id range for image-digest key tokens.
