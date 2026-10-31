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
  `qwen3_5_moe`. The processors are `Gemma4Processor` and `Qwen3VLProcessor`.
  Both render through a message generator (`Gemma4MessageGenerator`,
  `Qwen3VLMessageGenerator`). Both pass a `.messages` prompt through, with
  images taken from `UserInput.images`.
- VLM `Gemma4.prepare`, text-only input: `gemma4PrepareTextOnly`, with
  chunked prefill and any cache offset. VLM Gemma4 also has an MTP emit path,
  which the text path's `Gemma4Model` lacks. The VLM sanitizer drops the
  audio tower.
- VLM `Qwen35.prepare` continues a warm cache only when `state` carries
  `qwen35.ropeDeltas`. Otherwise it throws
  `ContinuationStateError.missingState` (`QwenVL.continuationAnchor`). For an
  image-free prefix the delta is zero. `TokenIterator` has an `init` that
  takes `state:`. `LMOutput.Key` is public, so the key can be built from its
  id. The upstream constant is private.
- VLM `Qwen35` conforms to `SpeculativeCacheRewindModel`. VLM `Gemma4` does
  not. Neither VLM class conforms to `LoRAModel`.

## Measured cost

Extra resident weights when these load through the VLM factory, measured from
the local checkpoints' safetensors headers:

| Checkpoint | Total | Vision tower (loaded) |
|---|---|---|
| gemma-4-e4b-it-8bit | 8.35 GiB | 0.31 GiB (the 0.57 GiB audio tower is dropped) |
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
   generation path.** `runLLMGeneration` works on `any LanguageModel`, so
   nothing in it depends on the model being an `LLMModel`. Chosen.

## Design

### Classification

`gemma4`, `gemma4_unified`, `qwen3_5`, and `qwen3_5_moe` become `.mlxVLM` only
when the checkpoint really carries the vision tower:

- `config.json` has `vision_config`, and
- the weights hold vision-tower tensors, as listed in
  `model.safetensors.index.json`, or else in the safetensors headers.

Everything else stays `.mlx`. `Qwen3.6-27B-AEON…-8Bit` has neither, so it is
the guard case. `gemma4_text` and `qwen3_5_text` are untouched.

The four types move out of `vlmTypesServedAsText` into a new set, the
"full-path VLM families". The #163 guard tests must still classify every
registry key exactly once.

### Engine

- `LoadedSupport.vlm` gains the message generator its processor renders with.
  For the full-path families that is `Gemma4MessageGenerator` or
  `Qwen3VLMessageGenerator`, picked by `model_type`. Other VLMs get `nil`, as
  today. The `inferToolCallFormatFallback` backfill runs on these containers
  too.
- `generate` dispatch:
  - full-path family → `runLLMGeneration`, images included in the
    `UserInput`;
  - every other VLM → `runVLMGeneration`, unchanged.
- **Prompt cache, image requests.** When any message in the request carries
  an image there is no cache lookup, no record and no history-boundary
  snapshot. Each image expands to the same placeholder token ids whatever
  its pixels, so a token-keyed hit would reuse another image's KV and answer
  about the wrong picture.
  - Text-only requests use the cache exactly as on the text path.
  - Cost: a chat that once had an image re-prefills every later turn. That
    matches every VLM today.
- **Qwen3.5 warm continuation.** When a restored prefix is reused, the
  iterator gets `state` with `qwen35.ropeDeltas` = 0. The prefix is image-free
  by the rule above, so zero is the right delta.
  - A test pins the key id against the resolved upstream source, the same way
    #163 pins the registry.
  - A second test drives a real warm continuation and checks it does not
    throw.
- **Speculative decoding.** It stays on for text requests when the model
  conforms to `SpeculativeCacheRewindModel` (Qwen3.5 yes, Gemma 4 no), and is
  ignored with the existing log line otherwise. It is always off for image
  requests.
- **LoRA adapters.** Neither VLM class is a `LoRAModel`. A request naming an
  adapter on these loads fails with a clear error, never silently ignored.
  Today, as `Gemma4Model`, Gemma 4 accepts adapters, so this is a regression
  and goes in the CHANGELOG.
- **Batching.** No change. VLM loads keep `clearBatchServingState()`.
- **MTP.** No change in this work. `MTPDrafterDetection`'s comments assume
  `gemma4` loads as `Gemma4Model`. They get updated, and the MTP work picks up
  the VLM emit path later.

### GUI and server

There is nothing new to build. The image button and the "Vision" badge follow
`.mlxVLM`, and the server already decodes images on all three APIs.

## Parity gate (each family, before it flips)

Gated real-model tests, run through xcodebuild with `TEST_RUNNER_` env, on
`gemma-4-e4b-it-8bit` and `Qwen3.6-27B-4bit`:

1. **Same tokens.** For a fixed set of chat prompts (system turn, tools,
   replayed reasoning), the VLM-loaded and LLM-loaded containers produce
   identical prompt token ids.
2. **Same text.** A greedy 64-token continuation is token-identical, and
   first-step logits agree within a tolerance the builder sets from a
   same-path rerun. Both loads use the same quantized weights.
3. **Cache.** A two-turn text chat hits the prompt cache on turn 2, on both
   families (the Qwen3.5 warm path does not throw).
4. **Image.** A fixture image is described correctly. An image request with
   tools returns a parsed tool call. A text-only request after an image
   request never gets a hit from the image request's KV.
5. **Memory.** Resident growth after load is the vision-tower size, within
   the noise.

If 1 or 2 fails for a family, that family stays on text, and the finding goes
upstream as a report only, not a code PR.

## Rollout, one small PR each

1. Engine seam: generator on `.vlm`, the dispatch, the image-request cache
   bypass and the Qwen3.5 state seed. Include unit tests and the gated parity
   tests. The set of families is still empty, so nothing changes for users.
2. Gemma 4 flips: classification with the weight check, plus CHANGELOG and
   README.
3. Qwen3.5/3.6 flips, the same way.
4. Later, and not committed: other VLMs move to the full path after their own
   parity run, and an image-aware cache key folds image digests into the hash.

## Goals, non-goals, anti-goals

- Goal: images work on these two families with tool calls, constraints and
  the text-request cache intact.
- Non-goals: image-aware prompt cache, batching, video and audio input,
  moving other VLM families, MTP.
- Anti-goals, each a failure even if images work:
  - any change to text-only output, cache hit rate or tool calls for these
    families;
  - resident memory beyond the vision tower;
  - an image answer built on another image's cache.

## Left to the builder

- The shape of the `LoadedSupport` change (an extra associated value or a
  new case).
- The logits tolerance.
- The prompt set for the parity tests.
- How the weight check reads the index.
- Whether the full-path set lives in `ModelLibraryManager` or the engine.
