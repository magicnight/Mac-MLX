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
   generation path.** `runLLMGeneration` works on `any LanguageModel`, so
   nothing in it depends on the model being an `LLMModel`. Chosen.

## Design

### Classification

The types in the full-path set become `.mlxVLM` only when the checkpoint
really carries the vision tower:

- `config.json` has `vision_config`, and
- the weights hold vision-tower tensors, as listed in
  `model.safetensors.index.json`, or else in the safetensors headers.

Everything else stays `.mlx`. `Qwen3.6-27B-AEON…-8Bit` has neither, so it is
the guard case. `gemma4_text` and `qwen3_5_text` are untouched.

A type moves out of `vlmTypesServedAsText` into a new set, the "full-path
VLM families", only after its own parity run (see the gate below). This work
moves `gemma4` and `qwen3_5`, the two types with a local checkpoint.
`gemma4_unified` (different class, processor, weight names and resident
audio) and `qwen3_5_moe` (never run here) stay on text until a checkpoint of
each has passed the gate. The #163 guard tests must still classify every
registry key exactly once.

### Engine

- `LoadedSupport.vlm` gains the message generator its processor renders with.
  For the full-path families that is `Gemma4MessageGenerator` (`gemma4`) or
  `Qwen3VLMessageGenerator` (`qwen3_5`), picked by `model_type`. Other VLMs get `nil`, as
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
- **Qwen3.5 warm continuation.** Every `TokenIterator` that starts on a
  non-empty cache gets `state` with `qwen35.ropeDeltas` = 0. That covers a
  restored prefix, and also the history-boundary path, which starts warm
  inside one request even on a cache miss: `prefillHistory` throws its
  iterator's state away, and the generation leg after it starts on a
  non-empty cache. The call sites are `prefillHistory`, `customTokenStream`
  and the stock `generateTokensTask` call. Any warm start carries an
  image-free prefix, by the rule above, so zero is the right delta.
  - A test pins the key id against the resolved upstream source, the same way
    #163 pins the registry.
  - A second test drives a real warm continuation and checks it does not
    throw.
- **Speculative decoding.** Text requests keep today's `canTrimPromptCache`
  gate unchanged: Gemma 4 keeps speculating, Qwen3.5 keeps falling back.
  Image requests never speculate (the existing log line).
- **Draft models.** `LocalModel.draftCandidates` keeps only `.mlx`, so a
  flipped small checkpoint (Gemma 4 E2B) would fall out of the draft picker.
  The filter also admits `.mlxVLM` models of a full-path family, and
  `ensureDraftContainer` keeps loading drafts through `LLMModelFactory`,
  text-only, as now.
- **LoRA adapters.** These loads keep accepting adapters. The layer path is
  the same, so an adapter trained against the text-path class should apply
  unchanged. The gate checks it when a local adapter for the family exists.
  Without one, a unit test pins that the VLM class's `loraLayers` paths match
  the text class's.
- **Batching.** No change. VLM loads keep `clearBatchServingState()`.
- **MTP.** No change in this work. `MTPDrafterDetection`'s comments assume
  `gemma4` loads as `Gemma4Model`. They get updated, and the MTP work picks up
  the VLM emit path later.

### GUI and server

There is nothing new to build. The image button and the "Vision" badge follow
`.mlxVLM`, and the server already decodes images on chat completions, Anthropic
messages and responses (Ollama chat does not, and stays that way).

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
5. **Agent-sized image prompt.** An image request behind a 16k-token prompt,
   the shape Xcode and Claude Code send. Gemma 4 prefills it in one shot,
   so record its peak memory. If it does not fit where the same text-only
   prompt fits, add a guard that refuses the request with a clear error
   above a measured size, rather than letting Metal fail.
6. **Memory.** Resident growth after load is the vision-tower size, within
   the noise.
7. **Unchanged text features.** Gemma 4 still speculates with a draft model
   on text requests, and a LoRA adapter, if one exists locally, gives the same
   output as on the text-path load.

If 1, 2 or 7 fails for a family, that family stays on text, and the finding goes
upstream as a report only, not a code PR.

## Rollout, one small PR each

1. Engine seam: generator on `.vlm`, the dispatch, the image-request cache
   bypass and the Qwen3.5 state seed. Include unit tests and the gated parity
   tests. The set of families is still empty, so nothing changes for users.
2. Gemma 4 flips: classification with the weight check, plus CHANGELOG and
   README.
3. Qwen3.5/3.6 flips, the same way.

Qwen3.5 MTP (v0.11, waiting on mlx-swift-lm 3.33) is wired after this flip
and targets whatever class the flip leaves loaded. MLXVLM has its own
`Qwen35MTP.swift`, so that is the VLM `Qwen35` for a checkpoint with a
vision tower.
4. Later, and not committed: `gemma4_unified` and `qwen3_5_moe` flip after
   their own parity runs, other VLMs move to the full path after their own
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

## Calls made without asking

A builder check raised these, and I settled them from decisions already
made. Each can be overruled.

- Only `gemma4` and `qwen3_5` flip in this work. `gemma4_unified` and
  `qwen3_5_moe` wait for their own parity runs.
- Speculative decoding keeps today's gate, so nothing about it changes for
  text requests.
- Adapters keep working rather than being refused.
- Qwen3.5 MTP comes after the flip and targets the VLM class.
- Agent-sized image prompts are measured. A size guard is added only if the
  measurement calls for one, and there is no upstream fix (no code PRs to
  mlx-swift-lm).

## Left to the builder

- The shape of the `LoadedSupport` change (an extra associated value or a
  new case).
- The logits tolerance.
- The prompt set for the parity tests.
- How the weight check reads the index.
- Whether the full-path set lives in `ModelLibraryManager` or the engine.
