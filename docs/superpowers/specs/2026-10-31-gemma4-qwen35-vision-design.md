# Images for Gemma 4 and Qwen3.5/3.6 on the full serving path — design

Status: approved 2026-10-31. Rollout PR 1 (the engine seam) is split in
two: the seam itself first (#165), with every image request running cold,
then image cache step 1 (#166). PR 2 flips Gemma 4 (#167); its parity run
changed three things, listed under "What the Gemma 4 parity run changed"
below. Qwen3.5/3.6 did not pass its run and stays on text (see "What the
Qwen3.6 parity run found"). E2B, the 26B-A4B and the 31B dense were run
too and stay text (see "What the other Gemma 4 sizes' runs found").

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

- **Step 1 (engine PR): reuse up to the first image block.**
  - Before the first image, KV and positions do not depend on pixels.
  - The block start is each family's opening token:
    - Qwen3.5: `<|vision_start|>`, which is how `getRopeIndex` finds images;
    - Gemma 4: `boi` (255999), which the processor inserts.
  - The ids come from `config.json`, with upstream's keys and defaults. The
    first placeholder token counts too, which can only cap reuse earlier.
    Qwen3.6 sets `vision_start_token_id` to 248053, not upstream's default.
  - Lookup is `fetchNearest(tokens: inputTokens[...blockStart])`, a capped
    key. It is not "longest common prefix, then clamp", because a `MambaCache`
    cannot be trimmed to a clamped length. The key ends with the opening
    token because the store always leaves a key's last token to prefill:
    reuse still ends at the block, and an entry stored under exactly the
    prefix is reused whole rather than trimmed by one.
  - The request records no entry and takes no history-boundary snapshot.
  - **Qwen3.5.** A warm `prepare` routes to `prepareContinuation`
    (`Qwen35.swift:1243-1248`). It runs the vision tower over the remainder
    and anchors positions at `cacheOffset + seed`, so a seed of 0 is right.
    The remainder's mask is sliced along the last axis together with its
    tokens, because `getRopeIndex` reads it.
  - **Gemma 4.** On a warm cache the mask is `[n, offset + n]`.
    `tokenTypeIds` covers only the remainder, so the bidirectional overlay's
    shape guard (`Gemma4.swift:309-311`) returns the plain causal mask, and
    no error is raised.
    - For a checkpoint whose `text_config.use_bidirectional_attention` is
      `"vision"`, a warm image request would therefore silently attend
      causally inside the image block. Step 1 is off for those checkpoints,
      and their image requests run cold.
    - The local e4b has the flag unset, so it is unaffected. No local
      checkpoint sets it, so a unit test drives the flag decision from a
      `config.json` fixture.
    - The overlay gap goes upstream as an issue report only, and only with
      the user's approval of the text.
- **Step 2: image-digest keys (a follow-up design, not this one).** Until it
  lands, a session that once had an image re-prefills from its first image
  on every turn, and gate 5 measures what that costs. The review found these
  constraints, and that design must meet all of them:
  - **Snapshots.** Cross-turn reuse for both families goes through
    history-boundary snapshots. Qwen's `MambaCache` cannot trim, and Gemma's
    sliding window cannot trim past 512 tokens. So image requests must take
    boundary snapshots, and `prefillHistory` must carry the images the
    boundary leg covers: only those after the reused prefix, which means a
    subset of the processed pixels, or reprocessing.
  - **Qwen rope delta.** It is read from a self-built `TokenIterator`'s
    `state` after `init` (`Evaluate.swift:720`), since `getRopeIndex` is
    internal and `generateTokenTask` consumes the iterator. It is stored with
    the entry as is (it is already offset-relative). It also seeds the
    generation leg after an image-bearing history leg in the same request.
  - **Key ids.** One per placeholder, so the key keeps its length and its
    alignment with the real tokens. Each comes from (image SHA-256, index in
    the block), mapped into `[vocabSize, Int32.max)`, because
    `PromptCacheKey.hash` converts with `Int32(tok)` and traps on overflow.
  - **Processor config.** The digest also covers the bytes of the processor
    config.
  - **Gemma 4 flag.** A new image on a warm cache hits the same overlay gap
    as step 1, so the same `"vision"` flag rule applies.
  - **Gate.** The same text with different images reuses exactly up to the
    block start.

**Qwen3.5 warm continuation.** Every `TokenIterator` that starts on a
non-empty cache gets `state` carrying `qwen35.ropeDeltas`:

- zero for an image-free prefix (all of step 1, and every text request);
- for step 2, the delta its design specifies.

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
2. **Same distribution.** On those prompts, compare the two loads in float32
   log-softmax over the text load's top 20 tokens:
   - the largest |Δ log p| stays within 4 bf16 ULPs of the text load's top-1
     logit for that prompt, written δ below, or within 1.25 times the text
     load's own |Δ log p| when the same prompt is prefilled in 128-token
     chunks, whichever is larger (changed on 2026-11-01, see "What the
     other Gemma 4 sizes' runs found");
   - the KL over those tokens stays below δ²/2, the second-order KL of a
     δ-sized shift;
   - a greedy 64-token continuation is token-identical up to the first
     position where the text load's top-2 margin is below δ.

   Both checkpoints compute in bf16, and Gemma 4 softcaps logits at 30, so one
   ULP is 0.0625–0.125 at typical top-logit sizes. The two paths are separate
   implementations (norm classes, prefill chunking, M-RoPE vs 1-D RoPE), so
   bit-identity is not expected. The builder records the observed values
   alongside the bounds.
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
   - step 1: an image request after a long text prefix, warm vs cold, meets
     the gate 2 bounds on both families. This is compared on logits, not
     judged by the answer.
5. **Image session.** Five turns of a 16k-token agent conversation with an
   image in turn 1, recording the prefill time and peak memory of each turn.
   Gemma 4 prefills an image remainder in one shot. If a turn does not fit
   where the same conversation without the image fits, a guard refuses image
   requests above a measured size with a clear error, before Metal fails.
   The gate states what the guard does to a session mid-way.
6. **Memory.** Resident growth after load is the vision-tower size, within
   the noise.
7. **Unchanged speculation.** Gemma 4 E4B, with itself as a draft loaded
   through `LLMModelFactory`, still speculates on a text request on the
   full-path load. Under greedy, its output is identical to the full-path
   load decoding without a draft.

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
4. **Image cache step 2**, after its own design is approved.

Qwen3.5 MTP (v0.11, waiting on mlx-swift-lm 3.33) is wired after Qwen
flips. MLXVLM has its own `Qwen35MTP.swift`, so it targets the VLM `Qwen35`
for a checkpoint with a vision tower.

Later, and not committed: `gemma4_unified` and `qwen3_5_moe` flip after
their own parity runs, and other VLMs move to the full path after theirs.

## What the Gemma 4 parity run changed

The gates ran on `gemma-4-e4b-it-8bit` (suites `FullPathVLMParitySmokeTests`
and `FullPathVLMImageGateSmokeTests`). Three findings changed the engine
before the flip:

- **Rendering (gate 1).** Upstream's family generators render every
  non-system message as content parts. Gemma 4's template keeps the model
  turn open after a tool result only when the calling assistant message has
  empty content, and a one-part list is not empty, so a prompt ending in a
  tool result closed the turn and opened none (|Δ log p| 5.7). The full path
  now renders a message without media with `DefaultMessageGenerator`, as the
  text load does, the family's generator only for a message with media
  (`FullPathMessageGenerator`), and always hands the processor `.messages`,
  since the processors render `.chat` with their own generator.
- **Speculation (gate 7).** mlx-swift-lm 3.32.3's `SpeculativeTokenIterator`
  never emits the first token when the target's `prepare` returns logits,
  which VLM `Gemma4.prepare` does, and the draft runs a token behind. The
  full path wraps its speculative target (`LastPromptTokenTarget`) so
  `prepare` leaves the last prompt token, as an LLM's does. A tiny-model test
  keeps the upstream drop pinned, so the wrapper can go when it is fixed.
  This is an upstream report candidate, sent only with the user's approval.
- **Image session (gate 5).** An image request's processor input runs in one
  shot. In the five-turn session, turn 5 (22.7k tokens) took 23.5 s to its
  first token at a 43.9 GB peak, against 3.2 s and 9.5 GB without the image
  in the same run. Rather than the guard the gate allowed for, Gemma 4 now
  prefills up to the last media token with the pixels and the text after it
  in chunks on the warm cache (`FullPathVLM.mediaPrefillSplit`): 3.9 s and
  11.1 GB, flat across turns, against 3.0 s and 9.5 GB without the image in
  that run. Attention past the block is causal. Qwen3.5 does not split;
  its text after an image needs the rope delta the image leaves, which the
  Qwen flip has to carry or measure without.

The weight check requires both `vision_tower.*` and `embed_vision.*`, since
the VLM class loads both.

The flip is narrowed to the shape the run covered
(`FullPathVLM.Family.parityRunCovers`): a Gemma 4 text config with
`attention_k_eq_v`, `use_bidirectional_attention: "vision"`,
`enable_moe_block` or `use_double_wide_mlp` stays text. The 31B dense sets
the first two, the 26B-A4B the third and E2B the fourth; each is read by
separate code in the two classes and needs its own run (each had one on
2026-11-01 and stays text; see "What the other Gemma 4 sizes' runs found"). A missing key reads
as both classes decode it, which for `use_double_wide_mlp` is `true`, so only
an explicit `false` (E4B's value) passes. Step 1's bidirectional rule still applies to them under the
forcing seam.

`HistoryBoundaryPromptCacheSmokeTests`' SSD-tier case assumed a snapshot
larger than the default 512 MiB in-memory tier. E4B's 7k-token snapshot is
smaller (sliding window, shared KV layers), so it failed on both loads alike;
the case now runs with a 32 MiB in-memory tier, so it holds on both models.

## What the Qwen3.6 parity run found

The same suites ran on `Qwen3.6-27B-4bit` through the forcing seam on
2026-10-31. Qwen3.5/3.6 stays on text.

- **Passed.**
  - Gate 1: the agent prompts render the same tokens on both loads.
  - Gate 3: turn 2 reuses 1,662 tokens on both.
  - Gate 6: growth is 921.7 MB against 921.5 MB of vision-tower tensors.
  - Gate 7: neither load speculates, since the `MambaCache` cannot be
    trimmed, and the text is the same.
  - Gate 5: the five-turn image session completes at an 18.6 GB peak,
    against 18.8 GB without the image. Upstream's `Qwen35.prepare` already
    runs an image remainder in windows (`prepareContinuation`), so this
    family needs no split.
- **Gate 2 failed on its absolute bound.**
  - Max |Δ log p| over the text load's top 20 is 1.44, 2.01 and 0.62 against
    δ = 0.5. KL is at most 2.0e-2, and greedy decoding agrees up to the first
    near-tie.
  - The text load against itself, prefilled in 128-token chunks instead,
    differs by 1.37, 1.73 and 0.87: the same size as the gap between the
    two classes. That suggests the gap is the hybrid model's own chunking
    noise in the low-probability tail, plausibly the linear-attention
    recurrence; the cause was not isolated.
  - On Gemma 4 E4B the same read is 0.24 to 0.50, under the bound.
  - Gate 2 for this family should compare against that noise rather than a
    fixed δ. On 2026-11-01 the user chose that bound (δ or 1.25 times the
    rechunked noise, whichever is larger); these numbers are inside it, but
    Qwen3.6 was not run again under it.
- **Gate 4 failed on reading.**
  - Over the API, Qwen3.6 read the fixture "MACMLX 2026" as "MAXLX COROS",
    and as "MAXX 2000" in a tool call.
  - On renders of the same sizes and text, the stock VLM path read exactly
    as the full path did, and Python mlx-vlm 0.7.11 on the same checkpoint
    misread too: "MAX MACHINES" for a 1280×320 render of "MACMLX 2026", and
    "HILL" for "HELLO WORLD" at 1024×1024. Both read colors right. The
    Swift and Python runs rendered their own PNGs.
  - The vision tower is unquantized bf16, so its quantization is ruled out.
    Swapping the patch kernel's H/W or T axes at runtime made the reading
    worse, so the loaded layout is not transposed by either of those. The
    misread is not the Swift port's alone.
  - Found on 2026-11-01: the checkpoint misreads these renders itself.
    transformers 5.22 on the original bf16 weights (`Qwen/Qwen3.6-27B`)
    reads the 1280×320 render as "maxsong", on MPS and on CPU alike, and
    as "MAX 2016" with thinking on. mlx-community's 8-bit conversion
    misreads it in mlx-vlm too ("maxson"), so neither the 4-bit language
    model nor the conversion is the cause, and neither MLX implementation.
    Qwen3.5-4B (8-bit) misreads it as "LILX 2026". The same render scaled
    up 2× (2560×640) reads "MACMLX 2026", and a tight crop nearly does
    ("MACMLX2020"): at this resolution each 32-pixel merged patch carries
    too little of each glyph for this family.
  - So gate 4 for this family needs a fixture this family reads in the
    reference implementation; the 2× render is one. The fixture change is
    left to the Qwen flip.
  - Step 1, warm against cold, differs by 0.75: comparable to the chunking
    noise above, which was measured on the text prompts, not this one.

The suites take the family through `MACMLX_FULL_PATH_PARITY_MODEL` and
`MACMLX_FULL_PATH_IMAGE_MODEL`. Gate 7 asserts unchanged speculation (the
full path speculates exactly where the text load does). Gate 2 prints the
rechunked read next to the bound.

## What the other Gemma 4 sizes' runs found

On 2026-11-01 the parity and image suites ran on the three other mlx-community
8-bit checkpoints: `gemma-4-e2b-it-8bit`, `gemma-4-26b-a4b-it-8bit` and
`gemma-4-31b-it-8bit`.

- **Gate 2 changed.** On all three, and on Qwen3.6, the first run missed the
  fixed δ while KL stayed far under its bound and greedy decoding agreed. The
  gap was of the same order as the text load's own difference when the same
  prompt is prefilled in 128-token chunks (up to 1.6 times it). The user chose a bound of δ or
  1.25 times that rechunked difference, whichever is larger. KL and greedy
  agreement are unchanged.
- **The cold tier, when a snapshot outgrows the hot tier.** A 31B
  history-boundary snapshot at 1.5k tokens is about 0.9 GB (50 sliding
  layers of 16 KV heads), more than the 512 MiB hot tier, and the suites ran
  without a cold tier, so turn 2 reused nothing on either load. With
  `MACMLX_FULL_PATH_COLD_TIER=1` each engine gets the default cold tier in a
  throwaway directory, and both loads reuse 1,452 tokens.
- **Image requests run cold on 31B and 26B-A4B, as designed.** Both use
  bidirectional vision attention, so `reusesPrefixBeforeMedia` is off. The
  image gate's step 1 and its cancel test now expect no reuse on such a
  checkpoint instead of failing on it.
- **31B dense: passes the gates, stays text.** Gate 2's largest |Δ log p| is 0.57, 0.60
  and 0.43, under bounds of 0.64, 0.78 and 0.50; KL at most 2.0e-7; greedy
  identical. Gate 3 reuses 1,452 on both loads with the cold tier; gate 6
  grows by 1,146.7 MB against 1,145.7 MB of vision tower; gate 7 speculates
  24/24 on both loads with the same text. The image is read, with `kv_bits` 8
  and 4 too, through a tool call and behind replayed reasoning, and the
  five-turn session completes at a 36,786 MiB peak.
- **An image late in a long session is the costly case, and it blocks the
  31B.** Gate 5 put the image in turn 1. With it in turn 5 instead (four
  text turns, then the image; `testAnImageLateInALongSessionCompletes`),
  everything before the image went through `prepare` in one shot, with no
  cancel point, because the split only chunked the text after the last
  image (as it still does on a bidirectional checkpoint):
  - E4B: 22,684 tokens, 1,450 reused, first token after 17.4 s at a 41,957
    MiB peak, 32,620 MiB above turn 4's.
  - 31B: 22,688 tokens, none reused (its image requests run cold), first
    token after 304.6 s at an 88,476 MiB peak, which a 64 GB Mac cannot
    hold.

  For E4B the text before the first image now prefills in chunks first
  (`FullPathVLM.textPrefillBeforeMedia`), as text, so `prepare` starts at
  the block as it does on a warm hit: turn 5 then took 3.6 s at a 9,486 MiB
  peak, 149 MiB above turn 4's, and the case asserts the peak stays within
  1 GiB of the text turn's. Since the cold run in step 1 now prepares the
  block on a cached prefix too, step 1 also compares the warm run with the
  one-shot `prepare` the logits read does, as it already compared the cold
  run. For a
  bidirectional checkpoint that is the case `reusesPrefixBeforeMedia` turns
  off, so the 31B needs a bounded head another way before it flips.
- **E2B: one prompt over, stays text.** On "after a tool result" the gap is
  0.42 against a bound of 0.37 (rechunked difference 0.30). Gates 1 and 3-7
  pass and the image gate passes. Sampled over more chunkings (64 to 4,096
  tokens), the text load moves by up to 0.46 on that prompt, so the gap is
  inside what chunking does to this checkpoint; the rule as chosen measures
  one chunking, and E2B misses it.
- **26B-A4B: text verified, images stay off.** The load guard that refused
  Gemma 4 MoE checkpoints "until a run of one has been verified" is removed:
  on the three agent prompts the text load's top-1 token matches mlx-lm
  0.32.2's, greedy decoding is token-identical through each tool call
  (19, 19 and 20 tokens), and |Δ log p| against mlx-lm's one-shot prefill
  (1.91, 0.70 and 1.11 per prompt) is no larger than mlx-lm's own when it
  prefills in 128-, 512- or 2,048-token chunks (up to 2.06, 1.00 and
  2.27). On the full path, gate 2 misses on "opening" (1.23 against 0.98)
  and passes the other two; the other gates and the image gate pass. With
  itself as a draft, the text load's greedy output differs from its own
  output without one (22 of 24 drafts accepted); the full path's does not.
- Qwen3.6's numbers from the first run are inside the new gate 2 bound, but
  it was not run again; its reading is covered above.

## Goals, non-goals, anti-goals

- **Goal:** images work on these two families with tool calls, constraints,
  reasoning replay and the text-request cache intact.
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
- Image-digest cache keys get a design of their own, with the constraints
  listed above. Until then, an agent session that once had an image
  re-prefills from that image on every turn, and gate 5 puts a number on it.
- Gemma 4 checkpoints with bidirectional vision attention run image requests
  cold rather than reuse a warm prefix.
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
