# Porting DeepSeek V3.2 to Pure Swift

How macMLX brings a frontier model architecture — DeepSeek V3.2, with its
DSA sparse attention and absorbed Multi-head Latent Attention — to Apple
Silicon in **pure Swift**, as a zero-fork overlay on top of Apple's
`mlx-swift-lm`.

> Status: in progress. This document tracks the approach and the reasoning
> behind it; live per-component progress lives in the implementation plan.

## Why do this at all

When macMLX started, "MLX-native inference" was itself the differentiator.
That moat is gone: Ollama and LM Studio now ship MLX backends too. What
they *can't* easily do is **own a model architecture** — take a brand-new
frontier model that upstream `mlx-swift-lm` doesn't support yet, and make
it run, correctly, without waiting for anyone.

DeepSeek V3.2 is the proof. It is also deliberate groundwork: V3.2's DSA
sparse attention (the "lightning indexer") is shared with DeepSeek V4, so
a correct V3.2 port is the foundation V4 layers hash-cluster routing,
per-layer compressed KV, o-LoRA and MXFP4 on top of.

## Zero-fork: the overlay approach

We do **not** fork `mlx-swift-lm`. The model registers itself into the
library's model factory under `model_type = "deepseek_v32"` via
`ModelOverlay.registerAll()`, flipped on only once every component passes
its parity gate. Everything lives in one file,
`MacMLXCore/Sources/MacMLXCore/Models/DeepseekV32.swift`.

A feasibility audit up front confirmed the library exposes the primitives
we need: `argPartition`, `putAlong`, `takeAlong`, `top`, `sigmoid`,
`quantized`/`dequantized`, plus the public `SwitchGLU` (MoE expert switch)
and `CacheList` (the per-layer composite cache). The one internal type we
needed — `MultiLinear` (batched per-head linear) — isn't public, so we
copied ~40 lines into our own `DeepseekMultiLinear` rather than fork the
package. One capability is deferred: `mlx_from_fp8` has no Swift binding,
so we target pre-converted bf16/int4 weights first and skip the native-FP8
`weight_scale_inv` path.

## What the architecture actually is

Three pieces stacked into each decoder layer:

- **DSA lightning indexer** — a small, cheap attention-like scorer that,
  for each query, selects the top-`index_topk` key positions worth
  attending to. This is the sparsity mechanism.
- **Absorbed MLA** — Multi-head Latent Attention that keeps K/V compressed
  in a `kv_lora_rank` latent space instead of materialising full per-head
  K/V, attending only over the indexer's selected positions.
- **MoE** — a `SwitchGLU` expert stack with noaux_tc sigmoid routing
  (`e_score_correction_bias` + group-select + `norm_topk_prob` +
  `routed_scaling_factor`), interleaved with dense layers per
  `first_k_dense_replace` / `moe_layer_freq`.

We port bottom-up — leaf components first — so each has a passing parity
test before the thing that uses it.

## The hard parts

### The lightning indexer

`wq_b → q`, `wk → k` (with `k_norm`), RoPE on both, score, then
`argPartition` for the top-k indices; `weights_proj` produces the per-head
mixing weights. When `seq_len <= index_topk` the indexer short-circuits to
`nil` (nothing to prune) — a real path the tests exercise separately.

One subtle correctness bug lived here. Stock `mlx-lm` 0.31.3 hardcoded the
indexer's RoPE to `traditional=True`. Upstream PR #1431 corrected this to a
config flag, `indexer_rope_interleave` (default **false**) — the hardcoded
mode silently degrades long-sequence quality. We caught it by cross-reading
the upstream reference against the PR during review, fixed the Swift port
to `traditional: config.indexerRopeInterleave`, and regenerated the parity
fixture against a patched reference. The lesson: on a numeric port, an
independent review pass plus an upstream diff earns its keep.

### Absorbed MLA — the crux

The absorbed form never materialises full K/V. Instead it keeps the latent
`kvLatent` (`kv_lora_rank`-wide) and projects per head with two
`DeepseekMultiLinear` weights: `embed_q` maps the query's no-position part
into the latent space, `unembed_out` maps the attention output back out.
The RoPE contribution is computed separately as `pe_scores` and handed to
SDPA as an **additive mask**, so the full score
`scale·(qNope·kᵀ) + pe_scores` resolves in a single softmax.

The genuinely tricky bit is that decode and prefill are *different code
paths*:

- **`L == 1` (decode):** `qNope = embed_q(qNope)`, `k = v = kvLatent`;
  gather the selected positions with `takeAlong`; project the output back
  through `unembed_out`.
- **`L > 1` (prefill):** `k = embed_q(kvLatent, transpose: false)`,
  `v = unembed_out(kvLatent, transpose: false)`; scatter the top-k into a
  sparse mask with `putAlong`, AND it with the causal mask, and attend.

Each branch got its own parity fixture. Prefill split further into a dense
case (`s <= index_topk`, indexer short-circuits) and a sparse case
(`s > index_topk`, the scatter path runs).

### The two-cache-per-layer trick

Each layer carries a `CacheList(KVCacheSimple, KVCacheSimple)`: sub-cache 0
is the main MLA KV, sub-cache 1 is the indexer's keys. Attention reads its
position offset from sub-cache 0 and threads sub-cache 1 into the indexer.
The decode-step parity test primes both by prefilling 6 tokens, then
decodes 1 (total 7 > `index_topk` 4) so the indexer returns a real top-k
and the `L == 1` gather branch actually runs — the path prefill tests never
touch.

## Parity: the correctness backbone

We can't run the full 671B model in CI, and don't need to. Correctness is
proven **per component, numerically**, not by a full forward pass:

1. In a throwaway offline `uv` venv (never `pip`) — `mlx-lm` 0.31.3 +
   `transformers<5.13` (5.13 breaks the mlx-lm import) + the PR #1431 rope
   patch — run the Python reference with `mx.random.seed(0)`, fixed tiny
   shapes, and deterministic weights.
2. Save weights + inputs + `expected_output` to a `.safetensors` fixture.
   Capture scripts live in `docs/reference/capture_*.py`. **No Python ever
   enters macMLX** — only the resulting fixture, checked into the test
   bundle.
3. The Swift test loads the same weights, runs the Swift component, and
   asserts `allClose(out, expected, atol: 1e-4)`.

These tests are Metal-gated (`requireMLXRuntimeOrSkip`) — they allocate
`MLXArray`s and run kernels, so they run under `xcodebuild` and skip under a
bare `swift test`. One operational gotcha: `mlx-swift` ships a build-tool
plugin (`CudaBuild`) that Xcode refuses to run unvalidated, so the tests
must be invoked with `xcodebuild test ... -skipPackagePluginValidation`.

## Status and what's left

Done and parity-verified (1e-4):

1. `DeepseekV32Configuration` — Codable, matches `config.json`.
2. `DeepseekMultiLinear` — batched per-head linear, both transpose directions.
3. `DeepseekV32Indexer` — DSA lightning indexer (rope-corrected).
4. `DeepseekV32Attention` — absorbed MLA, **all three branches green**:
   dense prefill, sparse prefill, and cached decode.

Remaining, bottom-up:

5. Assemble the decoder block's attention + `makeCache()`.
6. `DeepseekV32MLP` (SwiGLU), `MoEGate` (noaux_tc routing), `DeepseekV32MoE`.
7. `DeepseekV32DecoderLayer`, `DeepseekV32Model`, `sanitize` (expert
   stacking + `kv_b_proj` → `embed_q`/`unembed_out` split).
8. Register in `ModelOverlay.registerAll()`, then a real-model smoke test
   on a 64 GB+ Mac.

Component parity is the correctness gate; the smoke test is only the
integration confirmation. After that, DeepSeek V4 arrives as an increment
on this same foundation.
