# DeepSeek V3.2 pure-Swift port (overlay) — implementation plan

Written 2026-05-11. First macMLX-owned model architecture, proving the
zero-fork `ModelOverlay` path with a real cutting-edge model. V3.2 is
the **foundation for DeepSeek V4** — its DSA sparse attention (lightning
indexer) is shared with V4; V4 then adds hash-cluster routing +
per-layer compressed KV + o-LoRA + MXFP4 on top of our own V3.2.

**Reference:** `docs/reference/deepseek_v32_mlx_lm_reference.py`
(Apple's Python mlx-lm implementation, 655 lines — the numerical
ground truth we translate from and validate against).

**Target file:** `MacMLXCore/Sources/MacMLXCore/Models/DeepseekV32.swift`
**Registration:** `ModelOverlay.registerAll()` → `"deepseek_v32"`
(flipped on only once component parity passes).

---

## Feasibility audit (done — all green except one separable gap)

**mlx-swift primitives** (all present):
`argPartition`, `putAlong`, `takeAlong`, `depends`, `top`,
`unflattened`, `sigmoid`, `dequantized` / `quantized`.

**mlx-swift-lm components:**
- `SwitchGLU` — **public** ✅ (`SwitchLayers.swift`), MoE expert switch.
- `CacheList: BaseKVCache` — **public** ✅ (`KVCache.swift`), holds the
  2 sub-caches (main KV + indexer) V3.2 needs per layer.
- `MultiLinear` — **internal** upstream (`GLM4MOELite.swift`). Not
  usable from our module ⇒ **copy ~40 lines** into our file (batched
  per-head linear + its `Quantizable` conformance).

**Separable gap — `from_fp8`:** the C API has `mlx_from_fp8` but
mlx-swift exposes no Swift binding. Only used in `sanitize()` to load
DeepSeek's **native FP8 release** directly (`weight_scale_inv` block
format). Deferred: target a pre-converted (bf16 / int4) mlx-community
weight set first, which skips the fp8 path entirely. Adding the Swift
binding (or a small C shim) is a self-contained follow-up if we want
to load the raw FP8 release.

---

## Component checklist (top-down, each with a parity gate)

Port order is **bottom-up** (leaf components first so each has a
parity test before the thing that uses it):

- [x] **1. `DeepseekV32Configuration`** — Codable, matches
  `config.json` exactly. Unambiguous; no parity test needed (decode
  test only). **DONE** (`8c2df46`) — `DeepseekV32ConfigurationTests`
  2 decode tests pass under SPM.
- [x] **2. `MultiLinear`** (copied as `DeepseekMultiLinear`) — batched
  per-head linear. `callAsFunction(_:transpose:)` mirrors mlx-lm's
  `mla.MultiLinear.__call__(x, transpose=True)`: `true` ⇒ `x @ wᵀ`
  (in→out), `false` ⇒ `x @ w` (out→in, the absorbed-MLA reverse).
  **DONE** (transpose param `78766e3`).
- [x] **3. `Indexer`** (`DeepseekV32Indexer`) — the novel DSA lightning
  indexer. wq_b / wk / k_norm / weights_proj / rope → score →
  `argPartition` top-k indices. **DONE + numerically verified**
  (`59b2f6f`) — `DeepseekV32IndexerParityTests` selects the *same*
  sorted top-k as Python mlx-lm 0.31.3 (fixture
  `Fixtures/indexer_parity_fixture.safetensors`, captured offline via
  `scratchpad/capture_indexer.py`). Both structural tests + the
  short-circuit path (`s <= index_topk → nil`) covered. **Rope
  corrected 2026-07-07:** now `traditional: config.indexerRopeInterleave`
  (default false) per upstream mlx-lm PR #1431 — stock 0.31.3's
  hardcoded `True` was a regression that silently degrades long-sequence
  quality. Fixture regenerated against the patched venv; parity re-green
  under xcodebuild.
- [ ] **4. `DeepseekV32Attention`** — MLA (q_a/q_b, kv_a_proj_with_mqa,
  kv_a_layernorm, embed_q / unembed_out `DeepseekMultiLinear`, o_proj)
  + indexer sparse-mask integration. The two branches (L==1 decode vs
  L>1 prefill) differ; both need parity. **← NEXT (S2). Hardest
  component.** Kickoff recipe below.
- [ ] **5. `DeepseekV32MLP`** — plain SwiGLU. Trivial; parity quick.
- [ ] **6. `group_expert_select` + `MoEGate`** — noaux_tc routing:
  sigmoid scoring + `e_score_correction_bias` + group select +
  norm_topk_prob + routed_scaling_factor. Parity: fixed gates vs
  Python `group_expert_select`.
- [ ] **7. `DeepseekV32MoE`** — SwitchGLU experts + gate + optional
  shared experts. Parity vs Python.
- [ ] **8. `DeepseekV32DecoderLayer`** — attn + (MoE|dense per
  `first_k_dense_replace` / `moe_layer_freq`) + 2 RMSNorms.
- [ ] **9. `DeepseekV32Model` + `Model`** — embed + layers + norm +
  lm_head. `makeCache()` → `[CacheList(KVCache, KVCache)]` per layer.
  `kvHeads` for `KVCacheDimensionProvider`.
- [ ] **10. `sanitize`** — expert stacking (experts.N.{gate,up,down}
  → switch_mlp) + kv_b_proj split into embed_q / unembed_out. **Skip
  the fp8 dequant branch** (targeting pre-converted weights). Parity:
  a synthetic weight dict round-trips to the expected stacked keys.
- [ ] **11. Register** in `ModelOverlay.registerAll()` — only after
  1-10 pass. `LoRAModel` conformance (`loraLayers`) for adapter reuse.
- [ ] **12. Real-model smoke** (manual, 64GB+ Mac) — load a converted
  DeepSeek V3.2 and generate. Component parity is the correctness
  gate; this is the integration confirmation.

---

## Parity-test strategy (the crux of a correct port)

We can't run the full 671B model in CI, and don't need to. Each
component gets a **numerical parity test** with tiny synthetic
weights + inputs, comparing our Swift output against values captured
from the Python reference (run once, offline, hard-coded as expected
arrays). Standard practice for model porting: verify the *math*
per-component, not the full forward.

- Metal-gated (`requireMetalOrSkip`) since these allocate `MLXArray` +
  run kernels — run under `xcodebuild`, skip under bare `swift test`.
- Capture Python expected values by running the reference with
  `mx.random.seed(0)` + fixed shapes; paste the resulting arrays as
  test fixtures. (Do this in a throwaway Python venv **outside** the
  repo — no Python enters macMLX.)
- Tolerance: `1e-4` abs for bf16-range activations.

---

## Session milestones

- **S1:** ✅ **DONE** — feasibility audit, plan, `Configuration`,
  `DeepseekMultiLinear`, `Indexer`. Compiles; SPM 153/153.
- **S1.5:** ✅ **DONE** — MLX test gate fixed (`requireMLXRuntimeOrSkip`,
  discriminates `.build/` path so MLX tests actually run under
  xcodebuild) + **Indexer numerical parity** vs Python (逐位一致).
- **S2 groundwork:** ✅ **DONE** — `DeepseekMultiLinear` gains the
  `transpose` param (absorbed-MLA needs both directions).
- **S2:** ← **NEXT** — `DeepseekV32Attention` (both branches) + parity.
  The hard one. See kickoff recipe below.
- **S3:** MoE stack (MLP, gate, MoE) + parity.
- **S4:** assemble model + sanitize + register + real-model smoke.
- **S5+:** DeepSeek V4 as the increment (hash routing, compressed KV,
  o-LoRA, MXFP4) on top of our V3.2.

---

## S2 kickoff recipe (next focused session — start here)

**Branch:** `feat/deepseek-v32-overlay` (5 commits on merged main
`4575768`; all green). Read the reference **first**:
`docs/reference/deepseek_v32_mlx_lm_reference.py` lines **117–260**
(`DeepseekV32Attention`).

**What the attention does (absorbed MLA + DSA sparse mask):**
1. `q = q_b(q_a_layernorm(q_a_proj(x)))` → reshape `[B,S,nHeads,qHeadDim]`,
   split into `q_nope` (`qk_nope_head_dim`) + `q_pe` (`qk_rope_head_dim`).
2. `kv = kv_a_proj_with_mqa(x)` → split `kv_latent` (`kv_lora_rank`) +
   `k_pe` (`qk_rope_head_dim`); `kv_latent = kv_a_layernorm(kv_latent)`.
3. RoPE on `q_pe` and `k_pe` (offset = cache length).
4. **Indexer** produces top-k key indices → build the sparse attention
   mask (only selected keys attend) — this is the DSA novelty wired in.
5. **Two branches** (the crux — port each with its own parity test):
   - **L==1 (decode):** `q_nope = embed_q(q_nope)` (transpose=**true**);
     `k = v = kv_latent`; scores over the sparse set; `output =
     unembed_out(output)` (transpose=**true**).
   - **L>1 (prefill):** `k = embed_q(kv_latent, transpose=false)`,
     `v = unembed_out(kv_latent, transpose=false)`; full/blocked attn
     over the sparse mask.
6. `o_proj(output)`.

**Cache:** per layer a `CacheList(KVCache, KVCache)` — sub-cache 0 = main
MLA KV, sub-cache 1 = indexer. `makeCache()` builds it; attention reads
offset from sub-cache 0.

**Parity plan (mirror the Indexer harness exactly):**
- New offline capture `scratchpad/capture_attention.py` — tiny config
  (reuse the Indexer's HIDDEN=16 family; add `kv_lora_rank`,
  `qk_nope_head_dim`, `v_head_dim`, `num_attention_heads` small, e.g.
  2–4). `mx.random.seed(0)`, deterministic `det()` weights, fixed x.
- **Capture BOTH branches:** one fixture at `S>1` (prefill), one at
  `S==1` with a primed cache (decode). Save weights + inputs +
  `expected_output` to `Fixtures/attn_prefill_fixture.safetensors` and
  `Fixtures/attn_decode_fixture.safetensors`.
- Swift `DeepseekV32AttentionParityTests` (Metal-gated via
  `requireMLXRuntimeOrSkip`): load weights with
  `update(parameters:verify:[.noUnusedKeys])`, run forward, assert
  `allClose(out, expected, atol: 1e-4)`.
- **uv, never pip** for the venv; pin `transformers<5.13` (5.13 breaks
  mlx-lm import — `'str' has no attribute '__module__'`). Reuse the
  existing `scratchpad/deepseek-ref-venv` if still present.

**Weight key names** (must match `sanitize`'s eventual output — see
reference lines 546–581 for the `kv_b_proj` → `embed_q`/`unembed_out`
split): `q_a_proj`, `q_a_layernorm`, `q_b_proj`, `kv_a_proj_with_mqa`,
`kv_a_layernorm`, `embed_q` (`[nHeads, kv_lora_rank, qk_nope_head_dim]`),
`unembed_out` (`[nHeads, v_head_dim, kv_lora_rank]`), `o_proj`, plus the
indexer sub-module keys under `indexer.*`.

**Gotchas already learned:**
- `DeepseekMultiLinear.callAsFunction(_:transpose:)` is ready — decode
  uses `transpose:true`, prefill `transpose:false`.
- Indexer rope follows `indexer_rope_interleave` (default **false**) —
  corrected per upstream PR #1431; do not reintroduce `traditional: true`.
  The **main attention's** rope is different: upstream keeps
  `traditional=True` there (reference line ~173) — confirm against the
  reference before capturing S2 fixtures; mismatch silently breaks parity.
- Sort-before-compare only applies to the indexer's index sets; the
  attention output is a dense tensor → compare with `allClose`, not sort.

---

## Notes / risks

- `MultiLinear` upstream is `Quantizable`; the `embed_q`/`unembed_out`
  absorbed-MLA weights come from splitting `kv_b_proj` in `sanitize`.
  Getting that split + the quantize round-trip right (Python lines
  546-581) is fiddly — parity-test it in isolation.
- The indexer's `k.shape[2] <= index_topk → return None` short-circuit
  means short contexts skip sparsity entirely. Test both paths
  (context ≤ topk and > topk).
- `@mx.compile` on `group_expert_select` is a Python perf hint; Swift
  has no direct equivalent needed — just translate the body.
- Pipeline / shard / distributed code in the reference is multi-GPU
  server infra — **omit entirely** (macMLX is single-process).
