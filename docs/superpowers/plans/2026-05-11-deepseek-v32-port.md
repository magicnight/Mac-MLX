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

- [ ] **1. `DeepseekV32Configuration`** — Codable, matches
  `config.json` exactly. Unambiguous; no parity test needed (decode
  test only). *(this session)*
- [ ] **2. `MultiLinear`** (copied) — batched per-head linear.
  Parity: a `[H, in]→[H, out]` forward vs a hand-computed expected.
- [ ] **3. `Indexer`** — the novel DSA lightning indexer. wq_b / wk /
  k_norm / weights_proj / rope → score → `argPartition` top-k
  indices. Parity: fixed small weights + input vs Python reference
  output indices. *(this session — the core novelty)*
- [ ] **4. `DeepseekV32Attention`** — MLA (q_a/q_b, kv_a_proj_with_mqa,
  kv_a_layernorm, embed_q / unembed_out `MultiLinear`, o_proj) +
  indexer sparse-mask integration. The two branches (L==1 decode vs
  L>1 prefill) differ; both need parity. Hardest component.
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

- **S1 (this):** feasibility audit (done), plan, `Configuration`,
  `MultiLinear`, `Indexer` + parity scaffolding. Compiles.
- **S2:** `DeepseekV32Attention` (both branches) + parity. The hard one.
- **S3:** MoE stack (MLP, gate, MoE) + parity.
- **S4:** assemble model + sanitize + register + real-model smoke.
- **S5+:** DeepSeek V4 as the increment (hash routing, compressed KV,
  o-LoRA, MXFP4) on top of our V3.2.

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
