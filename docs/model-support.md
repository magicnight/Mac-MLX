# Model Support

macMLX's in-process engine runs every architecture bundled with
[mlx-swift-lm](https://github.com/ml-explore/mlx-swift-examples) — the full
text family plus 16 VLM architectures — with no work on our side. This page
tracks the **architectures macMLX itself ports in pure Swift** and registers
into the model factory as external overlays, together with their support
tier.

## Support tiers

**✅ Tested** — the smallest published quantization fits on ordinary Apple
Silicon (roughly ≤30 GB), so the port ran the full methodology end-to-end:
every component numerically parity-gated at `1e-4` against the Python
`mlx-lm` reference, **and** a real-checkpoint generation smoke test.

**⚠️ Theoretical** — the architecture is ported and passes the exact same
per-component parity gates, but **end-to-end generation has never run**:
either the smallest checkpoint is too large for our hardware, or a
toolchain gap blocks the load path (e.g. no `tokenizer.json` exists in any
published checkpoint — see the per-row notes for which). The math is
verified; the end-to-end model is untested. Treat it as "should work once
loadable" — if you get it running and hit a problem, please
[open an issue](../../../issues); theoretical-tier bugs are fixed
issue-driven.

## Pure-Swift ports (overlay-registered)

| Architecture | `model_type` | Tier | Notes |
|---|---|---|---|
| Mellum2-12B-A2.5B | `mellum` | ✅ Tested | Sliding/full attention interleave (21+7 layers) + 64-expert MoE; 68.9 tok/s on the 4-bit checkpoint |
| Seed-OSS-36B | `seed_oss` | ✅ Tested | Dense GQA with independent attention/o-proj/MLP bias switches; full 1e-4 parity suite on real Metal, plus a real-checkpoint generation smoke at 18.2 tok/s on the 4-bit checkpoint. Ships a built-in chat-template override (`SeedOssChatTemplate`): the checkpoint's `chat_template.jinja` builds its thinking-budget table as an integer-keyed Jinja dict that swift-jinja cannot parse, so macMLX substitutes a semantically-identical if/elif rewrite — applied before template compilation and proven byte-for-byte equivalent, ungated, by `SeedOssChatTemplateParityTests`. A per-model `macmlx.chat_template.jinja` file overrides it |
| Hunyuan V1 Dense (0.5B–7B) | `hunyuan_v1_dense` | ✅ Tested | Dense GQA whose twists are per-head q/k RMSNorm applied AFTER RoPE (the inverse of the Qwen3 lineage's pre-RoPE norm) and a DynamicNTKAlpha RoPE whose base is pre-scaled by `rope_scaling.alpha`; a single `attention_bias` drives q/k/v/o and the embeddings are tied. Full 1e-4 parity suite on real Metal (two adversarial configs inverting every switch), plus a real-checkpoint generation smoke at 80.3 tok/s on the 1.8B 4-bit checkpoint. No chat-template override needed: the checkpoint's own `chat_template.jinja` renders natively under swift-jinja, proven byte-for-byte on the standard conversation path by `HunyuanV1DenseChatTemplateParityTests` |
| Cohere Command R7B | `cohere2` | ✅ Tested | Cohere-family decoder with a PARALLEL residual block (attention and MLP both read the same single `input_layernorm(x)`; output is `attn + mlp + x`, no post-attention norm), `LayerNorm` (not RMSNorm) with a `layer_norm_bias` switch, tied embeddings scaled by `logit_scale`, and — the Command R7B twist — interleaved sliding-window / global attention on a `sliding_window_pattern`: traditional (GPT-J) RoPE runs only on the sliding-window layers while the global layers get NO positional encoding (NoPE), and the KV cache is mixed (RotatingKVCache on sliding layers, KVCacheSimple on global). Full 1e-4 parity suite on real Metal (two adversarial configs inverting every switch — bias, ln_bias, logit_scale, pattern, window — with seq_len > window so the sliding mask genuinely differs from full-causal), plus a real-checkpoint generation smoke at 21.7 tok/s on the 7B 4-bit checkpoint (`c4ai-command-r7b-12-2024` — the `12-2024` is the December 2024 release date, not a parameter count). Ships a built-in chat-template override (`Cohere2ChatTemplate`): the checkpoint's `default` named template embeds a tool/RAG branch swift-jinja cannot parse, so macMLX drops that (dead-for-`default`) branch and keeps the standard-conversation branch byte-for-byte — applied before template compilation and proven byte-for-byte equivalent, ungated, by `Cohere2ChatTemplateParityTests`. A per-model `macmlx.chat_template.jinja` file overrides it |
| MiniCPM3-4B | `minicpm3` | ✅ Tested | Llama-family decoder with NON-ABSORBED (materialized) Multi-head Latent Attention — low-rank q/kv projections that materialize full multi-head q/k/v, a single-head RoPE key broadcast to all heads, a distinct value head dim (`hidden/heads`), and a `q_head_dim`-based (`(qk_nope+qk_rope)^-0.5`) softmax scale; `attention_bias` gates only q_a/kv_a/o (q_b/kv_b stay bias-free). Plus muP's three numerical scalings, each a parity chokepoint: embedding × `scale_emb`, both residual branches × `scale_depth/√layers`, and — when untied (the shipped checkpoint) — the head input ÷ `hidden/dim_model_base`. RoPE is the longrope `SuScaledRoPE` (always `long_factor`; `short_factor` ignored; mscale non-trivial only when `max_position > original_max`) — the same stock module Phi3 uses, so parity holds without a bespoke rope. Full 1e-4 parity suite on real Metal (two adversarial configs inverting every switch/scaling — bias, tie, scale_emb, scale_depth, and the longrope mscale path), plus a real-checkpoint generation smoke at 18.7 tok/s on the 4-bit checkpoint. No chat-template override needed: the checkpoint's own tool-use `chat_template` (recursive Jinja macros) renders natively under swift-jinja on the standard conversation path, proven byte-for-byte by `MiniCPM3ChatTemplateParityTests` |
| DeepSeek V3.2 | `deepseek_v32` | ⚠️ Theoretical | DSA sparse attention + absorbed MLA + noaux_tc MoE; full parity suite; smallest real checkpoint is 671B-class |
| Solar-Open-100B | `solar_open` | ⚠️ Theoretical | Config re-skin of upstream GLM4-MoE (~100B) |
| GLM-5.1 (GLM-DSA) | `glm_moe_dsa` | ⚠️ Theoretical | Subclass of our DeepSeek V3.2 port, no IndexShare; smallest quant ~405 GB. GLM-5.2 IndexShare checkpoints are detected and rejected with a clear error (unsupported until the upstream reference lands) |
| Kimi K2.5 | `kimi_k25` | 🚫 Blocked | Wraps upstream's DeepSeek V3 core, whose initializer is currently internal; registration errors loudly until upstream makes it public |
| InternLM3-8B-Instruct | `internlm3` | ⚠️ Theoretical | Dense Llama-family decoder — aggressive 16:1 GQA (32 query / 2 KV heads), SwiGLU, RMSNorm, standard serial pre-norm blocks — with two MISLEADINGLY-named bias switches (`qkv_bias` gates q/k/v AND o_proj; `bias` gates the MLP's gate/up/down) and `head_dim` fixed at `hidden/heads` (the config's explicit `head_dim` is never read). Its DynamicNTK RoPE is an INTENTIONAL DIVERGENCE from mlx-lm's `internlm3.py`, which carries four verified defects — a hard-coded `2.0` position scale, an unconsumed `rope_scaling.factor`, a `seq_len` read off the heads axis, and an NTK base rewrite wrongly applied to the `linear` type on long sequences — all corrected here to match the reference `modeling_internlm3.py` (upstream issue pending); the parity fixtures are captured from a minimally-patched mlx-lm (`docs/reference/capture_internlm3.py`). The architecture is FULLY 1e-4 parity-verified on real Metal (two adversarial configs inverting every switch, the `dynamic_active` config pinning all three RoPE corrections at once), plus decode / sanitize / registration / native-ChatML-render gates. GENERATION IS BLOCKED, though: every published InternLM3 checkpoint (`internlm/internlm3-8b-instruct` and the `mlx-community/*` conversions) ships ONLY a SentencePiece `tokenizer.model` + a custom Python tokenizer and NO `tokenizer.json`, which swift-transformers requires (it has no SentencePiece fallback), so the model cannot tokenize or generate in macMLX. Unblocks when a checkpoint ships a `tokenizer.json` (or the engine gains a SentencePiece path) — a trustworthy `tokenizer.json` cannot be produced without executing the checkpoint's custom remote code |

## Validated upstream weights

New model releases that reuse an existing architecture don't need a port —
just compatibility validation with real weights:

| Model | `model_type` | Status |
|---|---|---|
| Qwen3.6 (27B dense / 35B-A3B MoE) | `qwen3_5` / `qwen3_5_moe` | ✅ Validated — 20.6 tok/s (27B 4-bit) |
