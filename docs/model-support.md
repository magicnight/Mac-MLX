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
per-component parity gates, but the smallest checkpoint is too large to run
on our hardware, so **no real weights have ever been loaded**. The math is
verified; the end-to-end model is untested. Treat it as "should work" — if
you have the hardware and hit a problem, please
[open an issue](../../../issues); theoretical-tier bugs are fixed
issue-driven.

## Pure-Swift ports (overlay-registered)

| Architecture | `model_type` | Tier | Notes |
|---|---|---|---|
| Mellum2-12B-A2.5B | `mellum` | ✅ Tested | Sliding/full attention interleave (21+7 layers) + 64-expert MoE; 68.9 tok/s on the 4-bit checkpoint |
| Seed-OSS-36B | `seed_oss` | ✅ Tested | Dense GQA with independent attention/o-proj/MLP bias switches; full 1e-4 parity suite on real Metal, plus a real-checkpoint generation smoke at 18.2 tok/s on the 4-bit checkpoint. Ships a built-in chat-template override (`SeedOssChatTemplate`): the checkpoint's `chat_template.jinja` builds its thinking-budget table as an integer-keyed Jinja dict that swift-jinja cannot parse, so macMLX substitutes a semantically-identical if/elif rewrite — applied before template compilation and proven byte-for-byte equivalent, ungated, by `SeedOssChatTemplateParityTests`. A per-model `macmlx.chat_template.jinja` file overrides it |
| Hunyuan V1 Dense (0.5B–7B) | `hunyuan_v1_dense` | ✅ Tested | Dense GQA whose twists are per-head q/k RMSNorm applied AFTER RoPE (the inverse of the Qwen3 lineage's pre-RoPE norm) and a DynamicNTKAlpha RoPE whose base is pre-scaled by `rope_scaling.alpha`; a single `attention_bias` drives q/k/v/o and the embeddings are tied. Full 1e-4 parity suite on real Metal (two adversarial configs inverting every switch), plus a real-checkpoint generation smoke at 80.3 tok/s on the 1.8B 4-bit checkpoint. No chat-template override needed: the checkpoint's own `chat_template.jinja` renders natively under swift-jinja, proven byte-for-byte on the standard conversation path by `HunyuanV1DenseChatTemplateParityTests` |
| DeepSeek V3.2 | `deepseek_v32` | ⚠️ Theoretical | DSA sparse attention + absorbed MLA + noaux_tc MoE; full parity suite; smallest real checkpoint is 671B-class |
| Solar-Open-100B | `solar_open` | ⚠️ Theoretical | Config re-skin of upstream GLM4-MoE (~100B) |
| GLM-5.1 (GLM-DSA) | `glm_moe_dsa` | ⚠️ Theoretical | Subclass of our DeepSeek V3.2 port, no IndexShare; smallest quant ~405 GB. GLM-5.2 IndexShare checkpoints are detected and rejected with a clear error (unsupported until the upstream reference lands) |
| Kimi K2.5 | `kimi_k25` | 🚫 Blocked | Wraps upstream's DeepSeek V3 core, whose initializer is currently internal; registration errors loudly until upstream makes it public |

## Validated upstream weights

New model releases that reuse an existing architecture don't need a port —
just compatibility validation with real weights:

| Model | `model_type` | Status |
|---|---|---|
| Qwen3.6 (27B dense / 35B-A3B MoE) | `qwen3_5` / `qwen3_5_moe` | ✅ Validated — 20.6 tok/s (27B 4-bit) |
