# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx-lm", "numpy"]
# ///
"""Capture the GLM-DSA (`glm_moe_dsa`) Indexer reference for the Swift parity test.

`glm_moe_dsa.py` is `class Model(DSV32Model)` — it reuses DeepSeek V3.2's
architecture wholesale and differs only in `ModelArgs` defaults. The one default
that reaches a numeric code path is `indexer_rope_interleave` (GLM-DSA default
`True`; base V3.2 default `False`, per mlx-lm PR #1431). With interleave `True`
the lightning indexer's RoPE runs in the *traditional* (interleaved) layout.

This script captures that `indexer_rope_interleave=True` path so the Swift
`GlmMoeDsaIndexerParityTests` can prove `DeepseekV32Indexer` selects the same
top-k keys when driven by a GLM-DSA config. It is the sibling of
`capture_indexer.py` (which captures the `False` path); the ONLY functional
difference here is `indexer_rope_interleave=True`.

No Python enters macMLX — this is an offline, throwaway capture. Run with:

    uv run docs/reference/capture_glm_moe_dsa_indexer.py

It writes the fixture next to the Swift tests.
"""
import os

import mlx.core as mx
from mlx_lm.models.deepseek_v32 import Indexer, ModelArgs

mx.random.seed(0)

# Identical tiny config to `capture_indexer.py` so the two fixtures differ ONLY
# in the rope interleave mode. qk_rope_head_dim (4) < index_head_dim (8) so the
# partial RoPE path is exercised; s (10) > index_topk (4) so the indexer returns
# a real top-k selection rather than the None short-circuit.
HIDDEN = 16
Q_LORA = 12
INDEX_HEAD_DIM = 8
INDEX_N_HEADS = 2
INDEX_TOPK = 4
QK_ROPE_HEAD_DIM = 4
MAXPOS = 64
ROPE_THETA = 10000.0
B, S = 1, 10

args = ModelArgs(
    model_type="glm_moe_dsa",
    hidden_size=HIDDEN,
    q_lora_rank=Q_LORA,
    index_head_dim=INDEX_HEAD_DIM,
    index_n_heads=INDEX_N_HEADS,
    index_topk=INDEX_TOPK,
    qk_rope_head_dim=QK_ROPE_HEAD_DIM,
    max_position_embeddings=MAXPOS,
    rope_theta=ROPE_THETA,
    # The GLM-DSA difference surface: interleaved (traditional) indexer RoPE.
    indexer_rope_interleave=True,
)

indexer = Indexer(args)


# Deterministic weights (override the random init) so Swift can load the exact
# same values. Shapes come from the module definition:
#   wq_b:          Linear(q_lora_rank -> n_heads*index_head_dim)  -> [16, 12]
#   wk:            Linear(hidden -> index_head_dim)               -> [8, 16]
#   k_norm:        LayerNorm(index_head_dim)                      -> weight[8], bias[8]
#   weights_proj:  Linear(hidden -> n_heads)                      -> [2, 16]
def det(shape, scale=0.1):
    n = 1
    for d in shape:
        n *= d
    return (mx.arange(n).reshape(shape).astype(mx.float32) % 7 - 3) * scale


wq_b_w = det([INDEX_N_HEADS * INDEX_HEAD_DIM, Q_LORA])
wk_w = det([INDEX_HEAD_DIM, HIDDEN])
knorm_w = mx.ones([INDEX_HEAD_DIM]) * 1.1
knorm_b = det([INDEX_HEAD_DIM], 0.05)
wproj_w = det([INDEX_N_HEADS, HIDDEN], 0.07)

indexer.wq_b.weight = wq_b_w
indexer.wk.weight = wk_w
indexer.k_norm.weight = knorm_w
indexer.k_norm.bias = knorm_b
indexer.weights_proj.weight = wproj_w

# Fixed inputs.
x = det([B, S, HIDDEN], 0.03)
qr = det([B, S, Q_LORA], 0.04)

# Reference forward (prefill path: cache=None, mask=None).
topk = indexer(x, qr, None)  # [B, 1, S, INDEX_TOPK]
mx.eval(topk)

# argpartition output order is unstable; the Swift test compares SORTED
# per-position index sets. Store sorted indices so the fixture is
# order-canonical on both sides.
topk_sorted = mx.sort(topk, axis=-1).astype(mx.int32)
mx.eval(topk_sorted)

out = {
    "wq_b.weight": wq_b_w,
    "wk.weight": wk_w,
    "k_norm.weight": knorm_w,
    "k_norm.bias": knorm_b,
    "weights_proj.weight": wproj_w,
    "x": x,
    "qr": qr,
    "expected_topk_sorted": topk_sorted,
}

here = os.path.dirname(os.path.abspath(__file__))
dest = os.path.join(
    here,
    "..",
    "..",
    "MacMLXCore",
    "Tests",
    "MacMLXCoreTests",
    "Fixtures",
    "glm_moe_dsa_indexer_parity_fixture.safetensors",
)
dest = os.path.normpath(dest)
mx.save_safetensors(dest, out)
print("saved", dest)
print("expected_topk_sorted shape", topk_sorted.shape)
print("expected_topk_sorted[0,0]:")
print(topk_sorted[0, 0])
