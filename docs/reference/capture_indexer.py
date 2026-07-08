"""Capture DeepSeek V3.2 Indexer reference values for the Swift parity test.

Runs the mlx-lm reference Indexer with fixed weights + input, saves
{weights, inputs, expected top-k indices} to a safetensors fixture that
the Swift DeepseekV32IndexerParityTests loads. No Python enters macMLX —
this is an offline, throwaway capture.

Environment: uv venv with mlx-lm 0.31.3 + transformers<5.13, PLUS the
one-line patch from mlx-lm PR #1431 (merged upstream 2026-06-24,
unreleased as of capture): the Indexer rope is
`traditional=args.indexer_rope_interleave` (default False), NOT the
hardcoded `traditional=True` that stock 0.31.3 ships — upstream deemed
the hardcoded mode a regression that silently degrades long-sequence
quality. The Swift port follows the corrected behavior.
"""
import mlx.core as mx
from mlx_lm.models.deepseek_v32 import Indexer, ModelArgs

mx.random.seed(0)

# Tiny config. qk_rope_head_dim (4) < index_head_dim (8) so the partial
# RoPE path is exercised. s (10) > index_topk (4) so the indexer returns
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
    model_type="deepseek_v32",
    hidden_size=HIDDEN,
    q_lora_rank=Q_LORA,
    index_head_dim=INDEX_HEAD_DIM,
    index_n_heads=INDEX_N_HEADS,
    index_topk=INDEX_TOPK,
    qk_rope_head_dim=QK_ROPE_HEAD_DIM,
    max_position_embeddings=MAXPOS,
    rope_theta=ROPE_THETA,
    # Explicit: non-interleaved rope (the corrected default per PR #1431).
    indexer_rope_interleave=False,
)

indexer = Indexer(args)

# Deterministic weights (override the random init) so Swift can load the
# exact same values. Shapes come from the module definition:
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
path = "indexer_parity_fixture.safetensors"
mx.save_safetensors(path, out)
print("saved", path)
print("expected_topk_sorted shape", topk_sorted.shape)
print("expected_topk_sorted[0,0]:")
print(topk_sorted[0, 0])
