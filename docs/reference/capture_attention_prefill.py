"""Capture DeepSeek V3.2 Attention (prefill, short-sequence) reference.

S2.1 parity target: the absorbed-MLA core in isolation. We choose
s (3) <= index_topk (4) so the Indexer short-circuits to None and NO
sparsification happens — this fixture validates ONLY the absorbed-MLA
math (q_a/q_b + kv_a_proj + rope + embed_q/unembed_out + pe_scores as
the SDPA additive mask + o_proj), not the top-k gather/scatter (that's
S2.2 / S2.3).

Environment: the same uv venv used for the indexer fixture — mlx-lm
0.31.3 + transformers<5.13 + the PR #1431 one-line rope patch. No
Python enters macMLX; this is an offline capture.

Weights are deterministic (det()) so the Swift side loads identical
values. Shapes (tiny config):
  q_a_proj:            Linear(hidden=16 -> q_lora=12)          [12,16]
  q_a_layernorm:       RMSNorm(12)                             [12]
  q_b_proj:            Linear(12 -> n_heads*q_head_dim=2*8=16) [16,12]
  kv_a_proj_with_mqa:  Linear(16 -> kv_lora+qk_rope=6+4=10)    [10,16]
  kv_a_layernorm:      RMSNorm(6)                              [6]
  embed_q:   MultiLinear(qk_nope=4, kv_lora=6, n_heads=2)      [2,6,4]
  unembed_out: MultiLinear(kv_lora=6, v_head=4, n_heads=2)     [2,4,6]
  o_proj:    Linear(n_heads*v_head=2*4=8 -> hidden=16)         [16,8]
  indexer.wq_b/wk/k_norm/weights_proj  (unused on this path but the
    module is constructed + weight-loaded so the forward is real)
"""
import mlx.core as mx
from mlx_lm.models.deepseek_v32 import DeepseekV32Attention, ModelArgs

mx.random.seed(0)

HIDDEN = 16
Q_LORA = 12
NUM_HEADS = 2
QK_NOPE = 4
QK_ROPE = 4
KV_LORA = 6
V_HEAD = 4
# indexer knobs (short-circuits: s <= index_topk)
INDEX_HEAD_DIM = 8
INDEX_N_HEADS = 2
INDEX_TOPK = 4
MAXPOS = 64
ROPE_THETA = 10000.0
B, S = 1, 3  # S=3 <= INDEX_TOPK=4 -> indexer returns None

args = ModelArgs(
    model_type="deepseek_v32",
    hidden_size=HIDDEN,
    num_attention_heads=NUM_HEADS,
    q_lora_rank=Q_LORA,
    qk_nope_head_dim=QK_NOPE,
    qk_rope_head_dim=QK_ROPE,
    kv_lora_rank=KV_LORA,
    v_head_dim=V_HEAD,
    index_head_dim=INDEX_HEAD_DIM,
    index_n_heads=INDEX_N_HEADS,
    index_topk=INDEX_TOPK,
    max_position_embeddings=MAXPOS,
    rope_theta=ROPE_THETA,
    indexer_rope_interleave=False,
    attention_bias=False,
)

attn = DeepseekV32Attention(args)


def det(shape, scale=0.05):
    n = 1
    for d in shape:
        n *= d
    return (mx.arange(n).reshape(shape).astype(mx.float32) % 7 - 3) * scale


# Deterministic weights for every submodule the forward touches.
weights = {
    "q_a_proj.weight": det([Q_LORA, HIDDEN]),
    "q_a_layernorm.weight": mx.ones([Q_LORA]) * 1.05,
    "q_b_proj.weight": det([NUM_HEADS * (QK_NOPE + QK_ROPE), Q_LORA]),
    "kv_a_proj_with_mqa.weight": det([KV_LORA + QK_ROPE, HIDDEN]),
    "kv_a_layernorm.weight": mx.ones([KV_LORA]) * 0.95,
    "embed_q.weight": det([NUM_HEADS, KV_LORA, QK_NOPE], 0.06),
    "unembed_out.weight": det([NUM_HEADS, V_HEAD, KV_LORA], 0.06),
    "o_proj.weight": det([HIDDEN, NUM_HEADS * V_HEAD], 0.04),
    # indexer weights (constructed + loaded even though unused on this path)
    "indexer.wq_b.weight": det([INDEX_N_HEADS * INDEX_HEAD_DIM, Q_LORA]),
    "indexer.wk.weight": det([INDEX_HEAD_DIM, HIDDEN]),
    "indexer.k_norm.weight": mx.ones([INDEX_HEAD_DIM]) * 1.1,
    "indexer.k_norm.bias": det([INDEX_HEAD_DIM], 0.05),
    "indexer.weights_proj.weight": det([INDEX_N_HEADS, HIDDEN], 0.07),
}
attn.load_weights(list(weights.items()))

x = det([B, S, HIDDEN], 0.03)
# Bool causal mask [S, S]: True = attend (lower triangle incl. diagonal),
# False = masked out. The reference's `pe_scores = where(mask, pe_scores,
# min)` needs a bool array — the installed create_attention_mask returns
# the string "causal", which that where() cannot consume. This is exactly
# the mask the Swift side constructs for prefill.
mask = mx.tril(mx.ones((S, S), dtype=mx.bool_))

out = attn(x, mask=mask, cache=None)  # [B, S, HIDDEN]
mx.eval(out)

fixture = dict(weights)
fixture["x"] = x
# uint8 is safetensors-friendly; Swift reloads and compares != 0 for bool.
fixture["mask"] = mask.astype(mx.uint8)
fixture["expected_output"] = out

path = "attn_prefill_fixture.safetensors"
mx.save_safetensors(path, fixture)
print("saved", path)
print("output shape", out.shape)
print("output[0,0,:6]", out[0, 0, :6])
