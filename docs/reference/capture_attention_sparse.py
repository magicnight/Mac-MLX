"""Capture DeepSeek V3.2 Attention (sparse prefill) reference.

S2.2 parity target: the absorbed-MLA prefill path WITH sparsification.
We choose s (8) > index_topk (4) so the Indexer returns a real top-k
selection (not the None short-circuit), which drives the
`put_along_axis` sparse-mask scatter branch (`L > 1`, topk != None) that
S2.1 did not exercise. Everything else matches capture_attention_prefill.

Environment: the same uv venv (mlx-lm 0.31.3 + transformers<5.13 + the
PR #1431 rope patch). Offline capture; no Python enters macMLX.
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
INDEX_HEAD_DIM = 8
INDEX_N_HEADS = 2
INDEX_TOPK = 4
MAXPOS = 64
ROPE_THETA = 10000.0
B, S = 1, 8  # S=8 > INDEX_TOPK=4 -> indexer returns top-k (sparse branch)

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


weights = {
    "q_a_proj.weight": det([Q_LORA, HIDDEN]),
    "q_a_layernorm.weight": mx.ones([Q_LORA]) * 1.05,
    "q_b_proj.weight": det([NUM_HEADS * (QK_NOPE + QK_ROPE), Q_LORA]),
    "kv_a_proj_with_mqa.weight": det([KV_LORA + QK_ROPE, HIDDEN]),
    "kv_a_layernorm.weight": mx.ones([KV_LORA]) * 0.95,
    "embed_q.weight": det([NUM_HEADS, KV_LORA, QK_NOPE], 0.06),
    "unembed_out.weight": det([NUM_HEADS, V_HEAD, KV_LORA], 0.06),
    "o_proj.weight": det([HIDDEN, NUM_HEADS * V_HEAD], 0.04),
    "indexer.wq_b.weight": det([INDEX_N_HEADS * INDEX_HEAD_DIM, Q_LORA]),
    "indexer.wk.weight": det([INDEX_HEAD_DIM, HIDDEN]),
    "indexer.k_norm.weight": mx.ones([INDEX_HEAD_DIM]) * 1.1,
    "indexer.k_norm.bias": det([INDEX_HEAD_DIM], 0.05),
    "indexer.weights_proj.weight": det([INDEX_N_HEADS, HIDDEN], 0.07),
}
attn.load_weights(list(weights.items()))

x = det([B, S, HIDDEN], 0.03)
# Bool causal mask [S, S] (True = attend). The attention ANDs the
# indexer's scattered sparse mask with this.
mask = mx.tril(mx.ones((S, S), dtype=mx.bool_))

out = attn(x, mask=mask, cache=None)  # [B, S, HIDDEN]
mx.eval(out)

fixture = dict(weights)
fixture["x"] = x
fixture["mask"] = mask.astype(mx.uint8)
fixture["expected_output"] = out

path = "attn_sparse_fixture.safetensors"
mx.save_safetensors(path, fixture)
print("saved", path)
print("output shape", out.shape)
print("output[0,0,:6]", out[0, 0, :6])
