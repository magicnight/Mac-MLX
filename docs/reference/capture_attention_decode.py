"""Capture DeepSeek V3.2 Attention DECODE-step reference (S2.3).

Two-stage: build a CacheList(KVCache, KVCache), prefill S_PRE=6 tokens to
prime both the main MLA cache (cache[0]) and the indexer cache (cache[1]),
then decode ONE token. With S_PRE+1 = 7 > index_topk = 4, the indexer
returns a real top-k on the decode step, exercising the `L == 1`
`take_along_axis` gather branch that S2.1/S2.2 never touched.

We save the weights + the prefill inputs + the decode input, so the Swift
test reproduces the exact same two-stage cache state and compares only the
decode-step output.

Environment: the same uv venv (mlx-lm 0.31.3 + transformers<5.13 + the
PR #1431 rope patch). Offline; no Python enters macMLX.
"""
import mlx.core as mx
from mlx_lm.models.deepseek_v32 import DeepseekV32Attention, ModelArgs
from mlx_lm.models.cache import CacheList, KVCache

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
B, S_PRE = 1, 6  # prefill 6, then decode 1 → total 7 > INDEX_TOPK 4

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

# One CacheList holds both sub-caches for the layer.
cache = CacheList(KVCache(), KVCache())

# --- stage 1: prefill (fills cache[0] main-KV and cache[1] indexer) ---
x_prefill = det([B, S_PRE, HIDDEN], 0.03)
mask_prefill = mx.tril(mx.ones((S_PRE, S_PRE), dtype=mx.bool_))
_ = attn(x_prefill, mask=mask_prefill, cache=cache)

# --- stage 2: decode ONE token (mask=None: attend all cached top-k) ---
x_decode = det([B, 1, HIDDEN], 0.037)
out_decode = attn(x_decode, mask=None, cache=cache)
mx.eval(out_decode)

fixture = dict(weights)
fixture["x_prefill"] = x_prefill
fixture["mask_prefill"] = mask_prefill.astype(mx.uint8)
fixture["x_decode"] = x_decode
fixture["expected_decode"] = out_decode

path = "attn_decode_fixture.safetensors"
mx.save_safetensors(path, fixture)
print("saved", path)
print("decode out shape", out_decode.shape)
print("decode out[0,0,:6]", out_decode[0, 0, :6])
