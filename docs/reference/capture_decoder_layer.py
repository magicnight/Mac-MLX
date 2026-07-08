"""Capture DeepSeek V3.2 DecoderLayer (dense, prefill) reference (S2.4).

The dense decoder block = pre-norm absorbed-MLA attention + residual, then
pre-norm SwiGLU MLP + residual. `n_routed_experts=None` forces the dense
`DeepseekV32MLP` branch (routed-expert MoE is S3). We prefill s=6 tokens
(> index_topk=4, so the indexer is live inside the attention), but this
fixture only asserts the *block's* final output — the attention's own
per-branch parity is already proven by the S2.1-S2.3 fixtures. This adds
the residual wiring, the two `config.rms_norm_eps` layer norms, and the
dense MLP on top.

Environment: the same offline uv venv used for the attention/indexer
fixtures — mlx-lm 0.31.3 + transformers<5.13 + the PR #1431 one-line rope
patch. No Python enters macMLX; this is an offline capture.

Weights are deterministic (det()) so the Swift side loads identical values.
Shapes (tiny config, intermediate_size=32):
  self_attn.*   the same 13 keys as capture_attention_prefill.py
  mlp.gate_proj: Linear(hidden=16 -> intermediate=32)   [32,16]
  mlp.up_proj:   Linear(hidden=16 -> intermediate=32)   [32,16]
  mlp.down_proj: Linear(intermediate=32 -> hidden=16)   [16,32]
  input_layernorm / post_attention_layernorm: RMSNorm(16)  [16]
"""
import mlx.core as mx
from mlx_lm.models.deepseek_v32 import DeepseekV32DecoderLayer, ModelArgs
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
INTERMEDIATE = 32
MAXPOS = 64
ROPE_THETA = 10000.0
B, S = 1, 6  # s=6 > INDEX_TOPK=4 -> indexer live inside the attention

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
    intermediate_size=INTERMEDIATE,
    n_routed_experts=None,  # forces the dense DeepseekV32MLP branch
    num_hidden_layers=1,
    rms_norm_eps=1e-6,
    max_position_embeddings=MAXPOS,
    rope_theta=ROPE_THETA,
    indexer_rope_interleave=False,
    attention_bias=False,
)

layer = DeepseekV32DecoderLayer(args, layer_idx=0)


def det(shape, scale=0.05):
    n = 1
    for d in shape:
        n *= d
    return (mx.arange(n).reshape(shape).astype(mx.float32) % 7 - 3) * scale


weights = {
    # --- self_attn.* : identical 13 keys to capture_attention_prefill.py ---
    "self_attn.q_a_proj.weight": det([Q_LORA, HIDDEN]),
    "self_attn.q_a_layernorm.weight": mx.ones([Q_LORA]) * 1.05,
    "self_attn.q_b_proj.weight": det([NUM_HEADS * (QK_NOPE + QK_ROPE), Q_LORA]),
    "self_attn.kv_a_proj_with_mqa.weight": det([KV_LORA + QK_ROPE, HIDDEN]),
    "self_attn.kv_a_layernorm.weight": mx.ones([KV_LORA]) * 0.95,
    "self_attn.embed_q.weight": det([NUM_HEADS, KV_LORA, QK_NOPE], 0.06),
    "self_attn.unembed_out.weight": det([NUM_HEADS, V_HEAD, KV_LORA], 0.06),
    "self_attn.o_proj.weight": det([HIDDEN, NUM_HEADS * V_HEAD], 0.04),
    "self_attn.indexer.wq_b.weight": det([INDEX_N_HEADS * INDEX_HEAD_DIM, Q_LORA]),
    "self_attn.indexer.wk.weight": det([INDEX_HEAD_DIM, HIDDEN]),
    "self_attn.indexer.k_norm.weight": mx.ones([INDEX_HEAD_DIM]) * 1.1,
    "self_attn.indexer.k_norm.bias": det([INDEX_HEAD_DIM], 0.05),
    "self_attn.indexer.weights_proj.weight": det([INDEX_N_HEADS, HIDDEN], 0.07),
    # --- dense MLP (SwiGLU) ---
    "mlp.gate_proj.weight": det([INTERMEDIATE, HIDDEN], 0.05),
    "mlp.up_proj.weight": det([INTERMEDIATE, HIDDEN], 0.04),
    "mlp.down_proj.weight": det([HIDDEN, INTERMEDIATE], 0.045),
    # --- block layer norms (config.rms_norm_eps) ---
    "input_layernorm.weight": mx.ones([HIDDEN]) * 1.02,
    "post_attention_layernorm.weight": mx.ones([HIDDEN]) * 0.98,
}
layer.load_weights(list(weights.items()))

x = det([B, S, HIDDEN], 0.03)
# Bool causal mask [S, S]: True = attend (lower triangle incl. diagonal).
mask = mx.tril(mx.ones((S, S), dtype=mx.bool_))
# One CacheList per layer (main MLA KV + indexer). Fresh, prefill offset 0.
cache = CacheList(KVCache(), KVCache())

out = layer(x, mask=mask, cache=cache)  # [B, S, HIDDEN]
mx.eval(out)

fixture = dict(weights)
fixture["x"] = x
# uint8 is safetensors-friendly; Swift reloads and compares != 0 for bool.
fixture["mask"] = mask.astype(mx.uint8)
fixture["expected_output"] = out

path = "decoder_layer_fixture.safetensors"
mx.save_safetensors(path, fixture)
print("saved", path)
print("output shape", out.shape)
print("output[0,0,:6]", out[0, 0, :6])
print("output[0,-1,:6]", out[0, -1, :6])
