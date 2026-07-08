"""Capture DeepSeek V3.2 full Model (embed → 2 layers → norm → lm_head) reference (S4).

End-to-end parity gate for the model assembly + the dense/MoE decoder-layer
predicate. Tiny 2-layer config with `first_k_dense_replace=1` and
`n_routed_experts=4`, so:
  - layer 0 is DENSE  (idx 0 <  first_k_dense_replace) → DeepseekV32MLP
  - layer 1 is MoE    (idx 1 >= first_k_dense_replace, 1 % moe_layer_freq == 0)
which exercises the layer-selection predicate top to bottom.

Sequence length s=4 == index_topk=4, so the DSA indexer short-circuits (every
key fits in top-k) → dense attention. The indexer's own sparse/decode parity
is proven separately by the S2.2 / S2.3 fixtures.

Environment: the same offline uv venv used for the attention/indexer/decoder/
MoE fixtures — mlx-lm 0.31.3 + the PR #1431 one-line rope patch. No Python
enters macMLX; this is an offline capture. Weights are deterministic (det())
so the Swift side loads identical values. The fixture stores the *module*
(post-sanitize) layout: layer 1's experts are already stacked into
`switch_mlp`, so the model-parity test loads directly without sanitize (that
is covered separately by the sanitize round-trip test).
"""
import mlx.core as mx
from mlx_lm.models.deepseek_v32 import Model, ModelArgs

mx.random.seed(0)

VOCAB = 32
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
MOE_INTERMEDIATE = 8
N_ROUTED = 4
N_SHARED = 1
TOP_K = 2
N_GROUP = 1
TOPK_GROUP = 1
NUM_LAYERS = 2
FIRST_K_DENSE = 1
MAXPOS = 64
ROPE_THETA = 10000.0

args = ModelArgs(
    model_type="deepseek_v32",
    vocab_size=VOCAB,
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
    moe_intermediate_size=MOE_INTERMEDIATE,
    n_routed_experts=N_ROUTED,
    n_shared_experts=N_SHARED,
    num_experts_per_tok=TOP_K,
    n_group=N_GROUP,
    topk_group=TOPK_GROUP,
    norm_topk_prob=True,
    routed_scaling_factor=2.5,
    topk_method="noaux_tc",
    scoring_func="sigmoid",
    num_hidden_layers=NUM_LAYERS,
    first_k_dense_replace=FIRST_K_DENSE,
    moe_layer_freq=1,
    rms_norm_eps=1e-6,
    max_position_embeddings=MAXPOS,
    rope_theta=ROPE_THETA,
    indexer_rope_interleave=False,
    attention_bias=False,
)

model = Model(args)


def det(shape, scale=0.05):
    n = 1
    for d in shape:
        n *= d
    return (mx.arange(n).reshape(shape).astype(mx.float32) % 7 - 3) * scale


def attn_weights(prefix):
    # The same 13 self_attn keys as capture_attention_prefill.py / decoder.
    return {
        f"{prefix}.q_a_proj.weight": det([Q_LORA, HIDDEN]),
        f"{prefix}.q_a_layernorm.weight": mx.ones([Q_LORA]) * 1.05,
        f"{prefix}.q_b_proj.weight": det([NUM_HEADS * (QK_NOPE + QK_ROPE), Q_LORA]),
        f"{prefix}.kv_a_proj_with_mqa.weight": det([KV_LORA + QK_ROPE, HIDDEN]),
        f"{prefix}.kv_a_layernorm.weight": mx.ones([KV_LORA]) * 0.95,
        f"{prefix}.embed_q.weight": det([NUM_HEADS, KV_LORA, QK_NOPE], 0.06),
        f"{prefix}.unembed_out.weight": det([NUM_HEADS, V_HEAD, KV_LORA], 0.06),
        f"{prefix}.o_proj.weight": det([HIDDEN, NUM_HEADS * V_HEAD], 0.04),
        f"{prefix}.indexer.wq_b.weight": det([INDEX_N_HEADS * INDEX_HEAD_DIM, Q_LORA]),
        f"{prefix}.indexer.wk.weight": det([INDEX_HEAD_DIM, HIDDEN]),
        f"{prefix}.indexer.k_norm.weight": mx.ones([INDEX_HEAD_DIM]) * 1.1,
        f"{prefix}.indexer.k_norm.bias": det([INDEX_HEAD_DIM], 0.05),
        f"{prefix}.indexer.weights_proj.weight": det([INDEX_N_HEADS, HIDDEN], 0.07),
    }


weights = {
    "model.embed_tokens.weight": det([VOCAB, HIDDEN], 0.02),
    "model.norm.weight": mx.ones([HIDDEN]) * 1.01,
    "lm_head.weight": det([VOCAB, HIDDEN], 0.03),
}

# --- layer 0: DENSE MLP ---
weights.update(attn_weights("model.layers.0.self_attn"))
weights["model.layers.0.input_layernorm.weight"] = mx.ones([HIDDEN]) * 1.02
weights["model.layers.0.post_attention_layernorm.weight"] = mx.ones([HIDDEN]) * 0.98
weights["model.layers.0.mlp.gate_proj.weight"] = det([INTERMEDIATE, HIDDEN], 0.05)
weights["model.layers.0.mlp.up_proj.weight"] = det([INTERMEDIATE, HIDDEN], 0.04)
weights["model.layers.0.mlp.down_proj.weight"] = det([HIDDEN, INTERMEDIATE], 0.045)

# --- layer 1: MoE (routed + shared), experts already stacked into switch_mlp ---
weights.update(attn_weights("model.layers.1.self_attn"))
weights["model.layers.1.input_layernorm.weight"] = mx.ones([HIDDEN]) * 1.03
weights["model.layers.1.post_attention_layernorm.weight"] = mx.ones([HIDDEN]) * 0.97
weights["model.layers.1.mlp.gate.weight"] = det([N_ROUTED, HIDDEN], 0.05)
weights["model.layers.1.mlp.gate.e_score_correction_bias"] = mx.array(
    [0.1, -0.15, 0.2, -0.05], dtype=mx.float32
)
weights["model.layers.1.mlp.switch_mlp.gate_proj.weight"] = det(
    [N_ROUTED, MOE_INTERMEDIATE, HIDDEN], 0.05
)
weights["model.layers.1.mlp.switch_mlp.up_proj.weight"] = det(
    [N_ROUTED, MOE_INTERMEDIATE, HIDDEN], 0.04
)
weights["model.layers.1.mlp.switch_mlp.down_proj.weight"] = det(
    [N_ROUTED, HIDDEN, MOE_INTERMEDIATE], 0.045
)
weights["model.layers.1.mlp.shared_experts.gate_proj.weight"] = det(
    [MOE_INTERMEDIATE, HIDDEN], 0.05
)
weights["model.layers.1.mlp.shared_experts.up_proj.weight"] = det(
    [MOE_INTERMEDIATE, HIDDEN], 0.04
)
weights["model.layers.1.mlp.shared_experts.down_proj.weight"] = det(
    [HIDDEN, MOE_INTERMEDIATE], 0.045
)

model.load_weights(list(weights.items()), strict=False)

# Token ids [1, 4]; int32 so the Swift Embedding gathers identical rows.
x = mx.array([[1, 5, 2, 8]], dtype=mx.int32)
out = model(x)  # [1, 4, VOCAB]
mx.eval(out)

fixture = dict(weights)
fixture["x"] = x
fixture["expected_output"] = out

path = "model_fixture.safetensors"
mx.save_safetensors(path, fixture)
print("saved", path)
print("output shape", out.shape)
print("output[0,0,:6]", out[0, 0, :6])
print("output[0,-1,:6]", out[0, -1, :6])
