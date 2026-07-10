# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlx==0.32.0",
#     "mlx-lm @ git+https://github.com/ml-explore/mlx-lm.git@e476a22246b86fb6e2a8d35c81953293ebf86a0f",
#     "transformers>=5.0,<5.13",
#     "numpy",
# ]
# ///
"""Capture Mellum-2 (12B-A2.5B) numerical-parity references (Track G).

Mellum (`model_type: mellum`) is a Qwen3-lineage sparse-MoE decoder: per-head
`q_norm`/`k_norm` attention, a 64-expert `SwitchGLU` MoE on every layer, and
28 layers that alternate 3 sliding-window + 1 full-attention. The two attention
families use *different* RoPE: full-attention layers use YaRN (factor 16),
sliding layers use plain RoPE — both at theta 500000.

This script produces four deterministic fixtures so the pure-Swift
`Mellum2.swift` overlay port can be proven at 1e-4 per component and end-to-end:

  mellum2_moe_fixture.safetensors               MellumSparseMoeBlock
  mellum2_attention_sliding_fixture.safetensors Attention (default RoPE + window)
  mellum2_attention_full_fixture.safetensors    Attention (YaRN RoPE + full causal)
  mellum2_model_fixture.safetensors             Model (embed -> layers -> norm -> lm_head)

The tiny config keeps `sliding_window (3) < seq_len (6)` so the sliding-window
mask genuinely differs from the full-causal mask (otherwise both collapse to a
plain causal mask and the distinction is untested). It carries both RoPE
families in `rope_parameters` so the YaRN path is exercised.

Environment (PEP 723, run with `uv run docs/reference/capture_mellum2.py`):
mlx 0.32.0 + mlx-lm @ e476a22 (the PR #1339 merge that first shipped mellum.py;
unreleased on PyPI as of 0.31.3). No Python enters macMLX — this is an offline,
one-shot capture; the safetensors are the durable artifact.

RoPE version note: mlx 0.32.0 carries the batched single-token RoPE fix
(ml-explore/mlx#3498). All fixtures here are single-sequence (B=1), so they are
insensitive to that fix — the Swift fork (mlx-swift + the same cherry-pick) and
stock mlx agree on the B=1 path regardless.

Weights are deterministic via det() so the Swift side loads identical values.
"""
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.mellum import Attention, MellumSparseMoeBlock, Model, ModelArgs

mx.random.seed(0)

# ---- tiny shared config -----------------------------------------------------
HIDDEN = 32
HEAD_DIM = 16
N_HEADS = 4
N_KV_HEADS = 2
N_EXPERTS = 6
N_EXPERTS_PER_TOK = 2
MOE_INTERMEDIATE = 24
INTERMEDIATE = 48  # dense intermediate_size; unused by the MoE forward
VOCAB = 40
RMS_EPS = 1e-6
MAXPOS = 64
SLIDING_WINDOW = 3
B, S = 1, 6  # S > SLIDING_WINDOW so the window mask differs from full causal

# Real Mellum RoPE families, verbatim: full-attention = YaRN, sliding = default.
# `attention_factor` is present in the checkpoint but ignored by mlx-lm's
# initialize_rope (only mscale/mscale_all_dim are read); it equals the derived
# default 0.1*ln(16)+1 = 1.2772..., so both sides compute the same mscale.
ROPE_PARAMETERS = {
    "full_attention": {
        "rope_type": "yarn",
        "rope_theta": 500000.0,
        "factor": 16.0,
        "original_max_position_embeddings": 8192,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "attention_factor": 1.2772588722239782,
    },
    "sliding_attention": {
        "rope_type": "default",
        "rope_theta": 500000.0,
    },
}

# Mini layer cadence: sliding, sliding, full, full. first_sliding = 0,
# first_full = 2 — both mask families are built and both RoPE paths run.
LAYER_TYPES = [
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "full_attention",
]
NUM_LAYERS = len(LAYER_TYPES)


def make_args() -> ModelArgs:
    return ModelArgs(
        model_type="mellum",
        hidden_size=HIDDEN,
        num_hidden_layers=NUM_LAYERS,
        intermediate_size=INTERMEDIATE,
        num_attention_heads=N_HEADS,
        num_experts=N_EXPERTS,
        num_experts_per_tok=N_EXPERTS_PER_TOK,
        moe_intermediate_size=MOE_INTERMEDIATE,
        rms_norm_eps=RMS_EPS,
        vocab_size=VOCAB,
        num_key_value_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        tie_word_embeddings=False,
        max_position_embeddings=MAXPOS,
        norm_topk_prob=True,
        sliding_window=SLIDING_WINDOW,
        layer_types=LAYER_TYPES,
        rope_parameters=ROPE_PARAMETERS,
    )


def det(shape, scale=0.05):
    n = 1
    for d in shape:
        n *= d
    return (mx.arange(n).reshape(shape).astype(mx.float32) % 7 - 3) * scale


def norm_w(shape, scale):
    return mx.ones(shape) * scale


FIXTURES_DIR = Path(__file__).resolve().parents[2] / (
    "MacMLXCore/Tests/MacMLXCoreTests/Fixtures"
)


def save(name: str, fixture: dict, out):
    path = FIXTURES_DIR / name
    fx = dict(fixture)
    fx["expected_output"] = out
    mx.save_safetensors(str(path), fx)
    print("saved", path.name, "output", tuple(out.shape))
    print("  out[0,0,:5]", out.reshape(-1, out.shape[-1])[0, :5])


# ---- attention weight keys (shared) -----------------------------------------
def attention_weights():
    return {
        "q_proj.weight": det([N_HEADS * HEAD_DIM, HIDDEN], 0.03),
        "k_proj.weight": det([N_KV_HEADS * HEAD_DIM, HIDDEN], 0.04),
        "v_proj.weight": det([N_KV_HEADS * HEAD_DIM, HIDDEN], 0.05),
        "o_proj.weight": det([HIDDEN, N_HEADS * HEAD_DIM], 0.035),
        "q_norm.weight": norm_w([HEAD_DIM], 1.05),
        "k_norm.weight": norm_w([HEAD_DIM], 0.95),
    }


def moe_weights():
    return {
        "gate.weight": det([N_EXPERTS, HIDDEN], 0.05),
        "switch_mlp.gate_proj.weight": det([N_EXPERTS, MOE_INTERMEDIATE, HIDDEN], 0.05),
        "switch_mlp.up_proj.weight": det([N_EXPERTS, MOE_INTERMEDIATE, HIDDEN], 0.04),
        "switch_mlp.down_proj.weight": det([N_EXPERTS, HIDDEN, MOE_INTERMEDIATE], 0.045),
    }


def causal_bool_mask(s):
    linds = mx.arange(s)[:, None]
    rinds = mx.arange(s)[None, :]
    return linds >= rinds


def sliding_bool_mask(s, window):
    linds = mx.arange(s)[:, None]
    rinds = mx.arange(s)[None, :]
    return (linds >= rinds) & (linds < rinds + window)


# ---- 1. MoE block -----------------------------------------------------------
def capture_moe():
    args = make_args()
    moe = MellumSparseMoeBlock(args)
    weights = moe_weights()
    moe.load_weights(list(weights.items()))
    # Seeded-random input rather than det(): the modular det() pattern can make
    # two experts' gate logits *exactly* equal for some token, and which of the
    # tied pair lands in the top-k is MLX-implementation-defined tie-order —
    # not part of Mellum's routing logic. Real activations never tie exactly, so
    # a random input tests the router + experts faithfully and deterministically
    # (the input is saved into the fixture and reloaded verbatim on the Swift
    # side). The end-to-end model fixture already covers det()-derived MoE input.
    mx.random.seed(1)
    x = mx.random.normal([B, S, HIDDEN]) * 0.3
    out = moe(x)
    mx.eval(out)
    save("mellum2_moe_fixture.safetensors", {**weights, "x": x}, out)


# ---- 2. Attention (sliding, default RoPE, windowed mask) ---------------------
def capture_attention_sliding():
    args = make_args()
    attn = Attention(args, layer_idx=0)  # layer 0 is "sliding_attention"
    weights = attention_weights()
    attn.load_weights(list(weights.items()))
    x = det([B, S, HIDDEN], 0.03)
    mask = sliding_bool_mask(S, SLIDING_WINDOW)
    out = attn(x, mask=mask, cache=None)
    mx.eval(out)
    save(
        "mellum2_attention_sliding_fixture.safetensors",
        {**weights, "x": x, "mask": mask.astype(mx.uint8)},
        out,
    )


# ---- 3. Attention (full, YaRN RoPE, full causal mask) -----------------------
def capture_attention_full():
    args = make_args()
    attn = Attention(args, layer_idx=2)  # layer 2 is "full_attention"
    weights = attention_weights()
    attn.load_weights(list(weights.items()))
    x = det([B, S, HIDDEN], 0.03)
    mask = causal_bool_mask(S)
    out = attn(x, mask=mask, cache=None)
    mx.eval(out)
    save(
        "mellum2_attention_full_fixture.safetensors",
        {**weights, "x": x, "mask": mask.astype(mx.uint8)},
        out,
    )


# ---- 4. Full model (embed -> layers -> norm -> lm_head) ---------------------
def capture_model():
    args = make_args()
    model = Model(args)

    weights = {"model.embed_tokens.weight": det([VOCAB, HIDDEN], 0.02)}
    for l in range(NUM_LAYERS):
        p = f"model.layers.{l}"
        for k, v in attention_weights().items():
            weights[f"{p}.self_attn.{k}"] = v
        for k, v in moe_weights().items():
            weights[f"{p}.mlp.{k}"] = v
        weights[f"{p}.input_layernorm.weight"] = norm_w([HIDDEN], 1.02)
        weights[f"{p}.post_attention_layernorm.weight"] = norm_w([HIDDEN], 0.98)
    weights["model.norm.weight"] = norm_w([HIDDEN], 1.0)
    weights["lm_head.weight"] = det([VOCAB, HIDDEN], 0.03)

    model.load_weights(list(weights.items()))

    tokens = mx.array([[1, 5, 2, 8, 3, 0]], dtype=mx.int32)  # [B, S]
    out = model(tokens)  # default cache -> prefill, offset 0; logits [B, S, VOCAB]
    mx.eval(out)
    save("mellum2_model_fixture.safetensors", {**weights, "x": tokens}, out)


if __name__ == "__main__":
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    capture_moe()
    capture_attention_sliding()
    capture_attention_full()
    capture_model()
    print("all mellum2 fixtures captured ->", FIXTURES_DIR)
