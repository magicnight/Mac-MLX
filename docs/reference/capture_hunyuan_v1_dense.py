# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlx-lm==0.31.3",
#     "mlx==0.32.0",
#     "transformers>=5.0,<5.13",
#     "numpy",
# ]
# ///
"""Capture Hunyuan V1 Dense (`model_type: hunyuan_v1_dense`) parity references (Track G).

Hunyuan V1 Dense (Tencent Hunyuan dense line — 0.5B/1.8B/4B/7B) is a dense
Llama-family decoder — GQA + SwiGLU MLP, RMSNorm — with two architecture-specific
twists that are THE port risk:

  1. per-head q/k RMSNorm applied AFTER RoPE (opposite order to the Qwen3 lineage,
     where q/k norm precedes RoPE), gated by `use_qk_norm`,
  2. a `DynamicNTKAlphaRoPE` whose base is pre-scaled by
     `alpha ** (head_dim / (head_dim - 2))`, consuming ONLY `rope_scaling.alpha`
     (a missing dict, or one without `alpha`, means alpha = 1.0 → plain RoPE).

Plus the usual dense switches: a single `attention_bias` drives q/k/v/o (unlike
Seed-OSS's three independent switches; MLP is always bias-free), an explicit
`head_dim` (falls back to `hidden_size / num_attention_heads`), and
`tie_word_embeddings` (the real checkpoints tie).

The two fixtures are adversarial — every switch takes the OPPOSITE value across
them, so any switch read backwards diverges on one side (parity fails) or fails
to load its weights:

  hunyuan_v1_dense_realistic  use_qk_norm=T, attention_bias=F, tie=T,
                              rope_scaling={type:dynamic, alpha:1000, factor:1},
                              EXPLICIT head_dim=16 (!= hidden/heads=8). Mirrors
                              the shipped checkpoints' switch shape and forces the
                              head_dim != hidden/heads o_proj path.
  hunyuan_v1_dense_inverse    use_qk_norm=F, attention_bias=T, tie=F,
                              rope_scaling OMITTED (alpha fallback -> 1.0),
                              head_dim OMITTED (fallback -> hidden/heads=8). Every
                              projection carries a bias; no q/k norm; untied lm_head.

alpha=1000 in the realistic fixture moves the RoPE base from 1e4 to ~2.68e7, so a
port that ignores alpha (or defaults it to 1.0) diverges immediately; the inverse
fixture pins the alpha=1.0 fallback.

Environment (PEP 723, run with `uv run docs/reference/capture_hunyuan_v1_dense.py`):
mlx-lm 0.31.3 (ships hunyuan_v1_dense.py) + mlx 0.32.0 + transformers 5.0-5.12.
No Python enters macMLX — this is an offline, one-shot capture; the safetensors are
the durable artifact. Weights are seeded random-normal (well-conditioned; see
`det()`) and saved into the fixture, so the Swift side loads identical values.
"""
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.hunyuan_v1_dense import Model, ModelArgs

mx.random.seed(0)

# ---- tiny shared config -----------------------------------------------------
# Dimensions match the parity-proven Seed-OSS fixture regime (hidden 32,
# intermediate 48) ON PURPOSE: these tiny reductions stay numerically
# well-conditioned, so the 1e-4 gate tests the architecture rather than float32
# matmul-kernel rounding. (Wider reductions — e.g. a 128-deep down_proj — combined
# with sign-varying activations amplify tiny kernel rounding differences well past
# 1e-4 even for a correct port; the Seed-OSS-scale dims avoid that.)
HIDDEN = 32
N_HEADS = 4
N_KV_HEADS = 2
INTERMEDIATE = 48
VOCAB = 64
RMS_EPS = 1e-5
MAXPOS = 128
NUM_LAYERS = 2
ROPE_THETA = 10000.0

# Realistic fixture uses an EXPLICIT head_dim that is deliberately NOT
# hidden/heads (=8), exercising the o_proj shape (n_heads*head_dim -> hidden)
# where head_dim != hidden/heads. The inverse fixture omits head_dim to exercise
# the hidden/heads fallback.
HEAD_DIM_EXPLICIT = 16
HEAD_DIM_FALLBACK = HIDDEN // N_HEADS  # 8

# Two token sequences per fixture ("multiple sequences" parity in one forward).
TOKENS = mx.array([[1, 5, 2, 8, 3, 0], [7, 0, 4, 1, 6, 2]], dtype=mx.int32)  # [B=2, S=6]


def det(shape, scale=0.05):
    # Seeded random-normal projection weights (deterministic via mx.random.seed
    # above; the exact values are the durable artifact — they are saved into the
    # fixture, so the Swift side loads identical numbers). Random rather than a
    # structured arange pattern ON PURPOSE: a structured `(arange % 7 - 3)`
    # pattern makes the large intermediate matmuls (e.g. the 128-wide down_proj)
    # sign-align into huge partial sums that catastrophically cancel, so the
    # final logits become dominated by float32 matmul-kernel rounding (~1e-3
    # relative) rather than the architecture — a well-conditioned Gaussian keeps
    # the cancellation ratio ~sqrt(n) so the parity gate actually tests the port.
    return mx.random.normal(shape).astype(mx.float32) * scale


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


def make_args(
    *,
    head_dim,
    attention_bias: bool,
    use_qk_norm: bool,
    tie_word_embeddings: bool,
    rope_scaling,
) -> ModelArgs:
    return ModelArgs(
        model_type="hunyuan_v1_dense",
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=NUM_LAYERS,
        intermediate_size=INTERMEDIATE,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        rms_norm_eps=RMS_EPS,
        rope_theta=ROPE_THETA,
        max_position_embeddings=MAXPOS,
        attention_bias=attention_bias,
        use_qk_norm=use_qk_norm,
        rope_scaling=rope_scaling,
        tie_word_embeddings=tie_word_embeddings,
        head_dim=head_dim,
    )


def model_weights(
    *, head_dim, attention_bias: bool, use_qk_norm: bool, tie_word_embeddings: bool
):
    weights = {"model.embed_tokens.weight": det([VOCAB, HIDDEN], 0.02)}
    for layer in range(NUM_LAYERS):
        p = f"model.layers.{layer}"
        # Attention projections (q/k/v/o) — all share the single attention_bias.
        weights[f"{p}.self_attn.q_proj.weight"] = det([N_HEADS * head_dim, HIDDEN], 0.03)
        weights[f"{p}.self_attn.k_proj.weight"] = det([N_KV_HEADS * head_dim, HIDDEN], 0.04)
        weights[f"{p}.self_attn.v_proj.weight"] = det([N_KV_HEADS * head_dim, HIDDEN], 0.05)
        weights[f"{p}.self_attn.o_proj.weight"] = det([HIDDEN, N_HEADS * head_dim], 0.035)
        if attention_bias:
            weights[f"{p}.self_attn.q_proj.bias"] = det([N_HEADS * head_dim], 0.02)
            weights[f"{p}.self_attn.k_proj.bias"] = det([N_KV_HEADS * head_dim], 0.03)
            weights[f"{p}.self_attn.v_proj.bias"] = det([N_KV_HEADS * head_dim], 0.04)
            weights[f"{p}.self_attn.o_proj.bias"] = det([HIDDEN], 0.025)
        if use_qk_norm:
            # Per-head RMSNorm over head_dim; scales != 1 so the norm actually bites.
            weights[f"{p}.self_attn.query_layernorm.weight"] = norm_w([head_dim], 1.05)
            weights[f"{p}.self_attn.key_layernorm.weight"] = norm_w([head_dim], 0.95)
        # MLP (always bias-free).
        weights[f"{p}.mlp.gate_proj.weight"] = det([INTERMEDIATE, HIDDEN], 0.05)
        weights[f"{p}.mlp.up_proj.weight"] = det([INTERMEDIATE, HIDDEN], 0.04)
        weights[f"{p}.mlp.down_proj.weight"] = det([HIDDEN, INTERMEDIATE], 0.045)
        weights[f"{p}.input_layernorm.weight"] = norm_w([HIDDEN], 1.02)
        weights[f"{p}.post_attention_layernorm.weight"] = norm_w([HIDDEN], 0.98)
    weights["model.norm.weight"] = norm_w([HIDDEN], 1.0)
    if not tie_word_embeddings:
        weights["lm_head.weight"] = det([VOCAB, HIDDEN], 0.03)
    return weights


def capture_model(
    name: str,
    *,
    head_dim,
    attention_bias: bool,
    use_qk_norm: bool,
    tie_word_embeddings: bool,
    rope_scaling,
):
    args = make_args(
        head_dim=head_dim,
        attention_bias=attention_bias,
        use_qk_norm=use_qk_norm,
        tie_word_embeddings=tie_word_embeddings,
        rope_scaling=rope_scaling,
    )
    model = Model(args)
    resolved_head_dim = head_dim if head_dim is not None else HEAD_DIM_FALLBACK
    weights = model_weights(
        head_dim=resolved_head_dim,
        attention_bias=attention_bias,
        use_qk_norm=use_qk_norm,
        tie_word_embeddings=tie_word_embeddings,
    )
    model.load_weights(list(weights.items()))

    out = model(TOKENS)  # default cache == None -> prefill, offset 0; logits [B, S, VOCAB]
    mx.eval(out)
    # `x` carries the tokens for the Swift side; expected_output is the logits.
    save(name, {**weights, "x": TOKENS}, out)


if __name__ == "__main__":
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Realistic: q/k norm ON, no attention bias, tied, dynamic RoPE alpha=1000,
    # explicit head_dim != hidden/heads.
    capture_model(
        "hunyuan_v1_dense_realistic_fixture.safetensors",
        head_dim=HEAD_DIM_EXPLICIT,
        attention_bias=False,
        use_qk_norm=True,
        tie_word_embeddings=True,
        rope_scaling={"type": "dynamic", "alpha": 1000.0, "factor": 1.0},
    )

    # Inverse: q/k norm OFF, attention bias ON (q/k/v/o), untied, RoPE alpha
    # fallback (rope_scaling omitted), head_dim fallback (omitted).
    capture_model(
        "hunyuan_v1_dense_inverse_fixture.safetensors",
        head_dim=None,
        attention_bias=True,
        use_qk_norm=False,
        tie_word_embeddings=False,
        rope_scaling=None,
    )

    print("all hunyuan_v1_dense fixtures captured ->", FIXTURES_DIR)
