# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlx-lm==0.31.3",
#     "mlx==0.32.0",
#     "numpy",
# ]
# ///
"""Capture MiniCPM3 (`model_type: minicpm3`, OpenBMB MiniCPM3-4B) parity references (Track G).

MiniCPM3 is a Llama-family decoder with two defining departures, BOTH of which are
the port risk this fixture pins:

  1. NON-ABSORBED Multi-head Latent Attention (MLA). Q is materialized from a
     low-rank latent (`q_a_proj → q_a_layernorm → q_b_proj`) then split nope/rope
     (64/32). K/V come from `kv_a_proj_with_mqa` → a single-head RoPE key `k_pe`
     plus a compressed latent that `kv_b_proj` materializes into per-head `k_nope`
     (64) and `values` (64, the DISTINCT value head dim). `k_pe` is broadcast to
     all heads; the softmax scale is `(qk_nope + qk_rope) ** -0.5` (= q_head_dim,
     NOT v_head_dim). `attention_bias` gates ONLY q_a / kv_a_with_mqa / o_proj —
     q_b and kv_b are always bias-free (an asymmetry the fixtures pin).
  2. muP THREE scalings — each a parity chokepoint: embedding × `scale_emb`; every
     layer scales BOTH residual branches by `scale_depth / √num_layers`; and — when
     UNTIED — the hidden state is divided by `hidden_size / dim_model_base` before
     `lm_head` (tied path skips that division).

RoPE is the longrope `SuScaledRoPE`: it ALWAYS uses `long_factor` (short_factor is
ignored) and its mscale is `1.0` iff `max_position_embeddings == original_max`.

The two fixtures are adversarial — every switch/scaling takes the OPPOSITE value
across them, so any one read backwards diverges on one side (parity fails) or fails
to load its weights:

  minicpm3_realistic  attention_bias=F, tie=F (UNTIED → head ÷ (32/8=4), dim_base
                      8), scale_emb=12, scale_depth=1.4, longrope long_factor
                      [1.2,1.7] with short_factor [9.9,9.9] (DELIBERATELY wrong — a
                      port that uses short_factor instead of long_factor diverges),
                      max_position=64 == original_max=64 → mscale 1.0.
  minicpm3_inverse    attention_bias=T (q_a/kv_a/o biased; q_b/kv_b still bias-free),
                      tie=T (TIED → NO head division; no lm_head key), scale_emb=3,
                      scale_depth=0.7, long_factor [1.5,1.1], max_position=256 >
                      original_max=64 → mscale = √(1+ln(4)/ln(64)) ≠ 1 (pins the
                      non-trivial mscale formula), dim_model_base=16 (must NOT be
                      consumed on the tied path — a port that divides anyway
                      diverges).

CRITICAL DESIGN POINTS:
  • The MLA dims are all NON-trivial and distinct: qk_nope 8, qk_rope 4 →
    v_head_dim 8 (= hidden/heads), q_head_dim 12; long_factor length = qk_rope/2 =
    2. A degenerate (equal) choice could mask a wrong split/scale.
  • rope_theta is set EXPLICITLY to 10000 (NOT the 1_000_000 dataclass default) so
    the fixture also pins that theta is actually read.

Environment (PEP 723, run with `uv run docs/reference/capture_minicpm3.py`):
mlx-lm 0.31.3 (ships minicpm3.py) + mlx 0.32.0. No Python enters macMLX — this is an
offline, one-shot capture; the safetensors are the durable artifact. Weights are
seeded random-normal (well-conditioned) and saved into the fixture, so the Swift
side loads identical values.
"""
import math
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.minicpm3 import Model, ModelArgs

mx.random.seed(0)

# ---- tiny shared config -----------------------------------------------------
# Dimensions match the parity-proven Seed-OSS / Hunyuan / Cohere2 fixture regime
# (hidden 32, intermediate 48) ON PURPOSE: these tiny reductions stay numerically
# well-conditioned, so the 1e-4 gate tests the architecture rather than float32
# matmul-kernel rounding. The MLA sub-dims are all non-trivial and distinct.
HIDDEN = 32
N_HEADS = 4
N_KV_HEADS = 4  # MLA materializes full K/V, so kv_heads == heads here
Q_LORA = 16
KV_LORA = 12
QK_NOPE = 8
QK_ROPE = 4  # -> V_HEAD_DIM = HIDDEN // N_HEADS = 8, Q_HEAD_DIM = 12
V_HEAD_DIM = HIDDEN // N_HEADS  # 8
Q_HEAD_DIM = QK_NOPE + QK_ROPE  # 12
INTERMEDIATE = 48
VOCAB = 64
RMS_EPS = 1e-5
ROPE_THETA = 10000.0  # explicit, NOT the 1_000_000 dataclass default -> pins theta
NUM_LAYERS = 2
SEQ_LEN = 8

# Two token sequences per fixture ("multiple sequences" parity in one forward).
TOKENS = mx.array(
    [[1, 5, 2, 8, 3, 0, 6, 4], [7, 0, 4, 1, 6, 2, 3, 5]], dtype=mx.int32
)  # [B=2, S=8]


def det(shape, scale=0.05):
    # Seeded random-normal weights (deterministic via mx.random.seed above; the
    # exact values are the durable artifact — saved into the fixture, so the Swift
    # side loads identical numbers). Well-conditioned Gaussian (not a structured
    # arange) keeps the cancellation ratio ~sqrt(n) so the parity gate tests the
    # port rather than float32 matmul rounding.
    return mx.random.normal(shape).astype(mx.float32) * scale


def norm_w(shape):
    # RMSNorm weights near 1 but NON-uniform (per-element perturbation) so the norm
    # actually bites and a per-element misalignment can't hide behind a uniform scale.
    return mx.ones(shape) + det(shape, 0.02)


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
    attention_bias: bool,
    tie_word_embeddings: bool,
    scale_emb: float,
    scale_depth: float,
    dim_model_base: int,
    max_position_embeddings: int,
    long_factor,
    short_factor,
    original_max_position_embeddings: int,
) -> ModelArgs:
    return ModelArgs(
        model_type="minicpm3",
        hidden_size=HIDDEN,
        dim_model_base=dim_model_base,
        num_hidden_layers=NUM_LAYERS,
        intermediate_size=INTERMEDIATE,
        num_attention_heads=N_HEADS,
        rms_norm_eps=RMS_EPS,
        vocab_size=VOCAB,
        num_key_value_heads=N_KV_HEADS,
        q_lora_rank=Q_LORA,
        qk_nope_head_dim=QK_NOPE,
        qk_rope_head_dim=QK_ROPE,
        kv_lora_rank=KV_LORA,
        scale_depth=scale_depth,
        scale_emb=scale_emb,
        max_position_embeddings=max_position_embeddings,
        attention_bias=attention_bias,
        rope_theta=ROPE_THETA,
        rope_scaling={
            "type": "longrope",
            "long_factor": long_factor,
            "short_factor": short_factor,
            "original_max_position_embeddings": original_max_position_embeddings,
        },
        tie_word_embeddings=tie_word_embeddings,
    )


def model_weights(*, attention_bias: bool, tie_word_embeddings: bool):
    weights = {"model.embed_tokens.weight": det([VOCAB, HIDDEN], 0.02)}
    for layer in range(NUM_LAYERS):
        p = f"model.layers.{layer}"
        a = f"{p}.self_attn"
        # MLA projections. Bias gates ONLY q_a / kv_a_with_mqa / o_proj.
        weights[f"{a}.q_a_proj.weight"] = det([Q_LORA, HIDDEN], 0.03)
        weights[f"{a}.q_a_layernorm.weight"] = norm_w([Q_LORA])
        weights[f"{a}.q_b_proj.weight"] = det([N_HEADS * Q_HEAD_DIM, Q_LORA], 0.03)
        weights[f"{a}.kv_a_proj_with_mqa.weight"] = det([KV_LORA + QK_ROPE, HIDDEN], 0.04)
        weights[f"{a}.kv_a_layernorm.weight"] = norm_w([KV_LORA])
        weights[f"{a}.kv_b_proj.weight"] = det(
            [N_HEADS * (QK_NOPE + V_HEAD_DIM), KV_LORA], 0.04
        )
        weights[f"{a}.o_proj.weight"] = det([HIDDEN, N_HEADS * V_HEAD_DIM], 0.035)
        if attention_bias:
            weights[f"{a}.q_a_proj.bias"] = det([Q_LORA], 0.02)
            weights[f"{a}.kv_a_proj_with_mqa.bias"] = det([KV_LORA + QK_ROPE], 0.03)
            weights[f"{a}.o_proj.bias"] = det([HIDDEN], 0.025)
        # MLP (always bias-free).
        weights[f"{p}.mlp.gate_proj.weight"] = det([INTERMEDIATE, HIDDEN], 0.05)
        weights[f"{p}.mlp.up_proj.weight"] = det([INTERMEDIATE, HIDDEN], 0.04)
        weights[f"{p}.mlp.down_proj.weight"] = det([HIDDEN, INTERMEDIATE], 0.045)
        weights[f"{p}.input_layernorm.weight"] = norm_w([HIDDEN])
        weights[f"{p}.post_attention_layernorm.weight"] = norm_w([HIDDEN])
    weights["model.norm.weight"] = norm_w([HIDDEN])
    # Untied checkpoints carry lm_head; tied ones do NOT (upstream never pops it,
    # so a stray key would fail strict load — the tied fixture simply omits it).
    if not tie_word_embeddings:
        weights["lm_head.weight"] = det([VOCAB, HIDDEN], 0.02)
    return weights


def capture_model(
    name: str,
    *,
    attention_bias: bool,
    tie_word_embeddings: bool,
    scale_emb: float,
    scale_depth: float,
    dim_model_base: int,
    max_position_embeddings: int,
    long_factor,
    short_factor,
    original_max_position_embeddings: int,
):
    args = make_args(
        attention_bias=attention_bias,
        tie_word_embeddings=tie_word_embeddings,
        scale_emb=scale_emb,
        scale_depth=scale_depth,
        dim_model_base=dim_model_base,
        max_position_embeddings=max_position_embeddings,
        long_factor=long_factor,
        short_factor=short_factor,
        original_max_position_embeddings=original_max_position_embeddings,
    )
    model = Model(args)
    weights = model_weights(
        attention_bias=attention_bias, tie_word_embeddings=tie_word_embeddings
    )
    model.load_weights(list(weights.items()))

    out = model(TOKENS)  # default cache == None -> prefill, offset 0; logits [B, S, VOCAB]
    mx.eval(out)
    save(name, {**weights, "x": TOKENS}, out)


if __name__ == "__main__":
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Realistic: untied (head ÷ hidden/dim_base = 32/8 = 4), no attention bias,
    # scale_emb 12, scale_depth 1.4, max_position == original_max -> mscale 1.0.
    # short_factor is deliberately wrong ([9.9, 9.9]) — ignored by MiniCPM3, so a
    # port that mistakenly uses it diverges.
    capture_model(
        "minicpm3_realistic_fixture.safetensors",
        attention_bias=False,
        tie_word_embeddings=False,
        scale_emb=12.0,
        scale_depth=1.4,
        dim_model_base=8,
        max_position_embeddings=64,
        long_factor=[1.2, 1.7],
        short_factor=[9.9, 9.9],
        original_max_position_embeddings=64,
    )

    # Inverse: tied (NO head division; no lm_head key), attention bias ON
    # (q_a/kv_a/o only), scale_emb 3, scale_depth 0.7, max_position 256 >
    # original_max 64 -> non-trivial mscale, dim_model_base 16 (must NOT be
    # consumed on the tied path).
    factor = 256 / 64
    expected_mscale = math.sqrt(1 + math.log(factor) / math.log(64))
    print(f"inverse mscale = {expected_mscale:.6f} (must be != 1.0)")
    capture_model(
        "minicpm3_inverse_fixture.safetensors",
        attention_bias=True,
        tie_word_embeddings=True,
        scale_emb=3.0,
        scale_depth=0.7,
        dim_model_base=16,
        max_position_embeddings=256,
        long_factor=[1.5, 1.1],
        short_factor=[9.9, 9.9],
        original_max_position_embeddings=64,
    )

    print("all minicpm3 fixtures captured ->", FIXTURES_DIR)
