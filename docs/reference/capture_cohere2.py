# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlx-lm==0.31.3",
#     "mlx==0.32.0",
#     "transformers>=5.0,<5.13",
#     "numpy",
# ]
# ///
"""Capture Cohere2 (`model_type: cohere2`, Cohere Command R7B) parity references (Track G).

Cohere2 is the Cohere-family decoder (parallel residual block, LayerNorm, tied
embeddings + logit scaling) with the Command R7B interleaved-attention twist that
is THE port risk:

  1. PARALLEL residual block: attention AND MLP both read the SAME
     `input_layernorm(x)`; the block output is `attn + mlp + x`. There is NO
     `post_attention_layernorm` (one norm per layer).
  2. Interleaved sliding-window / global attention on `sliding_window_pattern`.
     A layer is GLOBAL when `i % pattern == pattern - 1`, else sliding-window.
     RoPE (traditional / GPT-J, `position_embedding_type: rope_gptj`) is applied
     ONLY on the sliding-window layers; global layers get NO positional encoding
     at all (NoPE). Sliding layers use a windowed mask + a RotatingKVCache, global
     layers use the full-causal mask + a plain KVCache.
  3. `LayerNorm` (not RMSNorm) with a `layer_norm_bias` switch, a single
     `attention_bias` for q/k/v/o, and a `logit_scale` multiplier on the tied-head
     logits.

The two fixtures are adversarial — every switch takes the OPPOSITE value across
them, so any switch read backwards diverges on one side (parity fails) or fails
to load its weights:

  cohere2_realistic  attention_bias=F, layer_norm_bias=F, logit_scale=0.25 (the
                     shipped checkpoint's value, NOT the 0.0625 dataclass default),
                     sliding_window_pattern=4, sliding_window=4. With 4 layers this
                     gives L0/L1/L2 sliding (+RoPE) and L3 global (NoPE) — the full
                     three-sliding-then-global boundary the real model repeats.
  cohere2_inverse    attention_bias=T (q/k/v AND o biased), layer_norm_bias=T
                     (LayerNorm carries a bias term), logit_scale=0.0625 (the
                     Swift config OMITS logit_scale, so its fallback is pinned),
                     sliding_window_pattern=2, sliding_window=5. With 4 layers this
                     gives L0/L2 sliding (+RoPE) and L1/L3 global (NoPE) — a
                     different interleave that pins the `i % pattern == pattern-1`
                     global test.

CRITICAL DESIGN POINT: both fixtures use seq_len (8) > sliding_window (4 / 5), so
the sliding-window mask genuinely differs from the full-causal mask. A shorter
sequence would collapse the two masks and leave a mis-read sliding window
undetected. logit_scale differs across the two fixtures so a port that hard-codes
or drops it diverges.

Environment (PEP 723, run with `uv run docs/reference/capture_cohere2.py`):
mlx-lm 0.31.3 (ships cohere2.py) + mlx 0.32.0 + transformers 5.0-5.12. No Python
enters macMLX — this is an offline, one-shot capture; the safetensors are the
durable artifact. Weights are seeded random-normal (well-conditioned; see `det()`)
and saved into the fixture, so the Swift side loads identical values.
"""
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cohere2 import Model, ModelArgs

mx.random.seed(0)

# ---- tiny shared config -----------------------------------------------------
# Dimensions match the parity-proven Seed-OSS / Hunyuan fixture regime (hidden 32,
# intermediate 48) ON PURPOSE: these tiny reductions stay numerically
# well-conditioned, so the 1e-4 gate tests the architecture rather than float32
# matmul-kernel rounding. head_dim is EXPLICIT (8 = hidden/heads) because the
# upstream dataclass default is 128; a fixture that omitted it would trip the
# head_dim * n_heads == hidden precondition.
HIDDEN = 32
N_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = 8  # 4 * 8 == 32 == HIDDEN
INTERMEDIATE = 48
VOCAB = 64
LN_EPS = 1e-5
ROPE_THETA = 50000.0
NUM_LAYERS = 4
SEQ_LEN = 8  # > every sliding_window below so the window mask bites

# Two token sequences per fixture ("multiple sequences" parity in one forward).
TOKENS = mx.array(
    [[1, 5, 2, 8, 3, 0, 6, 4], [7, 0, 4, 1, 6, 2, 3, 5]], dtype=mx.int32
)  # [B=2, S=8]


def det(shape, scale=0.05):
    # Seeded random-normal projection weights (deterministic via mx.random.seed
    # above; the exact values are the durable artifact — saved into the fixture,
    # so the Swift side loads identical numbers). Random rather than a structured
    # arange pattern ON PURPOSE: a structured `(arange % 7 - 3)` pattern makes the
    # large intermediate matmuls sign-align into huge partial sums that
    # catastrophically cancel, so the final logits become dominated by float32
    # matmul-kernel rounding (~1e-3 relative) rather than the architecture; a
    # well-conditioned Gaussian keeps the cancellation ratio ~sqrt(n) so the parity
    # gate actually tests the port.
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
    attention_bias: bool,
    layer_norm_bias: bool,
    logit_scale: float,
    sliding_window: int,
    sliding_window_pattern: int,
) -> ModelArgs:
    return ModelArgs(
        model_type="cohere2",
        hidden_size=HIDDEN,
        head_dim=HEAD_DIM,
        num_hidden_layers=NUM_LAYERS,
        intermediate_size=INTERMEDIATE,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        rope_theta=ROPE_THETA,
        vocab_size=VOCAB,
        layer_norm_eps=LN_EPS,
        logit_scale=logit_scale,
        attention_bias=attention_bias,
        layer_norm_bias=layer_norm_bias,
        sliding_window=sliding_window,
        sliding_window_pattern=sliding_window_pattern,
    )


def model_weights(*, attention_bias: bool, layer_norm_bias: bool):
    weights = {"model.embed_tokens.weight": det([VOCAB, HIDDEN], 0.02)}
    for layer in range(NUM_LAYERS):
        p = f"model.layers.{layer}"
        # Attention projections (q/k/v/o) — all share the single attention_bias.
        weights[f"{p}.self_attn.q_proj.weight"] = det([N_HEADS * HEAD_DIM, HIDDEN], 0.03)
        weights[f"{p}.self_attn.k_proj.weight"] = det([N_KV_HEADS * HEAD_DIM, HIDDEN], 0.04)
        weights[f"{p}.self_attn.v_proj.weight"] = det([N_KV_HEADS * HEAD_DIM, HIDDEN], 0.05)
        weights[f"{p}.self_attn.o_proj.weight"] = det([HIDDEN, N_HEADS * HEAD_DIM], 0.035)
        if attention_bias:
            weights[f"{p}.self_attn.q_proj.bias"] = det([N_HEADS * HEAD_DIM], 0.02)
            weights[f"{p}.self_attn.k_proj.bias"] = det([N_KV_HEADS * HEAD_DIM], 0.03)
            weights[f"{p}.self_attn.v_proj.bias"] = det([N_KV_HEADS * HEAD_DIM], 0.04)
            weights[f"{p}.self_attn.o_proj.bias"] = det([HIDDEN], 0.025)
        # MLP (always bias-free).
        weights[f"{p}.mlp.gate_proj.weight"] = det([INTERMEDIATE, HIDDEN], 0.05)
        weights[f"{p}.mlp.up_proj.weight"] = det([INTERMEDIATE, HIDDEN], 0.04)
        weights[f"{p}.mlp.down_proj.weight"] = det([HIDDEN, INTERMEDIATE], 0.045)
        # LayerNorm: scale != 1 so the norm actually bites; bias only when enabled.
        weights[f"{p}.input_layernorm.weight"] = norm_w([HIDDEN], 1.02)
        if layer_norm_bias:
            weights[f"{p}.input_layernorm.bias"] = det([HIDDEN], 0.02)
    weights["model.norm.weight"] = norm_w([HIDDEN], 1.0)
    if layer_norm_bias:
        weights["model.norm.bias"] = det([HIDDEN], 0.015)
    # No lm_head — embeddings are always tied.
    return weights


def capture_model(
    name: str,
    *,
    attention_bias: bool,
    layer_norm_bias: bool,
    logit_scale: float,
    sliding_window: int,
    sliding_window_pattern: int,
):
    args = make_args(
        attention_bias=attention_bias,
        layer_norm_bias=layer_norm_bias,
        logit_scale=logit_scale,
        sliding_window=sliding_window,
        sliding_window_pattern=sliding_window_pattern,
    )
    model = Model(args)
    weights = model_weights(
        attention_bias=attention_bias, layer_norm_bias=layer_norm_bias
    )
    model.load_weights(list(weights.items()))

    out = model(TOKENS)  # default cache == None -> prefill, offset 0; logits [B, S, VOCAB]
    mx.eval(out)
    # `x` carries the tokens for the Swift side; expected_output is the logits.
    save(name, {**weights, "x": TOKENS}, out)


if __name__ == "__main__":
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Realistic: no attention bias, no LayerNorm bias, logit_scale 0.25 (shipped),
    # pattern 4 / window 4 -> L0,L1,L2 sliding (+RoPE), L3 global (NoPE).
    capture_model(
        "cohere2_realistic_fixture.safetensors",
        attention_bias=False,
        layer_norm_bias=False,
        logit_scale=0.25,
        sliding_window=4,
        sliding_window_pattern=4,
    )

    # Inverse: attention bias ON (q/k/v/o), LayerNorm bias ON, logit_scale 0.0625
    # (the Swift config OMITS it -> dataclass fallback), pattern 2 / window 5 ->
    # L0,L2 sliding (+RoPE), L1,L3 global (NoPE).
    capture_model(
        "cohere2_inverse_fixture.safetensors",
        attention_bias=True,
        layer_norm_bias=True,
        logit_scale=0.0625,
        sliding_window=5,
        sliding_window_pattern=2,
    )

    print("all cohere2 fixtures captured ->", FIXTURES_DIR)
