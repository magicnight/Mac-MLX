# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlx-lm==0.31.3",
#     "mlx==0.32.0",
#     "numpy",
# ]
# ///
"""Capture InternLM3 (`model_type: internlm3`, InternLM3-8B-Instruct) parity refs (Track G).

InternLM3 is a conventional dense Llama-family decoder (aggressive 16:1 GQA, SwiGLU
MLP, RMSNorm, standard serial pre-norm blocks) whose only real port risk is its
DynamicNTK-scaling RoPE — because mlx-lm's `internlm3.py` implementation carries
THREE VERIFIED DEFECTS. The macMLX port does NOT copy the bugs; it implements the
corrected semantics aligned with InternLM's official HuggingFace reference
`modeling_internlm3.py`. So the ground truth here is a MINIMALLY-PATCHED mlx-lm:
this script monkey-patches only the three defective lines (each annotated below with
its defect id), leaving the rest of the upstream module untouched.

======================================================================================
THE FOUR UPSTREAM DEFECTS (mlx-lm 0.31.3 `mlx_lm/models/internlm3.py`; upstream
`main` still carries them — this docstring is the reference for the pending issue).
Reference semantics confirmed against `internlm/internlm3-8b-instruct`'s
`modeling_internlm3.py` (repo `main`, checked 2026-07-11):

  Defect A — POSITION SCALE HARD-CODED TO 2.0.
    `Attention.__init__` computes
        rope_scale = (1 / factor) if (rope_scaling and rope_type == "linear") else 2.0
    and passes it as `mx.fast.rope(scale=rope_scale)`. For EVERY non-linear config
    (dynamic AND no-scaling) the position scale is 2.0 — silently DOUBLING every
    position. The reference `modeling_internlm3.py` scales positions by `1/factor`
    ONLY for the linear type and by 1.0 everywhere else.

  Defect B — CONFIG `factor` NEVER CONSUMED.
    `DynamicNTKScalingRoPE.__call__` grows the NTK base with
        base = orig_base * ((self.scale * seq_len / max_pos) - (self.scale - 1)) ** (d/(d-2))
    i.e. it uses `self.scale` (the buggy 2.0) where the config `rope_scaling.factor`
    (6.0 for the shipped 8B) belongs. The configured factor has NO effect on the base.

  Defect C — SEQUENCE LENGTH READ OFF THE HEADS AXIS.
    The same `__call__` computes `seq_len = x.shape[1] + offset`, but at the call
    site `x` is the transposed `[B, H, L, D]` tensor, so `shape[1]` is the HEAD count,
    not the sequence length. The dynamic-base threshold therefore triggers on the
    number of heads, not the actual sequence position.

  Defect D — THE NTK BASE REWRITE ALSO FIRES FOR THE `linear` TYPE.
    The `if seq_len > max_position_embeddings:` branch in the same `__call__` does
    not discriminate on rope_type, so a `linear` config ALSO rewrites the base on
    long sequences (using self.scale = 1/factor in the NTK formula). The reference
    keeps the base static for linear scaling — only the 1/factor position scale
    applies. (This port: `linear` → static base, pinned by the decode tests.)

CORRECTED SEMANTICS (what this script's patch — and the Swift port — implement):
  • linear   → position scale 1/factor, static base = rope_theta.
  • dynamic  → position scale 1.0; when seqLen > max_position_embeddings (seqLen from
               the SEQUENCE axis + offset) the base grows by
               rope_theta * ((factor * seqLen / max_pos) - (factor - 1)) ** (d/(d-2))
               with `factor` from config; otherwise base = rope_theta.
  • none / unknown type → position scale 1.0, static base = rope_theta (plain RoPE).
======================================================================================

The two fixtures are adversarial — every switch takes the OPPOSITE value across them,
and together they pin all three corrections:

  internlm3_dynamic_active  UNTIED, qkv_bias=F, bias=F, rope_scaling
                            {rope_type: dynamic, factor: 4.0}, max_position=6 < seq
                            len 8 → the dynamic base fires IN PREFILL. This one row
                            pins all three corrections at once: factor really enters
                            the base (B — factor 4, not the buggy 2.0), seqLen is the
                            sequence axis (C — heads=4 ≠ seqLen=8; reading the heads
                            axis yields 4 < 6, so the dynamic base would NOT fire and
                            the output would diverge), and positions are NOT doubled
                            (A — scale 1.0). The captured config ALSO carries a bogus
                            `head_dim: 999` decoy on the Swift side (this script never
                            passes it — upstream/ours both use hidden/heads); a port
                            that consumes it mis-shapes q_proj and fails to load.
  internlm3_inverse_plain   TIED (no lm_head key), qkv_bias=T (q/k/v/o all biased),
                            bias=T (MLP gate/up/down all biased), NO rope_scaling
                            (plain RoPE, position scale 1.0 — pins defect A on the
                            no-scaling path: the buggy upstream would use 2.0 here),
                            max_position=128 (never triggers the dynamic base).

The `linear` branch is NOT given a numeric fixture (its position scale 1/factor is
the one branch upstream got right); the decode unit tests pin that computation.

CRITICAL DESIGN POINTS:
  • head_dim = hidden/heads = 32/4 = 8 (even, > 2, so d/(d-2) is well-defined).
  • GQA is genuinely non-trivial: 4 query heads vs 2 KV heads (a 2:1 ratio) so a
    wrong KV-head reshape/broadcast diverges.
  • rope_theta is an explicit 10000 (the small tiny-fixture value) so the base math
    stays well-conditioned; the real 8B uses 5e7.

Environment (PEP 723, run with `uv run docs/reference/capture_internlm3.py`):
mlx-lm 0.31.3 (ships internlm3.py) + mlx 0.32.0. No Python enters macMLX — this is an
offline, one-shot capture; the safetensors are the durable artifact. Weights are
seeded random-normal (well-conditioned) and saved into the fixture, so the Swift side
loads identical values.
"""
from pathlib import Path

import mlx.core as mx
import mlx_lm.models.internlm3 as internlm3
from mlx_lm.models.internlm3 import Model, ModelArgs

mx.random.seed(0)

# ---- minimal ground-truth patch (the three defects, each annotated) ----------


def _corrected_rope_call(self, x, offset: int = 0):
    """Corrected `DynamicNTKScalingRoPE.__call__` — fixes defects B and C.

    Reads `self._dynamic_factor` (attached per-instance by `correct_ropes`, carrying
    the config factor — defect B) and derives the sequence length from the SEQUENCE
    axis `x.shape[-2]` rather than `x.shape[1]` (the heads axis — defect C). The
    position scale `self.scale` is corrected per-instance by `correct_ropes` (defect
    A), so it is passed through unchanged here.
    """
    # Defect C: sequence length from the SEQUENCE axis of [B, H, L, D], not shape[1].
    seq_len = x.shape[-2] + offset
    if self._dynamic_factor is not None and seq_len > self.max_position_embeddings:
        # Defect B: the NTK base consumes the CONFIG factor, not self.scale (2.0).
        f = self._dynamic_factor
        base = self.original_base * (
            (f * seq_len / self.max_position_embeddings) - (f - 1)
        ) ** (self.dims / (self.dims - 2))
    else:
        base = self.original_base
    return mx.fast.rope(
        x,
        self.dims,
        traditional=self.traditional,
        base=base,
        # Defect A: self.scale is the corrected position scale set by correct_ropes.
        scale=self.scale,
        offset=offset,
    )


# Patch the class method once (fixes B and C for every rope instance).
internlm3.DynamicNTKScalingRoPE.__call__ = _corrected_rope_call


def correct_ropes(model, rope_scaling):
    """Per-instance fixup of defect A (position scale) + inject the factor for B.

    Upstream `Attention.__init__` set every non-linear rope's `scale` to 2.0; here it
    is corrected to the reference value, and `_dynamic_factor` is attached (the config
    factor for the dynamic type, else None) so `_corrected_rope_call` can consume it.
    """
    rope_type = rope_scaling.get("rope_type") if rope_scaling else None
    factor = rope_scaling.get("factor") if rope_scaling else None
    for layer in model.model.layers:
        rope = layer.self_attn.rope
        if rope_type == "linear" and factor:
            rope.scale = 1.0 / factor  # A: linear position scale (upstream: correct)
            rope._dynamic_factor = None  # linear keeps a static base
        elif rope_type == "dynamic":
            rope.scale = 1.0  # A: dynamic must NOT double positions (upstream: 2.0)
            rope._dynamic_factor = factor  # B: config factor drives the NTK base
        else:
            rope.scale = 1.0  # A: no scaling → plain rope (upstream: 2.0)
            rope._dynamic_factor = None


# ---- tiny shared config ------------------------------------------------------
# Dimensions match the parity-proven Seed-OSS / Hunyuan / Cohere2 / MiniCPM3 fixture
# regime (hidden 32, intermediate 48) ON PURPOSE: these tiny reductions stay
# numerically well-conditioned, so the 1e-4 gate tests the architecture rather than
# float32 matmul-kernel rounding. GQA is a genuine 4:2 (heads:kv) split.
HIDDEN = 32
N_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = HIDDEN // N_HEADS  # 8 (upstream/ours never read config head_dim)
INTERMEDIATE = 48
VOCAB = 64
RMS_EPS = 1e-5
ROPE_THETA = 10000.0
NUM_LAYERS = 2
SEQ_LEN = 8

# Two token sequences per fixture ("multiple sequences" parity in one forward).
TOKENS = mx.array(
    [[1, 5, 2, 8, 3, 0, 6, 4], [7, 0, 4, 1, 6, 2, 3, 5]], dtype=mx.int32
)  # [B=2, S=8]


def det(shape, scale=0.05):
    # Seeded random-normal weights (deterministic via mx.random.seed above; the exact
    # values are the durable artifact — saved into the fixture, so the Swift side
    # loads identical numbers). Well-conditioned Gaussian keeps the cancellation ratio
    # ~sqrt(n) so the parity gate tests the port rather than float32 matmul rounding.
    return mx.random.normal(shape).astype(mx.float32) * scale


def norm_w(shape):
    # RMSNorm weights near 1 but NON-uniform so the norm actually bites.
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
    qkv_bias: bool,
    bias: bool,
    tie_word_embeddings: bool,
    max_position_embeddings: int,
    rope_scaling,
) -> ModelArgs:
    return ModelArgs(
        model_type="internlm3",
        hidden_size=HIDDEN,
        num_hidden_layers=NUM_LAYERS,
        intermediate_size=INTERMEDIATE,
        num_attention_heads=N_HEADS,
        rms_norm_eps=RMS_EPS,
        vocab_size=VOCAB,
        bias=bias,
        qkv_bias=qkv_bias,
        max_position_embeddings=max_position_embeddings,
        num_key_value_heads=N_KV_HEADS,
        rope_theta=ROPE_THETA,
        rope_traditional=False,
        rope_scaling=rope_scaling,
        tie_word_embeddings=tie_word_embeddings,
    )


def model_weights(*, qkv_bias: bool, bias: bool, tie_word_embeddings: bool):
    weights = {"model.embed_tokens.weight": det([VOCAB, HIDDEN], 0.02)}
    for layer in range(NUM_LAYERS):
        p = f"model.layers.{layer}"
        a = f"{p}.self_attn"
        # Attention: qkv_bias gates q/k/v AND o_proj (the misleading-name trap).
        weights[f"{a}.q_proj.weight"] = det([N_HEADS * HEAD_DIM, HIDDEN], 0.03)
        weights[f"{a}.k_proj.weight"] = det([N_KV_HEADS * HEAD_DIM, HIDDEN], 0.03)
        weights[f"{a}.v_proj.weight"] = det([N_KV_HEADS * HEAD_DIM, HIDDEN], 0.03)
        weights[f"{a}.o_proj.weight"] = det([HIDDEN, N_HEADS * HEAD_DIM], 0.035)
        if qkv_bias:
            weights[f"{a}.q_proj.bias"] = det([N_HEADS * HEAD_DIM], 0.02)
            weights[f"{a}.k_proj.bias"] = det([N_KV_HEADS * HEAD_DIM], 0.02)
            weights[f"{a}.v_proj.bias"] = det([N_KV_HEADS * HEAD_DIM], 0.02)
            weights[f"{a}.o_proj.bias"] = det([HIDDEN], 0.025)
        # MLP: `bias` gates gate/up/down (the OTHER misleading-name trap).
        weights[f"{p}.mlp.gate_proj.weight"] = det([INTERMEDIATE, HIDDEN], 0.05)
        weights[f"{p}.mlp.up_proj.weight"] = det([INTERMEDIATE, HIDDEN], 0.04)
        weights[f"{p}.mlp.down_proj.weight"] = det([HIDDEN, INTERMEDIATE], 0.045)
        if bias:
            weights[f"{p}.mlp.gate_proj.bias"] = det([INTERMEDIATE], 0.02)
            weights[f"{p}.mlp.up_proj.bias"] = det([INTERMEDIATE], 0.02)
            weights[f"{p}.mlp.down_proj.bias"] = det([HIDDEN], 0.02)
        weights[f"{p}.input_layernorm.weight"] = norm_w([HIDDEN])
        weights[f"{p}.post_attention_layernorm.weight"] = norm_w([HIDDEN])
    weights["model.norm.weight"] = norm_w([HIDDEN])
    # Untied checkpoints carry lm_head; tied ones do NOT (upstream never pops it, so a
    # stray key would fail strict load — the tied fixture simply omits it).
    if not tie_word_embeddings:
        weights["lm_head.weight"] = det([VOCAB, HIDDEN], 0.02)
    return weights


def capture_model(
    name: str,
    *,
    qkv_bias: bool,
    bias: bool,
    tie_word_embeddings: bool,
    max_position_embeddings: int,
    rope_scaling,
):
    args = make_args(
        qkv_bias=qkv_bias,
        bias=bias,
        tie_word_embeddings=tie_word_embeddings,
        max_position_embeddings=max_position_embeddings,
        rope_scaling=rope_scaling,
    )
    model = Model(args)
    correct_ropes(model, rope_scaling)  # apply defect-A/B corrections per instance
    weights = model_weights(
        qkv_bias=qkv_bias, bias=bias, tie_word_embeddings=tie_word_embeddings
    )
    model.load_weights(list(weights.items()))

    out = model(TOKENS)  # default cache == None -> prefill, offset 0; logits [B, S, VOCAB]
    mx.eval(out)
    save(name, {**weights, "x": TOKENS}, out)


if __name__ == "__main__":
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Dynamic-active: untied, no biases, dynamic scaling with factor 4, and
    # max_position 6 < seq len 8 so the NTK base fires in prefill — pinning defects
    # A (scale 1.0 not 2.0), B (factor 4 enters the base), and C (seqLen is the
    # sequence axis 8, not the heads axis 4).
    capture_model(
        "internlm3_dynamic_active_fixture.safetensors",
        qkv_bias=False,
        bias=False,
        tie_word_embeddings=False,
        max_position_embeddings=6,
        rope_scaling={"rope_type": "dynamic", "factor": 4.0},
    )

    # Inverse-plain: tied (no lm_head), qkv_bias ON (q/k/v/o), bias ON (MLP
    # gate/up/down), NO rope_scaling (plain RoPE, position scale 1.0 — pins defect A
    # on the no-scaling path), max_position 128 (never triggers the dynamic base).
    capture_model(
        "internlm3_inverse_plain_fixture.safetensors",
        qkv_bias=True,
        bias=True,
        tie_word_embeddings=True,
        max_position_embeddings=128,
        rope_scaling=None,
    )

    print("all internlm3 fixtures captured ->", FIXTURES_DIR)
