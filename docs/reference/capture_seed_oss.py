# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlx-lm==0.31.3",
#     "mlx==0.32.0",
#     "transformers>=5.0,<5.13",
#     "numpy",
# ]
# ///
"""Capture Seed-OSS (`model_type: seed_oss`) numerical-parity references (Track G).

Seed-OSS (ByteDance Seed-OSS-36B) is a standard dense Llama-family decoder — GQA
+ SwiGLU MLP, one RoPE family, RMSNorm — whose only architecture-specific twists
are three bias switches:

  • attention_bias      → q/k/v_proj bias,
  • attention_out_bias  → o_proj bias, a SEPARATE switch (Llama drives o_proj off
                          its single attention_bias; Seed-OSS gives o its own),
  • mlp_bias            → gate/up/down_proj bias.

The bias trio is THE port risk, so the fixtures are chosen adversarially to
triangulate the two INDEPENDENT attention switches rather than merely toggling
"all on / all off" (which cannot tell whether o_proj is wired to attention_bias
or to attention_out_bias):

  seed_oss_attention_qkv_bias_fixture   attn_bias=T, out_bias=F  (real asymmetry:
                                        q/k/v biased, o NOT)
  seed_oss_attention_o_bias_fixture     attn_bias=F, out_bias=T  (inverse: q/k/v
                                        NOT biased, o biased)
  seed_oss_mlp_bias_fixture             mlp_bias=T
  seed_oss_mlp_nobias_fixture           mlp_bias=F
  seed_oss_model_fixture                real-like: attn_bias=T, out_bias=F,
                                        mlp_bias=F, untied lm_head
  seed_oss_model_allbias_fixture        every bias on (attn/out/mlp), untied

The two asymmetric attention fixtures TOGETHER pin both switches: if o_proj were
wired to attention_bias, the qkv-bias fixture would wrongly add an o bias and the
o-bias fixture would wrongly drop it — either diverges at 1e-4.

The config carries `rope_scaling: {"rope_type": "default"}` and
`rope_theta: 1e7`, exactly like the real checkpoint, so the "default" RoPE path
(plain RoPE, scale 1.0) is exercised.

Environment (PEP 723, run with `uv run docs/reference/capture_seed_oss.py`):
mlx-lm 0.31.3 (first PyPI release shipping seed_oss.py) + mlx 0.32.0 +
transformers 5.0-5.12 (5.13+ breaks mlx-lm 0.31.3's tokenizer registration; the
checkpoint itself was made with transformers 4.55). No Python enters macMLX —
this is an offline, one-shot capture; the safetensors are the durable artifact.

Weights are deterministic via det() so the Swift side loads identical values.
"""
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.seed_oss import MLP, Attention, Model, ModelArgs

mx.random.seed(0)

# ---- tiny shared config -----------------------------------------------------
HIDDEN = 32
HEAD_DIM = 16
N_HEADS = 4
N_KV_HEADS = 2
INTERMEDIATE = 48
VOCAB = 40
RMS_EPS = 1e-6
MAXPOS = 64
NUM_LAYERS = 2
B, S = 1, 6

# Real Seed-OSS RoPE: a single "default" family at theta 1e7 (verbatim from the
# checkpoint's config.json). initialize_rope's "default" branch → plain RoPE.
ROPE_THETA = 1.0e7
ROPE_SCALING = {"rope_type": "default"}


def make_args(
    attention_bias: bool,
    attention_out_bias: bool,
    mlp_bias: bool,
    tie_word_embeddings: bool = False,
) -> ModelArgs:
    return ModelArgs(
        model_type="seed_oss",
        hidden_size=HIDDEN,
        num_hidden_layers=NUM_LAYERS,
        intermediate_size=INTERMEDIATE,
        num_attention_heads=N_HEADS,
        rms_norm_eps=RMS_EPS,
        vocab_size=VOCAB,
        num_key_value_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        max_position_embeddings=MAXPOS,
        attention_bias=attention_bias,
        attention_out_bias=attention_out_bias,
        mlp_bias=mlp_bias,
        rope_theta=ROPE_THETA,
        rope_traditional=False,
        rope_scaling=ROPE_SCALING,
        tie_word_embeddings=tie_word_embeddings,
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


def causal_bool_mask(s):
    linds = mx.arange(s)[:, None]
    rinds = mx.arange(s)[None, :]
    return linds >= rinds


# ---- attention weight keys --------------------------------------------------
def attention_weights(input_bias: bool, output_bias: bool):
    w = {
        "q_proj.weight": det([N_HEADS * HEAD_DIM, HIDDEN], 0.03),
        "k_proj.weight": det([N_KV_HEADS * HEAD_DIM, HIDDEN], 0.04),
        "v_proj.weight": det([N_KV_HEADS * HEAD_DIM, HIDDEN], 0.05),
        "o_proj.weight": det([HIDDEN, N_HEADS * HEAD_DIM], 0.035),
    }
    if input_bias:
        w["q_proj.bias"] = det([N_HEADS * HEAD_DIM], 0.02)
        w["k_proj.bias"] = det([N_KV_HEADS * HEAD_DIM], 0.03)
        w["v_proj.bias"] = det([N_KV_HEADS * HEAD_DIM], 0.04)
    if output_bias:
        w["o_proj.bias"] = det([HIDDEN], 0.025)
    return w


def mlp_weights(bias: bool):
    w = {
        "gate_proj.weight": det([INTERMEDIATE, HIDDEN], 0.05),
        "up_proj.weight": det([INTERMEDIATE, HIDDEN], 0.04),
        "down_proj.weight": det([HIDDEN, INTERMEDIATE], 0.045),
    }
    if bias:
        w["gate_proj.bias"] = det([INTERMEDIATE], 0.02)
        w["up_proj.bias"] = det([INTERMEDIATE], 0.03)
        w["down_proj.bias"] = det([HIDDEN], 0.015)
    return w


# ---- 1./2. Attention (two asymmetric-bias configs) --------------------------
def capture_attention(name: str, input_bias: bool, output_bias: bool):
    args = make_args(
        attention_bias=input_bias, attention_out_bias=output_bias, mlp_bias=False
    )
    attn = Attention(args)
    weights = attention_weights(input_bias, output_bias)
    attn.load_weights(list(weights.items()))
    x = det([B, S, HIDDEN], 0.03)
    mask = causal_bool_mask(S)
    out = attn(x, mask=mask, cache=None)
    mx.eval(out)
    save(name, {**weights, "x": x, "mask": mask.astype(mx.uint8)}, out)


# ---- 3./4. MLP (bias on/off) ------------------------------------------------
def capture_mlp(name: str, bias: bool):
    mlp = MLP(HIDDEN, INTERMEDIATE, bias=bias)
    weights = mlp_weights(bias)
    mlp.load_weights(list(weights.items()))
    x = det([B, S, HIDDEN], 0.04)
    out = mlp(x)
    mx.eval(out)
    save(name, {**weights, "x": x}, out)


# ---- 5./6. Full model -------------------------------------------------------
def capture_model(
    name: str, input_bias: bool, output_bias: bool, mlp_bias: bool
):
    args = make_args(
        attention_bias=input_bias,
        attention_out_bias=output_bias,
        mlp_bias=mlp_bias,
        tie_word_embeddings=False,
    )
    model = Model(args)

    weights = {"model.embed_tokens.weight": det([VOCAB, HIDDEN], 0.02)}
    for layer in range(NUM_LAYERS):
        p = f"model.layers.{layer}"
        for k, v in attention_weights(input_bias, output_bias).items():
            weights[f"{p}.self_attn.{k}"] = v
        for k, v in mlp_weights(mlp_bias).items():
            weights[f"{p}.mlp.{k}"] = v
        weights[f"{p}.input_layernorm.weight"] = norm_w([HIDDEN], 1.02)
        weights[f"{p}.post_attention_layernorm.weight"] = norm_w([HIDDEN], 0.98)
    weights["model.norm.weight"] = norm_w([HIDDEN], 1.0)
    weights["lm_head.weight"] = det([VOCAB, HIDDEN], 0.03)

    model.load_weights(list(weights.items()))

    tokens = mx.array([[1, 5, 2, 8, 3, 0]], dtype=mx.int32)  # [B, S]
    out = model(tokens)  # default cache -> prefill, offset 0; logits [B, S, VOCAB]
    mx.eval(out)
    save(name, {**weights, "x": tokens}, out)


if __name__ == "__main__":
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    # Asymmetric attention pair — triangulates the two independent switches.
    capture_attention(
        "seed_oss_attention_qkv_bias_fixture.safetensors",
        input_bias=True,
        output_bias=False,
    )
    capture_attention(
        "seed_oss_attention_o_bias_fixture.safetensors",
        input_bias=False,
        output_bias=True,
    )
    # MLP bias on/off.
    capture_mlp("seed_oss_mlp_bias_fixture.safetensors", bias=True)
    capture_mlp("seed_oss_mlp_nobias_fixture.safetensors", bias=False)
    # Full model: real-like (qkv bias, no o bias, no mlp bias) + everything-on.
    capture_model(
        "seed_oss_model_fixture.safetensors",
        input_bias=True,
        output_bias=False,
        mlp_bias=False,
    )
    capture_model(
        "seed_oss_model_allbias_fixture.safetensors",
        input_bias=True,
        output_bias=True,
        mlp_bias=True,
    )
    print("all seed_oss fixtures captured ->", FIXTURES_DIR)
