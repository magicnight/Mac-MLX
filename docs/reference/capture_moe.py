"""Capture DeepSeek V3.2 MoE (routed + shared experts) reference (S3).

The MoE block replaces the dense MLP on routed layers: a `SwitchGLU` bank of
`n_routed_experts`, gated by `MoEGate` (the `noaux_tc` sigmoid router), plus
one always-on shared expert. This fixture asserts the block's final output
so the Swift `DeepseekV32MoE` reproduces it within 1e-4.

The router has five subtleties versus stock DeepSeek-V3 (final-weight gather
from the bias-free sigmoid, unconditional `routed_scaling_factor`, the
`n_group == 1` skip, no `+1e-20` epsilon, float32 cast). We use `n_group=1`
(V3.2's default group path) and a *non-zero* `e_score_correction_bias` so the
gather-from-orig-scores subtlety is actually exercised: with a zero bias the
selection and weighting tensors would coincide and hide the difference.

Environment: the same offline uv venv used for the attention/indexer/decoder
fixtures — mlx-lm 0.31.3. No Python enters macMLX; this is an offline capture.

Weights are deterministic (det()) so the Swift side loads identical values.
Note the stacked [n_experts, out, in] shape for the switch_mlp projections.
Shapes (tiny config, hidden=16, moe_intermediate=8, 4 routed + 1 shared):
  gate.weight:                     [4, 16]
  gate.e_score_correction_bias:    [4]     (non-zero — see above)
  switch_mlp.gate_proj.weight:     [4, 8, 16]
  switch_mlp.up_proj.weight:       [4, 8, 16]
  switch_mlp.down_proj.weight:     [4, 16, 8]
  shared_experts.gate_proj.weight: [8, 16]
  shared_experts.up_proj.weight:   [8, 16]
  shared_experts.down_proj.weight: [16, 8]
"""
import mlx.core as mx
from mlx_lm.models.deepseek_v32 import DeepseekV32MoE, ModelArgs

mx.random.seed(0)

HIDDEN = 16
MOE_INTERMEDIATE = 8
N_ROUTED = 4
N_SHARED = 1
TOP_K = 2
N_GROUP = 1
TOPK_GROUP = 1
B, S = 1, 4

args = ModelArgs(
    model_type="deepseek_v32",
    hidden_size=HIDDEN,
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
)

moe = DeepseekV32MoE(args)


def det(shape, scale=0.05):
    n = 1
    for d in shape:
        n *= d
    return (mx.arange(n).reshape(shape).astype(mx.float32) % 7 - 3) * scale


weights = {
    # Router: weight + a deliberately non-zero, asymmetric correction bias so
    # the "gather final weights from the bias-free sigmoid" fix is exercised.
    "gate.weight": det([N_ROUTED, HIDDEN], 0.05),
    "gate.e_score_correction_bias": mx.array([0.1, -0.15, 0.2, -0.05], dtype=mx.float32),
    # Routed experts — stacked [n_experts, out, in].
    "switch_mlp.gate_proj.weight": det([N_ROUTED, MOE_INTERMEDIATE, HIDDEN], 0.05),
    "switch_mlp.up_proj.weight": det([N_ROUTED, MOE_INTERMEDIATE, HIDDEN], 0.04),
    "switch_mlp.down_proj.weight": det([N_ROUTED, HIDDEN, MOE_INTERMEDIATE], 0.045),
    # Shared expert — intermediate = moe_intermediate * n_shared = 8.
    "shared_experts.gate_proj.weight": det([MOE_INTERMEDIATE, HIDDEN], 0.05),
    "shared_experts.up_proj.weight": det([MOE_INTERMEDIATE, HIDDEN], 0.04),
    "shared_experts.down_proj.weight": det([HIDDEN, MOE_INTERMEDIATE], 0.045),
}
moe.load_weights(list(weights.items()), strict=False)

x = det([B, S, HIDDEN], 0.03)
out = moe(x)  # [B, S, HIDDEN]
mx.eval(out)

fixture = dict(weights)
fixture["x"] = x
fixture["expected_output"] = out

path = "moe_fixture.safetensors"
mx.save_safetensors(path, fixture)
print("saved", path)
print("e_score_correction_bias", weights["gate.e_score_correction_bias"])
print("output shape", out.shape)
print("output[0,0,:6]", out[0, 0, :6])
print("output[0,-1,:6]", out[0, -1, :6])
