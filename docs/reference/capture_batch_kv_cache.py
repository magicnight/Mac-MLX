# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["mlx==0.32.0", "mlx-lm==0.31.3"]
# ///
"""Capture mlx-lm BatchKVCache reference for the Swift port (A2b).

Pure array-op parity: BatchKVCache.update_and_fetch / make_mask / filter /
extract / extend involve NO RoPE, so the fixture is safe against the
mlx 0.32.0 (capture) vs mlx-swift 0.31.1 (Swift core) batched-RoPE delta —
these are deterministic concatenate / slice / compare / pad ops only.

The cohort is deliberately RAGGED: real prompt lengths [3, 5, 2] left-padded to
Lmax = 5, i.e. ``left_padding = [2, 0, 3]``. This exercises the per-row offset
book-keeping and the left-padding attention mask that stock KVCacheSimple cannot
express.

Env: the scratchpad uv venv — mlx 0.32.0 + mlx-lm 0.31.3, Python 3.13.
Offline; no Python enters macMLX. Mirrors the DeepSeek capture precedent in
docs/reference/.
"""

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache

B, H, DK, DV = 3, 2, 4, 4
LEFT_PADDING = [2, 0, 3]  # real lengths [3, 5, 2], Lmax = 5
LMAX = 5


def det(shape, scale=0.05):
    """Deterministic weight tensor — identical formula on the Swift side."""
    n = 1
    for d in shape:
        n *= d
    return (mx.arange(n).reshape(shape).astype(mx.float32) % 7 - 3) * scale


def snap(x):
    """Detach + materialize a snapshot.

    ``BatchKVCache`` mutates ``offset``/``keys``/``values`` in place across
    steps, so a bare reference would alias the final state. ``x + 0`` forces a
    fresh buffer and ``mx.eval`` fixes its values at this instant.
    """
    y = x + 0
    mx.eval(y)
    return y


fixture = {}

cache = BatchKVCache(LEFT_PADDING)
fixture["init_offset"] = snap(cache.offset)  # [-2, 0, -3]
fixture["init_left_padding"] = snap(cache.left_padding)  # [2, 0, 3]

# --- stage 1: prefill the whole left-padded [B, H, Lmax, D] block ---
kp = det([B, H, LMAX, DK], 0.05)
vp = det([B, H, LMAX, DV], 0.03)
mask_prefill = cache.make_mask(LMAX)  # idx == 0 → shape [B, 1, Lmax, Lmax]
kfetch, vfetch = cache.update_and_fetch(kp, vp)
fixture["prefill_keys"] = snap(kp)
fixture["prefill_values"] = snap(vp)
fixture["prefill_mask"] = snap(mask_prefill.astype(mx.uint8))
fixture["prefill_fetch_keys"] = snap(kfetch)
fixture["prefill_fetch_values"] = snap(vfetch)
fixture["offset_after_prefill"] = snap(cache.offset)  # [3, 5, 2]

# --- stage 2: decode step 1 ---
kd1 = det([B, H, 1, DK], 0.07)
vd1 = det([B, H, 1, DV], 0.09)
mask_d1 = cache.make_mask(1)  # idx == 5 → shape [B, 1, 1, 6]
kf1, vf1 = cache.update_and_fetch(kd1, vd1)
fixture["decode1_keys"] = snap(kd1)
fixture["decode1_values"] = snap(vd1)
fixture["decode1_mask"] = snap(mask_d1.astype(mx.uint8))
fixture["decode1_fetch_keys"] = snap(kf1)
fixture["decode1_fetch_values"] = snap(vf1)
fixture["offset_after_decode1"] = snap(cache.offset)  # [4, 6, 3]

# --- stage 3: decode step 2 ---
kd2 = det([B, H, 1, DK], 0.11)
vd2 = det([B, H, 1, DV], 0.13)
mask_d2 = cache.make_mask(1)  # idx == 6 → shape [B, 1, 1, 7]
kf2, vf2 = cache.update_and_fetch(kd2, vd2)
fixture["decode2_keys"] = snap(kd2)
fixture["decode2_values"] = snap(vd2)
fixture["decode2_mask"] = snap(mask_d2.astype(mx.uint8))
fixture["decode2_fetch_keys"] = snap(kf2)
fixture["decode2_fetch_values"] = snap(vf2)
fixture["offset_after_decode2"] = snap(cache.offset)  # [5, 7, 4]

# --- state snapshot (idx == 7): the 4-tuple (keys, values, offset, left_padding) ---
sk, sv, soff, slp = cache.state
fixture["state_keys"] = snap(sk)
fixture["state_values"] = snap(sv)
fixture["state_offset"] = snap(soff)
fixture["state_left_padding"] = snap(slp)

# --- filter: keep rows [0, 2], drop row 1 (min-left-pad left-shift kicks in) ---
cache.filter(mx.array([0, 2]))
fixture["filter_keys"] = snap(cache.keys[..., : cache._idx, :])
fixture["filter_values"] = snap(cache.values[..., : cache._idx, :])
fixture["filter_offset"] = snap(cache.offset)  # [5, 4]
fixture["filter_left_padding"] = snap(cache.left_padding)  # [0, 1]
fixture["filter_idx"] = mx.array([cache._idx])  # 5

# --- extract row 1 of the filtered cache (original row 2) into a plain KVCache ---
ext = cache.extract(1)
fixture["extract_keys"] = snap(ext.keys)
fixture["extract_values"] = snap(ext.values)
fixture["extract_offset"] = mx.array([ext.offset])  # 4 (row 2 real length)

# --- extend: merge two tightly-packed caches. State-set so buffer == idx, which
#     checks the right-justify-to-max_idx math without 256-step-buffer noise. ---
ea_keys, ea_values = det([2, H, 4, DK], 0.02), det([2, H, 4, DV], 0.04)
ea_offset, ea_left_padding = mx.array([3, 4]), mx.array([1, 0])
cache_a = BatchKVCache([1, 0])
cache_a.state = (ea_keys, ea_values, ea_offset, ea_left_padding)  # idx == 4

eb_keys, eb_values = det([1, H, 5, DK], 0.06), det([1, H, 5, DV], 0.08)
eb_offset, eb_left_padding = mx.array([3]), mx.array([2])
cache_b = BatchKVCache([2])
cache_b.state = (eb_keys, eb_values, eb_offset, eb_left_padding)  # idx == 5

cache_a.extend(cache_b)
fixture["extendA_keys"] = snap(ea_keys)
fixture["extendA_values"] = snap(ea_values)
fixture["extendA_offset"] = snap(ea_offset)
fixture["extendA_left_padding"] = snap(ea_left_padding)
fixture["extendB_keys"] = snap(eb_keys)
fixture["extendB_values"] = snap(eb_values)
fixture["extendB_offset"] = snap(eb_offset)
fixture["extendB_left_padding"] = snap(eb_left_padding)
fixture["extend_keys"] = snap(cache_a.keys[..., : cache_a._idx, :])
fixture["extend_values"] = snap(cache_a.values[..., : cache_a._idx, :])
fixture["extend_offset"] = snap(cache_a.offset)  # [3, 4, 3]
fixture["extend_left_padding"] = snap(cache_a.left_padding)  # [2, 1, 2]
fixture["extend_idx"] = mx.array([cache_a._idx])  # 5

mx.eval(*fixture.values())

path = "batch_kv_cache_fixture.safetensors"
mx.save_safetensors(path, fixture)
print("saved", path)
print("prefill_mask", tuple(mask_prefill.shape), "decode1_mask", tuple(mask_d1.shape))
print("init_offset", fixture["init_offset"].tolist())
print("offset_after_prefill", fixture["offset_after_prefill"].tolist())
print("offset_after_decode1", fixture["offset_after_decode1"].tolist())
print("offset_after_decode2", fixture["offset_after_decode2"].tolist())
print("filter_offset", fixture["filter_offset"].tolist(), "filter_lp", fixture["filter_left_padding"].tolist())
print("extract_offset", fixture["extract_offset"].tolist())
