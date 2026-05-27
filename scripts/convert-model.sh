#!/usr/bin/env bash
# scripts/convert-model.sh — Local dev tool: re-quantize an MLX
# checkpoint to a format mlx-swift-lm can load (affine quantization).
#
# Not shipped with Mac-MLX. Wraps `mlx_lm.convert` from the Python
# `mlx-lm` package in a uv-managed venv that lives under
# scripts/model-convert/. See scripts/model-convert/README.md for the
# full rationale and examples.
#
# Inputs: any flags accepted by `mlx_lm.convert` (passed through
# verbatim). Run with `--help` to see them.
#
# Outputs: a new MLX model directory at the path you pass via
# `--mlx-path`. Existing dirs are NOT overwritten — pick a fresh path.
#
# Example — re-quantize a local mxfp4 gpt-oss checkpoint to 4-bit
# affine so MLXSwiftEngine can load it:
#
#     ./scripts/convert-model.sh \
#         --hf-path "$HOME/Local Only Docs/models/nightmedia/naughty-gpt-oss-small" \
#         --mlx-path "$HOME/Local Only Docs/models/gpt-oss-20b-Heretic-q4-affine" \
#         -q --q-bits 4 --q-group-size 32

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$HERE/model-convert"

if [[ ! -d "$ENV_DIR" ]]; then
    echo "error: $ENV_DIR missing — scripts/model-convert/ was deleted." >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    cat <<'HINT' >&2
error: `uv` not found on PATH.

Install with:
    curl -LsSf https://astral.sh/uv/install.sh | sh

Then re-run this script.
HINT
    exit 1
fi

cd "$ENV_DIR"

if [[ ! -d .venv ]]; then
    echo "==> First run: bootstrapping uv env (Python 3.13 + mlx-lm)…" >&2
    uv sync
fi

exec uv run python -m mlx_lm convert "$@"
