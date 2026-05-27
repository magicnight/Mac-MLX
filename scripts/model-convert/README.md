# scripts/model-convert

Local developer tool for re-quantizing MLX checkpoints into a format
that mlx-swift-lm can load.

**Not shipped with Mac-MLX.** This is a dev convenience wrapped around
`mlx_lm.convert` from the Python `mlx-lm` package. Lives here so the
team has a single reproducible invocation; the `.venv` it creates is
gitignored and never bundled into `macMLX.app` or the `macmlx` CLI.

## When to use it

mlx-swift-lm 3.31.x only supports **affine** quantization. Checkpoints
quantized in **MX format** (`mxfp4`, `mxfp8` — Microscaling, common in
Unsloth output and OpenAI's gpt-oss native format) fail to load with a
cryptic `Key …biases not found in …QuantizedLinear` error.

Run this tool to re-quantize such a checkpoint to affine. Source can
be a local directory or a HuggingFace repo ID.

## Usage

```bash
# From repo root:
./scripts/convert-model.sh \
    --hf-path <src-path-or-hf-id> \
    --mlx-path <output-dir> \
    -q --q-bits 4 --q-group-size 32
```

First invocation bootstraps the uv env (~30s, downloads Python 3.13 +
`mlx-lm`). Subsequent invocations are instant.

## Examples

Re-quantize an mxfp4 gpt-oss-20b you already have on disk:

```bash
./scripts/convert-model.sh \
    --hf-path "$HOME/Local Only Docs/models/nightmedia/naughty-gpt-oss-small" \
    --mlx-path "$HOME/Local Only Docs/models/gpt-oss-20b-Heretic-q4-affine" \
    -q --q-bits 4 --q-group-size 32
```

Download an unquantized source from HF and produce a clean 4-bit
affine MLX checkpoint in one step:

```bash
./scripts/convert-model.sh \
    --hf-path ArliAI/gpt-oss-20b-Derestricted \
    --mlx-path "$HOME/Local Only Docs/models/gpt-oss-20b-Derestricted-q4-affine" \
    -q --q-bits 4 --q-group-size 32
```

Full `mlx_lm.convert` flag reference:

```bash
./scripts/convert-model.sh --help
```

## How it works

`scripts/convert-model.sh` is a thin wrapper. It locates this
directory, runs `uv sync` on first use (creating `.venv/`), then
`exec`s `uv run python -m mlx_lm.convert "$@"` so any flag passed
through reaches `mlx_lm.convert` unchanged.

## Maintenance

- Bump `mlx-lm` minimum in `pyproject.toml` when upstream adds a new
  quantization format Mac-MLX users hit (e.g. future ParoQuant
  variants).
- Pin to a specific `mlx-lm` version if conversion output starts
  drifting in ways mlx-swift-lm can't read.
- Delete `.venv/` to force a fresh bootstrap if Python or mlx-lm gets
  into a weird state.
