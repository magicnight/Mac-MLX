# Python Backend Conventions

## Role & Scope

Optional inference engine. A thin FastAPI wrapper around mlx-lm.
Default engine is mlx-swift-lm (Swift, in-process). Python engine is
for advanced users who need maximum model compatibility.

**Keep it simple. Do not build business logic here.**

## Python Version

**Pinned to Python 3.13.** Do not upgrade without testing mlx-lm compatibility.

```
# Backend/.python-version
3.13
```

uv reads this automatically. Never install Python manually.

## Package Management: uv (mandatory)

Never use `pip`, `pip install`, or `requirements.txt`.

```bash
# Setup (installs Python 3.13 + all deps automatically)
cd Backend && uv sync

# Add dependency
uv add fastapi

# Add dev dependency
uv add --dev pytest ruff

# Run server
uv run python server.py --port 8000

# Run tests
uv run pytest

# Lint + format
uv run ruff check .
uv run ruff format .
```

## pyproject.toml

```toml
[project]
name = "mac-mlx-backend"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "mlx-lm>=0.31.0",
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "rich>=13.0.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0.0",
    "httpx>=0.28.0",
    "ruff>=0.8.0",
]

[tool.ruff]
line-length = 100
target-version = "py313"

[tool.ruff.lint]
select = ["E", "F", "I"]
```

## Directory Structure

```
Backend/
├── pyproject.toml
├── uv.lock              # always commit
├── .python-version      # 3.13
├── server.py            # FastAPI entry point (<200 lines)
├── inference.py         # mlx-lm wrapper
├── models.py            # Pydantic schemas
└── config.py            # CLI args → config object
```

## Startup Protocol

Swift launches via uv and waits for `READY\n` on stdout:

```python
# server.py
if __name__ == "__main__":
    setup_logging()
    print("READY", flush=True)   # Swift reads this to confirm server is up
    uvicorn.run(app, host="127.0.0.1", port=args.port)
```

Swift timeout: 30 seconds. If `READY` not received, kill process and report error.

## How Swift Launches the Python Engine

```swift
// PythonMLXEngine.swift
private func makeProcess(port: Int) -> Process {
    let process = Process()
    process.executableURL = findUV()   // ~/.local/bin/uv or /opt/homebrew/bin/uv
    process.arguments = [
        "run",
        "--project", backendDirectory.path,
        "python", backendPath.path,
        "--port", "\(port)"
    ]
    return process
}
```

## uv Location Search Order

1. `~/.local/bin/uv` (default uv install)
2. `/opt/homebrew/bin/uv`
3. `which uv` result

If not found, show user-facing error:
> "uv is required for the Python engine. Install: `curl -LsSf https://astral.sh/uv/install.sh | sh`"

## Logging

Use Rich for beautiful stderr output. Swift captures and forwards to Pulse.

```python
# config_logging.py
from rich.logging import RichHandler
import logging, sys

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(
        rich_tracebacks=True,
        markup=True
    )]
)
```

## What NOT to do

- Never use `pip install` — always `uv add`
- Never create `requirements.txt`
- Never commit `.venv/` directory
- Never implement model download (Swift owns that)
- Never persist settings (Swift owns that)
- Never add admin UI or dashboard
- Keep `server.py` under 200 lines
