# Contributing to macMLX

## Before You Start

- Bug fixes → open PR directly
- New features → open Discussion first
- Roadmap items → check README before starting

## Development Setup

```bash
git clone https://github.com/magicnight/mac-mlx
cd mac-mlx

# Install dev tools
brew bundle

# Swift GUI app
open macMLX/macMLX.xcodeproj

# CLI + TUI
cd macmlx-cli && swift build

# Python engine (optional)
cd Backend && uv sync
```

## Code Style

**Swift:**
```bash
swiftformat .
swiftlint lint
```

**Python:**
```bash
cd Backend
uv run ruff format .
uv run ruff check .
```

Commit format: `type(scope): description`

## Adding Python Dependencies

```bash
cd Backend
uv add package-name          # runtime
uv add --dev package-name    # dev only
```
Always commit `uv.lock`. Never use `pip install`.

## Pull Request Checklist

- [ ] Follows `.claude/swift-conventions.md`
- [ ] Tests added or updated
- [ ] No force unwraps
- [ ] UI changes include screenshots

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
