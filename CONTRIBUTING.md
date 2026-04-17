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

## Local-First Verification (before every push)

Push cost real money — macOS runners on GitHub Actions bill at 10× the Linux rate.
The CI workflow caches SPM checkouts, so warm runs take ~2 min, but please run
the same checks locally before pushing so the first-pass signal is free.

Minimum pre-push verification:

```bash
# SPM tests (Core + CLI)
swift test --package-path MacMLXCore
swift test --package-path macmlx-cli

# SwiftUI app
xcodebuild -project macMLX/macMLX.xcodeproj -scheme macMLX \
           -configuration Debug -destination 'platform=macOS' build
```

All three must succeed before `git push`. If the Xcode build fails with a
"missing Metal Toolchain" error on a fresh Mac, run:

```bash
sudo xcodebuild -downloadComponent MetalToolchain
```

## Push Discipline

Run CI intentionally, not reflexively:

- **Commit per logical unit locally** — small is fine.
- **Push in batches** — once a feature or stage reaches a natural stopping point,
  push all its commits together. Not every commit needs CI on its own.
- **Skip CI on docs-only changes** — include `[skip ci]` in the commit message,
  or rely on `paths-ignore` in `.github/workflows/ci.yml` (covers `.omc/**`,
  `*.md`, `LICENSE`, `CITATION.*`, `CONTRIBUTING.md`, `docs/**`).
- **`concurrency.cancel-in-progress`** is enabled — pushing a new commit
  automatically cancels the older in-flight run on the same ref.
- **Use `workflow_dispatch`** (Actions → CI → Run workflow) for on-demand
  verification without a push.

## Pull Request Checklist

- [ ] Follows `.claude/swift-conventions.md`
- [ ] Tests added or updated
- [ ] No force unwraps
- [ ] UI changes include screenshots
- [ ] Local verification (SPM + Xcode) runs green before push

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
