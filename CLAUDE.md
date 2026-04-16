# CLAUDE.md
# macMLX — Master Context File

## Project Vision

Native macOS LLM inference desktop app for Apple Silicon.
Inspired by Swama's Swift-native approach, oMLX's feature depth,
and LM Studio's product experience — wrapped in a first-class SwiftUI GUI.

**Core differentiator: the only MLX inference tool with a truly native macOS GUI
AND an elegant CLI + TUI for developers.**

## Target Users

- **Newcomers**: people who want to run local LLMs on Mac without touching a terminal
- **Developers**: power users who want CLI integration, scripting, SSH access
- **Researchers**: benchmark-focused users running systematic model evaluations

## Competitive Landscape

| Tool | Strength | Gap we fill |
|------|---------|-------------|
| Swama | Swift-native inference, no GUI | We add GUI + TUI |
| oMLX | Feature depth | We add native GUI, better UX |
| LM Studio | Product polish | We add MLX-native, not GGUF |
| Ollama | CLI simplicity | We add GUI + MLX engine |
| SwiftLM | 100B+ MoE performance | We add all UX layers on top |

## Deliverables

Two products, one shared core:

```
macMLX.app     — SwiftUI GUI, menu bar, for all users
macmlx         — CLI + TUI (SwiftTUI), for developers
```

Both share `MacMLXCore` Swift package (inference, models, settings, HTTP server).

## Tech Stack (frozen)

### Core
- Language: **Swift 6** (strict concurrency)
- Inference default: **mlx-swift-lm** (Apple official SPM package, in-process)
- Inference optional: **SwiftLM** binary (100B+ MoE, subprocess)
- Inference optional: **mlx-lm Python** (max compatibility, subprocess)
- HTTP server: **Hummingbird** (Swift native, OpenAI-compatible API)
- Auto-update: **Sparkle 2.x** (EdDSA signed)
- Logging: **Pulse** (Swift) + **Rich** (Python engine side)

### GUI App
- Framework: **SwiftUI** (macOS 14+, Apple Silicon only)
- Distribution: GitHub Releases DMG (unsigned, Gatekeeper bypass doc)

### CLI + TUI
- CLI parsing: **swift-argument-parser** (Apple official)
- TUI framework: **SwiftTUI** (rensbreur/SwiftTUI)
- Distribution: **Homebrew tap** + bundled with DMG

### Python Engine (optional)
- Runtime: **Python 3.13** (managed by uv)
- Package manager: **uv** (never pip)
- Linter: **ruff**
- Web framework: **FastAPI + uvicorn**

### Platform
- macOS 14.0+ (Sonoma), Apple Silicon only
- No Windows, no Linux, no Intel Mac

## Architecture Principles

1. `MacMLXCore` owns all inference logic — GUI and CLI are thin shells
2. All engines conform to `InferenceEngine` Swift protocol — GUI never knows which engine runs
3. OpenAI-compatible HTTP always on at `localhost:8000` — external tools just work
4. Python engine is optional and advanced — never the default path
5. Settings at `~/.mac-mlx/settings.json`, logs at `~/.mac-mlx/logs/`

## Module Reference

Read the relevant file before starting any module:

- Architecture　　　→ `.claude/architecture.md`
- Swift conventions → `.claude/swift-conventions.md`
- UI guidelines　　 → `.claude/ui-guidelines.md`
- API contracts　　 → `.claude/api-contracts.md`
- Distribution　　　→ `.claude/distribution.md`
- Python backend　　→ `.claude/python-conventions.md`

Feature specs:
- Inference engines　→ `.claude/features/inference-engines.md`
- Onboarding　　　　→ `.claude/features/onboarding.md`
- Menu bar　　　　　→ `.claude/features/menubar.md`
- Model downloader　→ `.claude/features/model-downloader.md`
- Chat UI　　　　　 → `.claude/features/chat-ui.md`
- Parameters　　　　→ `.claude/features/parameters.md`
- Logs　　　　　　　→ `.claude/features/logs.md`
- Benchmark　　　　 → `.claude/features/benchmark.md`
- CLI + TUI　　　　 → `.claude/features/cli-tui.md`

## v0.1 Scope

Only implement items marked `v0.1` in each feature file.
Everything else: add `// TODO: v0.2` comment and skip.

## Universal Coding Rules

- All code comments and docs: **English**
- Commit format: `type(scope): description`
- No force unwrap (`!`), no `try!`
- Use `@Observable`, never `ObservableObject`
- Prefer `async/await` + structured concurrency, never callbacks
- One type per file, filename matches type name
- Every module needs unit tests
