# CLAUDE.md
# macMLX — Master Context File

## Project Vision

Native macOS LLM inference desktop app for Apple Silicon.
Inspired by Swama's Swift-native approach, oMLX's feature depth,
and LM Studio's product experience — wrapped in a first-class SwiftUI GUI.

**Core differentiator: the whole product over one in-process Swift MLX core —
native SwiftUI GUI, CLI/TUI, an always-on OpenAI/Anthropic/Ollama-compatible
server, model library, benchmark, the tiered SSD KV cache and audio — in one
dependency-free DMG with zero Python anywhere.** Two claims must NOT be made:
"only native GUI" (oMLX v0.4+ ships a SwiftUI shell, over a Python core), and
"only Swift in-process MLX engine" (Swama is Swift-native, and Apple's own
`MLXFoundationModels`, tagged in mlx-swift-lm 3.32.3, makes in-process Swift
MLX inference a first-party library capability). The defensible claim is the
combination, not the engine.

## Target Users

- **Newcomers**: people who want to run local LLMs on Mac without touching a terminal
- **Developers**: power users who want CLI integration, scripting, SSH access
- **Researchers**: benchmark-focused users running systematic model evaluations

## Competitive Landscape

| Tool | Strength | Gap we fill |
|------|---------|-------------|
| Swama | Swift-native inference, no GUI | We add GUI + TUI |
| oMLX | Feature depth, Python core | Zero-Python single DMG, native chat app, CLI/TUI |
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
- Inference optional: **SwiftLM** binary (100B+ MoE, subprocess) — not shipped, reopenable (#12)
- Inference optional: **mlx-lm Python** (max compatibility, subprocess) — not shipped, reopenable (#13)
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

### Python Engine (optional, not shipped)
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

## Scope

The project is past v0.9. The **Roadmap** section of `README.md` is the single
source of truth for what ships next, and `CHANGELOG.md` for what shipped. The
`.claude/features/*.md` files are the v0.1 design baseline: their `v0.1`
markers are historical, so never down-scope work to them and never add
`// TODO: v0.2` markers. Plan documents under `docs/superpowers/` are working
notes, not commitments.

## Universal Coding Rules

- All code comments and docs: **English**
- Commit format: `type(scope): description`
- No force unwrap (`!`), no `try!`
- Use `@Observable`, never `ObservableObject`
- Prefer `async/await` + structured concurrency, never callbacks
- One type per file, filename matches type name
- Every module needs unit tests
