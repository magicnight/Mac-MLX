# macMLX

**English** · [简体中文](README.zh-CN.md)

> Native macOS LLM inference, powered by Apple MLX.

macMLX brings local LLM inference to Apple Silicon with a first-class
native macOS experience. No cloud, no telemetry, no Electron — just
your Mac running models at full speed.

**macMLX is for everyone**: a polished SwiftUI app for newcomers, and a
proper CLI for developers.

---

## Why macMLX?

| | macMLX | LM Studio | Ollama | oMLX |
|--|--------|-----------|--------|------|
| Native macOS GUI | ✅ SwiftUI | ❌ Electron | ❌ | ❌ Web UI |
| MLX-native inference | ✅ | ❌ GGUF | ❌ GGUF | ✅ |
| CLI | ✅ | ❌ | ✅ | ✅ |
| Resumable downloads + mirrors | ✅ | ⚠ partial | ⚠ partial | ❌ |
| OpenAI-compatible API | ✅ always-on | ✅ | ✅ | ✅ |
| Zero Python required | ✅ | ✅ | ✅ | ❌ |

## Requirements

- macOS 14.0 (Sonoma) or later
- Apple Silicon (M1 / M2 / M3 / M4)
- No Python required

## Installation

Download `macMLX-vX.X.X.dmg` from [Releases](../../releases), mount it,
and drag `macMLX.app` to `/Applications`.

The DMG is **not notarized** (no paid Apple Developer account yet —
[#19](../../issues/19)), so Gatekeeper blocks it on first launch. Pick
one of the two unblocks:

**Option A — terminal (recommended, always works):**

```bash
xattr -cr /Applications/macMLX.app    # clear quarantine attribute
open /Applications/macMLX.app         # first launch
```

**Option B — right-click:** right-click `macMLX.app` → **Open** → then
click **Open** again in the dialog. On newer macOS versions this
fallback dialog sometimes doesn't appear — if so, use Option A.

Want to see what Gatekeeper thinks of the app?

```bash
spctl --assess --verbose /Applications/macMLX.app
```

## Feature highlights (v0.2 → v0.3.5)

Twelve-ish shipped releases since the v0.1 MVP. Pick the ones that matter:

**Downloads**
- Resumable downloads survive cancels AND app quits (background URLSession + persisted resume data) — #5/#6/#8
- Live speed (MB/s) + ETA + per-file progress bar — #7
- Configurable Hugging Face endpoint for mirrors like `https://hf-mirror.com` (GUI + CLI, both) — #21

**Chat**
- Conversation sidebar: switch between saved chats, rename, delete, **rewind to here** (truncate after any message) — v0.3.2
- Streaming Markdown rendering with paragraph breaks preserved — #10 (+ v0.3.1 fix)
- Right-click any message: Copy / Edit / Regenerate / Delete — #11
- Per-model **Parameters Inspector** (⌘⌥I) — temperature, top_p, max tokens, system prompt persist to disk — #15
- Chat model switcher in toolbar loads on tap — v0.3.1

**Benchmark** — v0.3.0 tab for local tok/s, TTFT, peak memory, and history, with `Share to Community` to a GitHub-issue leaderboard — #22

**Logs** — v0.3.4 tab reads Pulse's store directly: search, level filter, live tail, clear

**API (OpenAI-compat)**
- Cold-swap: `/v1/chat/completions` auto-loads any locally-downloaded model by ID, serialises concurrent swaps — v0.3.3
- `/x/status` reports real RSS

**CLI** — native ANSI dashboards (`macmlx pull`, `serve`, `run`), honours `preferredEngine` + per-model `ModelParameters` + HF mirror settings — v0.3.1 / v0.3.3 / v0.3.5

**Stability / polish** — chat survives sidebar tab switches (#1), single-instance enforcement (#2), Quit in menu bar (#17), `macmlx list` segfault fix (v0.3.1), ConversationStore date-precision fix (v0.3.3), and a 3-commit independent code-review sweep in v0.3.0

Full per-tag breakdown: [CHANGELOG.md](CHANGELOG.md).

## Quickstart

### GUI
1. Launch macMLX — the setup wizard points you at `~/.mac-mlx/models` and picks the MLX Swift engine
2. Download a model from the built-in HuggingFace browser (resumable, works through mirrors)
3. Load it and start chatting

### CLI

```bash
macmlx pull mlx-community/Qwen3-8B-4bit     # download
macmlx list                                  # local models
macmlx run Qwen3-8B-4bit "Hello, world"      # single prompt
macmlx run Qwen3-8B-4bit                     # interactive
macmlx serve                                 # start API on :8000
macmlx ps                                    # is serve running?
macmlx stop                                  # graceful SIGTERM
```

## Connecting external tools

macMLX's OpenAI-compatible server runs on `http://localhost:8000/v1`
whenever you load a model (or whenever `macmlx serve` is running).

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-8B-4bit","messages":[{"role":"user","content":"Hi"}],"stream":true}'
```

Any OpenAI-compatible client works — point it at
`http://localhost:8000/v1` with any key:

- Cursor / Continue / Cline: set the custom base URL in settings
- Open WebUI: add as an OpenAI provider
- Raycast, Zed, etc.: same pattern

## Inference engines

| Engine | Status | Notes |
|--------|--------|-------|
| **MLX Swift** (default) | ✅ Shipping | Apple's `mlx-swift-lm`, in-process. Supports models up to ~70B on 64 GB+ Macs. |
| **SwiftLM** (100B+ MoE) | 🕒 Deferred to v0.3 | Subprocess launch blocked by App Sandbox policy; revisit when there's a concrete user ask ([#12](../../issues/12)). |
| **Python mlx-lm** | 🕒 Deferred to v0.3 | Same sandbox blocker ([#13](../../issues/13)). |

Settings → Engine shows Install Guide links for the deferred engines;
selecting them today surfaces a graceful "engine not available" state.

## Architecture

```
macMLX.app (SwiftUI)        macmlx (CLI)
            │                    │
            └─────── MacMLXCore ─┘    (Swift SPM package)
                        │
               InferenceEngine
                        │
                  MLXSwiftEngine    (in-process, mlx-swift-lm 3.31.x)
                        │
                  HummingbirdServer  → http://localhost:8000/v1
                        │
                Apple Silicon (Metal / ANE)
```

Data lives under `~/.mac-mlx/`:

```
~/.mac-mlx/
├── models/              # weights (default, changeable in Settings)
├── conversations/       # chat history JSON
├── model-params/        # per-model parameter overrides
├── downloads/           # resume-data for interrupted downloads
├── logs/                # Pulse logs
├── settings.json        # user preferences
└── macmlx.pid           # CLI daemon coordination
```

This path is deliberately a dotfile under real `$HOME`: macOS App
Sandbox's dotfile exemption lets a sandboxed app read/write here
without `user-selected.read-write` entitlements or security-scoped
bookmarks, while staying visible to power users.

## Building from source

```bash
git clone https://github.com/magicnight/mac-mlx
cd mac-mlx
brew bundle                            # dev tools

# GUI app
open macMLX/macMLX.xcodeproj           # or: xcodebuild -scheme macMLX build

# CLI
swift build --package-path macmlx-cli

# Core + tests
swift test --package-path MacMLXCore   # 90 tests, runs in ~3s
```

## Roadmap

### Shipped

- **v0.1.0** — native SwiftUI GUI, menu bar, CLI (`serve` / `pull` / `run` / `list` / `ps` / `stop`), HuggingFace downloader, OpenAI-compatible API, Sparkle auto-update, memory-aware onboarding.
- **v0.2.0** — Download + chat polish (10 issues): resumable downloads, HF mirrors, Markdown rendering, message edit/regenerate, Parameters Inspector.
- **v0.3.x** — six patch releases: Benchmark feature, cross-cutting gap fixes, UX patches, Chat history sidebar, API cold-swap, Logs tab, native ANSI CLI dashboards. See `CHANGELOG.md` for the per-tag breakdown.

### Next (v0.3.6 — maintenance patch)

- `macmlx --version` auto-bumped from the release tag
- `macmlx search <query>` command (queries `mlx-community` by default)
- Release binary slim-down via `strip -S` + dynamic Swift stdlib
- CLI `--log-level` + `--log-stderr` flags so Pulse logging surfaces from the terminal

### Next minor (v0.4.0)

- [#23](../../issues/23) Vision-Language Model support — `MLXVLM` already in the dependency tree, 16 architectures (Qwen2.5-VL, SmolVLM, Gemma-3, Paligemma, …). Full plan in [`.omc/plans/v0.4-vlm-plan.md`](.omc/plans/v0.4-vlm-plan.md).

### Later (v0.5+)

- **v0.5** — LoRA adapter loading (drop in existing HF adapters, no training) + conversation/dataset export
- **v0.6** — Speech I/O: WhisperKit for ASR (mic input in chat) + AVSpeechSynthesizer for TTS (play assistant replies)
- [#20](../../issues/20) Homebrew tap for the CLI (scheduled around v0.3.6–v0.4 once the CLI tarball lands as a release asset)

### Deferred / blocked

- [#19](../../issues/19) Signed + notarized DMG — needs a paid Apple Developer account
- Full native-MLX Whisper in Swift — upstream `mlx-swift-lm` doesn't ship audio models yet; WhisperKit (Core ML) covers the UX in the meantime
- [#12](../../issues/12) / [#13](../../issues/13) Subprocess-based engines (SwiftLM, Python mlx-lm) — closed as *not planned* because App Sandbox blocks spawning external binaries. Reopenable if sandbox policy is revisited or a Swift-native 100B+ MoE inference path appears.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and PRs welcome.

## License

Apache 2.0 — see [LICENSE](LICENSE)

## Acknowledgements

- [MLX](https://github.com/ml-explore/mlx) and [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-examples) by Apple
- [Swama](https://github.com/Trans-N-ai/swama) — Swift inference architecture inspiration
- [SwiftLM](https://github.com/SharpAI/SwiftLM) — 100B+ MoE engine (future integration)
- [oMLX](https://github.com/jundot/omlx) — feature depth reference
- [Hummingbird](https://github.com/hummingbird-project/hummingbird) — Swift HTTP server
- [Sparkle](https://github.com/sparkle-project/Sparkle) — auto-update framework
- [Pulse](https://github.com/kean/Pulse) — logging framework
- [SwiftTUI](https://github.com/rensbreur/SwiftTUI) — TUI framework

Full BibTeX citations: [CITATIONS.bib](CITATIONS.bib)
