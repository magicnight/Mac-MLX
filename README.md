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

## Feature highlights (v0.2 → v0.5)

Sixteen-plus shipped releases since the v0.1 MVP. The v0.5.0 headline
first, then the rest by area:

**Engine & models (v0.5.0)** — the biggest minor yet:
- **Vision-Language Models** — 16 VLM architectures (Qwen2.5/3-VL, Gemma-3, SmolVLM/2, Pixtral, Idefics3, glm_ocr, …). Image picker (drag-drop + paste), OpenAI multimodal `content` arrays, images persisted per conversation.
- **Tiered KV prompt cache** (hot RAM + cold SSD) — repeat prefixes (coding assistants re-sending history every turn) skip prefill.
- **Multi-model pool** — resident-memory-capped, LRU auto-evict, pin a model from the Models tab; cold-swap between pinned models skips the weight reload.
- **LoRA adapter inference** — drop a HuggingFace PEFT adapter into `~/.mac-mlx/adapters/`, pick it in the Parameters Inspector; auto-converts to MLX format at load.
- **MCP server** — `macmlx mcp serve` exposes `list_models` + `chat` tools over stdio to Claude Desktop / Cursor.

**Downloads**
- Resumable downloads survive cancels AND app quits (background URLSession + persisted resume data) — #5/#6/#8
- Live speed (MB/s) + ETA + per-file progress bar — #7
- Configurable Hugging Face endpoint for mirrors like `https://hf-mirror.com` (GUI + CLI, both) — #21
- **HF update detection** — downloaded models track the Hub commit SHA via a `.macmlx-meta.json` sidecar; Models tab surfaces an "Update available" badge when the Hub head advances (throttled to once / 24h) — v0.3.7

**Chat**
- Conversation sidebar: switch between saved chats, rename, delete, **rewind to here** (truncate after any message) — v0.3.2
- Streaming Markdown rendering with paragraph breaks preserved — #10 (+ v0.3.1 fix)
- Right-click any message: Copy / Edit / Regenerate / Delete — #11
- Per-model **Parameters Inspector** (⌘⌥I) — temperature, top_p, max tokens, system prompt persist to disk — #15
- Chat model switcher in toolbar loads on tap — v0.3.1
- **Collapsible `<think>` renderer** for Qwen3 / DeepSeek-R1 / Gemma reasoning blocks — v0.3.6

**Benchmark** — v0.3.0 tab for local tok/s, TTFT, peak memory, and history, with `Share to Community` to a GitHub-issue leaderboard — #22

**Logs** — v0.3.4 tab reads Pulse's store directly: search, level filter, live tail, clear. **MLX stdout / stderr** are teed into the log store at launch (v0.3.7) so library-level prints from `mlx-swift-lm` are visible without a debugger.

**API (OpenAI- and Ollama-compat)**
- Cold-swap: `/v1/chat/completions` auto-loads any locally-downloaded model by ID, serialises concurrent swaps — v0.3.3
- `/x/status` reports real RSS
- **CORS middleware + request logger + alias routes** + probe endpoints (`GET /`, `/v1`, `/v1/health`, `/v1/status`) — v0.3.6
- **Ollama API compatibility layer** — `GET /api/tags`, `GET /api/version`, `POST /api/chat`, `POST /api/generate`, `POST /api/show` with NDJSON streaming (default when `stream` omitted). Covers Zed, Immersive Translate, Open WebUI's Ollama provider — v0.3.6
- **Generation serialised across requests** — FIFO binary semaphore around every chat/completion path prevents parallel clients from crashing the engine — v0.3.6

**CLI** — native ANSI dashboards (`macmlx pull`, `serve`, `run`), honours `preferredEngine` + per-model `ModelParameters` + HF mirror settings. GUI and CLI now share `~/.mac-mlx/macmlx.pid` and refuse to double-bind :8000 — v0.3.1 / v0.3.3 / v0.3.5 / v0.3.7

**Sandbox off** — v0.3.6 disabled App Sandbox so `~/.mac-mlx/` reads/writes no longer redirect to the container home. Matches LM Studio / Ollama / oMLX. Gatekeeper remains the user-trust layer.

**Stability / polish** — chat survives sidebar tab switches (#1), single-instance enforcement (#2), Quit in menu bar (#17), `macmlx list` segfault fix (v0.3.1), ConversationStore date-precision fix (v0.3.3), and 13 user-reported bugs plus a dozen post-QA hot patches in v0.3.6

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
| **MLX Swift** (default) | ✅ Shipping | Apple's `mlx-swift-lm`, in-process. Supports models up to ~70B on 64 GB+ Macs (text + 16 VLM architectures). Tiered KV prompt cache + multi-model pool + LoRA adapters since v0.5.0. |
| **SwiftLM** (100B+ MoE) | 🔓 Reopenable | Subprocess path was blocked by App Sandbox until v0.3.6; with sandbox off, [#12](../../issues/12) / [#13](../../issues/13) are candidates for v0.5/v0.6 — not yet committed. Fills the [mlx-swift-lm#219](https://github.com/ml-explore/mlx-swift-lm/issues/219) MoE gap. |
| **Python mlx-lm** | 🔓 Reopenable | Same subprocess path. Max model coverage from mlx-community's Python-only checkpoints in exchange for `uv` on PATH. |

Settings → Engine shows Install Guide links for the non-default engines;
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
swift test --package-path MacMLXCore   # runs in ~3s
```

## Roadmap

### Shipped

- **v0.1.0** — native SwiftUI GUI, menu bar, CLI (`serve` / `pull` / `run` / `list` / `ps` / `stop`), HuggingFace downloader, OpenAI-compatible API, Sparkle auto-update, memory-aware onboarding.
- **v0.2.0** — Download + chat polish (10 issues): resumable downloads, HF mirrors, Markdown rendering, message edit/regenerate, Parameters Inspector.
- **v0.3.0 → v0.3.5** — Benchmark feature, cross-cutting gap fixes, UX patches, Chat history sidebar, API cold-swap, Logs tab, native ANSI CLI dashboards.
- **v0.3.6** — 13 user-reported bugs + post-QA hot patches: collapsible `<think>` renderer, sandbox disabled, CORS + request logger + alias routes, Ollama API compatibility layer with NDJSON streaming, GUI/CLI state coordination via `LoadHook`, FIFO generation semaphore, chat rendering fixes, sidebar rebuild.
- **v0.3.7** — maintenance release: CI pinned to Node.js 24 (`actions/checkout@v5` / `actions/cache@v5`), MLX stdout/stderr teed into the Logs tab, HF model-update detection via `.macmlx-meta.json` sidecar, shared `~/.mac-mlx/macmlx.pid` between GUI and CLI.

See `CHANGELOG.md` for the per-tag breakdown.

### On `main` (unreleased — next release)

- **MCP client pool** — `MCPClientPool` spawns each configured MCP server as a subprocess and aggregates their tools (`connectAll` / `listAllTools` / `callTool`). Library-ready; chat-side tool-call routing is the next step. Ships with two dead-server robustness fixes: a process-wide SIGPIPE ignore, and a connect timeout + `disconnect()` working around a swift-sdk 0.12.1 busy-loop when a spawned server dies before `initialize`.
- **API reasoning separation** ([#30](../../issues/30)) — reasoning models' `<think>` chain-of-thought now lands in `reasoning_content` (the DeepSeek / mlx-lm / LM Studio convention) instead of leaking into `content`, for both non-streaming and streaming responses.

### In progress — DeepSeek V3.2 architecture (pure-Swift port)

Proving macMLX can own a frontier model architecture as an **external overlay** — registering a custom `model_type` into mlx-swift-lm's factory with **zero fork of the library**. DeepSeek V3.2's DSA sparse attention (the "lightning indexer") + absorbed Multi-head Latent Attention are being ported to pure Swift and validated numerically against the Python `mlx-lm` reference (per-component parity to 1e-4, under an xcodebuild Metal test job). Foundation for a later DeepSeek V4 port. This is macMLX's differentiation now that Ollama and LM Studio ship MLX backends too.

### Later

- **v0.6** — Speech I/O via [`DePasqualeOrg/mlx-swift-audio`](https://github.com/DePasqualeOrg/mlx-swift-audio): MLX-native STT (Whisper, Fun-ASR for Chinese) + TTS (Marvis streaming, Chatterbox voice cloning, CosyVoice 2). Kokoro deliberately excluded to avoid GPL-3 espeak-ng.
- **v0.7+** — Community Benchmarks service (opt-in `POST /v1/benchmarks` → anonymised leaderboard by chip × model × quant × macOS), and continuous batching once upstream `mlx-swift-lm` ships `BatchGenerator` + `BatchKVCache`.

Full roadmap: [`docs/superpowers/plans/2026-05-11-omlx-parity-roadmap.md`](docs/superpowers/plans/2026-05-11-omlx-parity-roadmap.md).

### Reopenable / deferred

App Sandbox was disabled in v0.3.6, so several previously-closed items are feasible again (none committed yet):

- [#12](../../issues/12) Python `mlx-lm` engine via subprocess — max model coverage at the cost of `uv` on PATH.
- [#13](../../issues/13) SwiftLM binary engine via subprocess — 100B+ MoE coverage (Gemma 4 MoE, Llama 4 MoE) where `mlx-swift-lm` can't.
- [#20](../../issues/20) Homebrew tap for the CLI — once the CLI tarball ships as a release asset.
- [#19](../../issues/19) Signed + notarized DMG — needs a paid Apple Developer account.

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
