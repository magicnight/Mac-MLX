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

## Feature highlights (v0.2 → v0.3.7)

Fifteen-ish shipped releases since the v0.1 MVP. Pick the ones that matter:

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
| **MLX Swift** (default) | ✅ Shipping | Apple's `mlx-swift-lm`, in-process. Supports models up to ~70B on 64 GB+ Macs. Tiered KV prompt cache + multi-model pool since v0.4.0. |
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

### In progress (v0.4.0 — engine parity with oMLX)

Pivot from the original VLM-first plan: after comparing macMLX against [oMLX](https://github.com/jundot/omlx) (10.6k★), the higher-leverage investment is closing the inference-engine gap first. VLM moves to v0.4.1. Three independent sub-features, same release:

- **Tiered KV cache (hot RAM + cold SSD)** — shipped to `main` (PR #26). Successive chat turns on the same model reuse the KV cache when the new prompt extends the previous one. Hot tier = last-K snapshots in an LRU dict; cold tier = safetensors at `~/.mac-mlx/kv-cache/` (16-way sharded) round-tripped through mlx-swift-lm's `savePromptCache` / `loadPromptCache`. Settings → "KV Cache" section exposes hot/cold budgets + Clear All. Coding-assistant workflows (Claude Code / Cursor / Zed re-sending history every turn) see reduced TTFT on repeat prefixes.
- **Multi-model pool with auto-swap** — in PR #27. `ModelPool` actor holds `[String: InferenceEngine]` keyed by model ID, bounded by a user-configurable resident-memory cap (Settings → Model Pool; default 50% of total RAM). Non-pinned models auto-evict LRU when over budget. Pin a model from its row in the Models tab (orange pin icon) to keep it resident. Cold-swap between pinned models no longer re-reads weights.
- **MCP server MVP** — next. `macmlx mcp serve` CLI subcommand over stdio via [`modelcontextprotocol/swift-sdk`](https://github.com/modelcontextprotocol/swift-sdk) v0.11.x, exposing `list_models` and `chat` tools. Drop into Claude Desktop / Cursor's `mcpServers` config and run local MLX inference through their tool ecosystems.

Full plan: [`docs/roadmap-post-v0.3.6.md`](docs/roadmap-post-v0.3.6.md).

### Next minor (v0.4.1 — VLM)

Original v0.4 scope intact, shifted one dot:

- [#23](../../issues/23) Vision-Language Model support via `MLXVLM` (already in the dependency tree). 16 architectures: Qwen2.5-VL, Qwen3-VL, Gemma-3, SmolVLM/2, Paligemma, Pixtral, Idefics3, FastVLM, LFM2-VL, glm_ocr, mistral3. Image picker (NSOpenPanel + drag-drop + paste), OpenAI multimodal `content`-array parsing, images persisted to `~/.mac-mlx/conversations/<uuid>/images/`.

### Later (v0.5+)

- **v0.5** — Continuous batching (blocked on upstream `mlx-swift-lm` shipping `BatchGenerator` + `BatchKVCache` — tracked against Python mlx-lm PRs [#941](https://github.com/ml-explore/mlx-lm/pull/941) / [#1101](https://github.com/ml-explore/mlx-lm/pull/1101)), LoRA adapter loading (drop in existing HF adapters, no training), MCP *client* (configure external MCP servers from inside macMLX so chat models tool-call through them).
- **v0.6** — Speech I/O via [`DePasqualeOrg/mlx-swift-audio`](https://github.com/DePasqualeOrg/mlx-swift-audio) (replaces the original WhisperKit plan). MLX-native STT (Whisper, Fun-ASR for Chinese) + TTS (Marvis streaming, Chatterbox voice cloning, CosyVoice 2). Kokoro deliberately excluded to avoid GPL-3 espeak-ng.
- **v0.7** — Community Benchmarks service. Opt-in `POST /v1/benchmarks` endpoint aggregates anonymised `BenchmarkResult` + `HardwareInfo` by chip × model × quant × macOS version into a public leaderboard on this website and inside the app.

### Reopenable after sandbox removal (v0.3.6)

App Sandbox was disabled in v0.3.6; several previously-closed "not planned" items are feasible again. None are committed yet:

- [#12](../../issues/12) Python `mlx-lm` engine via subprocess — max model coverage at the cost of `uv` on PATH + slower first-token.
- [#13](../../issues/13) SwiftLM binary engine via subprocess — 100B+ MoE coverage where `mlx-swift-lm` can't handle (Gemma 4 MoE, Llama 4 MoE, DeepSeek-V3).
- [#20](../../issues/20) Homebrew tap for the CLI — unblocked once the CLI tarball ships as a release asset.

### Still deferred / blocked

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
