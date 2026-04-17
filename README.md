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

## What's in v0.2

Download and chat got serious. Ten issues landed since v0.1:

**Download**
- Real per-chunk progress bars with speed (MB/s) and ETA (#7)
- Cancel button mid-download; partial files cleaned up (#5)
- **Resume across cancel + app restart** — `URLError` resume-data persists to `~/.mac-mlx/downloads/` and the next Download continues from the last byte (#6)
- **Background URLSession** — transfers survive App Nap and full app quits; pending files are simply there next time you open the app (#8)
- Configurable HuggingFace endpoint for mirrors (e.g. `https://hf-mirror.com`) — for regions where huggingface.co is slow (#21)

**Chat**
- Full Markdown rendering for assistant messages, including streaming (#10)
- Right-click any message for Copy / Edit / Regenerate / Delete (#11)
- Conversations auto-save to `~/.mac-mlx/conversations/` and reload on launch (#9)
- **Parameters Inspector** (⌘⌥I) — per-model temperature, top_p, max tokens, system prompt — persists to `~/.mac-mlx/model-params/` (#15)

**Polish**
- Chat inference now survives sidebar tab switches (#1)
- Single-instance enforcement — a second launch activates the existing window (#2)
- Menu bar popover has a Quit button (#17)

Full list: [CHANGELOG.md](CHANGELOG.md).

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
swift test --package-path MacMLXCore   # 60 tests, runs in ~3s
```

## Roadmap

### Shipped

- **v0.1.0** — native SwiftUI GUI, menu bar, CLI (`serve` / `pull` / `run` / `list` / `ps` / `stop`), HuggingFace downloader, OpenAI-compatible API, Sparkle auto-update, memory-aware onboarding.
- **v0.2.0** — see "What's in v0.2" above. Download + chat polish; 10 issues closed.

### Next (v0.3 candidates)

- [#12](../../issues/12) SwiftLM engine (100B+ MoE) — pending sandbox policy review
- [#13](../../issues/13) Python mlx-lm engine — pending sandbox policy review
- [#22](../../issues/22) Benchmark feature (tok/s, TTFT, peak memory)
- [#23](../../issues/23) Vision-Language Model support
- [#16](../../issues/16) Logs tab (PulseUI console)
- [#20](../../issues/20) Homebrew tap for the CLI

### Long-term

- [#18](../../issues/18) Rich SwiftTUI dashboards (blocked upstream on Swift 6 compatibility)
- [#19](../../issues/19) Signed + notarized DMG (when there's a paid Apple Developer account)

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
