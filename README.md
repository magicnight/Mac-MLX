# macMLX

**English** · [简体中文](README.zh-CN.md)

> Native macOS LLM inference, powered by Apple MLX.

macMLX runs local LLMs on Apple Silicon with a first-class native macOS
experience — no cloud, no telemetry, no Electron. A polished SwiftUI app for
newcomers, a proper CLI for developers, and an always-on OpenAI-compatible
API for everything else.

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

macOS 14.0 (Sonoma) or later · Apple Silicon (M1–M4) · no Python required.

## Installation

Download `macMLX-vX.X.X.dmg` from [Releases](../../releases), mount it, and
drag `macMLX.app` to `/Applications`. The DMG isn't notarized yet
([#19](../../issues/19)), so clear Gatekeeper on first launch:

```bash
xattr -cr /Applications/macMLX.app    # clear quarantine
open /Applications/macMLX.app
```

(Or right-click the app → **Open** → **Open**.)

## Feature highlights

Sixteen-plus releases since the v0.1 MVP, by area:

- **Engine & models** — in-process MLX Swift engine (text + 16 VLM architectures, models to ~70B); tiered KV prompt cache (RAM + SSD), multi-model pool with LRU eviction, LoRA adapter inference, MCP server (`macmlx mcp serve`).
- **Downloads** — resumable across cancels and app quits, live speed/ETA, HuggingFace mirror support, Hub-commit update detection.
- **Chat** — conversation sidebar (rename, delete, rewind), streaming Markdown, per-message actions, per-model Parameters Inspector, collapsible `<think>` reasoning blocks.
- **API** — always-on OpenAI-compatible server plus an Ollama compatibility layer (NDJSON), model cold-swap by ID, CORS + probe endpoints, generation serialized across clients.
- **CLI** — native ANSI dashboards for `pull` / `serve` / `run`, PID coordination shared with the GUI.
- **Benchmark & Logs tabs** — local tok/s · TTFT · peak memory with a community leaderboard; a Pulse-backed log viewer with MLX stdout/stderr teed in.

Full per-release detail: [CHANGELOG.md](CHANGELOG.md).

## Quickstart

**GUI** — launch macMLX; the setup wizard picks the engine and model
directory; download a model from the built-in HuggingFace browser; load and
chat.

**CLI**

```bash
macmlx pull mlx-community/Qwen3-8B-4bit     # download
macmlx run Qwen3-8B-4bit "Hello, world"      # single prompt
macmlx serve                                 # API on :8000
macmlx ps / stop                             # status / shutdown
```

## Connecting external tools

The OpenAI-compatible server runs on `http://localhost:8000/v1` whenever a
model is loaded (or `macmlx serve` is running). Point any OpenAI client
(Cursor, Continue, Cline, Open WebUI, Zed, Raycast, …) at that base URL with
any key.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-8B-4bit","messages":[{"role":"user","content":"Hi"}]}'
```

## Inference engines

| Engine | Status | Notes |
|--------|--------|-------|
| **MLX Swift** (default) | ✅ Shipping | Apple's `mlx-swift-lm`, in-process. Text + 16 VLM architectures, models to ~70B, tiered KV cache + model pool + LoRA. |
| **SwiftLM** (100B+ MoE) | 🔓 Reopenable | Subprocess path, unblocked since sandbox-off ([#12](../../issues/12) / [#13](../../issues/13)) — not yet committed. |
| **Python mlx-lm** | 🔓 Reopenable | Subprocess path for max model coverage, in exchange for `uv` on PATH. |

Every engine hides behind one `InferenceEngine` protocol — the GUI never
knows which one runs.

## Architecture

```
macMLX.app (SwiftUI)   macmlx (CLI)
        └──── MacMLXCore ────┘        (Swift SPM package)
                  │
           InferenceEngine → MLXSwiftEngine (in-process)
                  │
           HummingbirdServer → http://localhost:8000/v1
                  │
           Apple Silicon (Metal / ANE)
```

Data lives under `~/.mac-mlx/` (models, conversations, params, logs,
settings) — a dotfile under real `$HOME`, so the sandboxed app reads/writes
without entitlements while staying visible to power users.

## Building from source

```bash
git clone https://github.com/magicnight/mac-mlx && cd mac-mlx
brew bundle                              # dev tools
open macMLX/macMLX.xcodeproj             # GUI  (or: xcodebuild -scheme macMLX build)
swift build --package-path macmlx-cli    # CLI
swift test  --package-path MacMLXCore    # tests (~3s)
```

## Roadmap

- **Shipped (v0.1 → v0.5)** — native GUI + menu bar + CLI + OpenAI API (v0.1); download & chat polish (v0.2); Benchmark, Logs, chat history, API cold-swap, Ollama compat, sandbox-off (v0.3); and the v0.5 engine leap — VLMs, tiered KV cache, model pool, LoRA, MCP server. Per-tag detail in [CHANGELOG.md](CHANGELOG.md).
- **Next release (on `main`)** — MCP client pool (chat-side tool routing next) and `reasoning_content` API separation ([#30](../../issues/30)).
- **In progress — DeepSeek V3.2 architecture** — pure-Swift port of DSA sparse attention + absorbed MLA as an external overlay into mlx-swift-lm's factory (zero fork), validated numerically against the Python reference. macMLX's differentiation now that Ollama and LM Studio also ship MLX backends.
- **Later** — v0.6 speech I/O (MLX-native STT/TTS); v0.7+ community benchmarks service and continuous batching once upstream ships it.
- **Reopenable** (feasible since sandbox-off) — Python / SwiftLM subprocess engines ([#12](../../issues/12) / [#13](../../issues/13)), Homebrew tap ([#20](../../issues/20)), signed + notarized DMG ([#19](../../issues/19)).

## Contributing · License

Issues and PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Apache 2.0
([LICENSE](LICENSE)).

## Acknowledgements

[MLX](https://github.com/ml-explore/mlx) + [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-examples) (Apple), [Swama](https://github.com/Trans-N-ai/swama), [SwiftLM](https://github.com/SharpAI/SwiftLM), [oMLX](https://github.com/jundot/omlx), [Hummingbird](https://github.com/hummingbird-project/hummingbird), [Sparkle](https://github.com/sparkle-project/Sparkle), [Pulse](https://github.com/kean/Pulse), [SwiftTUI](https://github.com/rensbreur/SwiftTUI). Full citations: [CITATIONS.bib](CITATIONS.bib).
