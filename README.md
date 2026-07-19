# macMLX

**English** · [简体中文](README.zh-CN.md)

> Native macOS LLM inference, powered by Apple MLX.

macMLX runs local LLMs on Apple Silicon with a first-class native macOS
experience — no cloud, no telemetry, no Electron. A polished SwiftUI app for
newcomers, a proper CLI for developers, and an always-on OpenAI-compatible
API for everything else.

---

## Why macMLX?

MLX inference and a CLI used to be the whole pitch — but as of 2026 both
[LM Studio](https://github.com/lmstudio-ai/mlx-engine) and
[Ollama](https://ollama.com/blog/mlx) ship MLX engines on Apple Silicon, and
LM Studio has the `lms` CLI. So the honest comparison is the *combination*: a
genuinely native macOS GUI, an always-on API, and zero Python in one ~50 MB app.

| | macMLX | LM Studio | Ollama | oMLX |
|--|--------|-----------|--------|------|
| Native macOS GUI | ✅ SwiftUI | Electron | menu-bar only | ✅ SwiftUI (v0.4+) |
| **Swift-native in-process engine** | ✅ | ❌ | ❌ | ❌ (Python core) |
| MLX inference | ✅ | ✅ | ✅ (preview) | ✅ |
| CLI | ✅ | ✅ `lms` | ✅ | launcher only |
| Resumable downloads + mirrors | ✅ | ⚠ partial | ⚠ partial | ❌ |
| OpenAI-compatible API | ✅ always-on | ✅ | ✅ | ✅ |
| Zero Python required | ✅ | ✅ | ✅ | ❌ |

Where macMLX stands alone: the **inference engine itself is Swift, running
in-process** — oMLX's native app (v0.4+) fronts a Python core, ours has no
Python anywhere in one ~50 MB DMG. On top of that: a proper CLI/TUI sharing
the same Swift core, and owning frontier model architectures in pure Swift
(the DeepSeek V3.2 port) instead of waiting for upstream.

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

## Feature highlights (v0.2 → v0.7)

Sixteen-plus releases since the v0.1 MVP, by area. **This section tracks the
latest shipped state — new features land here first, then get a one-line
roadmap entry below.**

- **Engine & models** — in-process MLX Swift engine (text + 16 VLM architectures with dedicated OCR-model recognition, models to ~70B); **continuous batching** (2.5-3.2× aggregate throughput under concurrent clients, engages only under real concurrency); tiered KV prompt cache (RAM + SSD) with **longest-common-prefix reuse** across agent turns; **speculative decoding** (draft models + acceptance-rate telemetry); multi-model pool with LRU eviction, LoRA adapter inference, MCP server (`macmlx mcp serve`); pure-Swift architecture ports — **DeepSeek V3.2**, **Mellum2**, Solar-Open, GLM-5.1 — registered as external overlays and parity-verified against the Python reference ([support tiers](docs/model-support.md)). Runs on a controlled minimal fork of mlx-swift carrying a single upstream-merged fix, dropped at the next upstream release.
- **Downloads** — resumable across cancels and app quits, live speed/ETA, HuggingFace mirror support, Hub-commit update detection.
- **Chat** — conversation sidebar (rename, delete, rewind), streaming Markdown, per-message actions, per-model Parameters Inspector, collapsible `<think>` reasoning blocks.
- **API** — always-on OpenAI-compatible server plus Ollama (NDJSON) and Anthropic (`/v1/messages`) compatibility; **structured output** (`response_format` json_object / JSON-schema subset, constrained decoding); `tools` pass-through with `tool_calls` responses; `logit_bias`, `logprobs`, XTC sampler, per-request LoRA adapters, KV-cache quantization; `/v1/embeddings` + `/v1/rerank`, optional bearer auth, model aliases + idle TTL, `reasoning_content` separation, model cold-swap by ID, stall watchdog, CORS + probe endpoints; concurrent clients batch onto one model instead of queuing serially.
- **CLI** — native ANSI dashboards for `pull` / `serve` / `run`, PID coordination shared with the GUI.
- **Activity, Benchmark & Logs tabs** — a live **Activity panel** with sudoless Apple-Silicon readouts (GPU occupancy, memory bandwidth, thermal/memory pressure, per-rail power) and the current inference **bottleneck** with advice, fusing hardware counters with the engine's own prefill/decode phase (the signal an external monitor can't reach); Benchmark's local tok/s · TTFT · peak memory with a community leaderboard, now attributing each run's decode bottleneck; a Pulse-backed log viewer with MLX stdout/stderr teed in.

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

> Kept current per release: when a `0.x` ships, it moves from a future section
> up to **Shipped**, and the feature highlights above get updated to match.

- **Shipped (v0.1 → v0.7.0)** — native GUI + menu bar + CLI + OpenAI API (v0.1); download & chat polish (v0.2); Benchmark, Logs, chat history, API cold-swap, Ollama compat, sandbox-off (v0.3); the v0.5 engine leap — VLMs, tiered KV cache, model pool, LoRA, MCP server + client pool, chat tool routing; server hardening, embeddings + rerank, and the stability wave (v0.5.1-0.5.3); the **DeepSeek V3.2 pure-Swift port**, parity-verified at `1e-4`; the **v0.6 agent backend** — continuous batching (2.5-3.2× under concurrency), longest-common-prefix prompt-cache reuse, structured output, speculative decoding, the API-compat pack; the **Track G model wave** and per-model chat-template overrides (v0.6.1-0.6.2); and **v0.7.0 silicon-metrics observability** — the sudoless Activity panel, the phase-fused bottleneck classifier, per-run benchmark attribution, and OCR-model recognition. Per-tag detail in [CHANGELOG.md](CHANGELOG.md).
- **Next release (on `main`) — v0.8, the tiered SSD KV cache** — the cold (SSD) cache is now bounded (Hot/Cold budgets, oldest-first pruning); weight-safe (each entry fingerprinted against model identity, so re-downloaded or swapped weights at the same path can't restore stale KV into wrong output); restart-surviving (a persisted index gives cross-session longest-prefix reuse, not just exact re-hits); and non-blocking (cold writes move off the cache actor onto a serial background writer, so one request's spill no longer stalls another's lookup).
- **In progress** — later SSD-cache stages (a lock-free serialize/write split, off-actor cold reads, and a block-hash index for partial cross-session reuse); DeepSeek real-checkpoint smoke; a true cross-encoder reranker.
- **Later (v0.8+)** — speech I/O (MLX-native STT/TTS); a community benchmarks service; custom Metal kernels for our DeepSeek DSA path if profiling demands it.
- **Reopenable** (feasible since sandbox-off) — Python / SwiftLM subprocess engines ([#12](../../issues/12) / [#13](../../issues/13)), Homebrew tap ([#20](../../issues/20)), signed + notarized DMG ([#19](../../issues/19)).

## Contributing · License

Issues and PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Apache 2.0
([LICENSE](LICENSE)).

## Acknowledgements

[MLX](https://github.com/ml-explore/mlx) + [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-examples) (Apple), [Swama](https://github.com/Trans-N-ai/swama), [SwiftLM](https://github.com/SharpAI/SwiftLM), [oMLX](https://github.com/jundot/omlx), [Hummingbird](https://github.com/hummingbird-project/hummingbird), [Sparkle](https://github.com/sparkle-project/Sparkle), [Pulse](https://github.com/kean/Pulse), [SwiftTUI](https://github.com/rensbreur/SwiftTUI). Full citations: [CITATIONS.bib](CITATIONS.bib).
