# macMLX

> Native macOS LLM inference, powered by Apple MLX.

macMLX brings local LLM inference to Apple Silicon with a first-class native macOS experience. No cloud, no telemetry, no Electron — just your Mac running models at full speed.

**macMLX is for everyone**: a polished GUI for newcomers, and a powerful CLI + TUI for developers.

---

## Why macMLX?

| | macMLX | LM Studio | Ollama | oMLX |
|--|--------|-----------|--------|------|
| Native macOS GUI | ✅ | ❌ Electron | ❌ | ❌ Web UI |
| MLX-native inference | ✅ | ❌ GGUF | ❌ GGUF | ✅ |
| CLI + TUI | ✅ | ❌ | ✅ | ✅ |
| 100B+ MoE models | ✅ SwiftLM | ❌ | ❌ | ❌ |
| Zero Python required | ✅ | ✅ | ✅ | ❌ |

## Requirements

- macOS 14.0 (Sonoma) or later
- Apple Silicon (M1 / M2 / M3 / M4)
- No Python required for default usage

## Installation

### GUI App

Download `macMLX-vX.X.X.dmg` from [Releases](../../releases).

**First launch**: Right-click → Open → Open (Gatekeeper bypass, one-time).

### CLI

```bash
# Bundled with the app — symlink during onboarding
macmlx --version

# Homebrew (coming in v0.2)
brew install magicnight/mac-mlx/macmlx
```

## Quickstart

### GUI
1. Launch macMLX — the setup wizard guides you through everything
2. Download a model from the built-in HuggingFace browser
3. Load it and start chatting

### CLI
```bash
macmlx pull Qwen3-8B-4bit          # download model
macmlx run Qwen3-8B-4bit           # interactive chat TUI
macmlx serve                        # start OpenAI-compatible API
macmlx run Qwen3-8B-4bit "Hello"   # single prompt
```

## Connecting External Tools

macMLX exposes an OpenAI-compatible API at `http://localhost:8000/v1`.

```bash
# Claude Code
claude --model macmlx/Qwen3-8B-4bit

# curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-8B-4bit","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

Any OpenAI-compatible client works: Cursor, Continue, Open WebUI, etc.

## Inference Engines

| Engine | Use case | Setup |
|--------|---------|-------|
| **MLX Swift** (default) | Most models, best integration | None — built-in |
| **SwiftLM** | 100B+ MoE models, SSD streaming | Install separately |
| **Python mlx-lm** | Maximum model compatibility | uv + mlx-lm |

## Architecture

```
macMLX.app / macmlx CLI
        │
MacMLXCore (Swift SPM package)
        │
   Engine Protocol
   ├── mlx-swift-lm (default, in-process)
   ├── SwiftLM (optional, 100B+ MoE)
   └── Python mlx-lm (optional, max compat)
        │
Hummingbird HTTP → localhost:8000/v1
        │
Apple Silicon (Metal / ANE / NVMe)
```

## Building from Source

```bash
git clone https://github.com/magicnight/mac-mlx
cd mac-mlx
brew bundle                    # dev tools
open macMLX/macMLX.xcodeproj  # GUI app
cd macmlx-cli && swift build  # CLI
cd Backend && uv sync          # Python engine (optional)
```

## Roadmap

**v0.1**
- [x] Native SwiftUI GUI + menu bar
- [x] mlx-swift-lm default engine
- [x] CLI + TUI (serve, pull, run, list, ps)
- [x] HuggingFace model downloader
- [x] OpenAI-compatible API (Hummingbird)
- [x] Memory-aware onboarding
- [x] Sparkle auto-update

**v0.2**
- [ ] Homebrew tap
- [ ] SwiftLM engine integration (100B+ MoE)
- [ ] Python mlx-lm engine
- [ ] VLM support
- [ ] Benchmark + community leaderboard
- [ ] HuggingFace mirror endpoint
- [ ] Conversation history persistence

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions welcome.

## License

Apache 2.0 — see [LICENSE](LICENSE)

## Acknowledgements

- [MLX](https://github.com/ml-explore/mlx) and [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) by Apple
- [Swama](https://github.com/Trans-N-ai/swama) — Swift inference architecture inspiration
- [SwiftLM](https://github.com/SharpAI/SwiftLM) — 100B+ MoE engine
- [oMLX](https://github.com/jundot/omlx) — feature depth reference
- [Hummingbird](https://github.com/hummingbird-project/hummingbird) — Swift HTTP server
- [Sparkle](https://github.com/sparkle-project/Sparkle) — auto-update framework
- [Pulse](https://github.com/kean/Pulse) — logging framework
- [SwiftTUI](https://github.com/rensbreur/SwiftTUI) — TUI framework

Full BibTeX citations: [CITATIONS.bib](CITATIONS.bib)
