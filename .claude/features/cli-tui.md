# Feature: CLI + TUI

## Overview

`macmlx` — a developer-first CLI with optional TUI interactive mode.
Same model as Ollama: CLI for scripting + TUI for interactive use.
Shares MacMLXCore with the GUI app — identical inference quality.

## Installation

```bash
# Via Homebrew (shipping post-#20)
brew tap magicnight/mac-mlx
brew install macmlx

# Bundled with macMLX.app DMG
# /Applications/macMLX.app/Contents/MacOS/macmlx
# Symlinked to /usr/local/bin/macmlx during onboarding (optional)
```

Pipeline lives in `.claude/distribution.md` (Homebrew Tap section):
release CI builds an arm64 tarball, renders `Formula/macmlx.rb`,
and pushes it to `magicnight/homebrew-mac-mlx` when
`HOMEBREW_TAP_TOKEN` is configured.

## Command Design

```
macmlx <command> [options]

Commands:
  serve     Start the inference server
  pull      Download a model from HuggingFace
  run       Run a model (interactive chat TUI or single prompt)
  list      List downloaded models
  search    Search mlx-community on HuggingFace (v0.3.6)
  ps        Show running server status
  bench     Run benchmark on a model
  stop      Stop the running server
  logs      Open log viewer (launches Pulse or prints to stdout)

Options (global):
  --json               Output as JSON (for scripting)
  --quiet              Suppress non-essential output
  --log-level <level>  debug | info | warning | error | critical | off
                       (v0.3.6 — default: warning; writes to LoggerStore)
  --log-stderr         Tee log output to stderr as plain text (v0.3.6)
  --help               Show help
```

### `macmlx search` (v0.3.6, planned)

Thin wrapper over `HFDownloader.search(query:limit:)`. Queries the
`mlx-community` author on Hugging Face Hub (other authors intentionally
not exposed — matches GUI Hugging Face tab behaviour).

```
macmlx search <query> [--limit N] [--sort <key>] [--json]

Options:
  --limit N             Max results to return (default: 10)
  --sort <key>          Sort by: downloads (default) | likes | recent
  --json                JSON output for scripting

Examples:
  macmlx search qwen3
  macmlx search llama-3 --limit 20 --sort likes
  macmlx search smol --json | jq '.[] | .id'
```

Plain-text output: NAME, DOWNLOADS, LIKES columns.
`--json` output: `[HFModel]` array serialised with camelCase keys.

### `--log-level` + `--log-stderr` (v0.3.6, planned)

All subcommands will accept these global flags. `--log-level` filters
what the `LogManager` (Pulse-backed) records — unchanged from the
GUI's behaviour, just exposed at the CLI boundary. `--log-stderr`
additionally mirrors entries as plain coloured lines on stderr for
terminal debugging without opening the GUI Logs tab.

```
$ macmlx serve --log-level debug --log-stderr
[DEBUG  engine    ] Loaded model: Qwen3-8B-4bit
[INFO   http      ] Serving on http://127.0.0.1:8000
[DEBUG  http      ] chat_completion model=Qwen3-8B-4bit tokens=512
```

## Command Details

### `macmlx serve`

```bash
macmlx serve
macmlx serve --model Qwen3-8B-4bit
macmlx serve --port 8080 --engine mlx-swift

Options:
  --model <id>      Auto-load this model on start
  --port <n>        Port (default: 8000)
  --engine <id>     Engine: mlx-swift | swift-lm | python (default: mlx-swift)
  --no-tui          Print logs to stdout instead of TUI
```

TUI mode (default when terminal is interactive):
```
┌─ macMLX Server ──────────────────────────────────────┐
│                                                       │
│  Status   ● Running                                   │
│  Engine   mlx-swift-lm 3.x                           │
│  Model    Qwen3-8B-4bit                               │
│  Port     8000                                        │
│  Memory   8.2 / 36 GB                                 │
│                                                       │
│  Requests today: 42    Tokens generated: 18,420       │
│                                                       │
│  ─────────────────────────────────────────────────   │
│  [10:23:45] Model loaded in 3.2s                      │
│  [10:24:01] POST /v1/chat/completions  142ms  68t/s   │
│  [10:24:18] POST /v1/chat/completions   98ms  71t/s   │
│                                                       │
│  q: quit   r: reload model   ?: help                  │
└───────────────────────────────────────────────────────┘
```

### `macmlx pull`

```bash
macmlx pull Qwen3-8B-4bit
macmlx pull mlx-community/Qwen3-8B-4bit
macmlx pull Qwen3-8B-4bit --dir ~/my-models

Options:
  --dir <path>    Download to this directory (default: settings.modelDirectory)
```

TUI progress:
```
Pulling mlx-community/Qwen3-8B-4bit

  config.json              ████████████████  100%   2 KB
  tokenizer.json           ████████████████  100%  12 KB
  model-00001-of-00002.safetensors
                           ████████░░░░░░░░   52%  2.3 GB / 4.5 GB
  model-00002-of-00002.safetensors
                           ░░░░░░░░░░░░░░░░    0%  pending

  Overall  ████████░░░░░░░░   48%  2.3 GB / 4.7 GB  12 MB/s  ETA 3m 20s

  ^C to cancel (partial download resumable)
```

### `macmlx run`

```bash
# Interactive TUI chat
macmlx run Qwen3-8B-4bit

# Single prompt, print response and exit
macmlx run Qwen3-8B-4bit "Explain quantum entanglement"

# Pipe input
echo "Write a haiku" | macmlx run Qwen3-8B-4bit

# With options
macmlx run Qwen3-8B-4bit --temperature 0.5 --max-tokens 500 "Hello"

Options:
  --temperature <f>   Sampling temperature (default: 0.7)
  --max-tokens <n>    Max generation tokens (default: 2048)
  --system <text>     System prompt
  --no-stream         Wait for full response before printing
  --json              Output as JSON (for scripting)
```

Interactive TUI mode:
```
┌─ macMLX Chat — Qwen3-8B-4bit ───────────────────────┐
│                                                       │
│  System: You are a helpful assistant.                 │
│  ─────────────────────────────────────────────────── │
│                                                       │
│  You: What is the unified memory architecture?        │
│                                                       │
│  Assistant: Apple's Unified Memory Architecture       │
│  (UMA) allows the CPU, GPU, and Neural Engine to      │
│  share a single pool of high-bandwidth memory...      │
│  ▌                                                    │
│                                                       │
│  ─────────────────────────────────────────────────── │
│  > Type your message...                               │
│                                                       │
│  68 tok/s  •  142ms TTFT  •  Esc: exit  •  ↑: history│
└───────────────────────────────────────────────────────┘
```

### `macmlx list`

```bash
macmlx list
macmlx list --json

Output:
NAME                        SIZE    QUANT  MODIFIED
Qwen3-8B-4bit               4.5 GB  4bit   2 days ago
Llama-3.2-3B-Instruct-4bit  2.1 GB  4bit   1 week ago
Qwen3-14B-4bit              8.2 GB  4bit   1 week ago
```

### `macmlx ps`

```bash
macmlx ps

Output:
STATUS   ENGINE         MODEL              PORT   MEMORY
running  mlx-swift-lm   Qwen3-8B-4bit      8000   8.2 GB
```

### `macmlx bench`

```bash
macmlx bench Qwen3-8B-4bit
macmlx bench Qwen3-8B-4bit --runs 5 --prompt-tokens 1024

TUI progress, then results table.
--json outputs submittable benchmark JSON.
```

### `macmlx stop`

```bash
macmlx stop
# Gracefully stops the running server
```

### `macmlx logs`

```bash
macmlx logs          # Print last 50 log lines to stdout
macmlx logs --follow # tail -f style
macmlx logs --level error  # filter by level
```

## TUI Implementation

Uses SwiftTUI (rensbreur/SwiftTUI) via SPM.
TUI is shown when stdout is a TTY (`isatty(STDOUT_FILENO)`).
Non-interactive mode (pipes, scripts) always prints plain text.

```swift
// macmlx-cli/Sources/macmlx/Commands/RunCommand.swift
import ArgumentParser
import SwiftTUI
import MacMLXCore

struct RunCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "run",
        abstract: "Run a model interactively or with a single prompt"
    )

    @Argument var modelName: String
    @Argument var prompt: String?
    @Flag var json = false

    func run() async throws {
        let core = try await MacMLXCore.shared
        let model = try core.library.find(modelName)
        try await core.engine.load(model)

        if let prompt {
            // Single-shot mode
            try await runSinglePrompt(prompt, core: core)
        } else if isatty(STDOUT_FILENO) != 0 {
            // Interactive TUI
            Application(rootView: ChatTUIView(core: core)).start()
        } else {
            // Pipe mode — REPL stdin
            try await runStdinLoop(core: core)
        }
    }
}
```

## SwiftTUI Dependency

```swift
// macmlx-cli/Package.swift
dependencies: [
    .package(path: "../MacMLXCore"),
    .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
    .package(url: "https://github.com/rensbreur/SwiftTUI", branch: "main"),
],
targets: [
    .executableTarget(
        name: "macmlx",
        dependencies: [
            "MacMLXCore",
            .product(name: "ArgumentParser", package: "swift-argument-parser"),
            .product(name: "SwiftTUI", package: "SwiftTUI"),
        ]
    )
]
```

Note: SwiftTUI is pinned to `main` branch (no stable release tag).
Pin to a specific commit hash for reproducible builds:
```swift
.package(url: "https://github.com/rensbreur/SwiftTUI",
         revision: "COMMIT_HASH_HERE")
```

## CLI Design Principles

- Commands are **composable**: `macmlx pull X && macmlx run X "hello"`
- JSON output (`--json`) for all commands enables scripting
- TUI is **opt-in by environment** (TTY detection), not a flag
- Error messages go to stderr, output to stdout
- Exit codes: 0 success, 1 user error, 2 engine error, 3 network error

## v0.1 Scope

- `serve`, `pull`, `run`, `list`, `ps`, `stop` commands
- TUI for `serve` (status dashboard) and `run` (chat)
- TUI for `pull` (download progress)
- `--json` flag for `list` and `ps`
- Homebrew formula: v0.2

Not in v0.1:
- `bench` command (v0.2)
- `logs` command (v0.2)
- Shell completions (v0.2)

## Historical: SwiftTUI — considered, abandoned in v0.3.5

Original v0.1 plan called for
[rensbreur/SwiftTUI](https://github.com/rensbreur/SwiftTUI) — a
third-party library that ports the SwiftUI DSL (`VStack`, `Text`,
etc.) to terminal rendering — for the `serve` / `pull` / `run`
dashboards. It shipped v0.1 through v0.3.4 as a **linked-but-unused
stub**: three `_*View` types existed only to keep the product
referenced so the build wouldn't drop it.

### Why it didn't work for us

- **Swift 6 incompatibility.** SwiftTUI's `View` protocol is declared
  `nonisolated`. Under Swift 6 strict concurrency that clashes
  immediately with our `@MainActor` state classes — the compiler
  outright rejects the combination.
- **Upstream unmaintained.** The repo has been stagnant for over a
  year as of 2026-04. No Swift 6 adaptation work in sight.
- **No viable drop-in replacement.** Swift's TUI ecosystem is thin;
  nothing else in SPM land offers the SwiftUI-DSL aesthetic. The
  available C libraries (ncurses, notcurses) would require writing a
  bridging shim of comparable complexity to just doing ANSI ourselves.

### What replaced it (v0.3.5)

- New `macmlx-cli/Sources/macmlx/Shared/CLITerm.swift` — a ~100-line
  ANSI helper (colours, bold/dim, unicode block progress bars,
  box-drawing header/footer). TTY detection so piped output stays
  clean.
- All three dashboards (`PullDashboard`, `ServeDashboard`, `ChatTUI`)
  reimplemented against `CLITerm`. Progress bar gets sub-cell
  precision (U+258x), serve status is boxed with colour-coded
  key/value rows, chat REPL has a tidy header.
- SwiftTUI package reference removed from `macmlx-cli/Package.swift`.
- GitHub issue #18 closed — the underlying goal (live CLI
  dashboards) is met without SwiftTUI as the vehicle.

### Reintroduction criteria

If SwiftTUI (a) resumes active development AND (b) ships Swift 6
strict-concurrency compatibility, it may be reevaluated for richer
full-screen dashboards (think htop/btop aesthetic). Until then,
`CLITerm` is the shape.
