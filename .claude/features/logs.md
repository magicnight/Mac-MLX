# Feature: Logs

## Overview

Beautiful, filterable log viewer using Pulse (Swift) + Rich (Python engine).
Reference: Proxyman log viewer aesthetic, but for LLM inference events.

## Architecture

```
Engine output (stdout/stderr)
    ↓
LogManager (Swift Actor)
    ↓ stores to
PulseLogStore (~/.mac-mlx/logs/pulse.sqlite)
    ↓ displayed by
PulseUI.ConsoleView (embedded in Logs tab)
```

## Pulse Integration

```swift
// Package.swift dependency
.package(url: "https://github.com/kean/Pulse", from: "5.0.0")

// LogManager.swift
import Pulse
import PulseUI

actor LogManager {
    static let shared = LogManager()
    private let store = LoggerStore.shared

    func log(_ message: String, level: LoggerStore.Level, category: String) {
        store.storeMessage(
            label: category,
            level: level,
            message: message,
            metadata: nil
        )
    }

    func logEngineEvent(_ event: EngineEvent) {
        switch event {
        case .modelLoaded(let model, let duration):
            log("Model loaded: \(model.name) in \(duration)s",
                level: .info, category: "engine")
        case .tokenGenerated(let tps):
            log("Generation: \(String(format: "%.1f", tps)) tok/s",
                level: .debug, category: "inference")
        case .error(let error):
            log(error.localizedDescription,
                level: .error, category: "engine")
        }
    }
}
```

## Logs View (SwiftUI)

```swift
import PulseUI

struct LogsView: View {
    var body: some View {
        ConsoleView()  // Full Pulse console, built-in search/filter
            .navigationTitle("Logs")
    }
}
```

Pulse ConsoleView provides out of the box:
- Full-text search across all logs
- Filter by level (debug / info / warning / error)
- Filter by category (engine / inference / download / http)
- Timeline view
- Network request inspection (for HF downloads)
- Share logs as file
- Pin important messages

## Log Categories

| Category | Content |
|----------|---------|
| `engine` | Model load/unload, engine start/stop |
| `inference` | Token generation speed, request start/end |
| `download` | HF download progress, file verification |
| `http` | Incoming API requests (method, path, duration) |
| `system` | Memory usage, port binding |
| `error` | All errors with full context |

## Rich Integration (Python Engine)

When Python engine is active, Rich formats its stderr output:

```python
# Backend/logging_config.py
from rich.logging import RichHandler
from rich.console import Console
import logging, sys

console = Console(
    stderr=True,
    force_terminal=True,
    width=120
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            console=console,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            markup=True,
            log_time_format="[%H:%M:%S]"
        )
    ]
)
```

Swift captures this stderr output and feeds it to Pulse as raw text messages in the `python-engine` category, preserving ANSI color codes where possible.

## Log Viewer UI Location

Accessible from:
1. Main window sidebar → "Logs" item
2. Menu bar popover → "View Logs..." button
3. Settings → "Open Log Viewer"

## Menu Bar Quick Access

```
┌──────────────────────────────┐
│  macMLX            ● Running │
│  ─────────────────────────── │
│  Qwen3-8B-4bit    68 tok/s  │
│  Memory: 8.2 / 36 GB        │
│  ─────────────────────────── │
│  ▶ Stop Service              │
│  📋 View Logs                │  ← opens Logs tab
│  ⚙ Settings                 │
│  ─────────────────────────── │
│  Quit macMLX                 │
└──────────────────────────────┘
```

## Log Persistence

- Pulse SQLite store: `~/.mac-mlx/logs/pulse.sqlite`
- Max size: 50 MB (Pulse handles rotation automatically)
- Retention: last 7 days (configurable in Settings)
- Export: Pulse built-in "Share" exports as `.pulse` file (openable in Pulse Pro)

## v0.1 Scope

- Pulse ConsoleView embedded in Logs tab
- LogManager capturing all engine events
- Python engine Rich formatted logs forwarded to Pulse
- Menu bar "View Logs" shortcut
- Does NOT include: custom log visualization, charts (v0.2)
