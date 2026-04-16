# Architecture

## System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Apple Silicon Mac                         │
│                                                                  │
│  ┌─────────────────────────┐   ┌────────────────────────────┐   │
│  │      macMLX.app          │   │         macmlx             │   │
│  │      (SwiftUI GUI)       │   │       (CLI + TUI)          │   │
│  │                          │   │   swift-argument-parser    │   │
│  │  Menu Bar   Main Window  │   │        SwiftTUI            │   │
│  └──────────┬───────────────┘   └─────────────┬──────────────┘  │
│             │                                  │                  │
│             └──────────────┬───────────────────┘                 │
│                            │ imports                             │
│  ┌─────────────────────────▼──────────────────────────────────┐  │
│  │                    MacMLXCore (SPM)                         │  │
│  │                                                             │  │
│  │  EngineCoordinator      ModelLibraryManager                 │  │
│  │  HFDownloader           SettingsManager                     │  │
│  │  BenchmarkRunner        LogManager                          │  │
│  │  HummingbirdServer      (OpenAI-compatible API)             │  │
│  │                                                             │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │              Engine Layer                             │  │  │
│  │  │                                                       │  │  │
│  │  │  InferenceEngine (protocol)                           │  │  │
│  │  │  ├── MLXSwiftEngine    (default, in-process)          │  │  │
│  │  │  ├── SwiftLMEngine     (optional, subprocess, 100B+)  │  │  │
│  │  │  └── PythonMLXEngine   (optional, subprocess, Python) │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                            │                                      │
│              Metal + ANE + NVMe (Apple Silicon)                   │
└──────────────────────────────────────────────────────────────────┘

External tools (Claude Code, Cursor, etc.)
  └── http://localhost:8000/v1  (OpenAI-compatible)
```

## Package Structure

```
mac-mlx/                          ← GitHub repo root
│
├── MacMLXCore/                   ← SPM package, shared by both products
│   ├── Package.swift
│   └── Sources/
│       └── MacMLXCore/
│           ├── Engine/
│           │   ├── InferenceEngine.swift      # protocol
│           │   ├── EngineCoordinator.swift
│           │   ├── MLXSwiftEngine.swift
│           │   ├── SwiftLMEngine.swift
│           │   └── PythonMLXEngine.swift
│           ├── Models/
│           │   ├── LocalModel.swift
│           │   ├── HFModel.swift
│           │   └── BenchmarkResult.swift
│           ├── Managers/
│           │   ├── ModelLibraryManager.swift
│           │   ├── HFDownloader.swift
│           │   ├── SettingsManager.swift
│           │   ├── BenchmarkRunner.swift
│           │   └── LogManager.swift
│           └── Server/
│               └── HummingbirdServer.swift
│
├── macMLX/                       ← SwiftUI App target (Xcode project)
│   ├── macMLX.xcodeproj
│   ├── App/
│   │   ├── macMLXApp.swift
│   │   └── AppDelegate.swift     # Sparkle + LSUIElement
│   ├── Views/
│   │   ├── Onboarding/
│   │   ├── MenuBar/
│   │   ├── ModelLibrary/
│   │   ├── Chat/
│   │   ├── Parameters/
│   │   ├── Logs/
│   │   ├── Benchmark/
│   │   └── Settings/
│   └── Resources/
│       └── Info.plist            # LSUIElement=YES, min macOS 14.0
│
├── macmlx-cli/                   ← CLI + TUI executable (SPM)
│   ├── Package.swift
│   └── Sources/
│       └── macmlx/
│           ├── main.swift        # swift-argument-parser entry
│           ├── Commands/
│           │   ├── ServeCommand.swift
│           │   ├── PullCommand.swift
│           │   ├── RunCommand.swift
│           │   ├── ListCommand.swift
│           │   ├── PSCommand.swift
│           │   └── BenchCommand.swift
│           └── TUI/
│               ├── ChatTUI.swift      # SwiftTUI chat view
│               ├── DownloadTUI.swift  # SwiftTUI download progress
│               └── BenchTUI.swift    # SwiftTUI benchmark view
│
├── Backend/                      ← Python optional engine
│   ├── pyproject.toml
│   ├── .python-version           # 3.13
│   ├── uv.lock
│   └── server.py
│
├── Tests/
│   ├── MacMLXCoreTests/
│   └── macmlx-cliTests/
│
├── scripts/
│   ├── build.sh
│   ├── package-dmg.sh
│   └── update_appcast.py
│
├── appcast.xml                   ← Sparkle update feed
├── CLAUDE.md
├── .claude/
└── .github/
```

## InferenceEngine Protocol

```swift
// MacMLXCore/Engine/InferenceEngine.swift

public protocol InferenceEngine: Actor {
    var engineID: EngineID { get }
    var status: EngineStatus { get }
    var loadedModel: LocalModel? { get }
    var version: String { get }

    func load(_ model: LocalModel) async throws
    func unload() async throws
    func generate(_ request: GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error>
    func embed(_ texts: [String]) async throws -> [[Float]]
    func healthCheck() async -> Bool
}

public enum EngineID: String, Codable {
    case mlxSwift  = "mlx-swift-lm"   // default, in-process
    case swiftLM   = "swift-lm"       // 100B+ MoE, subprocess
    case pythonMLX = "python-mlx"     // max compat, subprocess
}

public enum EngineStatus: Equatable {
    case idle
    case loading(model: String)
    case ready(model: String)
    case generating
    case error(String)
}
```

## Settings Schema

```json
// ~/.mac-mlx/settings.json
{
  "modelDirectory": "~/models",
  "preferredEngine": "mlx-swift-lm",
  "serverPort": 8000,
  "autoStartServer": false,
  "lastLoadedModel": "mlx-community/Qwen3-8B-4bit",
  "onboardingComplete": true,
  "pythonPath": null,
  "swiftLMPath": null,
  "sparkleUpdateChannel": "release",
  "logRetentionDays": 7
}
```

## File Layout (User Data)

```
~/.mac-mlx/
├── settings.json
├── model-cache.json          # HF search results cache (TTL 24h)
├── model-params/             # per-model parameter overrides
│   └── {model-id}.json
├── benchmarks/               # local benchmark history
│   └── {uuid}.json
├── engines/
│   └── mlx-lm-0.31.x/       # Python engine venv (if installed)
├── logs/
│   └── pulse.sqlite          # Pulse log store
└── models/                   # default model directory (optional)
```

## Concurrency Model

- `MacMLXCore` managers: `@Observable` final classes, `MainActor` where needed
- Engines: `actor` types (thread-safe by construction)
- Swift 6 strict concurrency: zero data races
- `async/await` throughout — no Combine, no callbacks

## Info.plist Required Entries

```xml
<key>LSUIElement</key>
<true/>
<!-- Hides Dock icon — menu bar app only -->

<key>LSMinimumSystemVersion</key>
<string>14.0</string>

<key>SUFeedURL</key>
<string>https://raw.githubusercontent.com/magicnight/mac-mlx/main/appcast.xml</string>
<!-- Sparkle update feed -->
```
