# Feature: Inference Engines

## Overview

macMLX supports multiple inference engines behind a unified Swift protocol.
The user can select engines from Settings. Engine switching requires model reload.

## Engine Registry

### Engine 1: MLXSwiftEngine (Default)

- Package: `mlx-swift-lm` (Apple official, SPM)
- Mode: In-process (same memory space as GUI)
- Best for: Most users, models up to ~70B on 64GB+ Mac
- Dependencies: None (bundled via SPM)
- Install: Automatic (SPM dependency)

```swift
import MLXLLM
import MLXLMHuggingFace
import MLXLMTokenizers

actor MLXSwiftEngine: InferenceEngine {
    private var model: LLMModel?
    private var session: ChatSession?

    func load(_ localModel: LocalModel) async throws {
        let container = try await loadModelContainer(
            from: localModel.directory,
        )
        self.model = container
        self.session = ChatSession(container)
    }

    func generate(_ request: GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error> {
        AsyncThrowingStream { continuation in
            Task {
                guard let session else {
                    continuation.finish(throwing: EngineError.modelNotLoaded)
                    return
                }
                let response = try await session.respond(to: request.lastUserMessage)
                continuation.yield(GenerateChunk(text: response))
                continuation.finish()
            }
        }
    }
}
```

### Engine 2: SwiftLMEngine (Optional, Advanced)

- Binary: `SwiftLM` (downloaded separately by user)
- Mode: Subprocess, OpenAI API on random localhost port
- Best for: MoE models 30B+, users with SSD streaming needs
- Install: User downloads from SwiftLM GitHub releases

Detection:
```swift
func isSwiftLMAvailable() -> Bool {
    let paths = [
        "/usr/local/bin/SwiftLM",
        "/opt/homebrew/bin/SwiftLM",
        (NSHomeDirectory() as NSString).appendingPathComponent(".mac-mlx/engines/SwiftLM")
    ]
    return paths.contains { FileManager.default.fileExists(atPath: $0) }
}
```

### Engine 3: PythonMLXEngine (Optional, Power Users)

- Runtime: Python + mlx-lm via uv
- Mode: Subprocess, OpenAI API on localhost
- Best for: Models not yet supported by mlx-swift-lm
- Install: User installs uv + mlx-lm manually
- See: `python-conventions.md` for details

## Engine Selection UI

Settings → Engine panel:

```
┌─────────────────────────────────────────────────────┐
│  Inference Engine                                    │
│                                                      │
│  ● MLX Swift (Recommended)              Built-in    │
│    Apple official · In-process · No setup needed    │
│                                                      │
│  ○ SwiftLM                              Not found   │
│    100B+ MoE · SSD streaming · [Install Guide ↗]   │
│                                                      │
│  ○ Python MLX                           Not found   │
│    Maximum compatibility · Requires uv             │
│    [Setup Guide ↗]                                  │
└─────────────────────────────────────────────────────┘
```

## Port Conflict Detection

When subprocess engines start an HTTP server:

```swift
func findAvailablePort(starting: Int = 8000) async throws -> Int {
    for port in starting...(starting + 20) {
        if await isPortAvailable(port) { return port }
    }
    throw EngineError.noAvailablePort
}

func isPortAvailable(_ port: Int) async -> Bool {
    // Attempt TCP bind, immediately release
    var addr = sockaddr_in()
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = in_port_t(port).bigEndian
    addr.sin_addr.s_addr = INADDR_ANY
    let sock = socket(AF_INET, SOCK_STREAM, 0)
    defer { close(sock) }
    var reuse: Int32 = 1
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))
    return withUnsafePointer(to: &addr) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            bind(sock, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) == 0
        }
    }
}
```

**User-facing conflict handling (macOS HIG):**

If preferred port 8000 is taken:
1. Silently try 8001–8010 first
2. If all taken: show non-blocking notification banner (NOT modal alert)
   > "Port 8000 is in use. macMLX is using port 8003."
3. Log the actual port to status bar tooltip
4. Never block app launch with a modal for port conflicts

## Model Compatibility Check

Before loading a model, run preflight checks:

```swift
struct CompatibilityReport {
    let model: LocalModel
    let engine: any InferenceEngine
    var warnings: [CompatibilityWarning] = []
    var canLoad: Bool = true
}

enum CompatibilityWarning {
    case insufficientMemory(required: Double, available: Double)
    case architectureUnknown(String)
    case engineVersionMismatch(modelRequires: String, engineVersion: String)
    case largeMoERecommendSwiftLM(paramCount: Int)
}
```

**Compatibility UI — macOS HIG compliant:**

Show as **inline warning in model row**, NOT as blocking modal:

```
┌─────────────────────────────────────────────────────┐
│  Qwen3-72B-4bit                          [Load]      │
│  72B · 4bit · 42 GB                                  │
│  ⚠ Requires ~42 GB · You have 48 GB · May be slow   │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Qwen3.5-122B-MoE-4bit                   [Load]      │
│  122B MoE · 4bit · 72 GB                             │
│  ✦ Recommended: SwiftLM engine for MoE models        │
│    [Switch to SwiftLM]  [Load Anyway]                │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Llama-3.1-405B-4bit                  [Load]         │
│  405B · 4bit · 238 GB                                │
│  ✕ Insufficient memory (238 GB required, 48 GB      │
│    available). This model cannot run on this Mac.    │
└─────────────────────────────────────────────────────┘
```

Rules:
- Green / no warning: model fits with >20% headroom
- Yellow warning: model fits but <20% headroom, or MoE better engine available
- Red blocking: model provably won't fit (required > available × 1.1)
- Never show modal dialogs for warnings, only for hard blocks

## Memory Detection

```swift
func getAvailableMemoryGB() -> Double {
    var size: Int = 0
    var sizeOfSize = MemoryLayout<Int>.size
    sysctlbyname("hw.memsize", &size, &sizeOfSize, nil, 0)
    return Double(size) / 1_073_741_824
}
```

## v0.1 Scope

- MLXSwiftEngine: full implementation
- SwiftLMEngine: detection + launch only, no advanced config
- PythonMLXEngine: detection only, manual setup by user
- Port conflict: silent auto-retry, notification banner
- Compatibility check: memory check + MoE detection
