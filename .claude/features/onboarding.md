# Feature: Onboarding

## Overview

First-launch experience. Shown once, skippable, resumable.
Design reference: oMLX welcome screen + LM Studio setup wizard.
macOS HIG: use sheets or dedicated window, never full-screen takeover.

## Trigger Conditions

Show onboarding when:
- `~/.mac-mlx/settings.json` does not exist
- OR `settings.onboardingComplete = false`

Skip onboarding when:
- User has existing models detected on first scan
- User clicks "Skip" at any step (save partial state)

## Step Flow

```
Step 1: Welcome
    ↓
Step 2: Model Directory
    ↓
Step 3: Engine Check (auto, no user action needed usually)
    ↓
Step 4: Download First Model (optional, skippable)
    ↓
Step 5: Done
```

## Step 1: Welcome

```
┌────────────────────────────────────────────────┐
│                                                 │
│           [macMLX icon - large]                 │
│                                                 │
│              Welcome to macMLX                  │
│                                                 │
│   Local LLM inference, native on your Mac.     │
│   Powered by Apple MLX · No cloud required.    │
│                                                 │
│                                                 │
│              [Get Started →]                    │
│                                                 │
│           Skip setup and open app              │
└────────────────────────────────────────────────┘
```

## Step 2: Model Directory

Auto-scan known locations first, present findings:

```swift
let scanLocations: [(path: String, source: String)] = [
    ("~/models", "Common"),
    ("~/.mac-mlx/models", "macMLX default"),
    ("~/.lmstudio/models", "LM Studio"),
    ("~/.ollama/models", "Ollama"),
]
```

UI:

```
┌────────────────────────────────────────────────┐
│  Step 1 of 3 ●●○○                              │
│                                                 │
│  Where are your models?                         │
│                                                 │
│  We found these locations on your Mac:          │
│                                                 │
│  ✅ ~/models             2 MLX models found     │
│  ⚠️  ~/.lmstudio/models  3 models (GGUF format) │
│     LM Studio models use GGUF format,           │
│     not compatible with macMLX.                 │
│     [Find MLX versions on HuggingFace ↗]        │
│  ⚠️  ~/.ollama/models    Not MLX format          │
│                                                 │
│  Use:  [~/models ▼]                [Browse...] │
│                                                 │
│  Or create default:  ~/.mac-mlx/models          │
│                      [Create & Use]             │
│                                                 │
│              [← Back]      [Continue →]         │
└────────────────────────────────────────────────┘
```

Key behavior:
- Auto-select best detected directory
- Clearly explain GGUF vs MLX incompatibility
- Provide direct link to mlx-community on HuggingFace
- Never hide or shame the user's existing setup

## Step 3: Engine Check (Auto)

No user input needed in most cases.
Show briefly while app verifies mlx-swift-lm is ready.

```
┌────────────────────────────────────────────────┐
│  Step 2 of 3  ●●●○                             │
│                                                 │
│  Checking inference engine...                   │
│                                                 │
│  ✅ MLX Swift Engine    Ready                   │
│  ○  SwiftLM             Not installed           │
│  ○  Python MLX          Not installed           │
│                                                 │
│  MLX Swift is the recommended engine for       │
│  most models. You can install additional       │
│  engines later in Settings.                    │
│                                                 │
│              [← Back]      [Continue →]         │
└────────────────────────────────────────────────┘
```

If MLX Swift engine fails to initialize (rare):
Show error with specific fix instructions, link to GitHub issues.

## Step 4: Download First Model

Memory-aware recommendation. Read `hw.memsize` and filter:

```swift
struct RecommendedModel {
    let id: String
    let name: String
    let sizeGB: Double
    let params: String
    let minMemoryGB: Double
    let description: String
}

let recommendations: [RecommendedModel] = [
    RecommendedModel(
        id: "mlx-community/Qwen3-1.7B-4bit",
        name: "Qwen3 1.7B",
        sizeGB: 1.1,
        params: "1.7B",
        minMemoryGB: 8,
        description: "Fastest · Great for coding tasks"
    ),
    RecommendedModel(
        id: "mlx-community/Qwen3-8B-4bit",
        name: "Qwen3 8B",
        sizeGB: 4.5,
        params: "8B",
        minMemoryGB: 8,
        description: "Best balance of speed and quality"
    ),
    RecommendedModel(
        id: "mlx-community/Qwen3-14B-4bit",
        name: "Qwen3 14B",
        sizeGB: 8.2,
        params: "14B",
        minMemoryGB: 16,
        description: "Higher quality · Needs 16GB+"
    ),
    RecommendedModel(
        id: "mlx-community/Qwen3-32B-4bit",
        name: "Qwen3 32B",
        sizeGB: 19.0,
        params: "32B",
        minMemoryGB: 32,
        description: "Near-frontier quality · Needs 32GB+"
    ),
]
```

UI:

```
┌────────────────────────────────────────────────┐
│  Step 3 of 3  ●●●●                             │
│                                                 │
│  Download your first model                      │
│  (You can skip and browse manually later)       │
│                                                 │
│  Recommended for your Mac (M3 Pro · 36 GB):    │
│                                                 │
│  ○ Qwen3 1.7B    1.1 GB   Fastest              │
│  ● Qwen3 8B      4.5 GB   Best balance  ★       │
│  ○ Qwen3 14B     8.2 GB   Higher quality        │
│  ○ Qwen3 32B    19.0 GB   Near-frontier         │
│                                                 │
│  Models exceeding your RAM are hidden.          │
│  [Browse all models]                            │
│                                                 │
│         [Skip]     [Download Qwen3 8B →]        │
└────────────────────────────────────────────────┘
```

If user already has MLX models in selected directory:
Skip step 4 entirely, go straight to Done.

## Step 5: Done

```
┌────────────────────────────────────────────────┐
│                                                 │
│              ✅                                  │
│                                                 │
│           You're all set!                       │
│                                                 │
│   macMLX is running in your menu bar.          │
│   The inference server starts automatically    │
│   when you load a model.                       │
│                                                 │
│   Connect external tools:                      │
│   Base URL: http://localhost:8000/v1           │
│   [Copy]                                        │
│                                                 │
│              [Start Using macMLX]               │
│                                                 │
└────────────────────────────────────────────────┘
```

## Implementation Notes

- Onboarding is a `NSWindow` (not a sheet) — standalone, resizable
- State persisted after each step so user can quit and resume
- "Skip" saves current partial state, shows onboarding reminder in Settings
- All steps are accessible from Settings → "Re-run Setup" for returning users
- Window closes and reveals main window + menu bar on completion

## v0.1 Scope

All 5 steps implemented. GGUF detection and explanation included.
Memory-aware model recommendations included.
