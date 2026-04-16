# Feature: Benchmark

## Overview

Two-tier benchmark system:
1. **Local benchmark** — test your own hardware/model combinations
2. **Community leaderboard** — compare with other Mac users via GitHub

Reference: oMLX benchmark tool, extended with community sharing.

## Metrics

```swift
struct BenchmarkResult: Codable, Identifiable {
    let id: UUID

    // Hardware (auto-detected)
    let chip: String           // "Apple M4 Max"
    let memoryGB: Int          // 128
    let macOSVersion: String   // "15.3.1"

    // Software (auto-detected)
    let macMLXVersion: String  // "0.1.0"
    let engineID: EngineID     // .mlxSwift
    let engineVersion: String  // "mlx-swift-lm 3.x"

    // Model
    let modelID: String        // "mlx-community/Qwen3-8B-4bit"
    let modelParams: String    // "8B"
    let modelQuant: String     // "4bit"

    // Results (averages over 3 runs)
    let prefillTPS: Double     // tokens/sec during prompt processing
    let generationTPS: Double  // tokens/sec during generation
    let timeToFirstTokenMS: Double  // latency to first token
    let memoryUsedGB: Double   // peak unified memory during inference
    let modelLoadTimeS: Double // cold start time

    // Test conditions
    let promptTokens: Int      // 512
    let generationTokens: Int  // 200
    let runs: Int              // 3

    let timestamp: Date
    let notes: String          // user-added notes
}
```

## Hardware Auto-Detection

```swift
func detectHardwareInfo() -> HardwareInfo {
    var chip = ""
    var size = 0
    var sizeOfSize = MemoryLayout<Int>.size

    // Chip name
    var chipBuffer = [CChar](repeating: 0, count: 256)
    var bufferSize = chipBuffer.count
    sysctlbyname("machdep.cpu.brand_string", &chipBuffer, &bufferSize, nil, 0)
    chip = String(cString: chipBuffer)
        .replacingOccurrences(of: "Apple ", with: "")

    // Memory
    sysctlbyname("hw.memsize", &size, &sizeOfSize, nil, 0)
    let memoryGB = size / 1_073_741_824

    return HardwareInfo(chip: chip, memoryGB: memoryGB)
}
```

## Benchmark Runner

```swift
actor BenchmarkRunner {
    private let engine: any InferenceEngine

    func run(
        model: LocalModel,
        promptTokens: Int = 512,
        generationTokens: Int = 200,
        runs: Int = 3
    ) async throws -> BenchmarkResult {

        var prefillSamples: [Double] = []
        var generationSamples: [Double] = []
        var ttftSamples: [Double] = []
        var memorySamples: [Double] = []

        // Warm-up run (not counted)
        try await runSingleBenchmark(promptTokens: 64, generationTokens: 20)

        // Measured runs
        for _ in 0..<runs {
            let sample = try await runSingleBenchmark(
                promptTokens: promptTokens,
                generationTokens: generationTokens
            )
            prefillSamples.append(sample.prefillTPS)
            generationSamples.append(sample.generationTPS)
            ttftSamples.append(sample.ttftMS)
            memorySamples.append(sample.memoryGB)
        }

        // Drop highest and lowest, take middle (for 3 runs: just use middle)
        return BenchmarkResult(
            prefillTPS: prefillSamples.sorted()[1],
            generationTPS: generationSamples.sorted()[1],
            timeToFirstTokenMS: ttftSamples.sorted()[1],
            memoryUsedGB: memorySamples.max() ?? 0,
            // ...
        )
    }
}
```

## Benchmark UI

```
┌────────────────────────────────────────────────────┐
│  Benchmark                              [New Run +] │
├────────────────────────────────────────────────────┤
│  Model:   [Qwen3-8B-4bit ▼]                        │
│  Engine:  [MLX Swift     ▼]                        │
│  Prompt:  [512 tokens ▼]   Gen: [200 tokens ▼]     │
│  Runs:    [3 ▼]                                    │
│                                    [Run Benchmark] │
├────────────────────────────────────────────────────┤
│                                                     │
│  Last Result — Qwen3-8B-4bit · M4 Max 128GB        │
│                                                     │
│  Prefill          2,840 tok/s                       │
│  Generation          68 tok/s                       │
│  Time to first token  142 ms                        │
│  Memory used         9.2 GB                         │
│  Model load time      4.1 s                         │
│                                                     │
│  [Share to Community ↗]    [Copy as JSON]           │
├────────────────────────────────────────────────────┤
│  History                                            │
│                                                     │
│  Qwen3-8B-4bit    68 tok/s   M4Max 128GB  Today    │
│  Qwen3-14B-4bit   41 tok/s   M4Max 128GB  Yesterday│
│  Llama-3.2-3B     124 tok/s  M4Max 128GB  3d ago   │
└────────────────────────────────────────────────────┘
```

## Community Leaderboard

### Architecture

No backend server. Pure GitHub-based:

```
User runs benchmark
    ↓
App generates JSON result
    ↓
"Share to Community" opens GitHub issue with pre-filled template
    ↓
GitHub Action parses issue, appends to benchmarks/data.json
    ↓
GitHub Pages renders leaderboard at magicnight.github.io/mac-mlx/benchmarks
```

### Issue Template (auto-filled by app)

```yaml
# .github/ISSUE_TEMPLATE/benchmark_submission.yml
name: Benchmark Submission
labels: ["benchmark"]
body:
  - type: textarea
    id: data
    attributes:
      label: Benchmark Data (auto-generated, do not edit)
      value: |
        ```json
        {paste JSON here}
        ```
```

### GitHub Action Parser

```yaml
# .github/workflows/benchmark-collect.yml
on:
  issues:
    types: [opened, labeled]

jobs:
  collect:
    if: contains(github.event.issue.labels.*.name, 'benchmark')
    runs-on: ubuntu-latest
    steps:
      - name: Extract and append benchmark data
        # Parse JSON from issue body
        # Append to benchmarks/data.json
        # Commit and push
        # Close issue with thank-you comment
```

### Leaderboard Display (GitHub Pages)

Grouped by model, sorted by generation TPS:

```
Community Benchmarks — Qwen3-8B-4bit

Rank  Chip              Memory  Engine       Gen TPS  Prefill TPS  Submitted
  1   M4 Max            128 GB  MLX Swift      68.3    2,840       2026-04-10
  2   M4 Pro             48 GB  MLX Swift      61.2    2,210       2026-04-09
  3   M3 Ultra           96 GB  MLX Swift      58.1    1,980       2026-04-08
  4   M3 Max             64 GB  MLX Swift      52.4    1,740       2026-04-07
  5   M2 Ultra          192 GB  MLX Swift      49.8    1,620       2026-04-06

[Filter by model ▼]  [Filter by chip ▼]  [Submit your result ↗]
```

## Privacy

Benchmark submissions contain:
- Chip model and memory (no serial number, no username)
- macOS version
- App version + engine version
- Model ID and benchmark results

No personal information collected.
User reviews the JSON before submitting.
Submission is a voluntary GitHub issue — user sees exactly what is shared.

## v0.1 Scope

- Local benchmark runner (prefill TPS, generation TPS, TTFT, memory)
- History stored locally in `~/.mac-mlx/benchmarks/`
- "Share to Community" opens pre-filled GitHub issue
- Community leaderboard page (GitHub Pages, manual setup)
- Does NOT include: automated leaderboard pipeline (v0.2)
