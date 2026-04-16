# Feature: Model Downloader

## Overview

Browse and download MLX-format models from HuggingFace directly
within the app. Focus on the `mlx-community` organization.

## v0.1 Scope

### HuggingFace API Integration

Use the public HuggingFace API (no API key required for public models):

```
Search models:
GET https://huggingface.co/api/models
  ?author=mlx-community
  &search={query}
  &limit=20
  &sort=downloads
  &direction=-1

Model files:
GET https://huggingface.co/api/models/{model_id}
```

### Model List View

```
┌─────────────────────────────────────────────────┐
│  🔍 Search mlx-community...                      │
├─────────────────────────────────────────────────┤
│  Qwen3-8B-4bit                    ↓ 2.1k        │
│  mlx-community · 4.2 GB · 4bit                  │
│  [  Download  ]                                  │
├─────────────────────────────────────────────────┤
│  Llama-3.2-3B-Instruct-4bit       ↓ 1.8k        │
│  mlx-community · 2.1 GB · 4bit                  │
│  [  Download  ]                                  │
└─────────────────────────────────────────────────┘
```

Show: model name, download count, total size, quantization level.

### Download Management

```swift
actor DownloadManager {
    func startDownload(model: HFModel, to directory: URL) async throws
    func cancelDownload(modelId: String)
    func pauseDownload(modelId: String)   // v0.2
    var activeDownloads: [String: DownloadProgress] { get }
}

struct DownloadProgress {
    let modelId: String
    let bytesDownloaded: Int64
    let totalBytes: Int64
    let speed: Double        // bytes/sec
    var fractionCompleted: Double { Double(bytesDownloaded) / Double(totalBytes) }
}
```

### Download Process

HuggingFace models are multi-file (safetensors + config files).
Download each file individually:

1. Fetch file list from HF API
2. Create model directory in `modelDirectory/{model_name}/`
3. Download each file with `URLSession` download task
4. Show aggregate progress across all files
5. On completion, verify files exist and add to local library

Use `URLSessionDownloadTask` for proper background download support.

### Resume / Retry

- If download fails midway, allow resume from partial files
- Check existing file sizes against expected sizes
- Only re-download incomplete files

### Local Model Detection

When user sets model directory, scan for existing MLX models:
- Look for directories containing `config.json` + `*.safetensors`
- Cross-reference with HF metadata if available
- Mark already-downloaded models in the search results

## File Structure After Download

```
~/models/
├── Qwen3-8B-4bit/
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── model.safetensors          # small models: single file
│   └── model-00001-of-00004.safetensors  # large models: sharded
├── Llama-3.2-3B-Instruct-4bit/
│   └── ...
```

## Out of Scope (v0.1)

- Model card / README preview
- Automatic quantization of non-MLX models
- Upload to HuggingFace
- Custom HF endpoint (HF mirror for China — v0.2)
- Download queue ordering / prioritization
