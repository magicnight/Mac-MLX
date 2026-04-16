# Changelog

All notable changes to macMLX will be documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- Initial project setup
- Stage 1 bootstrap: `MacMLXCore` SPM library (mlx-swift-lm@3.31.3, hummingbird@2.22.0), `macmlx-cli` SPM executable (swift-argument-parser@1.7.1, SwiftTUI revision-pinned), `macMLX` SwiftUI Xcode project (Bundle ID `com.chaosdevops.macMLX`, macOS 14.0+, Swift 6, LSUIElement=YES). All three targets compile, smoke tests pass, GitHub Actions CI green.
- Stage 2 core types: `InferenceEngine` actor protocol, `EngineID`, `EngineStatus`, `LocalModel` (with `ModelFormat` heuristic), `HFModel`, `BenchmarkResult` (+ `SystemInfo`), `GenerateRequest` (with `ChatMessage`, `MessageRole`, `GenerationParameters`), `GenerateChunk` (with `FinishReason`, `TokenUsage`), `EngineError`, `InferenceServiceError`, `MemoryProbe`. All public, all `Sendable`, 29 Swift Testing tests passing.
- Stage 3 core modules (parallel worktrees, sequentially merged): `MLXSwiftEngine` (wraps mlx-swift-lm 3.31.3 + MLXLMCommon + swift-transformers 1.3.x), `ModelLibraryManager` + `HFDownloader` (Foundation-only, filesystem scan + HF Hub REST + multi-file URLSession download with DownloadError), `Settings` + `SettingsManager` (JSON persistence at `~/.mac-mlx/settings.json` with corrupt-file preservation) + `LogManager` + `LogLevel` + `LogCategory` (Pulse 5.1.4 wrapper with `flush()` for deterministic tests), `HummingbirdServer` (Hummingbird 2.22.0 actor, OpenAI-compatible routes + port retry + SSE streaming). 60 Swift Testing tests passing; CLI binary + Xcode app both build clean.

### Changed
- `CITATION.cff`: license MIT -> Apache-2.0; references list expanded from 4 entries to 17, in sync with `CITATIONS.bib`.

### Changed
- `.gitignore`: fixed `xcuserdata` pattern to recurse via `**/xcuserdata/`; added OMC runtime-state ignores (plans + project-memory remain tracked).
- `.github/workflows/ci.yml`: rewrote to test the actual SPM packages on `macos-15`; deferred Python backend, SwiftLint, signing to later stages.

---

<!-- Versions will be appended here automatically via GitHub Actions -->
<!-- Example entry:

## [0.1.0] - 2026-XX-XX

### Added
- Native SwiftUI GUI with sidebar navigation
- Menu bar app with service status
- mlx-lm backend process management
- HuggingFace model downloader (mlx-community)
- Built-in chat interface with streaming
- OpenAI-compatible REST API on localhost:8000
- Model library management

-->
