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
- Stage 4 SwiftUI surfaces: `AppState` + `EngineCoordinator` (`@Observable @MainActor` foundation), `macMLXApp` + `MainWindowView` (NavigationSplitView root with sidebar status footer), `MenuBarManager` + `MenuBarPopoverView` + `AppDelegate` (NSStatusItem with engine status), 5-step `OnboardingWindow` (Welcome → ModelDirectory → EngineCheck → DownloadModel → Done, memory-aware via `MemoryProbe`), `ModelLibraryView` (Local + Hugging Face tabs with search), streaming `ChatView` (model selector, Cmd+Return send, token counter), `SettingsView` (engine picker, model directory, server config, re-run wizard). `xcodebuild -scheme macMLX build` SUCCEEDED.
- Stage 5 CLI + TUI: `macmlx` binary with 6 subcommands — `serve` (OpenAI-compatible HTTP server, PIDFile coordination, SIGINT/SIGTERM handling), `pull` (HF Hub model download), `run` (single-shot, interactive stdin REPL, and non-TTY stdin loop modes), `list` (local model table + `--json`), `ps` (running serve status + `--json`), `stop` (SIGTERM via PIDFile + poll). Shared infrastructure: `CLIContext` (per-invocation Core actor bootstrap), `PIDFile` (JSON at `~/.mac-mlx/macmlx.pid`), `TTYDetect` (`isatty` wrapper). SwiftTUI linked and used; full TUI dashboards deferred to v0.2 (Swift 6 nonisolated View protocol incompatibility with `@MainActor` state). 16/16 Swift Testing tests passing.
- Stage 6 distribution scaffolding: `appcast.xml` template + `scripts/build.sh` + `scripts/package-dmg.sh` + `scripts/ExportOptions.plist` + `scripts/update_appcast.py` (Sparkle EdDSA signature injection, tested with mock data → valid Sparkle XML output) + rewritten `.github/workflows/release.yml` (Xcode 16.4, MARKETING_VERSION injection, Sparkle sign + appcast commit + GitHub Release creation). User-action items remaining before `v0.1.0` tag: Sparkle SPM dep in Xcode, AppDelegate updater wiring, EdDSA keypair generation, GitHub Secret `SPARKLE_PRIVATE_KEY`, billing restoration. Documented in `.omc/plans/v0.1-stage6-distribution.md` Tasks 1, 6, 7.

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
