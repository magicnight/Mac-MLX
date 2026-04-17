# Changelog

All notable changes to macMLX will be documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

(nothing yet — next release will be v0.3.x)

---

## [0.2.0] - 2026-04-17

v0.2 focused on **download and chat polish**. Subprocess engines
(SwiftLM, Python MLX) deferred to v0.3 pending App Sandbox policy
review.

### Added
- **Download: real per-chunk progress + UI progress bars.** `DownloadProgress` reports `currentFileBytesDownloaded`, `currentFileTotalBytes`, `completedFiles`, `totalFiles`. `HFModelRow` renders a determinate progress bar for the current file plus a file-count counter. (`902b25e`)
- **Download: speed (MB/s) + ETA** on `DownloadProgress` via `SpeedSampler` (EMA over URLSession didWriteData callbacks, NSLock-protected). Row shows `"12.5 MB/s · 2m 13s"`. (#7, `b02d11f`)
- **Download: Cancel button** in `HFModelRow` during in-flight downloads, replacing the Download icon. `ModelLibraryViewModel` tracks `downloadTasks: [String: Task<Void, Never>]` and cleans up partial directories on cancel. (#5, `8276ada`)
- **Download: resumeData persistence across cancel/restart.** Cancel captures `URLError.downloadTaskResumeData` and persists to `~/.mac-mlx/downloads/{encoded-modelID}/{current-file.txt, resume.dat}`. Next download for the same model skips completed files and resumes the interrupted file via `URLSession.download(resumeFrom:)`. (#6, `78f7769`)
- **Download: background URLSession** — transfers survive App Nap and full app quits. New `DownloadSessionRouter` (session-level delegate) + `withCheckedThrowingContinuation` + `withTaskCancellationHandler` bridge delegate callbacks to async/await. Session identifier `com.magicnight.macmlx.downloader`. (#8, `d4a9aae`)
- **Download: configurable HF endpoint for mirrors** (`https://hf-mirror.com` etc. for restricted regions). Settings → Downloads section exposes the endpoint TextField; `HFDownloader.setBaseURL(_:)` hot-swaps for in-flight downloads. (#21, `fc746f0`)
- **Chat: conversation persistence** to `~/.mac-mlx/conversations/{uuid}.json` via `ConversationStore` actor. Auto-save on every message; load-latest on launch. Sidebar UI for multiple conversations deferred. (#9, `2324f0f`)
- **Chat: Parameters Inspector panel** (⌘⌥I) — right-side SwiftUI inspector for temperature, topP, maxTokens, systemPrompt per model. Persists to `~/.mac-mlx/model-params/{model-id}.json` via `ModelParametersStore` + debounced `ParametersViewModel`. (#15, `1e8e5ba`)
- **Chat: Markdown rendering for assistant messages** via `AttributedString(markdown:options: .init(interpretedSyntax: .full, failurePolicy: .returnPartiallyParsedIfPossible))` — supports block-level markdown during streaming. (#10, `c746f2a`)
- **Chat: message editing + regeneration + delete** via right-click context menu on each message. `EditMessageSheet` for user messages; `regenerate(from:)` for assistant messages. Shared private `generate()` helper between send and regenerate. (#11, `301fb3c`)
- **Settings: Install Guide links** on the deferred engines in the engine picker, pointing at SwiftLM and mlx-lm install docs. (#14, `21361b0`)

### Changed
- **Downloader progress is now honest** — replaced the misleading aggregate-bytes bar with per-file bar + file-count counter. The HF manifest doesn't report LFS sizes, so the old aggregate would jump 0 → 100% mid-download. (`8521cb3`)
- **Default model directory is `~/.mac-mlx/models` under real user home**, not the sandbox container. `Settings.default` uses `NSHomeDirectoryForUser(NSUserName())` to bypass the sandbox redirect; App Sandbox's dotfile exemption makes this directory writable without `user-selected.read-write` entitlements or security-scoped bookmarks. (`df45bda`)
- **Chat view model hoisted to `AppState`** — the streaming Task now survives sidebar tab switches instead of being torn down with the view. (#1, `e834239`)
- **Single-instance enforcement** on app launch via `NSRunningApplication.runningApplications(withBundleIdentifier:)` — a second launch activates the existing window and exits. (#2, `e834239`)
- **Onboarding: removed LM Studio / Ollama directory detection** from the model directory step. macMLX is MLX-ecosystem only; surfacing unrelated tools' paths created confusion. (#3, `e834239`)
- **Menu bar popover: Quit button** (⌘Q). (#17, `e834239`)

### Fixed
- Various Swift 6 strict-concurrency warnings in the Chat and Settings surfaces.

---

## [0.1.0] - 2026-04-17

Initial release. Native macOS LLM inference desktop app for Apple
Silicon, built on mlx-swift-lm.

### Added
- **Stage 1 bootstrap**: `MacMLXCore` SPM library (mlx-swift-lm@3.31.3, hummingbird@2.22.0), `macmlx-cli` SPM executable (swift-argument-parser@1.7.1, SwiftTUI revision-pinned), `macMLX` SwiftUI Xcode project (Bundle ID `com.chaosdevops.macMLX`, macOS 14.0+, Swift 6, LSUIElement=YES). GitHub Actions CI green.
- **Stage 2 core types**: `InferenceEngine` actor protocol, `EngineID`, `EngineStatus`, `LocalModel` (with `ModelFormat` heuristic), `HFModel`, `BenchmarkResult` (+ `SystemInfo`), `GenerateRequest` (with `ChatMessage`, `MessageRole`, `GenerationParameters`), `GenerateChunk` (with `FinishReason`, `TokenUsage`), `EngineError`, `InferenceServiceError`, `MemoryProbe`. All public, all `Sendable`, 29 Swift Testing tests passing.
- **Stage 3 core modules**: `MLXSwiftEngine` (wraps mlx-swift-lm 3.31.3 + MLXLMCommon + swift-transformers 1.3.x), `ModelLibraryManager` + `HFDownloader` (Foundation-only, filesystem scan + HF Hub REST + multi-file URLSession download), `Settings` + `SettingsManager` (JSON persistence at `~/.mac-mlx/settings.json` with corrupt-file preservation), `LogManager` (Pulse 5.1.4 wrapper), `HummingbirdServer` (Hummingbird 2.22.0 actor, OpenAI-compatible routes + port retry + SSE streaming). 60 Swift Testing tests passing.
- **Stage 4 SwiftUI surfaces**: `AppState` + `EngineCoordinator` (`@Observable @MainActor`), `macMLXApp` + `MainWindowView` (NavigationSplitView with sidebar status footer), `MenuBarManager` + `MenuBarPopoverView` + `AppDelegate` (NSStatusItem with engine status), 5-step `OnboardingWindow`, `ModelLibraryView` (Local + Hugging Face tabs), streaming `ChatView`, `SettingsView`.
- **Stage 5 CLI + TUI**: `macmlx` binary with 6 subcommands — `serve` (OpenAI-compatible HTTP server, PIDFile coordination, SIGINT/SIGTERM handling), `pull` (HF Hub model download), `run` (single-shot, interactive stdin REPL, and non-TTY stdin loop modes), `list` (local model table + `--json`), `ps` (running serve status + `--json`), `stop` (SIGTERM via PIDFile + poll). Shared infrastructure: `CLIContext`, `PIDFile`, `TTYDetect`. SwiftTUI linked; full TUI dashboards deferred.
- **Stage 6 distribution scaffolding**: `appcast.xml` template + `scripts/build.sh` + `scripts/package-dmg.sh` + `scripts/ExportOptions.plist` + `scripts/update_appcast.py` (Sparkle EdDSA signature injection) + rewritten `.github/workflows/release.yml` (Xcode 16.4, MARKETING_VERSION injection, Sparkle sign + appcast commit + GitHub Release creation).

### Changed
- `CITATION.cff`: license MIT → Apache-2.0; references list expanded from 4 entries to 17, in sync with `CITATIONS.bib`.
- `.gitignore`: fixed `xcuserdata` pattern to recurse via `**/xcuserdata/`; added OMC runtime-state ignores (plans + project-memory remain tracked).
- `.github/workflows/ci.yml`: rewrote to test the actual SPM packages on `macos-15`; deferred Python backend, SwiftLint, signing to later stages.
