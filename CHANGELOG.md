# Changelog

All notable changes to macMLX will be documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

(nothing yet)

---

## [0.3.2] - 2026-04-17

Chat history management — the conversation-store backend that shipped
in v0.2 #9 finally has a UI. Power-user feature for people who keep
multiple ongoing threads and occasionally want to roll back mid-stream.

### Added
- **Conversation sidebar** (⌘⌃S to toggle). Collapsible left pane on
  the Chat tab lists saved conversations newest-first with title +
  relative timestamp. Single-click switches; double-click inline-
  renames; right-click menu offers Rename / Delete (with confirmation).
  Default collapsed — existing users' first impression unchanged.
- **"Rewind to here"** context menu on every chat message. Drops
  everything after the clicked message; keeps that message and every
  earlier one. User decides what to do next (often: edit the last
  kept message and resend).
- **New Chat ⌘N** inside the sidebar creates a fresh conversation
  (flushes the current one to disk first).
- `ChatViewModel` extensions: `reloadConversationList()`,
  `switchTo(_:)`, `createNew()`, `rename(_:to:)`,
  `deleteConversation(_:)`, `truncateAfter(_:)`, `currentConversationID`,
  `conversations` (sorted list), plus private `persistNow()` that
  skips empty-chat writes so the sidebar doesn't spam empty rows.
- `ChatMessageView` grows an optional `onTruncate` closure; every
  existing context-menu entry stays; `Rewind to here` with
  `arrow.uturn.backward` SF Symbol.
- `ConversationSidebar.swift` — standalone view. Inline-rename via
  `TextField` state; delete via `confirmationDialog`; empty-state
  via `ContentUnavailableView`.

### Changed
- Chat layout is now two columns (sidebar + main) with the existing
  Parameters Inspector as an optional third. Main column stays
  flush-left when the sidebar is collapsed — no layout shift for
  users who never open it. `HStack` + animation rather than a nested
  `NavigationSplitView` to avoid double disclosure chevrons under
  macOS.
- `ChatViewModel.clearHistory()` is now a one-line alias for
  `createNew()` (was the same thing minus sidebar refresh).
- `persist()` now fires `reloadConversationList()` after each save
  so `updatedAt` bumps re-sort the sidebar in real time.

### Notes
- No unit tests added for the new view-model methods — the app target
  doesn't have a test bundle set up, and the VM methods are thin
  wrappers over well-tested `ConversationStore` primitives
  (6 tests in `ConversationStoreTests` already cover save / list /
  delete / corrupt-file tolerance / ordering). Adding an app-side
  test bundle is a separate chore-commit candidate.

---

## [0.3.1] - 2026-04-17

Patch release — five UX fixes surfaced during v0.3 bring-up use, plus
a CLI segfault fix that would have hit any `macmlx list` user on a
non-empty model store.

### Fixed
- **`macmlx list` segfault** on any non-empty local model store
  (`%s` + Swift `String` UB in the printf-based table formatter).
  Replaced with Swift-native `padding(toLength:…)` helpers. `list`
  exit code now 0; `list --json` already worked. (`1d68a94`)
- **Chat "No model loaded" banner flicker** during generation.
  `EngineStatus.isLoaded` now returns `true` for both `.ready` and
  `.generating` — a model generating *is* loaded from the UI's
  perspective. Fixes the banner flashing on every send → first-token
  window. Test updated to reflect the correct behaviour.
- **Assistant-message Markdown renderer was collapsing paragraphs.**
  `AttributedString(markdown:)` with `interpretedSyntax: .full` consumed
  block-level markers AND the `\n\n` paragraph separators, and SwiftUI's
  `Text(AttributedString)` flattened the result into a single run.
  Switched to `.inlineOnlyPreservingWhitespace` — paragraph breaks
  preserved, inline bold/italic/code/links still highlighted, block
  markers pass through as literal text (better than losing them).
- **Manually-copied models not appearing in the Models tab** until the
  user toggled the directory in Settings. Models view now auto-rescans
  when `currentSettings.modelDirectory` changes, and the empty-state
  spells out the actual scanned path so users can tell immediately
  whether the app is looking where they expect.

### Changed
- **Max tokens control in Parameters Inspector** now a `TextField`
  with `format: .number` (direct entry, clamped to 128–32768) plus
  a side Stepper for ±128 nudges. Pre-v0.3.1 Stepper-only took ~112
  clicks to go from 128 to 16384.
- **Chat toolbar model selector is finally functional.** Previously a
  `.constant`-bound Picker that only displayed the loaded model. Now
  a Menu that lists local models, checkmarks the loaded one, and
  loads on tap. Disabled mid-generation to prevent mid-stream swaps;
  shows a ProgressView while load is in flight. Refresh action in the
  menu re-scans the model directory on demand.
- **Release workflow hardening** — appcast push now rebases against
  fresh `origin/main` + retries up to 3 times with backoff, and is
  `continue-on-error: true` so a push race cannot block the DMG from
  shipping as a GitHub Release. `Create GitHub Release` step is
  `if: always()`. Caught by the v0.3.0 publish attempt (DMG built +
  signed but never released; recovered by re-tagging on the fix
  commit). (`6d45083`)

---

## [0.3.0] - 2026-04-17

v0.3 shipped the **local benchmark feature**, a sweep of
cross-cutting defects flagged by an independent code review of the
v0.1+v0.2 surface, a second pass on CLI/engine parity with the GUI,
the VLM (#23) implementation plan for v0.4, and a release-pipeline
hardening that fixes the race this very tag tripped over on its
first publish attempt.

### Added
- **Benchmark tab** (⌥⌘ sidebar) — local benchmark runner with config (model / prompt tokens / gen tokens / runs / notes), last-result readout (prefill + generation TPS, TTFT, peak memory, load time), history list with delete + clear, `Share to Community` (pre-fills a GitHub issue via `benchmark_submission.yml`), `Copy as JSON`. (#22, `88545ad` / `e3cf815` / `e155a7a`)
- `MacMLXCore/Util/DataRoot.swift` — single source of truth for `~/.mac-mlx/` paths under App Sandbox (replaced 5 inline copies of the `NSHomeDirectoryForUser` dance).
- `MacMLXCore/Managers/BenchmarkStore.swift` — actor persisting results to `~/.mac-mlx/benchmarks/{uuid}.json`.
- `MacMLXCore/Managers/BenchmarkRunner.swift` — measurement actor (warm-up + N measured runs + median aggregation, peak RSS via Mach `task_info`).
- `MacMLXCore/Util/HardwareInfo.swift` — chip / memory / macOS version via `sysctlbyname`.
- `MemoryProbe` gained `residentMemoryBytes()` + `residentMemoryGB()` (used by benchmark sampler **and** `HummingbirdServer`'s `/v1/status`, which now reports real RSS instead of 0).
- **Simplified Chinese README** (`README.zh-CN.md`) with bilingual switcher header on both files.
- `.github/ISSUE_TEMPLATE/benchmark_submission.yml` — target template for the app's Share-to-Community link.
- `MacMLXCore/Util/JSONCoding.swift` — shared `precisionEncoder()` (`.secondsSince1970`) + `tolerantDecoder()` that accepts both legacy ISO-8601 and new Double-seconds date shapes. Enables sub-second ordering for rapid-save scenarios without breaking v0.2 users' saved conversations.
- `.omc/plans/v0.3-vlm-plan.md` — full research + implementation blueprint for VLM support (#23). Finding: `MLXVLM` already ships in our `mlx-swift-lm` dependency with 16 supported VLM architectures. 7-step build plan targeting v0.4.0.

### Changed
- **SettingsManager no longer writes to the sandbox container** (CRITICAL). Pre-v0.3 `SettingsManager.init()` used `FileManager.default.homeDirectoryForCurrentUser` → `~/Library/Containers/<bundle-id>/Data/…`, so the GUI's `settings.json` lived inside the container while the CLI (and `Settings.default.modelDirectory`) used real `~/.mac-mlx/`. GUI and CLI were quietly disagreeing. Routed through `DataRoot.macMLX`. (`9764628`)
- **CLI honours the user's HF endpoint mirror** (CRITICAL). `macmlx pull` was hitting `huggingface.co` even when the GUI had the user on `https://hf-mirror.com` — #21 only ever wired the GUI side. `CLIContext.bootstrap()` now calls `downloader.setBaseURL(_:)`. (`9764628`)
- **Parameters Inspector overrides auto-load on model load** (CRITICAL). Pre-v0.3 `loadForModel(_:)` only ran from the Inspector's `.onAppear`; users who chatted without opening the Inspector saw persisted per-model temperature/top_p/system-prompt ignored. `EngineCoordinator` gained an `onModelLoaded` callback, `AppState` wires it to `parameters.loadForModel`. (`9764628`)
- **Background URLSession identifier is process-scoped**. Suffixed with `.app` or `.cli` based on `Bundle.main.bundlePath.hasSuffix(".app")` so GUI + CLI don't fight for the same identifier when both run. (`9764628`)
- **`PeakMemorySampler.stopAndCollect()` is now deterministic** — stores the Task handle, cancels, and awaits its value. Pre-v0.3 the sampling loop could run for ~50ms after stop returned, holding `self` until the next tick. (`9764628`)
- **`EngineCoordinator` exposes `engineVersion`** synchronously on the @MainActor (refreshed on init + after every `switchTo(_:)`). Lets the benchmark view model attach the real engine version to the result without awaiting the engine actor. (`e3cf815`)
- **TUI deferral comments** now point at [#18](https://github.com/magicnight/Mac-MLX/issues/18) (upstream SwiftTUI Swift 6 blocker) instead of stale `// TODO: v0.2`.
- **CLI `macmlx run` / `macmlx serve` now honour `Settings.preferredEngine`** via a new `CLIContext.makeEngine()` helper. Previously both hard-coded `MLXSwiftEngine()` — CLI and GUI disagreed silently on engine choice.
- **CLI `macmlx run` layers explicit flags over persisted per-model `ModelParameters`** via `CLIContext.resolveParameters(for:…)`. A user who set `temperature=0.3` for `Qwen3-8B-4bit` in the GUI Parameters Inspector now sees that value in `macmlx run Qwen3-8B-4bit` unless they pass `--temperature` explicitly. `--temperature`, `--max-tokens`, and `--system` are now `Optional` so "unset" is distinguishable from the old compile-time defaults.
- **CLI `macmlx list` empty-state displays the real configured model directory** (via `ctx.settings.modelDirectory`), not a hard-coded `~/models` guess. Moved-directory users no longer get wrong instructions.
- **ConversationStore date precision** — encoder switched from `.iso8601` (whole seconds only) to `JSONCoding.precisionEncoder()` (`.secondsSince1970` Double); decoder accepts both for backward compatibility. Rapid autosaves during active chat now have deterministic sort order in `list()`.
- **Release pipeline hardening** (`.github/workflows/release.yml`) — appcast push now rebases + retries (main advances during the 15-20 min Xcode archive step), and is `continue-on-error` so a push race doesn't block the DMG from landing as a GitHub Release. A race here took out the first v0.3.0 publish attempt (DMG built + signed but no Release artifact created); fix lets the job proceed to `Create GitHub Release` regardless.
- `.gitignore` now covers Xcode 16's `xcshareddata/swiftpm/` editor state (was a persistent untracked-file source for every developer).

### Fixed
- **Missing test coverage for v0.2 stores** — `ConversationStoreTests` + `ModelParametersStoreTests` (+12 tests total) cover save/load round-trip, sort ordering, delete, corrupt-file tolerance, empty store, and the slash-in-model-ID filesystem-safety edge case. Top-level test functions wrapped in `@Suite` structs so identical names across store test files don't collide.
- Miscellaneous stale `// TODO: v0.2` markers that were never resolved: `ModelLibraryManager.parameterCount/architecture` now marked "v0.3+ requires config.json parser", `MLXSwiftEngine.toolCall` note cleaned, `PSCommandTests` phantom v0.2 integration-test TODO dropped.

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
