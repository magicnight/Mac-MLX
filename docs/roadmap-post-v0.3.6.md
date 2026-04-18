# macMLX Roadmap — post v0.3.6

Written 2026-04-18 after the v0.3.6 release branch stabilised. Captures
features and issues that were deferred, plus items that became feasible
once App Sandbox was turned off.

---

## v0.3.7 — "Ollama parity" maintenance release

Small patches + the two state-sync improvements the QA pass surfaced.
No breaking changes, no schema migrations.

### 1. GUI ↔ CLI state convergence (Ollama-style daemon)

**Problem:** The GUI app and `macmlx` CLI run as independent processes
and each owns its own `HummingbirdServer` + engine instance. A user
running `macmlx serve` while the GUI is open gets two servers (one
retries to :8001), two loaded copies of the model, two sets of
conversation logs. Ollama solves this by having one background daemon
that both its CLI and GUI talk to via HTTP.

**Approach (smallest-possible-change version):**

1. GUI writes `~/.mac-mlx/macmlx.pid` when its server starts, matching
   the CLI's existing `PIDFile` format (already used by
   `macmlx ps` / `macmlx stop`).
2. CLI commands check the PIDFile on startup:
   - `macmlx ps` — shows GUI's PID + port if present.
   - `macmlx serve` — refuses to start if the GUI is running, prints
     "Another macMLX server is already running on :8000 (GUI). Close
     it or use `macmlx stop` first."
   - `macmlx run <model>` — if a GUI server is up, proxy the chat
     through the HTTP API instead of booting its own engine.
   - `macmlx stop` — sends a graceful-shutdown signal (already
     understood by the CLI server; GUI needs to implement the same
     handler or a `/x/shutdown` endpoint).
3. GUI writes `lastLoadedModel` to `~/.mac-mlx/settings.json` when the
   loaded model changes; CLI reads on next use (already works).

**Scope:** ~3 days of work. Touches PIDFile, CLI commands (except
`pull` / `list`), AppState server lifecycle. No engine changes.

### 2. Cold-start model rehydration (already landed)

GUI bootstrap loads the last model when `autoStartServer = true` so the
server is immediately useful. Document this in README alongside the
auto-start explanation.

### 3. MLX stdout/stderr capture in the Logs tab

**Problem:** `mlx-swift-lm` and friends print to stdout (tokenisation
progress, warnings, etc.). These are invisible in the GUI Logs tab
because they bypass `LogManager`.

**Approach:** Dup the process's `STDOUT_FILENO` and `STDERR_FILENO` to
a `Pipe`, read the pipe in a background task, tee each line into both
the original stdout (so terminal output still works when launched from
CLI) and `LogManager.shared.debug(_, category: .system)`. Scope the
redirection to app launch in `macMLXApp.init`. Roughly 80 lines.

### 4. Model update detection

**Problem:** When a user-downloaded HF model has been updated on the
Hub (new quant, bug fix), macMLX has no way to know.

**Approach:** Store the resolved commit SHA from `HFDownloader.files`
in a sidecar `~/.mac-mlx/models/<name>/.macmlx-meta.json` on download.
On Models-tab open (throttled weekly), fetch the current SHA via HF's
`/api/models/{id}` and compare. Display an orange "Update available"
badge on affected rows, with a "Re-download" action. No auto-update —
user is always in control.

### 5. Appcast + DMG signing (out-of-band)

- Finalize `appcast.xml` for v0.3.6 post-DMG (edSignature + length)
- Signed + notarised DMG still blocked on a paid Apple Developer
  account — issue #19 remains open

---

## v0.4 — Vision-Language Models (already scoped)

See `.omc/plans/v0.4-vlm-plan.md`. Main work:

- MLXVLM integration for Qwen2.5-VL / Qwen3-VL / Gemma-3 /
  SmolVLM / Pixtral / Idefics3 / FastVLM / LFM2-VL / glm_ocr /
  mistral3 (16 architectures)
- Image picker in ChatInputView (NSOpenPanel + drag-drop + paste)
- OpenAI multimodal `content`-array parsing in HummingbirdServer
- Images persisted to `~/.mac-mlx/conversations/<uuid>/images/`

Nothing in v0.4 is sandbox-affected — the plan stands.

---

## v0.5 — LoRA + Export

- LoRA adapter loading (drop in existing HuggingFace adapters, no
  training UI)
- Conversation / dataset export to JSONL (ChatML + ShareGPT formats)

---

## v0.6 — Speech I/O

- WhisperKit (Core ML) for mic input in chat
- AVSpeechSynthesizer for assistant reply read-back
- Native MLX Whisper deferred until `mlx-swift-lm` ships audio
  models

---

## v0.7 — Community Benchmarks

- Opt-in `POST /v1/benchmarks` remote endpoint receiving anonymised
  `BenchmarkResult` + `HardwareInfo`
- Aggregated leaderboard by chip × model × quant × macOS version
- Served on the website (`macmlx.app`) and inside the app's
  Benchmark tab

---

## Sandbox-blocked issues — re-openable now that sandbox is off

App Sandbox was disabled in v0.3.6. Several previously-closed issues
become feasible again. None are committed yet; each needs a
brainstorming pass on UX before landing.

### #12 / #13 — Alternative inference engines

Both were closed as "not planned: macOS App Sandbox blocks spawning
external binaries." That constraint is gone.

- **#12 — Python `mlx-lm` engine.** Subprocess launches `uv run` +
  a FastAPI shim, GUI talks to it via its own OpenAI-compatible
  endpoint. Pros: max model coverage (mlx-community's Python-only
  checkpoints). Cons: needs `uv` on PATH, slower first-token.
- **#13 — SwiftLM engine for 100B+ MoE.** Subprocess to the SwiftLM
  binary for architectures `mlx-swift-lm` can't handle (Gemma 4 MoE,
  Llama 4 MoE, DeepSeek-V3). Pros: covers the MoE gap (see
  [mlx-swift-lm#219](https://github.com/ml-explore/mlx-swift-lm/issues/219)).
  Cons: extra binary to distribute.

**Proposed:** Open both issues for v0.5 or v0.6 scoping. The
subprocess machinery is the same for both — implement once, use twice.

### #20 — Homebrew tap for the CLI

Was scheduled for v0.3.6-0.4. Unblocked by sandbox-off (not that
sandbox affected Homebrew publishing, but the CLI itself is now
fully aligned with the GUI paths). The work:

- Write a Homebrew formula pulling the latest release tarball
- Set up a tap repo (`homebrew-macmlx`)
- Update CI to publish on release tag

### Multi-model loading (new ask)

**User request:** "If memory is large enough, support loading multiple
models simultaneously."

**Approach (v0.5-ish):**

1. Engine layer grows from single `container` to `[String: Container]`
   keyed by model ID.
2. `EngineCoordinator` exposes a `loadedModels` set + a
   `load(_:)` that either reuses a slot or evicts the LRU when
   over a configurable memory budget (default 50% of `HardwareInfo.totalMemoryGB`).
3. `generate(_:)` picks the container matching `request.model` instead
   of using the single current one.
4. Memory watchdog: on hitting 90% of the budget, evict oldest.

**UX:**

- Models tab shows a "Loaded" green dot on each row that's in memory.
- Toolbar model switcher becomes a multi-selection — click to load,
  click again to unload.
- External API automatic routing (no more cold-swap unloads unless
  memory budget is hit).

**Scope:** 1-2 weeks. Biggest risk is MLX allocator behaviour under
overlapping contexts — may need `mlx-swift-lm` hints.

---

## Legend of deferred bugs worth revisiting

From the v0.3.6 post-QA pass:

- **Streaming Ollama `done_reason`** — currently always `"stop"`.
  Should reflect actual finish reason (`length`, `stop_tokens`,
  `cancelled`) from `GenerateChunk.finishReason`.
- **Parameters Inspector** — reopen the "per-conversation
  systemPrompt override" note in `ChatViewModel.adopt(_:)`. The
  current model-level systemPrompt is fine but users have asked.
- **HF search rank tweak** — include `likes` as a secondary sort
  key after the user's typed-token match so well-known fine-tunes
  surface above obscure forks.
- **Logs tab live-stream view** — pair with the stdout capture
  work. A separate tab showing the last 10k raw tokens ring buffer
  with autoscroll. Cleared on model unload.
