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

## v0.4 — Engine parity with oMLX (revised 2026-04-18)

**Plan change**: v0.4 was originally "Vision-Language Models." After
comparing against [oMLX](https://github.com/jundot/omlx) (10.6k stars,
more mature engine) we decided the higher-leverage investment is
closing the inference-engine gap first. VLM moves to v0.4.1. Research
notes below each item came from a 2026-04-18 investigation (see
`.omc/plans/v0.3.7-research-notes.md` for raw reports).

### v0.4.0 — Tiered KV cache + multi-model pool + MCP server

Three independent features, same release. Each has low-to-medium risk
because the underlying `mlx-swift-lm` APIs already exist.

#### Tiered KV cache (hot RAM + cold SSD)

- **Why it matters:** Coding assistants (Claude Code, Cursor, Zed)
  re-send the entire conversation history on every request. Caching
  the shared prefix cuts prefill to near-zero on re-asks and matches
  oMLX's headline feature. 3–10× perceived speedup for real workflows.
- **Swift-side primitives already shipped in `mlx-swift-lm`:**
  `KVCache.swift` exposes `savePromptCache(url:cache:metadata:)`,
  `loadPromptCache(url:)`, `trimPromptCache(_:numTokens:)`,
  `canTrimPromptCache(_:)`. Round-trips safetensors in the same
  format Python mlx-lm uses.
- **What we build:**
  - vLLM-style chained SHA-256 block hashing keyed on
    `(modelID, parentHash, tokenIDs, extraKeys)` at 256-token
    granularity (matches oMLX, matches vLLM, proven design).
  - Prefix-hash LRU on `~/.mac-mlx/kv-cache/` (16-subdir fanout by
    first hex char of hash).
  - Hot tier = last K `[KVCache]` structs in memory, keyed by hash.
    Cold tier = safetensors on disk. Background-writer `DispatchQueue`
    flushes to disk without MLX synchronization (critical for perf).
  - Longest-common-prefix matcher: given new tokens `T`, walk the
    hash chain to find the deepest cached prefix, trim the cache to
    that length, prefill only the delta.
  - Defaults: 8 GB hot, 100 GB cold, configurable in Settings.
- **Risk:** LOW. No model-architecture changes; pure Swift plumbing
  over stable APIs with existing test coverage (`KVCacheTests.swift`
  in the mlx-swift-lm tree).

#### Multi-model pool with auto-swap

- **Why it matters:** Users pin a "small daily model" + an
  "occasionally-used big model" and have both reachable without
  manual load/unload. Today our cold-swap unloads before load, which
  means the small model has to be re-loaded on every switch.
- **Swift-side primitives already shipped:** `ModelContainer` is
  `Sendable` so multiple instances in one process are explicitly
  supported. `MLX.GPU.setCacheLimit` + `Memory.memoryLimit` /
  `Memory.cacheLimit` for budget control. `WiredMemoryUtils` +
  `WiredBudgetPolicy` already coordinate concurrent `generate()`
  across containers.
- **What we build:**
  - `actor ModelPool { var entries: [String: PooledContainer] }` with
    LRU + explicit pin flag + estimated size from safetensors
    pre-scan.
  - Sequential loads under a lock (disk bandwidth is the bottleneck;
    parallel loads just thrash).
  - Memory-pressure watcher (`DispatchSource.makeMemoryPressureSource`
    OR `os_proc_available_memory`) evicts LRU containers. On
    eviction call `MLX.GPU.clearCache()` — unified memory does not
    release weights until this runs.
  - OpenAI/Ollama API routes dispatch on request's `model` field to
    the right container; cold-load on miss under the same lock.
  - Settings UI: "max resident memory" slider (default 50% of
    `HardwareInfo.totalMemoryGB`), per-model pin toggle on the
    Models tab.
- **Risk:** LOW. All APIs exist; MLX allocator is process-wide but
  documented as safe for multi-container use.

#### MCP server MVP

- **Why it matters:** Claude Code / Cursor / Zed all speak MCP; the
  moment we expose MCP tools, we're reachable from their ecosystems
  without client-side plumbing per tool. oMLX is MCP-*client* only;
  macMLX starting as MCP-*server* is a complementary niche.
- **SDK:** [`modelcontextprotocol/swift-sdk`](https://github.com/modelcontextprotocol/swift-sdk)
  (1,350 stars, MIT→Apache-2.0, Swift 6 strict concurrency, stdio +
  streamable-HTTP + SSE). Pin `from: "0.11.0"` — pre-1.0 API is
  still settling. Wrap usage behind a thin `MCPBridge` type so the
  pin can move without touching callers.
- **What we build:**
  - New CLI subcommand: `macmlx mcp serve` — spawns a
    `Server(name: "macmlx")` over stdio, registers two tools:
    - `chat(model: String, messages: [...], ...)` — wraps
      `InferenceEngine.generate` with OpenAI-shaped input/output.
    - `list_models()` — returns currently-downloaded models.
  - Users add it to Claude Desktop's `claude_desktop_config.json`:
    ```json
    {
      "mcpServers": {
        "macmlx": { "command": "macmlx", "args": ["mcp", "serve"] }
      }
    }
    ```
- **Risk:** MEDIUM. Pre-1.0 SDK could break. Isolate behind
  `MCPBridge` so upgrading is a one-file change.
- **MCP client** (configuring external MCPs from inside macMLX so
  the chat can tool-call) is deferred to v0.5.1 — needs tool-call
  UI in the chat view first.

### v0.4.1 — VLM (moved from v0.4.0)

Original v0.4 scope intact, just shifted one dot:

- MLXVLM integration for 16 architectures (Qwen2.5/3-VL, Gemma-3,
  SmolVLM/2, Paligemma, Pixtral, Idefics3, FastVLM, LFM2-VL,
  glm_ocr, mistral3)
- Image picker in ChatInputView (NSOpenPanel + drag-drop + paste)
- OpenAI multimodal `content`-array parsing in HummingbirdServer
- Images persisted to `~/.mac-mlx/conversations/<uuid>/images/`

Separate release cadence keeps v0.4.0's engine-parity work on a
tight testing loop before adding visual modality churn.

---

## v0.5 — Continuous batching (depends on upstream) + LoRA + MCP client

### Continuous batching

**Upstream blocker:** `mlx-swift-lm`'s `TokenIterator` is strictly
single-request. All 85 model architectures in the tree call
`createAttentionMask(h:cache:)` with a single `[KVCache]` — adding a
batched path means auditing or wrapping every model. Apple has
shipped this for Python (`mlx-lm` PRs #941, #1101, #1129, #873,
#1090) but nothing analogous is merged in Swift yet.

**Approach:** Two-track.
- **Track A (preferred):** Wait / nudge Apple to port `BatchGenerator`
  to `mlx-swift-lm`. File an upstream issue. If they merge within
  2–3 months, our work collapses to the scheduler layer — FCFS
  waiting queue, token-budget admission, request demuxing — which
  oMLX has already demonstrated in ~300 Python LOC. Port to Swift
  is a weekend.
- **Track B (fallback):** Swift-side Llama/Qwen-only fork of
  `BatchTokenIterator` + `BatchKVCache` on top of `MLXFast.scaledDotProductAttention`
  (which already takes arbitrary leading batch dim). Expect 2–3
  weeks plus ongoing merge pain against upstream single-batch model
  code. Only pursue if upstream stalls and our traction justifies.

Shipping target depends entirely on which track activates.

### LoRA adapter loading

Same as pre-revision plan: drop-in HuggingFace adapter support, no
training UI. No dependency on continuous batching — land
independently within v0.5.x.

### MCP client

Counterpart to v0.4.0's server role. Users configure external MCP
servers (mirror of `claude_desktop_config.json` format at
`~/.mac-mlx/mcp.json`); chat models tool-call through them. Requires
tool-call UI in chat view, which is incidental v0.5 work anyway.

---

## v0.6 — Speech I/O (revised 2026-04-18)

**Plan changed**: the original proposal (WhisperKit + AVSpeechSynthesizer)
is superseded by [DePasqualeOrg/mlx-swift-audio](https://github.com/DePasqualeOrg/mlx-swift-audio)
— a native-MLX audio stack covering both directions:

- **STT**: Whisper, Fun-ASR (latter is strong on Chinese)
- **TTS**: Kokoro, Chatterbox / Chatterbox Turbo, CosyVoice 2 / 3,
  Marvis (streaming), Orpheus (emotion tags), OuteTTS

Replacing WhisperKit with MLX Whisper keeps the architecture
100% MLX-native — no Core ML detour, no Python, consistent with
our engine philosophy.

### Scope for v0.6

- **STT**: `MLXAudio.Whisper` (multilingual) + `MLXAudio.FunASR` for
  Chinese. Push-to-talk in ChatInputView, auto-stop on silence.
- **TTS**: Default to **Marvis** (streaming audio for low-latency
  chat reply) or **Chatterbox** (voice cloning from a user-recorded
  reference clip). Kokoro intentionally excluded from the core
  target — see licensing below.
- **UI**: Mic button in `ChatInputView`, playback button on every
  assistant bubble, user-configurable auto-speak toggle in
  Parameters Inspector.
- **Package wiring**: Add `mlx-swift-audio` as a dependency pinned
  to a commit SHA (not `main`) since the upstream warns
  "expect breaking changes." Import only the `MLXAudio` product
  (avoids GPL-3 espeak-ng transitive from Kokoro).

### Licensing gotcha

Kokoro depends on `espeak-ng` which is **GPL-3**. If we link
Kokoro, GPL-3 terms propagate. The upstream package splits
correctly: `MLXAudio` core is MIT-compatible, `Kokoro` is a
separate product. We only take `MLXAudio`. If a user specifically
wants Kokoro later, that's a separate add-on target.

### Risks

- **Early-development upstream**: DePasqualeOrg is actively breaking
  APIs. Require a smoke-test CI job that re-resolves the pin
  weekly so we notice regressions before users do.
- **espeak-ng C-library builds**: even core MLXAudio has some
  transitive C code — verify the xcframework situation on first
  pull, document any DMG-packaging adjustments.
- **Model sizes**: Whisper large-v3 ≈ 3 GB, Fun-ASR ≈ 1 GB.
  Probably default to small/medium and expose a size selector.

### ⚠️ Do not use `Adamiito0909/mlx-swift-audio`

A user asked about this repo (2026-04-18). It's a **copycat with
likely malware-drop pattern**:

- 6 stars, pushed same day the user checked
- Not a fork (`parent: null`) but code clearly copied from
  DePasqualeOrg's
- README aggressively promotes downloading
  `examples/TTS App/TTS App.xcodeproj/project.xcworkspace/mlx-swift-audio-1.3-beta.5.zip`
  — a `.zip` deep inside an Xcode workspace folder, which is a
  classic drive-by download vector

Avoid it. Use `DePasqualeOrg/mlx-swift-audio` exclusively. Consider
GitHub-reporting the copycat as impersonation.

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
