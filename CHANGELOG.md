# Changelog

All notable changes to macMLX will be documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- **MCP Client Config** (v0.5 MCP track, part 1 of 2). Pure-Swift
  data layer for connecting macMLX to external MCP servers (mirror
  of v0.4.0's MCP server role, but reversed: now we *are* the host
  and chat models tool-call out to other people's MCP servers).
  - `MCPClientConfig` Codable struct mirroring Claude Desktop's
    `~/.claude_desktop_config.json` `mcpServers` shape — users
    with existing MCP setups can copy the file across without
    reformatting.
  - `MCPClientConfigStore` actor reads / writes
    `~/.mac-mlx/mcp.json`. Missing-file → empty config (no error
    on first run). Malformed JSON → empty config; the bad bytes
    stay on disk for the user to fix manually.
  - 7 new unit tests cover Claude Desktop schema decode, JSON
    round-trip, missing-file behaviour, and malformed-JSON fall-
    through. Subprocess pool + chat-side tool-call routing follow
    in v0.5 MCP track part 2.
- **LoRA Parameters Inspector picker** (v0.5, part 4 of 4). User-
  facing surface for selecting and applying LoRA adapters.
  - `AppState.adapterStore` actor + `availableAdapters: [LocalAdapter]`
    @Observable property + `adaptersDirectory` (real
    `~/.mac-mlx/adapters/`). Initial scan runs after `bootstrap()`.
  - `AppState.refreshAdapters()` re-runs the scan; the parameters
    inspector exposes a refresh button so users can drop a new
    adapter and pick it up without restarting the app.
  - `EngineCoordinator.onModelLoaded` callback now resolves the
    pinned `ModelParameters.adapterName` against the adapter cache
    and calls `engine.applyAdapter(_:)` after the base model loads.
    Missing-adapter / apply-failure paths log to Pulse's `engine`
    category — the model still loads text-only.
  - `ParametersInspector` grows a "LoRA Adapter" section with:
    - menu picker bound to `params.parameters.adapterName` (None /
      every detected adapter)
    - refresh button to rescan `~/.mac-mlx/adapters/`
    - empty-state hint when the directory is empty
    - orange warning when the configured adapter name no longer
      exists on disk (load will skip it cleanly)
- **LoRA Engine integration** (v0.5, part 3 of 4). Engine-side
  glue tying the v0.5 LoRA Foundation (#36) and PEFT → mlx
  Converter (#37) together so a downloaded HuggingFace adapter
  works end-to-end via the InferenceEngine protocol.
  - `InferenceEngine.applyAdapter(_:)` — new protocol method with
    a default `extension`-level no-op so test stubs and future
    CPU/Python engines compile unchanged. Implementations that
    DO support adapters wire it to their LoRA loader.
  - `MLXSwiftEngine.applyAdapter(_:)` — auto-routes PEFT-format
    adapters through `LoRAAdapterConverter` (cached at
    `~/.mac-mlx/adapters/.cache/<adapter-name>/` so repeat loads
    skip conversion), then calls
    `LoRAContainer.from(directory:)` and
    `LanguageModel.load(adapter:)`. Throws
    `EngineError.adapterApplyFailed(reason:)` on either step.
  - `EngineCoordinator.load(_, adapter:)` — optional adapter
    parameter; default `nil` keeps every existing call site
    unchanged. When provided, the adapter is applied immediately
    after the base model loads.
  - `AdapterStore.scan(_:)` now detects mlx-native format
    (`adapters.safetensors` + mlx-schema `adapter_config.json`)
    in addition to PEFT, with mlx-native taking precedence when
    both files coexist (caller already has the converter output
    side-by-side with the source).
  - `LocalAdapter.format: Format` (`peft` / `mlx`) drives the
    engine's auto-conversion branch. Backwards-compatible decode
    defaults pre-v0.5 records to `.peft`.
  - `ModelParameters.adapterName: String?` persists the user's
    adapter pick per model. Custom decoder defaults to nil so
    pre-v0.5 `~/.mac-mlx/model-params/*.json` files load unchanged.
  - 4 new tests (2 LocalAdapter format/round-trip, 2 AdapterStore
    mlx detection / dual-format precedence). 137/137 Core green.
  - Parameters-inspector picker UI lands in v0.5 part 4.
- **LoRA PEFT → mlx Converter** (v0.5, part 2 of 3). Pure-Swift
  in-process converter that turns a HuggingFace PEFT-format adapter
  directory into the mlx-swift-lm native format that
  `MLXLMCommon.LoRAContainer.from(directory:)` expects. Three
  translations: PEFT `r` / `lora_alpha` / `target_modules` →
  mlx `lora_parameters.{rank, scale, keys}` (scale = alpha/rank);
  PEFT keys `base_model.model.<path>.lora_A.weight` → mlx
  `<path>.lora_a` (drop wrapper, lowercase, drop `.weight`);
  PEFT tensor shapes `[r, in]` / `[out, r]` → mlx `[in, r]` /
  `[r, out]` (transpose). `num_layers` auto-inferred from the
  deepest `model.layers.<N>` index in the input keys. 8 pure-Swift
  unit tests cover key rewrite + layer-index extraction + config
  translation; 6 MLX-backed XCTest tests cover end-to-end
  filesystem round-trip (gated on Metal — skip cleanly under
  `swift test`, run under `xcodebuild`). Engine application of the
  converted adapter (parameters-inspector picker + Settings
  adapters section) lands in v0.5 part 3.
- **LoRA Foundation** (v0.5, part 1 of 3). Pure-Swift Core layer
  for HuggingFace LoRA adapter discovery — no engine integration
  yet, no UI. `LocalAdapter` value type holds adapter metadata
  parsed from PEFT's `adapter_config.json` (`base_model_name_or_path`,
  `r`, `lora_alpha`, `target_modules`, `peft_type`). `AdapterStore`
  actor scans `~/.mac-mlx/adapters/<name>/` for directories that
  contain both `adapter_config.json` and `adapter_model.safetensors`
  — best-effort, malformed configs / missing weights silently drop.
  10 new unit tests cover decode + scan paths. Engine application
  via `MLXLMCommon.LoRAContainer.from(directory:)` (after PEFT →
  mlx conversion) and the parameters-inspector picker land in
  v0.5 part 3.
- **Prompt cache tiering** (v0.4.0 engine parity, part 1 of 3).
  Successive chat turns on the same model now reuse the KV cache
  when the new prompt extends the previous one — the shared prefix
  skips prefill. In-memory hot tier (LRU, 8 entries in MVP) backed
  by on-disk cold tier at `~/.mac-mlx/kv-cache/`, 16-way sharded
  safetensors round-tripped through mlx-swift-lm's `savePromptCache`
  / `loadPromptCache`. Coding-assistant workflows (Claude Code,
  Cursor, Zed re-sending conversation history each turn) see
  reduced time-to-first-token on repeat prefixes.
- Settings → "KV Cache" section with hot/cold budget steppers and
  a "Clear All KV Caches" button. Steppers currently inform future
  byte-accurate budgeting (v0.4.0.1) — today's enforcement is the
  8-entry hot LRU cap plus manual Clear.
- Debug-level Logs tab entries `Prompt cache HIT — restored N
  tokens` / `Prompt cache MISS — cold prefill of N tokens` under
  the `engine` category, so you can see cache effectiveness.
- **Multi-model pool** (v0.4.0 engine parity, part 2 of 3). Load
  multiple models at once — previously the engine had to unload
  the old model before loading a new one, which meant every API
  cold-swap paid the full weight-read cost. Pool is bounded by a
  user-configurable resident memory cap (Settings → Model Pool;
  default 50% of total RAM). Least-recently-used non-pinned
  models auto-evict when the cap is exceeded. Pin a model from
  its row in the Models tab (pin icon) to keep it resident
  regardless of LRU order. Pinned state is in-memory for this
  release; persistence across restarts is a follow-up.
- **MCP server** (v0.4.0 engine parity, part 3 of 3). New CLI
  subcommand `macmlx mcp serve` exposes macMLX's local MLX
  inference to MCP clients (Claude Desktop, Cursor, Zed, Claude
  Code, …) over stdio. Two tools ship in this MVP:
  - `list_models` — returns every locally-downloaded model with
    its id / displayName / sizeBytes / format / quantization /
    parameterCount / architecture.
  - `chat` — runs a buffered chat completion against a model id,
    lazy-loading the engine on first call and lazy-swapping the
    loaded model when the requested id changes. Optional
    `temperature` / `max_tokens` / `system` arguments. System
    prompt handling matches HummingbirdServer's OpenAI-compat
    path so Qwen3 / Gemma / DeepSeek strict Jinja templates don't
    reject the prompt.

  Built on [`modelcontextprotocol/swift-sdk`](https://github.com/modelcontextprotocol/swift-sdk)
  v0.12.x — pinned per-minor since the SDK is still pre-1.0. CLI-
  only dependency, so the GUI release stays lean. Logs (SDK +
  ours) go to stderr so stdout stays a pure JSON-RPC stream.
  Honours `Settings.preferredEngine` (same engine selection as
  `macmlx serve` and `macmlx run`). Drop into Claude Desktop's
  `claude_desktop_config.json` as
  `{ "mcpServers": { "macmlx": { "command": "macmlx", "args": ["mcp", "serve"] } } }`.
- **VLM UI + Persistence + HTTP** (v0.4.1, part 3 of 3). Lights up
  the user-facing surfaces for vision-language models. Closes the
  v0.4.1 work begun in PRs #33 (Foundation) and #34 (Engine).
  - **Chat input image picker.** New paperclip button in the chat
    input opens SwiftUI's `.fileImporter` (image UTTypes only:
    jpeg / png / webp / gif / heic / bmp), populating a horizontal
    thumbnail strip above the text field. Click the × on a thumbnail
    to drop it. The button is disabled when the loaded model isn't
    a VLM, with an explanatory tooltip ("Load a vision-capable model
    (Qwen-VL, Gemma-3, SmolVLM, …) to attach images"). Image-only
    messages (no text) are now valid sends on a VLM.
  - **Inline thumbnails on chat bubbles.** `ChatMessageView` renders
    a 96pt LazyVGrid of attached images above the text bubble for
    any turn that has images. Click a thumbnail to open the file in
    Preview via `NSWorkspace`.
  - **Conversation persistence.** `StoredMessage.images` round-trips
    through `ConversationStore`. On save, every external image URL
    is copied into `<conversations>/<conv-uuid>/images/` and the
    stored URL is rewritten to point there — chats survive the user
    moving the picked file. On `delete(id:)`, the per-conversation
    directory is torn down so images don't leak. Pre-v0.4.1
    conversations decode unchanged (missing key → empty array).
  - **OpenAI multimodal HTTP.** `/v1/chat/completions` now accepts
    OpenAI's `content` array shape:
    ```json
    {"role":"user","content":[
      {"type":"text","text":"What's this?"},
      {"type":"image_url","image_url":{"url":"data:image/png;base64,…"}}
    ]}
    ```
    Plain-string `content` continues to work — the decoder tries
    string first, falls through to `[Part]`. base64 data URLs
    decode to tmpfile-backed `ImageAttachment` values; caps: 10 MB
    per image, 4 images per message; `http(s)://` and `file://`
    URLs are not fetched (defence-in-depth on a localhost-bound
    server). Ollama's `/api/chat` / `/api/generate` stay text-only
    — Ollama's wire format uses a separate top-level
    `images: [base64]` field; revisit in a follow-up.
- **VLM Engine** (v0.4.1, part 2 of 3). MLXSwiftEngine now branches
  on `model.format` to load text-only models through
  `MLXLLM.LLMModelFactory` and vision-language models through
  `MLXVLM.VLMModelFactory`. Runtime modality stored in a new
  `LoadedSupport` enum (`.none / .llm / .vlm`).
  - `runGeneration(_:)` splits into `runLLMGeneration` (existing
    prompt-cache flow — hot/cold KV tier, suffix prefill, save
    extended cache after stream) and `runVLMGeneration` (fresh KV
    cache per call; bypasses the prompt cache for now since
    multimodal cache keys would need to fold image bytes into the
    chained hash).
  - `Chat.Message` mapping respects modality: VLM models receive
    `ChatMessage.images` as `UserInput.Image.url(URL)` so the VLM's
    `UserInputProcessor` can inject image tokens; LLM models drop
    accidental attachments with a debug-level Pulse warning.
  - `MLXVLM` added to `MacMLXCore` package dependencies (sibling
    product of `MLXLLM` already in our `mlx-swift-lm` 3.31.x pin —
    no new SPM dependency tree).
  - Three new unit tests cover unsupported-format rejection
    (gguf, unknown, missing-VLM-directory). 111/111 Core tests
    green. Real VLM smoke (loading e.g. SmolVLM-Instruct-4bit) is
    a manual-QA item — multi-GB download.
  - Image picker, multimodal HTTP, and conversation persistence
    land in the v0.4.1 part-3 PR. Plan:
    `docs/superpowers/plans/2026-05-10-v0.4.1-vlm.md`.
- **VLM Foundation** (v0.4.1, part 1 of 3). Pure-Swift Core changes
  for vision-language model support. No MLX integration yet, no UI,
  no HTTP changes.
  - `ImageAttachment` value type (`fileURL`, `mimeType`) sits next
    to `LocalModel` / `HFModel` and round-trips through Codable.
    MIME-type helper covers jpeg / png / webp / gif / heic / bmp.
  - `ChatMessage` gains an `images: [ImageAttachment]` field with a
    custom `init(from:)` that defaults to empty when the key is
    absent — pre-v0.4.1 conversation JSON loads unchanged, no
    migration step.
  - `ModelFormat.mlxVLM` distinguishes vision-language directories.
    `ModelLibraryManager.scan(_:)` peeks `config.json`'s
    `model_type` and tags 14 known VLM families: qwen2_vl,
    qwen2_5_vl, qwen3_vl, qwen3_5_vl, gemma3, smolvlm, smolvlm2,
    paligemma, pixtral, idefics3, fast_vlm, lfm2_vl, glm_ocr,
    mistral3. Malformed / missing `model_type` falls back to
    `.mlx`. 13 new unit tests cover detection edge cases.
  - Engine integration (MLXSwiftEngine VLM branch via
    `MLXVLM.VLMModelFactory`) and UI / HTTP work land in follow-up
    PRs — see
    `docs/superpowers/plans/2026-05-10-v0.4.1-vlm.md`.

---

## [0.3.7] - 2026-04-18

Maintenance release — four items agreed after the v0.3.6 post-QA pass.

### Added
- **MLX stdout/stderr → Logs tab.** `StdoutCapture.install()` dups
  the process's STDOUT/STDERR to a Pipe at launch, tees each line
  to both the original fd (terminal visibility preserved) and
  `LogManager.shared.debug(category: .system)`. Users reporting
  "model is slow" can now see what `mlx-swift-lm` is printing
  without attaching a debugger.
- **HF model-update detection.** Downloaded models now get a
  `.macmlx-meta.json` sidecar recording the Hub's commit SHA +
  `lastModified` at download time. Models tab throttle-checks
  (once per 24 h) each sidecar against `/api/models/{id}`. Rows
  whose Hub head has advanced get an orange "Update available"
  badge. Delete-and-re-download is the update action for now.
- **Shared `~/.mac-mlx/macmlx.pid`.** `PIDFile` moved from the CLI
  target into `MacMLXCore`; `Record.owner` enum (`.gui | .cli`)
  distinguishes the two. GUI writes the file on `startServer()`,
  clears it on `stopServer()`. CLI's `macmlx serve` pre-flights
  with `kill(pid, 0)` and refuses to double-bind when a live
  record exists, naming the owner in the error so users know
  which process to close. `macmlx ps` now shows "Owner: GUI | CLI".

### Changed
- **GitHub Actions pinned to Node.js 24.** `actions/checkout` and
  `actions/cache` bumped from `@v4` (Node 20, deprecated) to `@v5`
  (Node 24). Silences the deprecation warning that surfaced on the
  v0.3.6 release run.

### Notes
- **Full Ollama-style daemon mode** (CLI commands proxied through
  the GUI's HTTP API rather than spinning their own engine) remains
  a future scope. This release just gets them to stop stepping on
  each other at the port-binding layer.
- **Backward compatibility**: pre-v0.3.7 PID files (no `owner` key)
  decode as `.cli`. Pre-v0.3.7 downloads without sidecars simply
  don't get an update-check until the user re-downloads them once.

---

## [0.3.6] - 2026-04-18

Bug-fix and UX-polish release covering thirteen user-reported issues
across the Models, Chat, and Logs surfaces. No schema changes.

### Added
- **Collapsible `<think>` renderer** in the Chat view. Qwen3,
  DeepSeek-R1 and Gemma-style reasoning blocks now render as a
  default-expanded disclosure with a small braille spinner while
  streaming and an italic secondary-color body when finished. Click
  the chevron to collapse; re-click to expand. Handles the Qwen3
  implicit-opener case where the chat template injects `<think>` in
  the prompt and only `</think>answer` shows up in the stream.
- **Model size badge** on Hugging Face search results. Shows a
  hard-drive glyph + size (e.g. `4.5 GB`) once the sibling-enrichment
  fetch resolves. Sizes that the Hub leaves `null` (LFS-backed
  weights) fall back to HEAD against `/{id}/resolve/main/{file}`
  and prefer `x-linked-size` over `content-length`.
- **Persistent HF size cache** at `~/.mac-mlx/hf-cache/sizes.json`
  with a 7-day TTL. Re-searching for a model you've already seen
  surfaces its size instantly instead of re-hitting the Hub.
- **Copy-model-name button** next to the model switcher in the Chat
  toolbar. Keyboard shortcut ⇧⌘C. Useful when pasting the ID into
  Cursor / Continue / Open WebUI configs.
- **Generate() diagnostic logging** under the `inference` category —
  starts / chunk count / token count / errors. A user reporting
  "no output" can inspect the Logs tab and see whether the stream
  yielded zero chunks, threw mid-stream, or completed empty.
- **Empty-output fallback message** in the assistant bubble when the
  stream completes with zero chunks and empty content — the user
  sees `[No output — model returned zero tokens…]` instead of an
  ambiguous blank bubble.
- **Friendly Gemma 4 MoE error** at model-load time. Loading a
  `gemma-4-*-a4b-*` Mixture-of-Experts checkpoint previously threw
  a cryptic `Unhandled keys [experts, router, …]` error; now a
  preflight inspects `config.json` and surfaces an explicit message
  pointing at [mlx-swift-lm#219](https://github.com/ml-explore/mlx-swift-lm/issues/219)
  with a hint to use dense E2B / E4B variants until the upstream
  port lands.

### Changed
- **HF download state survives tab switches.** `ModelLibraryViewModel`
  lives on `AppState` (same pattern `ChatViewModel` uses) so switching
  from Models to Chat mid-download no longer resets the progress bar.
- **Smoother speed + ETA display.** `SpeedSampler` throttles EMA
  updates to ≈2 Hz and lowers the smoothing factor (alpha 0.3 → 0.15),
  eliminating the multi-Hz digit flicker.
- **Stricter HF search matching.** Typing `gemma-4` no longer returns
  gemma-3 / gemma-2 results. We post-filter the Hub response so every
  whitespace/hyphen-split query token must appear in the repo name
  (org prefix excluded).
- **Bottom-anchored sparse chat.** Short conversations sit above the
  input box like every other chat app instead of cramming the top.
- **Vertically-centered input cursor.** Replaced the multi-line
  `TextEditor` (cursor anchored top) with `TextField(axis: .vertical)`,
  which keeps the cursor on the single-line baseline and auto-grows
  up to five lines. macOS 14+ API.
- **Switching models auto-creates a new chat.** Prevents the new model
  from inheriting tokens produced by the previous model's chat
  template. The old conversation is preserved in the sidebar.
- **Conversation sidebar rebuilt.** Hand-built scroll + tap rows
  replace `List(selection:)` — macOS SwiftUI's selection binding
  silently swallowed right-click Delete actions on the currently
  selected row. New implementation uses plain views with
  `.onTapGesture` + `.contextMenu`, and deletes immediately instead
  of presenting a confirmation dialog (matches Mail / iMessage).
- **HF tab layout.** Eliminated the blank strip between the toolbar
  and the results list.
- **Pulse log store capped at 100 MB.** `LogManager` owns its own
  `LoggerStore` with an explicit `sizeLimit = 100 MB` — Pulse
  auto-evicts oldest entries once the cap is reached.
- **Initial library scan after bootstrap.** Users who skip the
  onboarding wizard now see their existing downloaded models in
  the Models tab on first open without having to toggle anything
  in Settings.

### Fixed
- Conversation delete context menu is honoured regardless of whether
  the row is currently selected.
- `<think>` blocks that contain the entire response no longer hide
  the content behind a collapsed disclosure — blocks default to
  expanded.

### Post-QA hot patches (2026-04-18 afternoon)

These landed after hands-on QA surfaced regressions in the initial
drop:

- **App Sandbox disabled.** Sandboxed reads of `~/.mac-mlx/models/`
  were being denied ("permission to view it") even though it's our
  own dotfile data root. Apple's "dotfile exemption" for `~/.<path>`
  is not reliable across macOS versions. Turn sandbox off to
  converge the GUI and CLI on the same `~/.mac-mlx/` — matches LM
  Studio / Ollama / oMLX. Gatekeeper remains the user-trust layer.
- **DataRoot now returns the real user home under sandbox.**
  `NSHomeDirectoryForUser(NSUserName())` was returning the sandbox
  container home (`~/Library/Containers/.../Data/`) rather than the
  real `/Users/<user>` despite the Foundation docs. Construct the
  path directly from `/Users/` + `NSUserName()`. Relevant if anyone
  re-enables sandbox in the future.
- **HTTP server now auto-starts in the GUI.** The `autoStartServer`
  setting existed since v0.1 but nothing in the GUI read it — users
  toggling "Auto-start server on launch" saw no effect. Wire a full
  `HummingbirdServer` lifecycle onto `AppState`: `startServer()` /
  `stopServer()`, observable `server` / `serverPort` /
  `isServerToggling` state, `bootstrap()` auto-starts when the
  setting is on (rehydrating last-loaded model first), and the
  Settings toggle now drives start/stop on change.
- **Chat rendering fixed.** Task 7's VStack+ForEach `renderedContent`
  collapsed to zero size under the bubble's padding/background stack
  when the response was a single plain-text segment — so the most
  common case (model replies "Hi!") rendered as an invisible bubble.
  Single-`.text` segments now go through `inlineMarkdown` directly,
  matching pre-v0.3.6 rendering exactly. The segmented VStack only
  kicks in when there's an actual think block.
- **Task 9 GeometryReader reverted.** Bottom-anchoring sparse
  messages via `GeometryReader { geo in … .frame(minHeight: geo.size.height) }`
  interacted badly with ScrollView's unbounded vertical space —
  `geo.size.height` reported zero, collapsing the entire message
  list. Reverted to a plain `LazyVStack`. Input-cursor centering
  (the user's real complaint) is handled separately by
  `TextField(axis: .vertical)`.
- **Sidebar row rebuilt three times before landing.** Final form is
  plain VStack + single `.onTapGesture` + `.contextMenu`, inside a
  `ScrollView` + `LazyVStack`. `List(.sidebar)` and `List(selection:)`
  both had modes where rows disappeared or right-click actions got
  swallowed. The true cause of the delete-on-focused-row bug turned
  out to be `switchTo(_:)` calling `persistNow()`, which bumped the
  outgoing conversation's `updatedAt` — the sidebar's
  `updatedAt`-desc sort then reordered between the left-click and
  the follow-up right-click, so Delete targeted a different row.
  `switchTo` no longer persists (sends, edits, and renames all
  persist on their own).
- **Port no longer shows as 8,000.** SwiftUI's `Text` interpolates
  `Int` with locale-aware thousand separators, so the Settings
  "HTTP Server" section was rendering `"http://localhost:8,000/v1"`.
  Switch the Stepper label and Base URL line to `String(serverPort)`
  to bypass locale formatting.
- **Models tab force-refreshes on scan completion.** The `@Observable`
  registrar occasionally missed the async `localModels` mutation on
  the hoisted `ModelLibraryViewModel`, leaving the tab showing
  "No Local Models" despite a successful scan. Force a view-identity
  change via `.id(appState.modelLibrary.localModels.count)` so
  SwiftUI rebuilds the subtree when results arrive.
- **Input cursor vertically centered.** Replaced `TextEditor` (cursor
  anchored top) with `TextField(axis: .vertical)` — cursor sits on
  the single-line baseline, auto-grows up to five lines. macOS 14+.
- **Content-preview log line.** `ChatViewModel.generate()` now dumps
  the first 240 chars of each completed response at `.debug` level
  so users reporting "no output despite chars=N" can see exactly
  what the stream produced (wrapper tags, invisible tokens, etc.).
- **Local model scan logs the path + subdirs.** Zero-result scans
  now log a warning listing the raw subdir names so it's clear
  whether the scan is looking at the wrong path, hit a permission
  error, or the content doesn't match any model format.

### Post-QA hot patches — server & external-client compat

These landed during a second QA pass when the user tried pointing
external tools (Zed, Immersive Translate, Open WebUI) at the
macMLX HTTP server:

- **CORS middleware** on every response. Browser-based clients
  enforce `Access-Control-Allow-Origin` on fetch and returned
  "NetworkError / fetch error" before. Allow-origin `.all` is the
  right setting for a localhost-only server — the reach boundary is
  the 127.0.0.1 bind, not the origin header.
- **Request-logging middleware.** Every inbound request logs at
  `.debug` level (`→ METHOD PATH`) under the `http` category. 404
  responses (both returned and thrown as `HTTPError(.notFound)`)
  re-log at `.warning` with a `"unhandled route"` tag — so when a
  client reports a generic "fetch error" the Logs tab shows exactly
  which endpoint it tried.
- **Discovery & alias routes.** Probe paths several clients hit
  before committing to real endpoints:
  `GET /`, `GET /v1`, `GET /v1/health`, `GET /v1/status` all return
  a tiny JSON ack. `POST /`, `POST /v1`, `POST /v1/completions`, and
  `POST /v1/chat/completions/chat/completions` (for users who mis-
  configure base URL as the full endpoint path) all route to the
  same chat-completions handler.
- **Ollama API compatibility layer** (non-exhaustive but covers
  probe + chat): `GET /api/version`, `GET /api/tags`,
  `POST /api/show`, `POST /api/chat`, `POST /api/generate`. Chat
  and generate support **NDJSON streaming** (default when
  `stream` is omitted — Ollama's convention, opposite of OpenAI).
  Covers Zed's Ollama provider, Immersive Translate, and the
  Ollama CLI's probe pattern.
- **Duplicate system-message bug fixed.** `handleChatCompletions`
  was leaving system messages in the messages array AND extracting
  the same text into systemPrompt — `GenerateRequest.allMessages`
  then re-prepended the systemPrompt so the engine saw
  `[system, system, user, …]`. Qwen3 / Gemma / DeepSeek's strict
  Jinja templates reject consecutive systems with a
  `Jinja.TemplateException`, which surfaced as a 500
  "Model failed to load: Jinja.TemplateException error 1" on the
  client. Filter system out of the downstream messages array.
- **Generation serialised across requests.** MLX model state
  (tokenizer, KV cache, allocator) isn't safe across overlapping
  generate calls. Hummingbird actor serialises method entry but
  `generate` returns an AsyncStream iterated outside the actor —
  so parallel clients stomped on each other and either crashed or
  hung. Added a FIFO binary semaphore around every chat/completion
  code path (OpenAI and Ollama, streaming and non-streaming).
  Requests queue under load instead of crashing.
- **GUI auto-start server** honours `settings.autoStartServer`.
  Rehydrates last-loaded model before starting so the server
  survives app restart in a useful state.
- **Menu-bar Start/Stop Server button.** Popover now exposes a
  server-level Start/Stop with a "Server" row showing the base
  URL (or "Stopped"). Status dot reflects server health (green
  running / gray stopped / orange toggling / red engine error).
- **Copy-model-name button** in the Chat toolbar (⇧⌘C) for pasting
  the loaded model ID into external tool configs.
- **Cold-swap routed through EngineCoordinator.** External API
  requests triggering a cold-swap used to go straight to
  `engine.load()`, bypassing the coordinator so the GUI / menu bar
  still showed "no model loaded" while the engine was generating.
  `HummingbirdServer` now takes an optional `LoadHook` closure; GUI
  installs one that routes through `coordinator.load(_:)` so
  observable state (`currentModel`, `status`, `onModelLoaded`) stays
  in sync.

### Notes
- **Gemma 4 MoE not runnable.** Confirmed upstream gap
  ([ml-explore/mlx-swift-lm#219](https://github.com/ml-explore/mlx-swift-lm/issues/219)).
  Dense E2B / E4B variants work fine.
- **Raw MLX stdout capture** (so library-level prints land in the
  Logs tab) is v0.3.7 — needs file-descriptor redirection at launch.
- **Model-update detection** (warn when a downloaded model has been
  updated on the Hub) is also v0.3.7.

### Tests
- New `SpeedSamplerTests` (4 cases) — throttle window, EMA lag on
  rate jump, convergence, negative-bytes guard.
- New `MessageSegmentTests` (9 cases) — balanced tags, streaming
  open, Qwen3 implicit opener, multiple blocks, edge cases.
- New `MLXSwiftEnginePreflightTests` (5 cases) — Gemma 4 MoE
  detection, dense-Gemma not flagged, Mixtral not flagged,
  nested `text_config`, missing config.
- HFDownloader gets one declared-sizes test; HEAD-fallback path
  verified at runtime (MockURLProtocol hangs on HEAD requests).

---

## [0.3.5] - 2026-04-17

CLI TUI refresh: the three dashboards (`macmlx pull` / `macmlx serve`
/ `macmlx run` interactive) now render via a tiny in-house ANSI
helper instead of a stub-linked SwiftTUI dependency. SwiftTUI has
been unmaintained for over a year and is incompatible with Swift 6
strict concurrency (its `View` protocol is `nonisolated`, clashes
with our `@MainActor` state classes). Shipped through v0.3.4 as a
zombie import; now cleanly removed.

### Added
- `macmlx-cli/Sources/macmlx/Shared/CLITerm.swift` — small ANSI
  toolkit: colour / bold / dim helpers, TTY detection (so piped
  output stays clean), unicode block progress bars with sub-cell
  precision (U+258x), and box-drawing header/footer for section
  titles.
- **Unicode progress bar** on `macmlx pull` — replaces the bare
  `[2/4]  47%` text line with `[2/4] ██████████▌           47%`
  plus speed and ETA. Sub-cell precision means the bar advances
  smoothly even for small percentage deltas.
- **Boxed startup banner** on `macmlx serve` — coloured key/value
  rows inside a unicode box, including the health and status URLs.
- **Tidier REPL header** on `macmlx run` interactive — dimmed hint
  line, cyan `>` prompt, red error lines.
- `.claude/features/cli-tui.md` gains a "Historical: SwiftTUI"
  section documenting the decision, rationale, and reintroduction
  criteria.

### Removed
- **SwiftTUI** dependency from `macmlx-cli/Package.swift`. No code
  actually used it; the three `_*View` stubs were linker decoys.
- The `_PullDashboardView` / `_ServeDashboardView` / `_ChatTUIView`
  stub types that existed only to keep the SwiftTUI product
  referenced.
- **PulseUI** + **PulseProxy** package products from the
  `macMLX.xcodeproj` (maintainer action). Added in v0.3.4 on the
  assumption we'd drop in `PulseUI.ConsoleView`, but Pulse 5.x
  gates that view behind `#if !os(macOS)` — we wrote the Logs
  viewer natively against `LoggerMessageEntity` instead and never
  imported PulseUI from Swift. Pulse core stays (still the backing
  store for `LogManager` + the Logs tab).

### Notes
- GitHub issue #18 closed — the underlying ask (real live CLI
  dashboards) is satisfied by the CLITerm-based rendering without
  requiring SwiftTUI as the vehicle. If SwiftTUI resumes
  development AND ships Swift 6 compatibility we can revisit for
  richer full-screen dashboards.
- CLI tests: 16/16 green. `macmlx list` smoke-tested against a
  local model directory. Core tests: 90/90 still green.

---

## [0.3.4] - 2026-04-17

Logs tab (#16). A native macOS log viewer built on top of Pulse's
`LoggerStore` Core Data stack — every log line from the coordinator,
downloader, HTTP server, benchmark runner, and chat pipeline shows
up here with search + level filter.

### Added
- **Logs tab** in the sidebar (`list.bullet.rectangle` icon, between
  Benchmark and Settings). SwiftUI `Table` with columns for time,
  coloured level badge, category, and message. Search field +
  level picker in the toolbar. "Clear" button for wiping the
  on-disk store.
- `LogManager.store` is now `public nonisolated let` (was private)
  so the UI can read the backing `LoggerStore` synchronously from
  `@MainActor` without an await hop.

### Notes on PulseUI
- Originally planned to drop in `PulseUI.ConsoleView` directly; Pulse
  5.x `ConsoleView` turns out to be `#if !os(macOS)`-gated. PulseUI
  on macOS is intended for use with the standalone *Pulse for Mac*
  app (users export a `.pulse` bundle and open it there). We built
  the viewer natively against `LoggerMessageEntity` instead so the
  Logs tab works in-process with no external app required. PulseUI
  stays linked to the app target for any macOS-compatible pieces
  we might adopt in a future Pulse release.

---

## [0.3.3] - 2026-04-17

Server-side change: the OpenAI-compatible `/v1/chat/completions`
endpoint now auto-loads any locally-downloaded model on demand, so
external clients (Claude Code, Cursor, Continue, raw `curl`, anything
OpenAI-compatible) can point at `localhost:8000` and pick whichever
model they need without pre-arranging a manual load.

### Added
- **Cold-swap model loading** on `HummingbirdServer` (v0.3.3). When a
  chat completion arrives naming a model that isn't currently loaded,
  the server resolves the ID against the user's model directory,
  unloads whatever was current, loads the requested model, and
  proceeds — no observable difference to the client except the first
  request on a cold model takes longer.
- `HummingbirdServer.ModelResolver` typealias + second init that
  accepts it. Existing single-arg init still works (cold-swap off —
  back-compat with any caller that relied on the pre-v0.3.3 "only
  explicitly-loaded models answer" contract).
- Concurrency guardrail: an actor-local `loadInFlight: Task` serialises
  concurrent cold-swap requests. Two requests for the same not-yet-
  loaded model share a single load (no double disk-read); requests
  for different models queue cleanly instead of thrashing. Matches
  strategy "a — serialise + wait" from the v0.3 UX plan.
- +2 tests (`coldSwapLoadsResolvedModel`, `coldSwapReturns404WhenModelMissing`).
  Both use a foreground HTTP round-trip through a `StubInferenceEngine`
  + closure resolver, on ports 19_200 / 19_210.

### Changed
- `macmlx serve` (CLI) now wires the resolver up against
  `ModelLibraryManager.scan(settings.modelDirectory)` so a running
  serve sees every locally-downloaded model, not just the one passed
  to `--model`. `--model` still loads-at-startup for cold-start
  latency; omitting it now works (first chat completion loads on
  demand).
- `GET /v1/models` behaviour note: still returns only the loaded
  model (compatibility), but the cold-swap feature means clients can
  ask for any model by ID anyway. Future change: populate this
  endpoint from the resolver's full list. Deferred — not every
  resolver can enumerate.

### Error shape
Missing model → HTTP 404 with OpenAI-style
`{"error": {"code": "model_not_found", …}}` body.
Load failure → HTTP 500 with `load_failed`.

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
