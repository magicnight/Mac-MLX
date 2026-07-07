# macMLX ↔ oMLX Parity Roadmap (v0.5.1 → v0.8.0)

Written 2026-05-11 after surveying oMLX v0.4.4 (17.5k★, +65% in
3 months) against macMLX v0.5.0. Goal: close the visible feature
gap while keeping the constraints below intact.

## Hard constraints (do not break)

- **Swift-native.** No Python runtime, no spaCy, no `venvstacks`.
  Users pick macMLX because there's nothing to install and the app
  is a single 45–60 MB DMG. Every parity feature must be pure Swift
  (Foundation + MLX + our existing SPM deps).
- **Apache-2.0 compatible.** No GPL-3 transitives. Rules out
  Kokoro's `espeak-ng` chain and any custom kernels that require a
  copyleft dep.
- **Platform floor stays macOS 14 + Apple Silicon.** No stretch to
  older macOS or Intel.
- **App Sandbox stays off.** Reverting sandbox was v0.3.6's biggest
  UX unlock; don't undo it just to ship a feature.
- **v0.4/v0.5 architecture stays intact.** ModelPool + KV tiered
  cache + LoRA + MCP client config already ship. Extensions layer on
  top; no rewrite passes.

## What we deliberately don't chase (documented no-go)

| oMLX feature | Why we skip |
|---|---|
| `mlx-lm.BatchGenerator` continuous batching | Blocked on upstream `mlx-swift-lm` porting `BatchGenerator` + `BatchKVCache` from Python. Documented in v0.5 plan Track C. Watch, don't fork. |
| Custom kernels (GLM-5.2 / MiniMax M3 sparse attention) | Requires C++/Metal kernel ownership. Upstream `mlx-swift` doesn't ship them; writing our own is a specialist track, not a UX gap. |
| `oQe` imatrix quantization | Offline pipeline territory (Python `mlx_lm` quant tool already fills the ecological niche). Users can quantize with Python and drop the artefact into `~/.mac-mlx/models/`. |
| Kokoro TTS | Needs `espeak-ng` (GPL-3). Use Marvis / Chatterbox instead (already in v0.6 plan). |
| spaCy G2P | Python-only. No Swift equivalent worth porting. |
| Full-featured web `/admin` dashboard | We have a real Mac app. A web dashboard would just duplicate SwiftUI surfaces. Keep read-only `/x/status` for automation, no web UI. |

Every "don't do" here is a boundary we'd cross only under a
specific external demand.

---

## v0.5.1 — Server hardening + API compat (small; ~1 week)

**Theme:** close the loose ends of the v0.5.0 HTTP surface. All
changes are inside `HummingbirdServer` + Settings + `ModelParameters`;
no engine work, no UI surgery.

### Track A1 — `--api-key` bearer auth
- `Settings.serverAPIKey: String?` — nil = open (localhost dev default).
- `HummingbirdServer` middleware: when non-nil, require
  `Authorization: Bearer <key>` on every `/v1/*` and `/api/*` route.
  `GET /health` stays open.
- CLI: `macmlx serve --api-key <k>`; Settings toggle.
- Tests: 401 on missing / wrong key, 200 on correct key, no-auth
  path unchanged when key is nil.

### Track A2 — Anthropic `POST /v1/messages` (drop-in Anthropic)
- New handler translating Anthropic request shape → `GenerateRequest`,
  and `GenerateChunk` stream → Anthropic event stream (`message_start`
  / `content_block_delta` / `message_delta` / `message_stop`).
- Handles `system` field (Anthropic puts it at top level, not in
  `messages`).
- Handles Anthropic vision (`image` content block → `ImageAttachment`,
  same base64 → tmpfile pipe as OpenAI multimodal).
- Handles `stream_options` mirror + optional `thinking` param.
- Tests: fixture translation both directions.

### Track A3 — Model aliases
- `ModelParameters.alias: String?` — user-visible name.
- `/v1/models` returns `{ id: alias ?? directoryName, root: directoryName }`.
- Chat completions accept both `alias` and `directoryName` in `model`
  field.
- No engine change — pure resolver-layer glue.

### Track A4 — Per-model TTL
- `ModelParameters.idleTTLSeconds: Int?` — nil = disabled.
- `ModelPool` grows a timer; on eviction pass, models past their
  TTL get unloaded even if pool byte budget isn't reached.
- Pinned models bypass TTL (matches existing pin semantics).
- Tests: fake clock; verify eviction triggers.

### Track A5 — Chat template kwargs
- `ModelParameters.templateKwargs: [String: JSONValue]` (JSON-blob
  free-form).
- Passed through `TokenizerBridge.applyChatTemplate` as
  `additionalContext`. Already-supported channel; we're just wiring
  the UI + persistence.
- Parameters Inspector grows a key/value editor.
- Common preset: `{"thinking": true}` for Qwen3.

**Deliverables:** 5 small PRs. All Core-side. No SwiftUI surgery except
the small kwargs editor. Ships v0.5.1.

---

## v0.5.2 — Embeddings + Reranker (RAG unlock)

**Theme:** open the RAG use-case. Half of oMLX's recent star growth
came from `bge-m3` + rerank users. `mlx-swift-lm` already ships an
`MLXEmbedders` product; wiring it takes ~200 LOC.

### Track B1 — Embedding engine
- New `EmbeddingEngine` actor conforming to a new `EmbeddingModel`
  protocol (not `InferenceEngine` — separate lifecycle).
- Backed by `MLXEmbedders` (BERT / BGE-M3 / ModernBERT auto-detected
  from `config.json`).
- `ModelFormat.embedder` — third format after `.mlx / .mlxVLM`.
  `ModelLibraryManager.scan` upgrades based on `architectures`
  / `model_type`.
- `EmbeddingPool` sibling to `ModelPool` (they don't share weights).

### Track B2 — `/v1/embeddings` endpoint
- OpenAI-shaped: `{ model, input, encoding_format }`. Returns
  `{ data: [{ embedding: [Float], index }], model, usage }`.
- Handles single-string + array-of-strings inputs.
- Tests: mock embedder returning fixed vector; verify wire shape.

### Track B3 — Reranker engine
- New `RerankerEngine` (`ModernBERT` / `XLM-RoBERTa`).
- `ModelFormat.reranker` — fourth format tag.
- `/v1/rerank` endpoint: `{ model, query, documents } → { results:
  [{ index, relevance_score }] }`.

### Track B4 — Settings + model tab UI
- Models tab grows section badges (Text / Vision / Embedder /
  Reranker) so the user knows what each row does.
- No new tab; reuses the existing Models list.

**Deliverables:** 4 PRs. Ships v0.5.2.

---

## v0.5.3 — Tool calling parsers + MCP client wire-up

**Theme:** close the tool-calling loop. `MCPClientPool` (merged to
main via PR #45, 2026-07-06) provides the subprocess pool. This minor
connects it to chat completions.

### Track C1 — Per-family tool-call parsers
- New `ToolCallParser` protocol with concrete parsers for:
  - `LlamaQwenParser` — JSON `<tool_call>`
  - `Qwen35Parser` — XML `<function=...>`
  - `GemmaParser` — `<start_function_call>`
  - `GLMParser` — `<arg_key>/<arg_value>`
  - `MiniMaxParser` — `<minimax:tool_call>`
  - `MistralParser` — `[TOOL_CALLS]`
  - `KimiK2Parser` — `<|tool_calls_section_begin|>`
  - `LongcatParser` — `<longcat_tool_call>`
- Auto-detected from tokenizer chat template hints or model
  architecture.
- Emits `[ToolCall(name: String, arguments: [String: Value])]`.
- Tests: fixture output from each family → structured tool calls.

### ~~Track C2 — MCP subprocess pool (unstash + finish)~~ ✅ DONE
- ~~Resume the `feat/v0.5-mcp-client-pool` work~~ **Shipped in PR #45
  (2026-07-06):** `MCPClientPool` actor — spawn via `Process`, pipes to
  `StdioTransport`, connectAll / listAllTools / callTool /
  disconnectAll, partial-failure tolerance. Includes two dead-server
  robustness fixes (SIGPIPE ignore; connect timeout + `disconnect()`
  against swift-sdk 0.12.1's pre-init-EOF busy-loop, upstream PR #221
  still open).
- Still open (rolls into C3 or a hardening PR): integration test using
  `npx -y @modelcontextprotocol/server-everything`; zombie reaping /
  `waitUntilExit` on teardown; `forwardStderr` off the cooperative
  pool; `connectAll` signature honesty (review findings 2026-07-07).

### Track C3 — Chat completion tool_calls emit + MCP round-trip
- Chat completions extract `tools` from request (OpenAI schema).
- If configured with MCP servers, inject discovered tools into the
  prompt via chat template.
- If assistant emits tool_calls, route to MCP pool, feed result back
  as `tool`-role message, continue generation. Loop with max-depth 5.
- New Settings UI: MCP server list + connect status.

### Track C4 — Structured output (JSON Schema validate)
- `response_format: { type: "json_schema", schema: {...} }` on the
  request.
- Post-generation validate assistant content against schema; retry
  once if invalid, else surface error.
- Bundle a minimal JSON Schema validator (no deps — walk types
  ourselves).

**Deliverables:** 4 PRs. Ships v0.5.3.

---

## v0.6.0 — Speech I/O (revised v0.6 plan)

Original v0.6 plan stays valid. Two tracks (STT + TTS) via
`DePasqualeOrg/mlx-swift-audio`, skipping Kokoro. Audio settings
schema already landed in `main` (PR #41). Runtime work — mic capture,
push-to-talk button, speaker button, `STTService` / `TTSService` —
follows.

Plan: `docs/superpowers/plans/2026-05-10-v0.6.md`. No changes needed.

---

## v0.7.0 — Model profiles + admin UX + OCR

**Theme:** oMLX's most-liked UX features that aren't life-or-death.

### Track D1 — Model profiles
- `ModelProfile: { name: String, parameters: ModelParameters }`.
- `ModelParameters` grows `profiles: [ModelProfile] = []` +
  `activeProfileName: String?`.
- `/v1/models` exposes `<baseID>:<profileName>` as a virtual model
  (same underlying engine, per-profile parameter overlay applied at
  generate time).
- Parameters Inspector: profile picker + save-as button.

### Track D2 — Process memory enforcer
- `MemoryEnforcer` actor watches `os_proc_available_memory` +
  `mach_task_basic_info`.
- On breach (default: total RAM − 8 GB), triggers `ModelPool` eviction
  of LRU non-pinned models until below threshold.
- Kill switch: `Settings.processMemoryLimitGB` (nil = default).

### Track D3 — OCR model detection + prompt
- `ModelFormat.mlxVLM` gains an `.ocr` sub-flag (or `LocalModel.isOCR`
  computed).
- Auto-detected from model_type (`deepseek_ocr` / `dots_ocr` /
  `glm_ocr`) + tokenizer signature.
- OCR models get an optimised default prompt (`"Extract all visible
  text from this image."`) and a "Copy result" button on OCR
  responses in Chat.

### Track D4 — SwiftUI i18n (Localizable.xcstrings)
- Introduce `Localizable.xcstrings` — Xcode 15+ string catalog.
- Extract every user-facing string to the catalog.
- Ship base languages: EN, ZH, JA, KO. FR / RU / ES / PT-BR
  contribute-welcome via `.xliff` export.

**Deliverables:** 4 PRs (some big — i18n is a repo-sweep). Ships v0.7.0.

---

## v0.8.0 — Community Benchmarks + Auto-update polish

Original v0.7 plan (Community Benchmarks) moves here. Nothing new
against oMLX — this is our differentiator.

- Opt-in `POST /v1/benchmarks` endpoint receiving anonymised
  `BenchmarkResult` + `HardwareInfo`.
- Aggregated leaderboard by chip × model × quant × macOS.
- Served on `macmlx.app` and inside the app's Benchmark tab.

Plus small housekeeping:
- Sparkle EdDSA appcast polish (delta updates now that Sparkle 2.9.4
  supports them).
- Model update detection UI (v0.3.7's `.macmlx-meta.json` sidecar has
  a badge but no "Update all" batch action).

---

## Upstream watch list (no code, just tracking)

File / update `docs/roadmap-post-v0.3.6.md` whenever these move:

- **`mlx-swift-lm` `BatchGenerator`** — enables our v0.9 continuous
  batching wrapper (~300 LOC scheduler over their iterator).
- **`mlx-swift-audio`** — pin bumps whenever DePasqualeOrg cuts a
  new commit; unblocks v0.6 iteration speed.
- **`modelcontextprotocol/swift-sdk` 1.0** — take the API-stability
  bump when it lands; drop the `MCPBridge` wrapper if the SDK's
  public surface freezes.
- **`sparkle-project/Sparkle` 3.x** — when it ships, evaluate the
  delta-update API for smaller upgrade downloads.

---

## Sequencing rationale

- **v0.5.1 (server hardening) first** because `--api-key` is a
  security item — any user pointing the port at their LAN today is
  wide open. Ship this before we grow the audience further.
- **v0.5.2 (embeddings / reranker) second** because it opens the RAG
  world, which is where oMLX got most of its recent stars. Two clean
  wire-up PRs, big user-visible payoff.
- **v0.5.3 (tool-calling) third** because it makes MCP client actually
  useful. Once we have per-family parsers, MCP tool_calls flow
  through to external servers.
- **v0.6.0 (speech) fourth** as originally planned. It's a big
  feature but independent — can slip to v0.7.x if v0.5.x tail is
  longer than expected.
- **v0.7.0 (profiles + i18n + OCR) fifth** — these are UX polish that
  users notice but don't block adoption. i18n specifically is a repo
  sweep so we group it with other UI passes.
- **v0.8.0 (community benchmarks)** stays anchored to the "macMLX
  differentiator" thesis: no other MLX runtime crowdsources hardware
  data.

## Approval questions

If any of the following is a "no", say so and we retarget:

1. Is `--api-key` bearer auth ok as a v0.5.1 pre-req? (Otherwise we
   ship security later and open users to a scan window.)
2. Is Anthropic `/v1/messages` valuable enough to spend v0.5.1 on
   vs. holding for v0.5.3 with the tool-calling wire-up?
3. Are embedders + reranker a v0.5.2 must, or a v0.6 nice-to-have?
   (The "must" case is that RAG is where the star growth is.)
4. Tool-calling parsers — 8 families is a lot. Ship 3 (Llama/Qwen/
   Gemma covers ~90% of user models) in v0.5.3 and roll the rest to
   v0.5.4 patches?
5. i18n language set — EN/ZH/JA/KO baseline, FR/RU/ES/PT-BR
   contribute-welcome. Objections?
6. Anything on the "don't chase" list you actually want us to
   reconsider? (Kokoro / imatrix / web dashboard / continuous
   batching fork)
