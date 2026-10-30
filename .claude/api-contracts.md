# API Contracts

## Overview

Swift and Python communicate exclusively via HTTP on `localhost`.
The protocol is OpenAI API compatible, with minimal custom extensions.

Base URL: `http://127.0.0.1:{port}`  
Default port: `8000`

---

## Standard Endpoints (OpenAI Compatible)

### POST /v1/chat/completions

Request:
```json
{
  "model": "Qwen3-8B-4bit",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello"}
  ],
  "stream": true,
  "stream_options": {"include_usage": true},
  "temperature": 0.7,
  "max_tokens": 2048
}
```

Streaming response (SSE). Without `stream_options.include_usage` no chunk
carries a `usage` key; with it, every chunk carries `"usage": null` and one
usage-only chunk (empty `choices`) precedes `[DONE]`:
```
data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hi"},"index":0}],"usage":null}

data: {"id":"...","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":12,"completion_tokens":8,"total_tokens":20,"prompt_tokens_details":{"cached_tokens":10}}}

data: [DONE]
```

Non-streaming response:
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "Qwen3-8B-4bit",
  "choices": [{
    "message": {"role": "assistant", "content": "Hi there!"},
    "finish_reason": "stop",
    "index": 0
  }],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 8,
    "total_tokens": 20,
    "prompt_tokens_details": {"cached_tokens": 10}
  }
}
```

`prompt_tokens_details.cached_tokens` is how many prompt tokens the prompt
cache served (0 on a miss); `prompt_tokens` includes them, as OpenAI counts.
The Anthropic `/v1/messages` shape reports the same figure as
`cache_read_input_tokens` and, as Anthropic counts, takes it out of
`input_tokens` — the two sum to the prompt. Both are withheld from a request
that carries an `Origin` header (a cross-origin browser caller), because the
figure is a prefix oracle against every cached prompt.

`max_completion_tokens` is accepted as the newer spelling of `max_tokens` and
wins when both are present (the precedence mlx-lm's and vLLM's servers use).

An assistant message's `reasoning_content` (else `reasoning`, OpenRouter's
spelling; a non-string value is ignored) is that turn's reasoning: it reaches
the chat template as the turn's `reasoning_content`, which thinking templates
(Qwen3.x, Mellum2, GLM, Gemma 4) render for the turns after the last user
query (Qwen3.6 for every turn with the template kwarg
`preserve_thinking: true`). On a user, tool or system message it is ignored.
Not rendered on a VLM, whose processor builds the prompt from messages without
it, nor on the Ollama endpoints.

### POST /v1/responses

OpenAI Responses API, stateless (`Server/HummingbirdServer+Responses.swift`,
`Server/Responses/`). The request is translated onto the same
`GenerateRequest` chat completions builds and generated on the single-stream
path (never the batched one).

Request (the fields acted on; anything else is accepted and ignored):
```json
{
  "model": "Qwen3.6-27B-4bit",
  "instructions": "optional",
  "input": [
    {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "..."}]},
    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Run echo"}]},
    {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "..."},
    {"type": "function_call", "name": "exec_command", "arguments": "{\"cmd\":\"echo hi\"}", "call_id": "call_1"},
    {"type": "function_call_output", "call_id": "call_1", "output": "hi\n"}
  ],
  "tools": [
    {"type": "function", "name": "exec_command", "description": "...", "parameters": {"type": "object"}},
    {"type": "namespace", "name": "multi_agent_v1", "tools": [{"type": "function", "name": "wait_agent", "parameters": {}}]},
    {"type": "custom", "name": "apply_patch", "description": "..."},
    {"type": "web_search"}
  ],
  "tool_choice": "auto",
  "text": {"format": {"type": "json_schema", "name": "x", "schema": {}, "strict": true}},
  "reasoning": {"effort": "minimal"},
  "max_output_tokens": 2048,
  "temperature": 0.7,
  "top_p": 0.95,
  "stream": true
}
```

- `input`: a string (one user turn) or items. `system`/`developer` messages
  before the first other item, after `instructions`, become the system
  prompt (joined with a blank line); a later one is folded in place
  (`MidConversationSystemFolding`): appended to the user turn just before
  it, or, after an assistant or tool turn, opening a user turn there that a
  user message right after it joins. Earlier turns never change, so folding
  does not force them to be prefilled again. On a model whose cache has a
  layer that cannot be trimmed (hybrid, e.g. Qwen3.6; sliding-window, e.g.
  Mellum2), the engine also stores the cache at the end of a request's last
  message before the generation prompt (its history boundary), so the next
  turn reuses everything up to there. Not stored: when the boundary adds
  fewer than 256 tokens to what the cache already served, and on the paths
  that bypass the prompt cache (VLM, `kv_bits`, adapters, speculative
  decoding, batched decode). When the user turn merged into an opened one carries
  images, a VLM may render them before the text, so that last message's
  tokens can shift; earlier messages still do not. Consecutive `function_call` /
  `custom_tool_call` items, with an assistant message directly before them,
  form one assistant turn; `*_call_output` items become tool turns.
  A `reasoning` item's text is the reasoning of the assistant turn the next
  `message` or call item builds (rendered as on chat completions): its
  `reasoning_text` content parts, else the text its `encrypted_content`
  carries when macMLX wrote it (see the response below). `summary` is never
  read, and another provider's `encrypted_content` carries nothing; a user,
  system or tool item before the assistant turn drops it, and several items
  before one turn join with a blank line. Content parts: `input_text`,
  `output_text`, `text`, `refusal`, and `input_image` with a base64 data URL.
- `tools`: `function` and `custom` tools are offered as chat function specs (a
  `custom` tool takes one required string `input`); a `namespace`'s tools are
  flattened under their bare names, qualified as `<namespace>__<name>` only
  when two would collide. Hosted tool types are not offered and not rejected.
  `tool_choice: "none"` offers no tools; any other value offers all.
- `text.format` is rebuilt as the chat `response_format` and decoded by
  `ResponseFormatDecoder`; with tools or images it is a 400, as on chat.
- `reasoning.effort` `"none"`/`"minimal"` sets `enable_thinking=false`.
- `max_output_tokens` absent → 32768 (`responsesDefaultMaxOutputTokens`), not
  the chat path's 2048: Codex never sends it, and a thinking model spends
  2048 before its tool call. An explicit value is honored.
- Body limit 64 MB (`responsesBodyLimitBytes`, 413 `request_too_large`
  above it), not the 2 MB route default chat uses: the whole conversation
  is replayed every turn.
- A custom tool's `format.definition` is appended to its description as
  `Input format (<syntax>):\n<definition>`.
- In a tool output, a non-text part becomes a placeholder line
  (`[image omitted: macMLX does not pass images in tool results]`,
  `[<type> omitted]`); an image on a system/developer item becomes
  `[image omitted from a system message]`.
- 400 (`invalid_request_error`): bad JSON, missing `model` or `input`,
  non-null `previous_response_id`, `background: true`, `item_reference`
  items, `input_file`/`input_audio` parts, an `input_image` that is a URL or
  `file_id`, an unknown item type, a broken tool-call history. An unknown
  model is a 404 (before the stream opens, when streaming).

Response object (the non-streaming body adds `output_text`):
```json
{
  "id": "resp_…", "object": "response", "created_at": 1761700000,
  "status": "completed", "model": "Qwen3.6-27B-4bit",
  "output": [
    {"type": "reasoning", "id": "rs_…", "summary": [{"type": "summary_text", "text": "…"}],
     "content": [{"type": "reasoning_text", "text": "…"}]},
    {"type": "message", "id": "msg_…", "status": "completed", "role": "assistant",
     "content": [{"type": "output_text", "text": "…", "annotations": []}]},
    {"type": "function_call", "id": "fc_…", "call_id": "call_…", "name": "wait_agent",
     "namespace": "multi_agent_v1", "arguments": "{}", "status": "completed"}
  ],
  "usage": {"input_tokens": 12, "input_tokens_details": {"cached_tokens": 10},
            "output_tokens": 8, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 20},
  "error": null, "incomplete_details": null,
  "instructions": null, "tools": [], "tool_choice": "auto", "parallel_tool_calls": true,
  "temperature": 0.7, "top_p": 0.95, "max_output_tokens": 2048,
  "text": {"format": {"type": "text"}}, "reasoning": {"effort": null, "summary": null},
  "store": false, "previous_response_id": null, "metadata": {}
}
```

With `include: ["reasoning.encrypted_content"]` (Codex sends it) the reasoning
item also carries `"encrypted_content": "macmlx.r1.<base64 of the reasoning>"`,
in the non-streaming body and in the stream's `output_item.done` and terminal
response. It is not encrypted — the server is local and the client holds the
text anyway — only opaque to a client that replays it verbatim, and it lets
the next turn recover the reasoning when `content` was not replayed.

The message item is left out when the model only called tools. A freeform
tool's call is a `custom_tool_call` (`ctc_…`) with `input` instead of
`arguments`. `status` is `completed` for a stop or tool call, `incomplete`
with `incomplete_details.reason: "max_output_tokens"` at the length cap, and
`failed` with `error: {"code", "message"}` when generation fails or stalls
(still HTTP 200): `code` is `server_error`, or `model_not_found` /
`load_failed` for a cold swap that fails after the stream opened. The engine
reports no distinct context-overflow error, so there is no
`context_length_exceeded`. A message item cut off by the length cap or a
failure has `status: "incomplete"`. `cached_tokens` is 0 for a request with
an `Origin` header, as on chat. Requests are served one at a time (the
generation lock), so concurrent agents queue.

Streaming (`text/event-stream`, frames `event: <type>\ndata: <json>\n\n`, every
payload with `type` and a `sequence_number` from 0, no `[DONE]`). Every event
type starts with `response.`, left off below:
```
created → in_progress
reasoning: output_item.added → reasoning_summary_part.added → content_part.added
           → (reasoning_summary_text.delta, reasoning_text.delta)*
           → reasoning_summary_text.done → reasoning_summary_part.done
           → reasoning_text.done → content_part.done → output_item.done
text:      output_item.added → content_part.added → output_text.delta*
           → output_text.done → content_part.done → output_item.done
each call: output_item.added → function_call_arguments.delta → .done → output_item.done
           (custom_tool_call_input.delta / .done for a custom tool)
then one of completed | incomplete | failed
```
`created` and `in_progress` are sent before the generation lock is taken;
tool calls follow the text because the engine reports them on the terminal
chunk.

### POST /v1/messages

Anthropic Messages API (`handleAnthropicMessages` in
`Server/HummingbirdServer.swift`), translated onto the same `GenerateRequest`
chat completions builds and generated on the single-stream path. A query
string (Claude Code posts to `/v1/messages?beta=true`) and the
`anthropic-version` / `anthropic-beta` headers are ignored.

Request (the fields acted on):
```json
{
  "model": "Qwen3.6-27B-4bit",
  "max_tokens": 32000,
  "system": [{"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}],
  "messages": [
    {"role": "user", "content": [{"type": "text", "text": "Run echo hi"}]},
    {"role": "system", "content": [{"type": "text", "text": "<system-reminder>…</system-reminder>"}]},
    {"role": "assistant", "content": [
      {"type": "thinking", "thinking": "", "signature": "macmlx.r1.…"},
      {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "echo hi"}}]},
    {"role": "user", "content": [
      {"type": "tool_result", "tool_use_id": "toolu_1", "content": "hi", "is_error": false}]}
  ],
  "tools": [{"name": "Bash", "description": "...", "input_schema": {"type": "object"}}],
  "tool_choice": {"type": "auto"},
  "thinking": {"type": "adaptive", "display": "omitted"},
  "temperature": 0.7,
  "top_p": 0.95,
  "stream": true
}
```

- `max_tokens` is required (400 without it).
- A top-level `system` text block (blocks form only; a plain-string
  `system` is left alone) whose text starts (after leading
  whitespace) with `x-anthropic-billing-header:` is dropped: Claude Code's
  billing metadata, which differs between sessions and would keep a new
  session from reusing a cached system prompt.
- `system` (string or text blocks, joined with a newline) and the
  `role:"system"` turns in `messages` are placed by
  `MidConversationSystemFolding`, `system` first: system turns before the
  first other turn join the system prompt; a later one is folded in place
  as on `/v1/responses`. A system turn's image block becomes
  `[image omitted from a system message]`.
- User turns: `text` and base64 `image` blocks; each `tool_result` becomes a
  tool turn ahead of the turn's text. Assistant turns: `text` and `tool_use`,
  and `thinking` blocks, whose `thinking` text — or, when it is empty, the
  text a macMLX `signature` carries — is the turn's reasoning (rendered as on
  chat completions; several blocks join with a blank line).
  `redacted_thinking` and another provider's signature carry nothing.
- `tool_choice`: `none` offers no tools; `auto`, `any` and `tool` (which
  needs a `name`) offer them all, not enforced; any other type is a 400.
  `disable_parallel_tool_use` is ignored.
- `thinking.type`: `disabled` sets `enable_thinking=false` over the per-model
  template kwargs; `enabled` / `adaptive` keep the model's default and return
  the reasoning as a `thinking` block, with empty `thinking` text when
  `display` is `"omitted"` (Claude Code's setting, and what Anthropic returns
  for it). `budget_tokens` is not enforced. Without `thinking`, reasoning is
  dropped.
- Accepted and ignored: `metadata`, `output_config` (`effort` is not mapped),
  `context_management`, `safeguards`, `cache_control`, `top_k`,
  `stop_sequences`, `service_tier`.
- Body limit 64 MB (`anthropicBodyLimitBytes`), not the 2 MB route default
  chat uses: Claude Code replays the whole session every turn.
- Errors use Anthropic's envelope, `{"type": "error", "error": {"type",
  "message"}}`: 400 `invalid_request_error` (bad JSON, missing `max_tokens`,
  bad `tool_choice`, a broken tool history), 413 `request_too_large` (a body
  over the limit), 404 `not_found_error` (unknown
  model, before the stream opens when streaming), 500 `api_error` (a failed
  load or generation), 504 on a stall. The bearer-auth 401 keeps the OpenAI
  shape.

Response content, in order: a `thinking` block (`{"type": "thinking",
"thinking": "…", "signature": "macmlx.r1.<base64 of the reasoning>"}`, only
as above and when the model reasoned; the signature is not a signature but
the reasoning, opaque to a client that replays the block, so the next turn
recovers it even when the text was omitted), a `text` block
(starting at the first non-whitespace answer text; left out when the model
only called tools), then one `tool_use` block per call. `usage` is
`{"input_tokens", "output_tokens", "cache_read_input_tokens"}`, with
`input_tokens` the uncached part of the prompt; `cache_read_input_tokens` is
left out (and `input_tokens` is the whole prompt) for a request carrying an
`Origin` header, as on chat.

Streaming events: `message_start` (usage `input_tokens: 0` — the count is
known only at the end), then each block as `content_block_start` → deltas →
`content_block_stop`, indexed in emission order — thinking
(`thinking_delta`*, then one `signature_delta`; with `display: "omitted"` no
`thinking_delta`, the block sent once the reasoning has ended), text (`text_delta`*),
each tool call (one `input_json_delta`) — then `message_delta` carrying
`stop_reason` and the real `usage`, then `message_stop`. No `ping`. A
failure once the stream is open is an in-band `event: error`.

### POST /v1/messages/count_tokens

Anthropic token counting (`Server/HummingbirdServer+CountTokens.swift`). The
body is a `/v1/messages` body; `max_tokens` is optional and `stream` is
ignored. It is built into the same `GenerateRequest` (turns, system
placement, tools, `thinking`, images) and counted by
`InferenceEngine.countPromptTokens`, which renders the prompt exactly as
generation tokenizes it.

```json
{"input_tokens": 1234}
```

The model is resolved and loaded as for a generation, under the generation
lock, so a count waits behind an in-flight generation. The body limit and
errors are `/v1/messages`'s: 400, 413, 404 for an unknown model, and 501 `api_error` ("token
counting is not supported by this engine") for an engine without counting.

### GET /v1/models

Response:
```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen3-8B-4bit",
      "object": "model",
      "owned_by": "local"
    }
  ]
}
```

---

## Custom Extensions (prefixed with x-)

These are non-standard extensions specific to this app.

### POST /x/models/load

Load a model into memory.

Request:
```json
{"model_path": "/Users/user/models/Qwen3-8B-4bit"}
```

Response:
```json
{"status": "loaded", "model": "Qwen3-8B-4bit", "load_time_ms": 3240}
```

### POST /x/models/unload

Request (optional; `model` names the chat model, or the resident reranker
or embedder — the two auxiliary engines the pool budgets alongside chat
models since #130):
```json
{"model": "Qwen3-8B-4bit"}
```

Response:
```json
{"status": "unloaded"}
```

`model` is matched exactly (no aliases); the reranker and embedder slots are
checked before the chat model. A `model` that is neither the active chat
model nor the resident reranker or embedder is a 404 (`model_not_loaded`); a
body that is not a JSON object with an optional string `model` is a 400.
Unloading the reranker or the embedder answers with the id it dropped:
```json
{"status": "unloaded", "model": "ms-marco-MiniLM-L-6-v2"}
```

### GET /x/status

Returns current server state. Swift polls this every 5 seconds.

Response:
```json
{
  "status": "running",
  "loaded_model": "Qwen3-8B-4bit",
  "memory_used_gb": 8.2,
  "memory_total_gb": 48.0,
  "uptime_seconds": 3600,
  "requests_total": 42,
  "tokens_generated_total": 18420,
  "reranker": "ms-marco-MiniLM-L-6-v2",
  "embedder": null
}
```

### GET /health

Simple health check. Swift calls this to detect if backend is ready.

Response:
```json
{"status": "ok"}
```

---

## Process Startup Protocol

Swift starts the Python process and waits for readiness:

1. Swift spawns Python subprocess
2. Python prints `READY\n` to stdout when HTTP server is accepting connections
3. Swift reads stdout, detects `READY`, marks service as running
4. If `READY` not received within 30 seconds, Swift kills process and reports error
5. Swift polls `/health` every 5 seconds during operation

---

## Error Format

All errors follow OpenAI error format:

```json
{
  "error": {
    "message": "No model is currently loaded",
    "type": "invalid_request_error",
    "code": "model_not_loaded"
  }
}
```

HTTP status codes:
- `200` Success
- `400` Bad request (invalid parameters)
- `503` Model not loaded / service unavailable
- `500` Internal server error

---

## Swift Client Interface

```swift
protocol InferenceAPIClientProtocol {
    func listModels() async throws -> [RemoteModel]
    func loadModel(path: String) async throws -> LoadModelResponse
    func unloadModel(name: String) async throws
    func getStatus() async throws -> ServerStatus
    func chatCompletion(_ request: ChatRequest) async throws -> ChatResponse
    func streamChatCompletion(_ request: ChatRequest) -> AsyncThrowingStream<ChatChunk, Error>
}
```

This protocol allows mocking in tests and future backend substitution.
