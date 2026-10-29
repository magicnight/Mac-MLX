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
  prompt (joined with a blank line); a later one is folded into the next user
  turn (`MidConversationSystemFolding`). Consecutive `function_call` /
  `custom_tool_call` items, with an assistant message directly before them,
  form one assistant turn; `*_call_output` items become tool turns.
  `reasoning` items are dropped. Content parts: `input_text`, `output_text`,
  `text`, `refusal`, and `input_image` with a base64 data URL.
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
