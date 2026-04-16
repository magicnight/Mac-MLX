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
  "temperature": 0.7,
  "max_tokens": 2048
}
```

Streaming response (SSE):
```
data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hi"},"index":0}]}

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
    "total_tokens": 20
  }
}
```

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

Request:
```json
{"model": "Qwen3-8B-4bit"}
```

Response:
```json
{"status": "unloaded"}
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
  "tokens_generated_total": 18420
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
