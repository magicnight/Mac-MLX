# Feature: Chat UI

## Overview

A native SwiftUI chat interface for direct interaction with loaded models.
Clean, focused, distraction-free.

## v0.1 Scope

### Layout

```
┌─────────────────────────────────────────────┐
│ Toolbar: [Model Selector ▼]    [New Chat +] │
├─────────────────────────────────────────────┤
│                                             │
│  ┌───────────────────────────────────────┐  │
│  │ System: You are a helpful assistant.  │  │
│  └───────────────────────────────────────┘  │
│                                             │
│                        ┌─────────────────┐  │
│                        │ Hello! How can  │  │
│                        │ I help you?     │  │
│                        │ 10:32 · 12 tok  │  │
│                        └─────────────────┘  │
│  ┌─────────────────┐                        │
│  │ What is MLX?    │                        │
│  │ 10:33           │                        │
│  └─────────────────┘                        │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │ Generating...                       │    │
│  │ █                                   │    │
│  └─────────────────────────────────────┘    │
│                                             │
├─────────────────────────────────────────────┤
│  ┌─────────────────────────────────────┐    │
│  │ Message...                    [Send]│    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

### Chat State

```swift
@Observable
final class ChatViewModel {
    var messages: [ChatMessage] = []
    var inputText: String = ""
    var isGenerating: Bool = false
    var selectedModel: String = ""
    var systemPrompt: String = "You are a helpful assistant."
    
    func send() async { ... }
    func stopGeneration() { ... }
    func clearHistory() { ... }
}

struct ChatMessage: Identifiable {
    let id: UUID
    let role: MessageRole      // .user | .assistant | .system
    var content: String
    let timestamp: Date
    var tokenCount: Int?
    var isGenerating: Bool
}
```

### Streaming Display

Stream tokens as they arrive via SSE:
- Append each token chunk to the current assistant message content
- Show a blinking cursor `█` while generating
- Auto-scroll to bottom as content grows
- Use `ScrollViewReader` to programmatically scroll

```swift
func streamResponse(for request: ChatRequest) async {
    let stream = apiClient.streamChatCompletion(request)
    var assistantMessage = ChatMessage(role: .assistant, content: "", isGenerating: true)
    messages.append(assistantMessage)
    
    do {
        for try await chunk in stream {
            assistantMessage.content += chunk.delta
            // Update message in array
        }
        assistantMessage.isGenerating = false
    } catch {
        assistantMessage.content = "Error: \(error.localizedDescription)"
        assistantMessage.isGenerating = false
    }
}
```

### Message Rendering

- User messages: right-aligned, accent color background, white text
- Assistant messages: left-aligned, system secondary background
- System prompt: top of conversation, subtle styling, editable on click
- Code blocks: monospaced font, slightly different background, copy button
- Markdown rendering: bold, italic, inline code (basic support only in v0.1)

### Input Area

- Multi-line `TextEditor` that grows with content (max 5 lines)
- `Cmd+Return` to send
- `Return` adds newline
- Disable input and show spinner while generating
- **Stop** button replaces Send while generating

### Model Selector

Dropdown in toolbar showing loaded models.
If no model loaded, show "No model loaded" with link to Model Library.

### Token Counter

Show token count per message (from API usage response).
Show running total in toolbar subtitle.

### Conversation Management

- Each conversation is a separate array of messages in memory
- **New Chat** button clears history and starts fresh
- No conversation persistence in v0.1 (memory only)

## Keyboard Shortcuts

| Action | Shortcut |
|--------|---------|
| Send message | Cmd+Return |
| New chat | Cmd+N |
| Stop generation | Escape |
| Focus input | Cmd+L |

## Out of Scope (v0.1)

- Conversation history persistence
- Export conversation
- Image / file attachments (VLM support — v0.2)
- Multiple conversation tabs
- Message editing / regeneration
- Full markdown rendering (tables, lists)
- Token usage graph
