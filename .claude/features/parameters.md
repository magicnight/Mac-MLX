# Feature: Model Parameters

## Overview

Per-model sampling parameter configuration.
Reference: oMLX per-model settings panel, improved with better UX.
Design: Inspector panel (right sidebar) or popover from model row.

## Parameter Schema

```swift
struct ModelParameters: Codable, Equatable {
    // Sampling
    var temperature: Double = 0.7        // 0.0 – 2.0
    var topP: Double = 0.9               // 0.0 – 1.0
    var topK: Int = 50                   // 1 – 200
    var repetitionPenalty: Double = 1.0  // 0.8 – 1.5
    var minP: Double = 0.0               // 0.0 – 1.0

    // Context
    var maxTokens: Int = 2048            // 128 – model max context
    var contextLength: Int = 4096        // 512 – model max

    // System
    var systemPrompt: String = ""

    // Model behavior
    var modelAlias: String = ""          // Custom API name
    var trustRemoteCode: Bool = false

    // Lifecycle
    var ttlMinutes: Int = 0              // 0 = never auto-unload
    var pinInMemory: Bool = false
}
```

## UI Layout

Parameters shown in Inspector panel (right sidebar, collapsible).
Follows macOS Settings / Xcode inspector aesthetic.

```
┌──────────────────────────────────────────┐
│  Model Parameters              [Reset]   │
│                                          │
│  ▼ Sampling                              │
│                                          │
│  Temperature        0.7  ────●────────  │
│  Top P              0.9  ──────────●──  │
│  Top K               50  ────●────────  │
│  Min P              0.0  ●────────────  │
│  Repetition Penalty 1.0  ────●────────  │
│                                          │
│  ▼ Context                               │
│                                          │
│  Max Tokens        2048  [     2048    ] │
│  Context Length    4096  [     4096    ] │
│                                          │
│  ▼ System Prompt                         │
│  ┌────────────────────────────────────┐  │
│  │ You are a helpful assistant.       │  │
│  │                                    │  │
│  └────────────────────────────────────┘  │
│                                          │
│  ▼ Advanced                              │
│                                          │
│  Model Alias      [________________]    │
│  TTL (min)        [   0  ]  0=never     │
│  Pin in memory    [ ○ ]                  │
│  Trust remote code[ ○ ]                  │
│                                          │
│  [Apply]          [Reset to Defaults]   │
└──────────────────────────────────────────┘
```

## UX Rules

- Sliders for continuous values (temperature, top_p, min_p)
- Steppers for discrete values (top_k, max_tokens)
- Changes are **live preview** — apply immediately if model is loaded
- Reset button restores model defaults (from model card if available)
- Parameters persist per-model in `~/.mac-mlx/model-params/{model-id}.json`
- Global defaults in Settings, per-model overrides in Inspector

## Tooltip Explanations

Every parameter has a `?` info button showing plain-English explanation:

| Parameter | Tooltip |
|-----------|---------|
| Temperature | Higher = more creative, lower = more focused. 0.7 is a good default. |
| Top P | Limits token selection to the most likely tokens summing to this probability. |
| Top K | Limits token selection to the K most likely tokens at each step. |
| Min P | Filters out tokens below this probability relative to the top token. |
| Repetition Penalty | Penalizes repeated tokens. Values above 1.0 reduce repetition. |
| Max Tokens | Maximum number of tokens to generate per response. |
| Context Length | Total context window size (prompt + response). |
| TTL | Auto-unload model after this many minutes of inactivity. 0 = never. |
| Pin in memory | Keep model loaded even when memory is low. |

---

## AI UI Design Tool Prompt

Use this prompt with **Figma AI**, **v0**, **Galileo**, or similar tools:

---

```
Design a macOS native parameter configuration inspector panel for an LLM inference app called macMLX.

CONTEXT:
- This is an Inspector panel that appears as the right sidebar of a 3-column macOS window
- The app uses native macOS design language (SwiftUI, macOS Ventura/Sonoma aesthetic)
- Target users: technical developers and AI enthusiasts
- The panel controls inference parameters for a locally running language model

VISUAL STYLE:
- Follow Apple Human Interface Guidelines strictly
- Use system fonts (SF Pro), system colors, native controls
- Match the aesthetic of Xcode's inspector panel or macOS System Settings
- Light mode primary, Dark mode must also look native
- Subtle section dividers, collapsible sections with disclosure triangles
- No custom colors except system accent color for interactive elements

LAYOUT (top to bottom):
1. Header: "Model Parameters" title (left) + "Reset" button (right, text button)
2. Section "Sampling" (collapsible):
   - Temperature: horizontal slider (0.0–2.0) with numeric value display right-aligned
   - Top P: horizontal slider (0.0–1.0) with numeric value
   - Top K: stepper control with numeric input (1–200)
   - Min P: horizontal slider (0.0–1.0) with numeric value
   - Repetition Penalty: horizontal slider (0.8–1.5)
   - Each row: label (left, 130pt fixed width) + control (right, fills remaining space) + value display
3. Section "Context" (collapsible):
   - Max Tokens: numeric text field with stepper arrows
   - Context Length: numeric text field with stepper arrows
4. Section "System Prompt" (collapsible):
   - Multiline NSTextView / TextEditor, min 3 lines, scrollable
5. Section "Advanced" (collapsible, collapsed by default):
   - Model Alias: text field with placeholder "Same as model name"
   - TTL Minutes: numeric field, "0 = never auto-unload" hint text
   - Pin in memory: toggle switch
   - Trust remote code: toggle switch with warning icon
6. Footer: [Apply] primary button + [Reset to Defaults] secondary text button

DETAILS:
- Each parameter row has a small ⓘ info button on the right that shows a tooltip
- Slider thumb uses system accent color
- Disabled controls appear grayed out (when no model is loaded)
- Show a subtle "Modified" indicator next to section header when defaults are changed
- Width: 260–280pt (standard macOS inspector width)
- Scrollable if content exceeds window height

INTERACTION NOTES:
- Sliders update the numeric value in real-time as user drags
- Numeric fields validate input on focus-out (clamp to valid range)
- Apply button is primary (filled, accent color) and enabled only when there are unsaved changes
- Reset shows confirmation popover before applying

OUTPUT:
Provide the design as a component with both light and dark mode variants.
Include a hover state for each interactive element.
Show one expanded state (all sections open) and one compact state (sections collapsed).
```

---

## v0.1 Scope

- All sampling parameters with sliders/steppers
- System prompt text editor
- Per-model persistence
- Reset to defaults
- Tooltips for each parameter
- Does NOT include: preset profiles, import/export (v0.2)
