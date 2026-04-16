# Feature: Menu Bar

## Overview

A persistent macOS menu bar item that gives instant access to service
status and quick actions without opening the main window.

## v0.1 Scope

### Menu Bar Icon
- SF Symbol: `cpu` (represents inference/compute)
- Show status color overlay:
  - Gray: service stopped
  - Orange: starting
  - Green: running
  - Red: error

### Popover Content

Clicking the menu bar icon shows a popover (not a dropdown menu):

```
┌─────────────────────────────┐
│  macMLX          ●   │  ← status dot
│                             │
│  Status:  Running           │
│  Model:   Qwen3-8B-4bit     │
│  Memory:  8.2 / 48 GB       │
│                             │
│  ┌─────────┐  ┌──────────┐  │
│  │  Stop   │  │  Open    │  │
│  └─────────┘  └──────────┘  │
│                             │
│  Tokens today: 18,420       │
└─────────────────────────────┘
```

Popover width: 280pt
Popover height: adaptive

### Quick Actions

- **Start / Stop** button (toggles based on current state)
- **Open** button — brings main window to front

### Implementation Notes

Use `NSStatusItem` + `NSPopover`:

```swift
final class MenuBarManager {
    private var statusItem: NSStatusItem?
    private var popover: NSPopover?
    
    func setup() {
        statusItem = NSStatusBar.system.statusItem(
            withLength: NSStatusItem.squareLength
        )
        // Set button image
        // Attach click handler to toggle popover
    }
}
```

Menu bar icon must be a template image so macOS can invert it
for light/dark menu bar automatically.

## Out of Scope (v0.1)

- Model switching from popover
- Token usage history graph
- Multiple backend profiles
- Launch at login toggle (v0.2)
