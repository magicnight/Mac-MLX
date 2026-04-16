# UI Guidelines

## Design Philosophy

Native macOS first. Every UI decision should feel like it belongs on macOS,
not like a web app wrapped in a window.

Reference apps for quality bar:
- **Reeder** — clean information density
- **Lasso** — native macOS tool aesthetic
- **Sleeve** — menubar integration done right

## macOS Design Principles

Follow Apple HIG (Human Interface Guidelines):
- Use system fonts (`Font.body`, `.headline`, not custom fonts)
- Use semantic colors (`Color.primary`, `.secondary`, `.accentColor`)
- Respect Dark Mode automatically (use system colors only)
- Use standard macOS controls (no custom-styled buttons that fight the OS)
- Support keyboard navigation
- Minimum click target: 44x44pt

## Window Layout

### Main Window
Three-column layout (standard macOS pattern):

```
┌──────────┬────────────────────┬─────────────┐
│ Sidebar  │   Content Area     │  Inspector  │
│          │                    │  (optional) │
│ Models   │  (Chat / Library   │             │
│ Settings │   / Downloader)    │  Model      │
│          │                    │  Details    │
└──────────┴────────────────────┴─────────────┘
```

Use `NavigationSplitView` — do not build custom split view.

### Sidebar
- Width: 200-220pt
- Items: Models, Download, Settings
- Use `List` with `.sidebar` list style
- Show service status indicator in sidebar header

### Toolbar
Use `toolbar` modifier with `ToolbarItem` placements.
Primary actions in toolbar, secondary in context menus.

## Color Usage

Never hardcode colors. Use only:

```swift
// Semantic system colors
Color.primary          // main text
Color.secondary        // secondary text
Color.accentColor      // interactive elements (user customizable)
Color(.windowBackground)
Color(.controlBackground)
Color(.separatorColor)

// Status colors (use sparingly)
Color.green   // running / success
Color.orange  // warning / loading
Color.red     // error / stopped
```

## Typography

System fonts only:

```swift
Font.largeTitle    // Page titles
Font.title         // Section headers
Font.headline      // Item titles
Font.body          // Default content
Font.callout       // Secondary info
Font.caption       // Timestamps, metadata
Font.system(.body, design: .monospaced)  // Token counts, technical values
```

## Status Indicators

Service status shown consistently across menubar and sidebar:

```swift
enum ServiceStatus {
    case stopped    // gray circle
    case starting   // orange pulsing circle
    case running    // green circle
    case error      // red circle
}
```

Use SF Symbols for status icons:
- `circle.fill` with appropriate color
- `bolt.fill` for active inference
- `arrow.down.circle` for downloading

## Component Patterns

### Model Row
```
[Model Icon] Model Name                    [Status Badge]
             7B · Q4 · 4.2 GB             [Load Button]
```

### Chat Message
```
[Avatar] Assistant
         Message content here...
         12:34 PM · 142 tokens
         
         User
         User message here...    [Avatar]
```

### Download Progress
```
[Model Name]                        [Cancel]
mlx-community/Qwen3-8B-4bit
████████████░░░░  67% · 2.1 GB / 3.2 GB · 12 MB/s
```

## Animations

Keep animations subtle and purposeful:
- Use `.animation(.easeInOut(duration: 0.2))` for state transitions
- Status indicator pulse: `.animation(.easeInOut(duration: 1).repeatForever())`
- Do not animate layout changes — jarring on desktop
- Never auto-play animations that the user did not trigger

## Accessibility

- All interactive elements must have `.accessibilityLabel`
- Support `reduceMotion` environment value
- Test with VoiceOver before each release

## v0.1 UI Scope

Implement only:
- MenuBar popover with status + quick actions
- Main window with sidebar navigation
- Model Library view (list of local models)
- Basic Chat view
- Settings view (model directory, port)

Do NOT implement in v0.1:
- Inspector panel
- Download progress animations
- Onboarding flow
- Keyboard shortcuts
