// CLITerm.swift
// macmlx
//
// Minimal ANSI terminal helper. Used by the CLI dashboards in place of
// SwiftTUI (removed in v0.3.5 — upstream has been unmaintained for over
// a year and is incompatible with Swift 6 strict concurrency).
//
// Scope kept deliberately small: colour + bold, cursor reset for
// \r-overwrite lines, one box-drawing helper for section headers.
// If the CLI ever grows a truly live-updating full-screen dashboard we
// can reach for ncurses or notcurses via a C bridge at that point —
// this file is designed to be deleted without regret in that scenario.

import Foundation

public enum CLITerm {

    // MARK: - ANSI escape constants

    private static let esc = "\u{1B}["

    /// Reset all attributes back to default.
    public static let reset = "\(esc)0m"

    /// Bold / increased intensity.
    public static let bold = "\(esc)1m"

    /// Dim / decreased intensity (greys out text on most terminals).
    public static let dim = "\(esc)2m"

    // MARK: - Foreground colours

    public static let red     = "\(esc)31m"
    public static let green   = "\(esc)32m"
    public static let yellow  = "\(esc)33m"
    public static let blue    = "\(esc)34m"
    public static let magenta = "\(esc)35m"
    public static let cyan    = "\(esc)36m"
    public static let white   = "\(esc)37m"

    // MARK: - Functions

    /// Whether stdout is attached to a TTY (i.e. a real terminal, not a
    /// pipe). Colour / redraw sequences should be suppressed when false
    /// so they don't pollute piped output.
    public static var isTTY: Bool {
        isatty(fileno(stdout)) != 0
    }

    /// Wrap `text` in colour+reset if running under a TTY; otherwise
    /// return `text` verbatim so piped output stays clean.
    public static func colourise(_ text: String, _ colour: String) -> String {
        isTTY ? "\(colour)\(text)\(reset)" : text
    }

    /// Render a unicode progress bar, e.g. `fraction=0.52, width=30` →
    /// `"██████████████▌              "` (fraction range [0,1], width in
    /// cells). Uses the U+258x block-element range for sub-cell precision.
    public static func progressBar(fraction: Double, width: Int) -> String {
        guard width > 0 else { return "" }
        let clamped = max(0, min(1, fraction))
        let totalEighths = Int((Double(width) * 8 * clamped).rounded(.down))
        let full = totalEighths / 8
        let remainder = totalEighths % 8
        let partial: String
        switch remainder {
        case 0: partial = ""
        case 1: partial = "▏"
        case 2: partial = "▎"
        case 3: partial = "▍"
        case 4: partial = "▌"
        case 5: partial = "▋"
        case 6: partial = "▊"
        case 7: partial = "▉"
        default: partial = ""
        }
        let filled = String(repeating: "█", count: full)
        let emptyCount = max(0, width - full - (partial.isEmpty ? 0 : 1))
        let empty = String(repeating: " ", count: emptyCount)
        return filled + partial + empty
    }

    /// Render a single-line box around `label`, useful as a section
    /// header at the top of a dashboard. Example:
    ///
    ///     ╭─ macmlx serve ────────────────╮
    ///
    /// `width` is the total character width including both `╭`/`╮` caps.
    public static func boxHeader(_ label: String, width: Int = 48) -> String {
        let padded = " \(label) "
        let dashCount = max(0, width - padded.count - 2)  // minus the two caps
        let dashes = String(repeating: "─", count: dashCount)
        return "╭─\(padded)\(dashes)╮"
    }

    /// Footer bar matching `boxHeader` width.
    public static func boxFooter(width: Int = 48) -> String {
        let dashes = String(repeating: "─", count: max(0, width - 2))
        return "╰\(dashes)╯"
    }
}
