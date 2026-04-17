// ServeDashboard.swift
// macmlx
//
// Boxed status block printed at `macmlx serve` startup. No live update —
// the ServeCommand thread sleeps on a signal wait after this, so a
// self-refreshing dashboard would need full-screen terminal control
// (raw mode + SIGWINCH). Kept simple: print once, nicely, then let the
// user run `macmlx ps` or hit `/x/status` for live stats.

import Foundation
import MacMLXCore

enum ServeDashboard {
    /// Print a one-shot status block at startup. Called after the server
    /// has bound a port.
    static func printStatus(port: Int, modelID: String) {
        let width = 54
        print(CLITerm.boxHeader("macmlx serve", width: width))
        printRow("● RUNNING", value: "http://127.0.0.1:\(port)",
                 keyColour: CLITerm.green, width: width)
        printRow("Model:",    value: modelID,                             width: width)
        printRow("Health:",   value: "http://127.0.0.1:\(port)/health",   width: width)
        printRow("Status:",   value: "http://127.0.0.1:\(port)/x/status", width: width)
        print(CLITerm.boxFooter(width: width))
        print(CLITerm.colourise("Press Ctrl+C to stop.", CLITerm.dim))
    }

    /// Single "│ key   value…         │" row. Width must match
    /// `boxHeader` / `boxFooter`. ANSI colour codes are zero-width so we
    /// have to pad based on the *visible* length, not raw string length.
    private static func printRow(
        _ key: String,
        value: String,
        keyColour: String = CLITerm.bold,
        width: Int
    ) {
        let keyCol = 12     // fixed key column
        let keyPadded = key.padding(toLength: keyCol, withPad: " ", startingAt: 0)
        let colouredKey = CLITerm.colourise(keyPadded, keyColour)

        // Available visual width for the value: width - 2 (side bars) - 2 (inner padding) - keyCol
        let valueWidth = max(1, width - 2 - 2 - keyCol)
        let truncatedValue = value.count > valueWidth
            ? String(value.prefix(max(0, valueWidth - 1))) + "…"
            : value.padding(toLength: valueWidth, withPad: " ", startingAt: 0)

        print("│ \(colouredKey)\(truncatedValue) │")
    }
}
