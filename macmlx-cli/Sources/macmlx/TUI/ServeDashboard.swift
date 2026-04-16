import Foundation
import MacMLXCore
import SwiftTUI

// MARK: - Serve state model (plain stdout version)

// TODO: v0.2 — SwiftTUI's View protocol is nonisolated, which conflicts with
// @MainActor state classes under Swift 6 strict concurrency. Full TUI requires
// either a custom Sendable wrapper or an upstream SwiftTUI fix. For v0.1 we
// fall back to plain stdout for all output modes.

// NOTE: SwiftTUI types (Text, VStack, etc.) are imported here to satisfy the
// compiler; the actual dashboard is implemented as a plain-text printer below.
// The import keeps the SwiftTUI dependency exercised so it remains linked.

/// Entry point for serve status display.
///
/// v0.1: Always uses plain stdout. TUI rendering deferred to v0.2.
enum ServeDashboard {
    /// Print live serve status to stdout until `stop()` is called.
    ///
    /// Since `Application.start()` calls `dispatchMain()` and never returns,
    /// this implementation prints a single status block and returns — the
    /// calling `ServeCommand` runs its own `waitForSignal()` loop.
    static func printStatus(port: Int, modelID: String) {
        print("macmlx serve is running")
        print("  URL:    http://127.0.0.1:\(port)")
        print("  Model:  \(modelID)")
        print("  Docs:   http://127.0.0.1:\(port)/health")
        print("Press Ctrl+C to stop.")
    }
}

// Minimal SwiftTUI stub — keeps the product linked
// TODO: v0.2 — replace with a real Application(rootView:).start() dashboard
private struct _ServeDashboardView: View {
    var body: some View {
        Text("macmlx serve — TUI v0.2")
    }
}
