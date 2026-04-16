import Foundation
import MacMLXCore
import SwiftTUI

// TODO: v0.2 — SwiftTUI's View protocol is nonisolated, conflicting with
// @MainActor state classes under Swift 6 strict concurrency. Full TUI deferred.
// For v0.1 we print indeterminate progress to stdout.

/// Entry point for pull download progress display.
///
/// v0.1: Always uses plain stdout. TUI rendering deferred to v0.2.
enum PullDashboard {
    /// Download `modelID` while printing indeterminate progress to stdout.
    ///
    /// NOTE: `HFDownloader.download(modelID:to:onProgress:)` takes a non-`@Sendable`
    /// closure, so we pass `nil` for `onProgress` and report completion only.
    ///
    /// TODO: v0.2 — mark `onProgress` as `@Sendable` in `HFDownloader`, then show
    /// per-file progress in the TUI dashboard.
    static func run(
        modelID: String,
        target: URL,
        downloader: HFDownloader
    ) async throws {
        print("Pulling \(modelID)…")
        print("Target: \(target.path(percentEncoded: false))")
        let dest = try await downloader.download(modelID: modelID, to: target)
        print("Downloaded to: \(dest.path(percentEncoded: false))")
    }
}

// Minimal SwiftTUI stub — keeps the product linked
// TODO: v0.2 — replace with a real Application(rootView:).start() dashboard
private struct _PullDashboardView: View {
    var body: some View {
        Text("macmlx pull — TUI v0.2")
    }
}
