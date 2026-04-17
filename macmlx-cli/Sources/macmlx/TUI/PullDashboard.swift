import Foundation
import MacMLXCore
import SwiftTUI

// TODO: v0.2 — SwiftTUI's View protocol is nonisolated, conflicting with
// @MainActor state classes under Swift 6 strict concurrency. Full TUI deferred.
// For v0.1 we print live progress to stdout via carriage-return overwrite.

/// Entry point for pull download progress display.
///
/// v0.1: Plain stdout with `\r`-overwritten progress line per chunk.
enum PullDashboard {
    /// Download `modelID` and report live progress to stdout.
    static func run(
        modelID: String,
        target: URL,
        downloader: HFDownloader
    ) async throws {
        print("Pulling \(modelID)")
        print("Target: \(target.path(percentEncoded: false))")
        print("")

        // Locked storage so the URLSession callback (background queue) and
        // the printer (main task) can share the latest snapshot without a race.
        let latest = ProgressBox()

        let handler: HFDownloader.ProgressHandler = { snapshot in
            latest.set(snapshot)
        }

        // Spawn a printer that wakes 4×/sec and refreshes the line.
        let printerTask = Task { @Sendable in
            while !Task.isCancelled {
                if let snap = latest.get() {
                    let line = formatLine(snap)
                    // \r returns to start of line; pad with spaces to clobber
                    // any leftover characters from a longer previous line.
                    FileHandle.standardOutput.write(Data("\r\(line)   ".utf8))
                }
                try? await Task.sleep(for: .milliseconds(250))
            }
        }
        defer {
            printerTask.cancel()
            // Newline so the next print() lands clean.
            FileHandle.standardOutput.write(Data("\n".utf8))
        }

        let dest = try await downloader.download(
            modelID: modelID,
            to: target,
            progress: handler
        )
        // Force one final 100% line.
        if let final = latest.get() {
            FileHandle.standardOutput.write(Data("\r\(formatLine(final))   \n".utf8))
        }
        print("Downloaded to: \(dest.path(percentEncoded: false))")
    }

    /// `"45%  2.10 GB / 4.50 GB  (2/4)  model-00002-of-00004.safetensors"`
    private static func formatLine(_ p: DownloadProgress) -> String {
        let pct = p.totalBytes > 0 ? p.humanPercent : "..."
        let bytes = p.totalBytes > 0 ? p.humanProgress : "(unknown size)"
        let files = "(\(p.completedFiles)/\(p.totalFiles))"
        let current = p.currentFileName.map { " " + $0 } ?? ""
        return "\(pct)  \(bytes)  \(files)\(current)"
    }
}

/// Tiny lock-protected box for sharing `DownloadProgress` between the
/// URLSession callback queue and the stdout-printer Task. Marked
/// `@unchecked Sendable` because `NSLock` carries the synchronisation; no
/// shared mutable state is exposed without the lock.
private final class ProgressBox: @unchecked Sendable {
    private let lock = NSLock()
    private var value: DownloadProgress?

    func set(_ p: DownloadProgress) {
        lock.lock(); defer { lock.unlock() }
        value = p
    }

    func get() -> DownloadProgress? {
        lock.lock(); defer { lock.unlock() }
        return value
    }
}

// Minimal SwiftTUI stub — keeps the product linked
// TODO: v0.2 — replace with a real Application(rootView:).start() dashboard
private struct _PullDashboardView: View {
    var body: some View {
        Text("macmlx pull — TUI v0.2")
    }
}
