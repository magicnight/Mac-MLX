import Foundation
import MacMLXCore
import SwiftTUI

// Deferred — SwiftTUI's View protocol is nonisolated, conflicting with
// @MainActor state classes under Swift 6 strict concurrency. Full TUI
// requires an upstream SwiftTUI fix (tracked as #18). We print live
// progress to stdout via carriage-return overwrite in the meantime.

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

    /// `"[2/4]  47%  2.10 GB / 4.50 GB  12.5 MB/s  2m 13s  model-...safetensors"`
    /// — shows per-file progress (always accurate) + file counter (always accurate)
    /// + EMA speed + ETA when available. Overall aggregate bytes is intentionally
    /// NOT displayed because HF manifests omit LFS file sizes.
    private static func formatLine(_ p: DownloadProgress) -> String {
        let files = "[\(p.completedFiles)/\(p.totalFiles)]"
        let pct = p.currentFileTotalBytes > 0 ? p.currentFilePercent : "..."
        let bytes = p.currentFileTotalBytes > 0 ? p.currentFileHuman : "(starting)"
        let speed = p.currentFileSpeedHuman.isEmpty ? "" : "  \(p.currentFileSpeedHuman)"
        let eta = p.currentFileETASeconds != nil ? "  \(p.currentFileETAHuman)" : ""
        let current = p.currentFileName.map { "  " + $0 } ?? ""
        return "\(files)  \(pct)  \(bytes)\(speed)\(eta)\(current)"
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

// Minimal SwiftTUI stub — keeps the product linked.
// Full dashboard deferred to #18.
private struct _PullDashboardView: View {
    var body: some View {
        Text("macmlx pull — TUI v0.2")
    }
}
