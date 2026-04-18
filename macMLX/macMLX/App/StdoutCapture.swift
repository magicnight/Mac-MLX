// StdoutCapture.swift
// macMLX
//
// Dup stdout + stderr to a pipe at launch, read the pipe on a background
// task, and tee every line into both the original fd (so terminal output
// still works when launched from CLI) and LogManager.

import Darwin
import Foundation
import MacMLXCore

enum StdoutCapture {
    /// True once installed so repeat calls (e.g. from SwiftUI previews)
    /// are no-ops instead of double-redirecting.
    private static var installed = false

    /// Redirect STDOUT_FILENO and STDERR_FILENO to a Pipe. Launch a
    /// background task that reads the pipe and forwards each line to:
    ///   - the original fd (preserving terminal visibility)
    ///   - LogManager.shared at .debug on category .system
    /// Call exactly once from App.init().
    static func install() {
        guard !installed else { return }
        installed = true

        redirect(fd: STDOUT_FILENO, label: "stdout")
        redirect(fd: STDERR_FILENO, label: "stderr")
    }

    private static func redirect(fd: Int32, label: String) {
        // Save the original fd so we can still write to it (terminal tee).
        let originalFD = dup(fd)
        guard originalFD >= 0 else { return }

        // Create a pipe and point the source fd at its write side.
        let pipe = Pipe()
        let writeFD = pipe.fileHandleForWriting.fileDescriptor
        guard dup2(writeFD, fd) >= 0 else { return }

        // Pipe's write fd is now duplicated to the original fd; close the
        // pipe's side to avoid double-ownership. Foundation's Pipe retains
        // its own reference so the fd stays live.
        _ = close(writeFD)

        // Read loop: buffer bytes until \n, forward each line. Runs on a
        // detached Task so it never blocks app lifecycle.
        let reader = pipe.fileHandleForReading
        Task.detached(priority: .utility) {
            var buffer = Data()
            while true {
                let chunk = reader.availableData
                if chunk.isEmpty {
                    // EOF — very unusual for stdout; bail so the task
                    // doesn't spin.
                    return
                }
                // Mirror to original fd so terminal still sees it.
                chunk.withUnsafeBytes { ptr in
                    _ = write(originalFD, ptr.baseAddress, chunk.count)
                }
                buffer.append(chunk)
                while let newlineIdx = buffer.firstIndex(of: 0x0A) {
                    let lineData = buffer[buffer.startIndex..<newlineIdx]
                    let line = String(data: lineData, encoding: .utf8) ?? ""
                    buffer.removeSubrange(buffer.startIndex...newlineIdx)
                    if !line.isEmpty {
                        await LogManager.shared.debug(
                            "[\(label)] \(line)",
                            category: .system
                        )
                    }
                }
            }
        }
    }
}
