import ArgumentParser
import Darwin
import Foundation

struct StopCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "stop",
        abstract: "Stop the running macmlx serve."
    )

    func run() async throws {
        guard let record = try PIDFile.read() else {
            print("No macmlx serve running.")
            return
        }

        // Verify the process is still alive before sending SIGTERM.
        guard kill(record.pid, 0) == 0 else {
            print("PID \(record.pid) is not running. Cleaning up stale PID file.")
            try? PIDFile.clear()
            return
        }

        print("Stopping macmlx serve (PID \(record.pid))…")
        kill(record.pid, SIGTERM)

        // Poll up to 5 seconds for the PID file to be cleared by the serve process.
        for _ in 0..<50 {
            try await Task.sleep(nanoseconds: 100_000_000)  // 100ms
            if (try? PIDFile.read()) == nil {
                print("Stopped.")
                return
            }
        }

        // If the PID file is still present after 5s, the process didn't exit cleanly.
        print("Warning: serve did not exit cleanly within 5 seconds.")
        throw ExitCode(2)
    }
}
