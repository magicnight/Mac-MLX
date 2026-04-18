import ArgumentParser
import Darwin
import Foundation
import MacMLXCore

struct PSCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ps",
        abstract: "Show running macmlx serve status."
    )

    @Flag(help: "Output as JSON for scripting.")
    var json = false

    func run() async throws {
        guard let record = try PIDFile.read(),
              kill(record.pid, 0) == 0 else {
            print("No macmlx serve running.")
            throw ExitCode(1)
        }

        let uptime = Date().timeIntervalSince(record.startedAt)
        let uptimeStr = formatDuration(uptime)

        if json {
            let output = PSOutput(
                pid: Int(record.pid),
                port: record.port,
                modelID: record.modelID,
                startedAt: record.startedAt,
                uptimeSeconds: uptime
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(output)
            FileHandle.standardOutput.write(data)
            print()
        } else {
            print("macmlx serve is running")
            print("  PID:     \(record.pid)")
            print("  Owner:   \(record.owner.rawValue.uppercased())")
            print("  Port:    \(record.port)")
            print("  Model:   \(record.modelID ?? "(none)")")
            print("  Uptime:  \(uptimeStr)")
            print("  Started: \(record.startedAt)")
        }
    }

    // MARK: - Helpers

    private func formatDuration(_ seconds: TimeInterval) -> String {
        let secs = Int(seconds)
        let h = secs / 3600
        let m = (secs % 3600) / 60
        let s = secs % 60
        if h > 0 { return "\(h)h \(m)m \(s)s" }
        if m > 0 { return "\(m)m \(s)s" }
        return "\(s)s"
    }
}

/// JSON-serialisable representation of a running serve process.
private struct PSOutput: Codable {
    var pid: Int
    var port: Int
    var modelID: String?
    var startedAt: Date
    var uptimeSeconds: TimeInterval
}
