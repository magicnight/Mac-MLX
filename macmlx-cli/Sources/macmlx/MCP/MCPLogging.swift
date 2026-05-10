import Foundation
import Logging

/// `LogHandler` that writes structured swift-log entries to stderr.
///
/// MCP stdio servers must keep stdout clean for JSON-RPC, so any
/// logging coming out of the SDK or our own emit paths gets redirected
/// to the user's terminal via stderr instead. The format is plain text
/// — Claude Desktop / Cursor surface the stderr stream verbatim in
/// their MCP debugger panels.
public struct StderrLogHandler: LogHandler {
    public var metadata: Logger.Metadata = [:]
    public var logLevel: Logger.Level = .info

    private let label: String

    public init(label: String) {
        self.label = label
    }

    public subscript(metadataKey key: String) -> Logger.Metadata.Value? {
        get { metadata[key] }
        set { metadata[key] = newValue }
    }

    public func log(
        level: Logger.Level,
        message: Logger.Message,
        metadata: Logger.Metadata?,
        source: String,
        file: String,
        function: String,
        line: UInt
    ) {
        let formatted = "[\(level)] \(label): \(message)\n"
        if let data = formatted.data(using: .utf8) {
            FileHandle.standardError.write(data)
        }
    }
}

/// Process-wide bootstrap helper for MCP stdio servers.
///
/// `LoggingSystem.bootstrap` may only be called once per process, and
/// the convention is to do it at the very top of the entry point. The
/// MCP serve subcommand is the only call site, so this single helper
/// keeps the policy in one place.
public enum MCPLogging {
    public static func bootstrap() {
        LoggingSystem.bootstrap { label in StderrLogHandler(label: label) }
    }
}
