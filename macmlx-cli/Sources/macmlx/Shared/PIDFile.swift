import Foundation

/// Coordinates between `macmlx serve` and `macmlx stop`/`macmlx ps` via a
/// JSON file at `~/.mac-mlx/macmlx.pid`.
///
/// The PID file stores the process ID, port, optional loaded model ID, and
/// start time. It is written by `serve` on startup and removed on clean exit.
///
/// NOTE: Race conditions between concurrent `serve` invocations are not
/// handled in v0.1 — the last writer wins. Document and skip.
public enum PIDFile {
    /// Persistent record stored in the PID file.
    public struct Record: Codable, Sendable {
        public var pid: Int32
        public var port: Int
        public var modelID: String?
        public var startedAt: Date

        public init(pid: Int32, port: Int, modelID: String?, startedAt: Date) {
            self.pid = pid
            self.port = port
            self.modelID = modelID
            self.startedAt = startedAt
        }
    }

    /// URL of the PID file: `~/.mac-mlx/macmlx.pid`.
    public static var url: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appending(path: ".mac-mlx/macmlx.pid")
    }

    /// Write `record` to the PID file atomically.
    ///
    /// Creates the parent directory (`~/.mac-mlx/`) if it does not exist.
    public static func write(_ record: Record) throws {
        let fileURL = url
        let parentDir = fileURL.deletingLastPathComponent()
        if !FileManager.default.fileExists(atPath: parentDir.path) {
            try FileManager.default.createDirectory(
                at: parentDir,
                withIntermediateDirectories: true
            )
        }
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted]
        let data = try encoder.encode(record)
        try data.write(to: fileURL, options: .atomic)
    }

    /// Read and decode the PID file.
    ///
    /// - Returns: The decoded `Record`, or `nil` if the file does not exist.
    /// - Throws: `DecodingError` if the file exists but cannot be decoded.
    public static func read() throws -> Record? {
        let fileURL = url
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            return nil
        }
        let data = try Data(contentsOf: fileURL)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(Record.self, from: data)
    }

    /// Remove the PID file.
    ///
    /// Silently succeeds if the file does not exist.
    public static func clear() throws {
        let fileURL = url
        guard FileManager.default.fileExists(atPath: fileURL.path) else { return }
        try FileManager.default.removeItem(at: fileURL)
    }
}
