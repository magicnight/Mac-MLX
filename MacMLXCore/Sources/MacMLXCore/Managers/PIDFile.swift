import Foundation

/// Coordinates server discovery between the GUI (`macMLX.app`) and
/// CLI (`macmlx`) — both write to `~/.mac-mlx/macmlx.pid` so either
/// side can tell whether the other is already serving on port 8000.
///
/// `Record.owner` distinguishes the two: GUI sets `.gui`, CLI sets
/// `.cli`. `macmlx ps` renders the owner; `macmlx serve` refuses to
/// start when a record whose PID is still alive is found.
///
/// Backward compat: pre-v0.3.7 PID files lacked the `owner` key and
/// were always written by the CLI. Decoding defaults missing `owner`
/// to `.cli` so upgrading in place doesn't require manually deleting
/// the pid file.
public enum PIDFile {
    /// Persistent record stored in the PID file.
    public struct Record: Codable, Sendable {
        /// Which process wrote this record — used by `macmlx serve`
        /// to name the conflicting owner in its error message, and
        /// by `macmlx ps` to show the user which side is serving.
        public enum Owner: String, Codable, Sendable {
            case gui
            case cli
        }

        public var pid: Int32
        public var port: Int
        public var modelID: String?
        public var startedAt: Date
        public var owner: Owner

        public init(
            pid: Int32,
            port: Int,
            modelID: String?,
            startedAt: Date,
            owner: Owner
        ) {
            self.pid = pid
            self.port = port
            self.modelID = modelID
            self.startedAt = startedAt
            self.owner = owner
        }

        private enum CodingKeys: String, CodingKey {
            case pid, port, modelID, startedAt, owner
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.pid = try c.decode(Int32.self, forKey: .pid)
            self.port = try c.decode(Int.self, forKey: .port)
            self.modelID = try c.decodeIfPresent(String.self, forKey: .modelID)
            self.startedAt = try c.decode(Date.self, forKey: .startedAt)
            self.owner = (try c.decodeIfPresent(Owner.self, forKey: .owner)) ?? .cli
        }
    }

    /// URL of the PID file. Uses `DataRoot.macMLX` so the GUI (which
    /// runs without sandbox since v0.3.6) and CLI both resolve to the
    /// same real-home `~/.mac-mlx/` path.
    public static var url: URL {
        DataRoot.macMLX.appending(path: "macmlx.pid", directoryHint: .notDirectory)
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
