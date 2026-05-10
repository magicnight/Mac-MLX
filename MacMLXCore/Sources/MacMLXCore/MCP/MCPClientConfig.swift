import Foundation

/// Codable mirror of Claude Desktop's `~/.claude_desktop_config.json`
/// `mcpServers` shape (and Cursor's identical schema). Persisted at
/// `~/.mac-mlx/mcp.json` so users with existing MCP setups can copy
/// the file across without translating.
///
/// Wire format:
///
/// ```json
/// {
///   "mcpServers": {
///     "everything": {
///       "command": "npx",
///       "args": ["-y", "@modelcontextprotocol/server-everything"],
///       "env": { "OPENAI_API_KEY": "…" }
///     }
///   }
/// }
/// ```
public struct MCPClientConfig: Codable, Hashable, Sendable, Equatable {

    /// Map of human-readable server name → spawn instructions.
    /// The map is unordered on the wire; `MCPClientPool.connectAll`
    /// iterates `mcpServers.keys.sorted()` for stable ordering in
    /// tool listings + UI.
    public var mcpServers: [String: ServerEntry]

    public init(mcpServers: [String: ServerEntry] = [:]) {
        self.mcpServers = mcpServers
    }

    public struct ServerEntry: Codable, Hashable, Sendable, Equatable {
        /// Executable to spawn (`npx`, `uvx`, an absolute path, …).
        public let command: String
        /// Arguments passed to the executable. Empty array is fine.
        public let args: [String]
        /// Optional environment variables merged into the
        /// subprocess's environment. Nil means "inherit ours
        /// unchanged"; an empty dict means "inherit + add nothing".
        public let env: [String: String]?

        public init(command: String, args: [String] = [], env: [String: String]? = nil) {
            self.command = command
            self.args = args
            self.env = env
        }
    }
}

// MARK: - Persistence

/// Filesystem-backed config loader / saver.
///
/// On-disk default: `~/.mac-mlx/mcp.json`. Atomic writes; tolerates
/// missing files (returns `MCPClientConfig()` with no servers) so
/// first-run users see an empty list rather than an error.
public actor MCPClientConfigStore {

    private let url: URL
    private let fileManager: FileManager

    /// Default URL: `~/.mac-mlx/mcp.json` (real home, dotfile data
    /// root).
    public init(url: URL? = nil, fileManager: FileManager = .default) {
        self.url = url ?? DataRoot.macMLX("mcp.json")
        self.fileManager = fileManager
    }

    /// Read the config from disk. Missing file → empty config (no
    /// servers). Malformed file → empty config + the bad bytes are
    /// preserved on disk so the user can fix manually.
    public func load() async -> MCPClientConfig {
        guard let data = try? Data(contentsOf: url) else {
            return MCPClientConfig()
        }
        return (try? JSONDecoder().decode(MCPClientConfig.self, from: data))
            ?? MCPClientConfig()
    }

    /// Persist `config` atomically. Creates the parent directory if
    /// it doesn't exist yet.
    public func save(_ config: MCPClientConfig) async throws {
        let dir = url.deletingLastPathComponent()
        if !fileManager.fileExists(atPath: dir.path) {
            try fileManager.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(config)
        try data.write(to: url, options: .atomic)
    }
}
