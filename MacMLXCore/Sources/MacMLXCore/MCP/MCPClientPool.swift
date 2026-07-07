import Foundation
import MCP
import System

/// Manages connections to the external MCP servers configured in
/// `MCPClientConfig`. One pool instance ↔ one config dict; spawn each
/// `mcpServers[name]` entry as a subprocess via `Foundation.Process`,
/// hand the process's stdin/stdout pipes to an MCP `StdioTransport`,
/// and connect a `Client` per server.
///
/// Lifecycle:
/// - `connectAll()` — spawn + connect every entry; partial failures
///   are tolerated (a broken server logs to stderr but doesn't take
///   the pool down).
/// - `listAllTools()` — flatten every connected server's `listTools()`
///   into a `[serverName: [Tool]]` map for the chat-side router.
/// - `callTool(server:name:arguments:)` — route one call to the named
///   server's `Client.callTool`.
/// - `disconnectAll()` — terminate subprocesses and tear down clients.
///
/// Errors surface as typed `MCPClientPool.Error`. Subprocess stderr
/// is captured and forwarded to the host's stderr (so we don't lose
/// MCP server diagnostics) but isn't blocking.
public actor MCPClientPool {

    public enum Error: Swift.Error, CustomStringConvertible {
        case spawnFailed(server: String, command: String, reason: String)
        case connectFailed(server: String, reason: String)
        case unknownServer(String)

        public var description: String {
            switch self {
            case .spawnFailed(let server, let command, let reason):
                return "Failed to spawn MCP server '\(server)' (command: \(command)): \(reason)"
            case .connectFailed(let server, let reason):
                return "Failed to connect to MCP server '\(server)': \(reason)"
            case .unknownServer(let server):
                return "No connected MCP server named '\(server)'"
            }
        }
    }

    /// One live entry in the pool — process + transport + client.
    /// Held across `connectAll` for the lifetime of the pool.
    private struct Entry {
        let server: String
        let process: Process
        let transport: StdioTransport
        let client: Client
    }

    private let config: MCPClientConfig
    private var entries: [String: Entry] = [:]

    public init(config: MCPClientConfig) {
        self.config = config
    }

    // MARK: - Public API

    /// Connect to every configured server. Failures on individual
    /// servers are caught + reported but don't prevent the rest of
    /// the pool from coming up. Returns the names of servers that
    /// successfully connected.
    @discardableResult
    public func connectAll(
        clientName: String = "macmlx",
        clientVersion: String = MacMLXCore.version,
        connectTimeout: Duration = .seconds(10)
    ) async throws -> [String] {
        // A spawned MCP server can die immediately (bad command, crash on
        // startup). Writing to its now-closed stdin would raise SIGPIPE and
        // take the whole host process down. Ignore it process-wide so the
        // write returns EPIPE instead, surfacing as a normal connectFailed.
        signal(SIGPIPE, SIG_IGN)

        var connected: [String] = []
        for serverName in config.mcpServers.keys.sorted() {
            guard let entry = config.mcpServers[serverName] else { continue }
            do {
                try await connect(
                    serverName: serverName,
                    entry: entry,
                    clientName: clientName,
                    clientVersion: clientVersion,
                    timeout: connectTimeout
                )
                connected.append(serverName)
            } catch {
                FileHandle.standardError.write(Data(
                    "[MCPClientPool] \(serverName) failed: \(error)\n".utf8
                ))
            }
        }
        return connected
    }

    /// Names of every server we currently hold a live connection to.
    public func connectedServerNames() -> [String] {
        Array(entries.keys).sorted()
    }

    /// Aggregate `listTools()` across every connected server.
    /// Returns `[serverName: [Tool]]`. Servers that fail mid-list
    /// are dropped from the result with a stderr note.
    public func listAllTools() async -> [String: [Tool]] {
        var out: [String: [Tool]] = [:]
        for (name, entry) in entries {
            do {
                let (tools, _) = try await entry.client.listTools()
                out[name] = tools
            } catch {
                FileHandle.standardError.write(Data(
                    "[MCPClientPool] \(name) listTools failed: \(error)\n".utf8
                ))
            }
        }
        return out
    }

    /// Route one tool call to the named server. Throws
    /// `Error.unknownServer` if `server` is not in the pool.
    public func callTool(
        server: String,
        name: String,
        arguments: [String: Value]? = nil
    ) async throws -> CallTool.Result {
        guard let entry = entries[server] else {
            throw Error.unknownServer(server)
        }
        let request = CallTool.Parameters(name: name, arguments: arguments)
        let response = try await entry.client.send(CallTool.request(request))
        return try await response.value
    }

    /// Tear down every subprocess + client. Idempotent — safe to
    /// call multiple times or against an empty pool.
    public func disconnectAll() async {
        for (_, entry) in entries {
            await entry.client.disconnect()
            entry.process.terminate()
        }
        entries.removeAll()
    }

    // MARK: - Private — spawn + connect

    private func connect(
        serverName: String,
        entry config: MCPClientConfig.ServerEntry,
        clientName: String,
        clientVersion: String,
        timeout: Duration
    ) async throws {
        // stdin: we write → child reads. Child uses the read end.
        // stdout: child writes → we read. We use the read end.
        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()

        let process = Process()
        // /usr/bin/env <cmd> respects PATH so npx / uvx / etc. resolve.
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [config.command] + config.args
        if let extra = config.env {
            process.environment = ProcessInfo.processInfo.environment
                .merging(extra) { _, new in new }
        }
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        do {
            try process.run()
        } catch {
            throw Error.spawnFailed(
                server: serverName,
                command: config.command,
                reason: error.localizedDescription
            )
        }

        // Forward subprocess stderr to ours so the user sees MCP
        // server diagnostics in their terminal / Console.
        forwardStderr(from: stderrPipe, label: serverName)

        // FileDescriptor wraps a raw int fd. Foundation's
        // Pipe.fileHandleForReading/Writing exposes that as
        // .fileDescriptor (Int32).
        let inputFD = FileDescriptor(rawValue: stdoutPipe.fileHandleForReading.fileDescriptor)
        let outputFD = FileDescriptor(rawValue: stdinPipe.fileHandleForWriting.fileDescriptor)
        let transport = StdioTransport(input: inputFD, output: outputFD)

        let client = Client(name: clientName, version: clientVersion)
        do {
            try await runConnect(client: client, transport: transport, timeout: timeout)
        } catch {
            // disconnect() cancels swift-sdk's internal message loop, which
            // otherwise busy-loops at 100% CPU when the server dies before
            // initialize completes (see runConnect for the upstream bug).
            await client.disconnect()
            process.terminate()
            throw Error.connectFailed(
                server: serverName,
                reason: "\(error)"
            )
        }

        entries[serverName] = Entry(
            server: serverName,
            process: process,
            transport: transport,
            client: client
        )
    }

    /// Background task that reads child stderr line-by-line and
    /// forwards each line to our stderr with a `[<server>]` prefix.
    /// Detached because the pipe lives for the subprocess's full
    /// lifetime; ends naturally on EOF when the child exits.
    private nonisolated func forwardStderr(from pipe: Pipe, label: String) {
        let handle = pipe.fileHandleForReading
        Task.detached {
            while true {
                let chunk = handle.availableData
                if chunk.isEmpty { return }
                let prefixed = "[\(label)] ".data(using: .utf8) ?? Data()
                FileHandle.standardError.write(prefixed)
                FileHandle.standardError.write(chunk)
            }
        }
    }

    /// Bounds `client.connect` with a timeout.
    ///
    /// swift-sdk 0.12.1's `Client.connect` spins at 100% CPU if the
    /// transport reaches EOF before `initialize` completes — e.g. the
    /// spawned server process dies on startup. Its internal message loop
    /// (`Client.swift`, `repeat { for try await … in stream } while true`)
    /// re-polls a finished stream forever instead of terminating, while
    /// `_initialize()` blocks on a response that never arrives.
    ///
    /// `disconnect()` is the antidote: it cancels that loop and resumes the
    /// pending initialize with an error. So we race connect against a
    /// timeout; on expiry we disconnect, which makes `connect` throw
    /// instead of hanging the whole process.
    private func runConnect(
        client: Client, transport: StdioTransport, timeout: Duration
    ) async throws {
        let connectTask = Task { _ = try await client.connect(transport: transport) }
        let timeoutTask = Task {
            try await Task.sleep(for: timeout)
            await client.disconnect()
        }
        defer { timeoutTask.cancel() }
        try await connectTask.value
    }
}
