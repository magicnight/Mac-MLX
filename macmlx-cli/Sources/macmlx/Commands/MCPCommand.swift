import ArgumentParser
import Foundation
import MacMLXCore
import MCP

/// Top-level `macmlx mcp` group. Currently has one subcommand —
/// `serve` — but kept as a group so future MCP-client / inspect
/// subcommands can land alongside without reorganising the CLI.
struct MCPCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mcp",
        abstract: "Model Context Protocol (MCP) server.",
        subcommands: [MCPServeCommand.self],
        defaultSubcommand: MCPServeCommand.self
    )
}

/// `macmlx mcp serve` — runs an MCP server over stdio.
///
/// Drop into Claude Desktop / Cursor / Zed via their `mcpServers`
/// config and the host will spawn this process per session, talking
/// JSON-RPC over the pipe pair.
struct MCPServeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Start an MCP server over stdio (JSON-RPC on stdin/stdout, logs on stderr)."
    )

    func run() async throws {
        // stdout is reserved for MCP JSON-RPC. Route any swift-log output
        // (from the SDK or our own emit paths) to stderr so we don't
        // corrupt the wire protocol with diagnostic noise.
        MCPLogging.bootstrap()

        let ctx = try await CLIContext.bootstrap()

        // Adapter: ModelLibraryManager.scan(_:) → ModelSource.
        let library = CLIModelSource(
            manager: ctx.library,
            modelDirectory: ctx.settings.modelDirectory
        )

        // Engine factory mirrors RunCommand / ServeCommand — honours
        // `Settings.preferredEngine`. Each MCP serve process owns one
        // engine; the bridge swaps the loaded model lazily.
        let bridge = MCPBridge(
            library: library,
            paramStore: ctx.paramStore,
            engineFactory: { try ctx.makeEngine() }
        )

        // Friendly note on stderr so users running `macmlx mcp serve`
        // by hand know the process is alive (handy when piping JSON-RPC
        // through manually). MCP hosts ignore stderr.
        FileHandle.standardError.write(Data("macmlx mcp serve: ready (stdio)\n".utf8))

        try await bridge.start(
            transport: StdioTransport(),
            serverVersion: MacMLXCore.version
        )
    }
}

/// Wraps `ModelLibraryManager.scan(_:)` so the bridge can stay
/// agnostic of the manager's concrete shape (the manager wants the
/// model directory passed every call; the bridge doesn't care).
struct CLIModelSource: ModelSource {
    let manager: ModelLibraryManager
    let modelDirectory: URL

    func scan() async throws -> [LocalModel] {
        try await manager.scan(modelDirectory)
    }
}
