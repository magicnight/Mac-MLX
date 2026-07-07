import Testing
import Foundation
@testable import MacMLXCore

/// Unit-level coverage for `MCPClientPool` — only the paths that
/// don't require a live MCP server subprocess. Real-server smoke
/// tests are deferred to v0.5 MCP track part 2.5 (we'd want a
/// scripted test fixture against
/// `@modelcontextprotocol/server-everything` or similar).
@Suite("MCPClientPool")
struct MCPClientPoolTests {

    @Test
    func emptyConfigConnectsToZeroServers() async throws {
        let pool = MCPClientPool(config: MCPClientConfig())
        let connected = try await pool.connectAll()
        #expect(connected.isEmpty)
        #expect(await pool.connectedServerNames().isEmpty)
    }

    @Test
    func badCommandTolerated_partialFailureDoesntCrashPool() async throws {
        // `definitely-not-a-real-binary-xyz` won't resolve through
        // /usr/bin/env — Process.run() throws, the pool catches it,
        // and connectAll() returns [] without raising.
        let cfg = MCPClientConfig(mcpServers: [
            "ghost": .init(
                command: "definitely-not-a-real-binary-xyz-9876",
                args: [],
                env: nil
            )
        ])
        let pool = MCPClientPool(config: cfg)
        // The ghost server dies instantly, so connect hits the dead-transport
        // path. Use a short timeout: we just need it to fail fast (and NOT
        // busy-loop — the regression this guards against), not wait 10s.
        let connected = try await pool.connectAll(connectTimeout: .milliseconds(500))
        #expect(connected.isEmpty, "spawn-failed servers must not appear in connected list")
        #expect(await pool.connectedServerNames().isEmpty)
    }

    @Test
    func callToolThrowsForUnknownServer() async throws {
        let pool = MCPClientPool(config: MCPClientConfig())
        await #expect(throws: MCPClientPool.Error.self) {
            _ = try await pool.callTool(server: "missing", name: "x")
        }
    }

    @Test
    func disconnectAllIsIdempotentOnEmptyPool() async {
        let pool = MCPClientPool(config: MCPClientConfig())
        await pool.disconnectAll()
        await pool.disconnectAll()  // second call is a no-op
        #expect(await pool.connectedServerNames().isEmpty)
    }

    @Test
    func errorDescriptionsAreReadable() {
        let spawn = MCPClientPool.Error.spawnFailed(
            server: "x", command: "missing-cmd", reason: "no such file"
        )
        #expect(spawn.description.contains("'x'"))
        #expect(spawn.description.contains("missing-cmd"))
        #expect(spawn.description.contains("no such file"))

        let connect = MCPClientPool.Error.connectFailed(server: "y", reason: "EOF")
        #expect(connect.description.contains("'y'"))
        #expect(connect.description.contains("EOF"))

        let unknown = MCPClientPool.Error.unknownServer("z")
        #expect(unknown.description.contains("'z'"))
    }
}
