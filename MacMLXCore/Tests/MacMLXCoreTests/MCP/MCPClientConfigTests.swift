import Testing
import Foundation
@testable import MacMLXCore

@Suite("MCPClientConfig")
struct MCPClientConfigTests {

    @Test
    func decodesClaudeDesktopShape() throws {
        let json = """
        {
          "mcpServers": {
            "everything": {
              "command": "npx",
              "args": ["-y", "@modelcontextprotocol/server-everything"]
            },
            "filesystem": {
              "command": "uvx",
              "args": ["mcp-server-filesystem", "/tmp"],
              "env": { "FOO": "bar" }
            }
          }
        }
        """
        let config = try JSONDecoder().decode(MCPClientConfig.self, from: Data(json.utf8))
        #expect(config.mcpServers.count == 2)

        let everything = try #require(config.mcpServers["everything"])
        #expect(everything.command == "npx")
        #expect(everything.args == ["-y", "@modelcontextprotocol/server-everything"])
        #expect(everything.env == nil)

        let fs = try #require(config.mcpServers["filesystem"])
        #expect(fs.env?["FOO"] == "bar")
    }

    @Test
    func roundTripsThroughJSON() throws {
        let original = MCPClientConfig(mcpServers: [
            "x": .init(command: "uvx", args: ["mcp-server-x"], env: nil),
            "y": .init(command: "node", args: ["server.js"], env: ["KEY": "val"]),
        ])
        let data = try JSONEncoder().encode(original)
        let back = try JSONDecoder().decode(MCPClientConfig.self, from: data)
        #expect(back == original)
    }

    @Test
    func emptyConfigDecodesAndEncodes() throws {
        let json = #"{"mcpServers": {}}"#
        let cfg = try JSONDecoder().decode(MCPClientConfig.self, from: Data(json.utf8))
        #expect(cfg.mcpServers.isEmpty)
    }
}

@Suite("MCPClientConfigStore", .serialized)
struct MCPClientConfigStoreTests {

    @Test
    func loadReturnsEmptyConfigWhenFileMissing() async throws {
        let temp = try TempDir()
        let store = MCPClientConfigStore(url: temp.url.appendingPathComponent("mcp.json"))
        let cfg = await store.load()
        #expect(cfg.mcpServers.isEmpty)
    }

    @Test
    func saveAndLoadRoundTrip() async throws {
        let temp = try TempDir()
        let store = MCPClientConfigStore(url: temp.url.appendingPathComponent("mcp.json"))
        let original = MCPClientConfig(mcpServers: [
            "everything": .init(command: "npx", args: ["-y", "@modelcontextprotocol/server-everything"])
        ])
        try await store.save(original)
        let loaded = await store.load()
        #expect(loaded == original)
    }

    @Test
    func saveCreatesParentDirectory() async throws {
        let temp = try TempDir()
        let nestedURL = temp.url
            .appendingPathComponent("never-existed", isDirectory: true)
            .appendingPathComponent("also-not", isDirectory: true)
            .appendingPathComponent("mcp.json")
        let store = MCPClientConfigStore(url: nestedURL)
        try await store.save(MCPClientConfig())
        #expect(FileManager.default.fileExists(atPath: nestedURL.path))
    }

    @Test
    func loadReturnsEmptyConfigOnMalformedJSON() async throws {
        let temp = try TempDir()
        let url = temp.url.appendingPathComponent("mcp.json")
        try Data("{not json".utf8).write(to: url)
        let store = MCPClientConfigStore(url: url)
        let cfg = await store.load()
        #expect(cfg.mcpServers.isEmpty, "malformed file must not crash; loader returns empty")
    }
}

private struct TempDir {
    let url: URL
    init() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("macmlx-mcp-config-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        self.url = base
    }
}
