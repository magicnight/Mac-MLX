import Testing
import Foundation
import MacMLXCore
@testable import macmlx

/// Stub `ModelSource` for tests — returns a fixed list without
/// touching the filesystem or `ModelLibraryManager`.
struct StubLibrary: ModelSource {
    let models: [LocalModel]
    func scan() async throws -> [LocalModel] { models }
}

@Suite("MCPBridge")
struct MCPBridgeTests {

    @Test
    func listModelsReturnsLibraryEntries() async throws {
        let stub = StubLibrary(models: [
            LocalModel(
                id: "Qwen3-8B-4bit",
                displayName: "Qwen3-8B-4bit",
                directory: URL(fileURLWithPath: "/tmp/Qwen3-8B-4bit"),
                sizeBytes: 4_500_000_000,
                format: .mlx,
                quantization: "4bit",
                parameterCount: "8B",
                architecture: "qwen3"
            )
        ])
        let bridge = MCPBridge(library: stub)

        let payload = try await bridge.listModels()
        #expect(payload.count == 1)
        #expect(payload[0].id == "Qwen3-8B-4bit")
        #expect(payload[0].sizeBytes == 4_500_000_000)
        #expect(payload[0].format == "mlx")
        #expect(payload[0].quantization == "4bit")
        #expect(payload[0].parameterCount == "8B")
        #expect(payload[0].architecture == "qwen3")
    }

    @Test
    func listModelsEmptyLibraryReturnsEmptyArray() async throws {
        let bridge = MCPBridge(library: StubLibrary(models: []))
        let payload = try await bridge.listModels()
        #expect(payload.isEmpty)
    }

    @Test
    func mcpModelEntryRoundTripsThroughJSON() throws {
        let entry = MCPModelEntry(
            id: "test",
            displayName: "Test",
            sizeBytes: 1_234_567,
            format: "mlx",
            quantization: nil,
            parameterCount: nil,
            architecture: nil
        )
        let data = try JSONEncoder().encode(entry)
        let decoded = try JSONDecoder().decode(MCPModelEntry.self, from: data)
        #expect(decoded == entry)
    }
}
