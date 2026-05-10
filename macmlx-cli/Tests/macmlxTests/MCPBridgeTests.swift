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

/// Tiny in-memory `InferenceEngine` so the bridge's `chat` flow can be
/// exercised without standing up MLX. Yields the configured chunks
/// verbatim, then finishes the stream.
actor StubEngine: InferenceEngine {
    nonisolated let engineID: EngineID = .mlxSwift

    var status: EngineStatus = .idle
    var loadedModel: LocalModel?
    nonisolated let version = "stub-1.0"

    private let chunks: [String]

    init(chunks: [String]) {
        self.chunks = chunks
    }

    func load(_ model: LocalModel) async throws {
        loadedModel = model
        status = .ready(model: model.id)
    }

    func unload() async throws {
        loadedModel = nil
        status = .idle
    }

    func healthCheck() async -> Bool { true }

    nonisolated func generate(_ request: GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error> {
        let chunks = chunks
        return AsyncThrowingStream { continuation in
            Task {
                for token in chunks {
                    continuation.yield(GenerateChunk(text: token))
                }
                continuation.finish()
            }
        }
    }
}

/// Convenience for tests that don't exercise `chat` — a factory that
/// would explode if accidentally invoked.
@Sendable
private func unreachableEngine() throws -> any InferenceEngine {
    fatalError("engine factory should not run for list-models-only tests")
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
        let bridge = MCPBridge(library: stub, engineFactory: unreachableEngine)

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
        let bridge = MCPBridge(
            library: StubLibrary(models: []),
            engineFactory: unreachableEngine
        )
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

    @Test
    func chatBuffersTokensAndReturnsAssistantText() async throws {
        let stub = StubLibrary(models: [
            LocalModel(
                id: "test-model",
                displayName: "test-model",
                directory: URL(fileURLWithPath: "/tmp/test-model"),
                sizeBytes: 1,
                format: .mlx,
                quantization: nil,
                parameterCount: nil,
                architecture: nil
            )
        ])
        let engine = StubEngine(chunks: ["Hello", " ", "world"])
        let bridge = MCPBridge(library: stub, engineFactory: { engine })

        let result = try await bridge.chat(
            model: "test-model",
            messages: [
                MCPChatMessage(role: "user", content: "Hi")
            ],
            temperature: nil,
            maxTokens: nil,
            system: nil
        )
        #expect(result == "Hello world")

        // Engine state should reflect the on-demand load.
        let loaded = await engine.loadedModel
        #expect(loaded?.id == "test-model")
    }

    @Test
    func chatLooksUpModelByDisplayName() async throws {
        let stub = StubLibrary(models: [
            LocalModel(
                id: "Org/test-model",
                displayName: "test-model",
                directory: URL(fileURLWithPath: "/tmp/test"),
                sizeBytes: 1,
                format: .mlx,
                quantization: nil,
                parameterCount: nil,
                architecture: nil
            )
        ])
        let engine = StubEngine(chunks: ["ok"])
        let bridge = MCPBridge(library: stub, engineFactory: { engine })

        let result = try await bridge.chat(
            model: "test-model", // displayName, not id
            messages: [MCPChatMessage(role: "user", content: "hi")],
            temperature: nil,
            maxTokens: nil,
            system: nil
        )
        #expect(result == "ok")
        let loaded = await engine.loadedModel
        #expect(loaded?.id == "Org/test-model")
    }

    @Test
    func chatThrowsOnUnknownModel() async throws {
        let bridge = MCPBridge(
            library: StubLibrary(models: []),
            engineFactory: { StubEngine(chunks: []) }
        )

        await #expect(throws: MCPBridgeError.self) {
            try await bridge.chat(
                model: "missing",
                messages: [MCPChatMessage(role: "user", content: "x")],
                temperature: nil,
                maxTokens: nil,
                system: nil
            )
        }
    }

    @Test
    func chatPromotesLeadingSystemMessageToSystemPrompt() async throws {
        // Capture the request the engine sees so we can assert that the
        // bridge correctly hoisted the system-role turn into systemPrompt.
        actor RequestCapture {
            var captured: GenerateRequest?
            func set(_ r: GenerateRequest) { captured = r }
        }
        let capture = RequestCapture()

        actor CapturingEngine: InferenceEngine {
            nonisolated let engineID: EngineID = .mlxSwift
            var status: EngineStatus = .idle
            var loadedModel: LocalModel?
            nonisolated let version = "capture"
            let capture: RequestCapture
            init(capture: RequestCapture) { self.capture = capture }
            func load(_ model: LocalModel) async throws {
                loadedModel = model
                status = .ready(model: model.id)
            }
            func unload() async throws { loadedModel = nil; status = .idle }
            func healthCheck() async -> Bool { true }
            nonisolated func generate(_ request: GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error> {
                let capture = capture
                return AsyncThrowingStream { continuation in
                    Task {
                        await capture.set(request)
                        continuation.yield(GenerateChunk(text: "ok"))
                        continuation.finish()
                    }
                }
            }
        }
        let engine = CapturingEngine(capture: capture)
        let stub = StubLibrary(models: [
            LocalModel(
                id: "m",
                displayName: "m",
                directory: URL(fileURLWithPath: "/tmp/m"),
                sizeBytes: 1,
                format: .mlx,
                quantization: nil,
                parameterCount: nil,
                architecture: nil
            )
        ])
        let bridge = MCPBridge(library: stub, engineFactory: { engine })

        _ = try await bridge.chat(
            model: "m",
            messages: [
                MCPChatMessage(role: "system", content: "be terse"),
                MCPChatMessage(role: "user", content: "hi")
            ],
            temperature: nil,
            maxTokens: nil,
            system: nil
        )

        let captured = await capture.captured
        #expect(captured?.systemPrompt == "be terse")
        #expect(captured?.messages.count == 1)
        #expect(captured?.messages.first?.role == .user)
    }
}
