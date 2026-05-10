import Foundation
import MacMLXCore

/// Source of locally-downloaded models the bridge can advertise.
///
/// In production this is wrapped around `ModelLibraryManager.scan(_:)`
/// (see `CLIModelSource` in `MCPCommand.swift`). The protocol exists so
/// tests can plug in a fixed list without standing up a real on-disk
/// model directory. Public so `MCPBridge.init` can stay public alongside
/// the rest of the bridge surface.
public protocol ModelSource: Sendable {
    func scan() async throws -> [LocalModel]
}

/// One row of the `list_models` MCP tool response.
///
/// Sendable + Codable so it round-trips cleanly through XCTest
/// assertions and through the MCP SDK's JSON-Value encoder. Mirrors
/// the public-facing fields of `LocalModel` minus the on-disk
/// `directory` (which leaks the user's home path and isn't useful to
/// remote tool callers).
public struct MCPModelEntry: Sendable, Codable, Equatable {
    public let id: String
    public let displayName: String
    public let sizeBytes: Int64
    public let format: String
    public let quantization: String?
    public let parameterCount: String?
    public let architecture: String?
}

/// One wire-format message in the MCP `chat` tool input.
///
/// Sendable + Codable so it round-trips cleanly through the SDK's
/// JSON-Value encoder and through XCTest assertions. Mirrors the shape
/// MCP clients (Claude Desktop, Cursor, Zed) already speak when calling
/// OpenAI-compatible chat APIs.
public struct MCPChatMessage: Sendable, Codable, Equatable {
    public let role: String
    public let content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

/// Errors surfaced by the bridge to MCP callers.
public enum MCPBridgeError: Error, CustomStringConvertible {
    case modelNotFound(String)

    public var description: String {
        switch self {
        case .modelNotFound(let id):
            return "Model not found: \(id). Run `macmlx list` to see available models."
        }
    }
}

/// Hosts the long-lived state for a `macmlx mcp serve` process.
///
/// Designed as an actor because the MCP SDK's handler closures run on
/// arbitrary executor queues; the actor's serialised access removes
/// the need for any external locking around model swaps or the engine
/// reference. Owns:
///
/// - the `ModelSource` (typically a `ModelLibraryManager` adapter)
/// - an `EngineFactory` that lazily produces the `InferenceEngine` on
///   the first `chat` call (mirrors the engine-construction pattern in
///   `RunCommand` / `ServeCommand` so `Settings.preferredEngine` is
///   honoured)
///
/// The MCP SDK's `Server` is built and started via `start(transport:)`
/// (Task 5 of the v0.4 MCP plan). The bridge stays decoupled from the
/// SDK in this file so the MVP can be unit-tested without piping
/// JSON-RPC end-to-end.
public actor MCPBridge {
    public typealias EngineFactory = @Sendable () throws -> any InferenceEngine

    private let library: ModelSource
    private let engineFactory: EngineFactory
    private var engine: (any InferenceEngine)?

    public init(library: ModelSource, engineFactory: @escaping EngineFactory) {
        self.library = library
        self.engineFactory = engineFactory
    }

    /// Implements the MCP `list_models` tool.
    ///
    /// Returns every locally-downloaded model with stable, JSON-shaped
    /// metadata. No engine load is required — this is a directory scan.
    public func listModels() async throws -> [MCPModelEntry] {
        let models = try await library.scan()
        return models.map { m in
            MCPModelEntry(
                id: m.id,
                displayName: m.displayName,
                sizeBytes: m.sizeBytes,
                format: m.format.rawValue,
                quantization: m.quantization,
                parameterCount: m.parameterCount,
                architecture: m.architecture
            )
        }
    }

    /// Implements the MCP `chat` tool.
    ///
    /// Buffers the entire generation into a single string before
    /// returning — MCP tool calls don't have a streaming surface in
    /// 0.12.x, so this is the right shape until the SDK adds chunked
    /// tool responses (revisit in v0.5+). Lazy-loads the engine and
    /// swaps the loaded model on demand.
    public func chat(
        model: String,
        messages: [MCPChatMessage],
        temperature: Double?,
        maxTokens: Int?,
        system: String?
    ) async throws -> String {
        let engine = try await engine(for: model)

        var chatMessages = messages.compactMap { wire -> ChatMessage? in
            guard let role = MessageRole(rawValue: wire.role) else { return nil }
            return ChatMessage(role: role, content: wire.content)
        }
        // Explicit `system` field wins; otherwise lift a leading system
        // message into `systemPrompt`. Matches HummingbirdServer's
        // OpenAI-compat handling, which prevents the [system, system,
        // user, …] sequence that broke Qwen3 / Gemma / DeepSeek's
        // strict Jinja templates in v0.3.6.
        var systemPrompt: String? = system
        if systemPrompt == nil,
           chatMessages.first?.role == .system {
            systemPrompt = chatMessages.removeFirst().content
        } else if systemPrompt != nil {
            chatMessages.removeAll(where: { $0.role == .system })
        }

        var params = GenerationParameters()
        if let temperature { params.temperature = temperature }
        if let maxTokens { params.maxTokens = maxTokens }
        params.stream = false

        let request = GenerateRequest(
            model: model,
            messages: chatMessages,
            systemPrompt: systemPrompt,
            parameters: params
        )

        var buffer = ""
        for try await chunk in engine.generate(request) {
            buffer.append(chunk.text)
        }
        return buffer
    }

    /// Lazy-create the engine and lazy-swap the loaded model.
    ///
    /// First `chat` call constructs the engine via `engineFactory`
    /// (which honours `Settings.preferredEngine`); subsequent calls
    /// reuse the same engine instance and only re-`load` when the
    /// requested model differs from the currently-loaded one.
    private func engine(for modelID: String) async throws -> any InferenceEngine {
        let models = try await library.scan()
        guard let local = models.first(where: { $0.id == modelID || $0.displayName == modelID }) else {
            throw MCPBridgeError.modelNotFound(modelID)
        }

        let engine = try (self.engine ?? engineFactory())
        self.engine = engine

        let current = await engine.loadedModel
        if current?.id != local.id {
            try await engine.load(local)
        }
        return engine
    }
}
