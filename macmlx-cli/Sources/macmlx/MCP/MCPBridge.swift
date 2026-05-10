import Foundation
import MacMLXCore
import MCP

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

// MARK: - MCP SDK wiring

extension MCPBridge {
    /// The MCP tool definitions exposed by `macmlx mcp serve`.
    ///
    /// Static so callers (smoke tests, the SDK Server registration in
    /// `start(transport:)`) can introspect the same shape without
    /// instantiating an actor.
    public static var tools: [Tool] {
        [
            Tool(
                name: "list_models",
                description: "List MLX-format models that have been downloaded locally to ~/.mac-mlx/models. Returns id, displayName, sizeBytes, format, and metadata for each.",
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([:])
                ])
            ),
            Tool(
                name: "chat",
                description: "Run a chat completion against a locally-installed MLX model and return the assistant's reply as plain text. The model is loaded into memory on demand if not already loaded.",
                inputSchema: .object([
                    "type": .string("object"),
                    "required": .array([.string("model"), .string("messages")]),
                    "properties": .object([
                        "model": .object([
                            "type": .string("string"),
                            "description": .string("Model id or displayName (see list_models).")
                        ]),
                        "messages": .object([
                            "type": .string("array"),
                            "description": .string("OpenAI-shaped chat history."),
                            "items": .object([
                                "type": .string("object"),
                                "required": .array([.string("role"), .string("content")]),
                                "properties": .object([
                                    "role": .object([
                                        "type": .string("string"),
                                        "enum": .array([
                                            .string("system"),
                                            .string("user"),
                                            .string("assistant")
                                        ])
                                    ]),
                                    "content": .object([
                                        "type": .string("string")
                                    ])
                                ])
                            ])
                        ]),
                        "temperature": .object([
                            "type": .string("number"),
                            "description": .string("Sampling temperature (0.0 – 2.0). Defaults to 0.7.")
                        ]),
                        "max_tokens": .object([
                            "type": .string("integer"),
                            "description": .string("Maximum tokens to generate. Defaults to 2048.")
                        ]),
                        "system": .object([
                            "type": .string("string"),
                            "description": .string("Optional system prompt; overrides any leading system message.")
                        ])
                    ])
                ])
            )
        ]
    }

    /// Build a configured `Server`, register `tools/list` + `tools/call`
    /// handlers, then start it against the supplied transport. Returns
    /// only when the transport signals completion (typically EOF on
    /// stdin for `StdioTransport`).
    public func start(transport: any Transport, serverVersion: String) async throws {
        let server = Server(
            name: "macmlx",
            version: serverVersion,
            capabilities: .init(tools: .init(listChanged: false))
        )

        await server.withMethodHandler(ListTools.self) { _ in
            ListTools.Result(tools: MCPBridge.tools)
        }

        // The actor itself is captured via implicit `self`; the SDK's
        // handler closures are @Sendable so this is safe.
        let bridge = self
        await server.withMethodHandler(CallTool.self) { params in
            switch params.name {
            case "list_models":
                do {
                    let entries = try await bridge.listModels()
                    let json = try JSONEncoder().encode(entries)
                    let text = String(decoding: json, as: UTF8.self)
                    return CallTool.Result(
                        content: [.text(text: text, annotations: nil, _meta: nil)],
                        isError: false
                    )
                } catch {
                    return CallTool.Result(
                        content: [.text(text: "Error: \(error)", annotations: nil, _meta: nil)],
                        isError: true
                    )
                }

            case "chat":
                let args = params.arguments ?? [:]
                guard let model = args["model"]?.stringValue else {
                    return CallTool.Result(
                        content: [.text(text: "Missing required argument: model", annotations: nil, _meta: nil)],
                        isError: true
                    )
                }
                guard let rawMessages = args["messages"]?.arrayValue else {
                    return CallTool.Result(
                        content: [.text(text: "Missing required argument: messages", annotations: nil, _meta: nil)],
                        isError: true
                    )
                }
                let messages = rawMessages.compactMap { v -> MCPChatMessage? in
                    guard let dict = v.objectValue,
                          let role = dict["role"]?.stringValue,
                          let content = dict["content"]?.stringValue
                    else { return nil }
                    return MCPChatMessage(role: role, content: content)
                }
                let temperature = args["temperature"]?.doubleValue
                let maxTokens = args["max_tokens"]?.intValue
                let system = args["system"]?.stringValue

                do {
                    let reply = try await bridge.chat(
                        model: model,
                        messages: messages,
                        temperature: temperature,
                        maxTokens: maxTokens,
                        system: system
                    )
                    return CallTool.Result(
                        content: [.text(text: reply, annotations: nil, _meta: nil)],
                        isError: false
                    )
                } catch {
                    return CallTool.Result(
                        content: [.text(text: "\(error)", annotations: nil, _meta: nil)],
                        isError: true
                    )
                }

            default:
                return CallTool.Result(
                    content: [.text(text: "Unknown tool: \(params.name)", annotations: nil, _meta: nil)],
                    isError: true
                )
            }
        }

        try await server.start(transport: transport)
        // `Server.start` returns once the receive-loop task is wired up;
        // for stdio that loop runs to EOF on stdin. Block here so the
        // CLI process stays alive until the host closes its end.
        await server.waitUntilCompleted()
    }
}
