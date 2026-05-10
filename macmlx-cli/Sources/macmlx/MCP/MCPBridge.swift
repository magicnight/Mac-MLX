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

/// Hosts the long-lived state for a `macmlx mcp serve` process.
///
/// Designed as an actor because the MCP SDK's handler closures run on
/// arbitrary executor queues; the actor's serialised access removes
/// the need for any external locking around model swaps or the engine
/// reference. Owns:
///
/// - the `ModelSource` (typically a `ModelLibraryManager` adapter)
/// - the lazy `InferenceEngine` (created on the first `chat` call)
///
/// The MCP SDK's `Server` is built and started via `start(transport:)`
/// — see Task 5 of the v0.4 MCP plan. The bridge stays decoupled from
/// the SDK in this file so the MVP can be unit-tested without piping
/// JSON-RPC end-to-end.
public actor MCPBridge {
    private let library: ModelSource

    public init(library: ModelSource) {
        self.library = library
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
}
