// ToolCallRequest.swift
// MacMLXCore
//
// A tool call the model asked to make, in macMLX's own wire vocabulary.
//
// This deliberately MIRRORS mlx-swift-lm's `MLXLMCommon.ToolCall` rather than
// re-exporting it: the wire types (`GenerateChunk`, `ChatMessage`) stay free of
// an `MLXLMCommon` import, so they remain plain `Codable` values the server,
// GUI, and on-disk conversation store can move around without linking the
// inference engine. The engine converts upstream ⇄ macMLX at its boundary
// (`MLXSwiftEngine.toolCallRequest(from:)` / `upstreamToolCall(from:)`).

import Foundation

/// A single tool invocation requested by the model during generation.
///
/// Shapes 1:1 to an OpenAI `tool_call`: a stable `id` (used to correlate the
/// call with the tool result message that answers it), the tool `name`, and
/// the decoded `arguments` object.
public struct ToolCallRequest: Codable, Hashable, Sendable {
    /// Correlation id, echoed back on the answering `.tool` message. Always
    /// populated — the engine synthesises `call_<uuid>` if the model's format
    /// didn't carry one.
    public let id: String
    /// The tool name the model wants to call.
    public let name: String
    /// Decoded call arguments as a JSON object.
    public let arguments: [String: JSONValue]

    public init(id: String, name: String, arguments: [String: JSONValue]) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }
}
