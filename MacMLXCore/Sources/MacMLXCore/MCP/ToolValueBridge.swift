// ToolValueBridge.swift
// MacMLXCore
//
// Pure value converters that bridge the three JSON-shaped value trees the
// chat-side MCP tool router has to move data between:
//
// - `MacMLXCore.JSONValue` (Util/JSONValue.swift) — the wire type macMLX owns
//   and persists. Cases: string/int/double/bool/null/array/object.
// - `MCP.Value` (swift-sdk) — what `MCPClientPool.callTool(arguments:)` accepts
//   and what a `Tool.inputSchema` is expressed in. Adds a `.data(mimeType:,Data)`
//   case with no macMLX counterpart.
// - `MLXLMCommon.JSONValue` (mlx-swift-lm) — what a parsed `ToolCall`'s
//   `function.arguments` dictionary uses. Cases are identical to macMLX's, so
//   the conversion is total and lossless.
//
// All functions are pure and free of MLX/Metal, so the suite exercises them
// without a model. End-to-end an upstream tool call reaches an MCP server via
// two hops that compose here: `jsonValue(from: MLXLMCommon.JSONValue)` (in the
// engine, building a `ToolCallRequest`) then `mcpValue(from:)` (in
// `ToolCallingSession`'s pool wrapper, building the `callTool` arguments).

import Foundation
import MCP
import MLXLMCommon

/// Namespace for the tool-routing value converters. An enum with only static
/// members so it can't be instantiated.
public enum ToolValueBridge {

    // MARK: - MCP.Value → macMLX JSONValue

    /// Convert an `MCP.Value` (e.g. a tool's `inputSchema` or a tool call's
    /// structured result) into macMLX's `JSONValue`.
    ///
    /// - Note: `MCP.Value` has a `.data(mimeType:,Data)` case with no macMLX
    ///   equivalent. It is mapped to a **base64 string** of the bytes; the MIME
    ///   type is dropped. This is lossy but keeps the value tree fully
    ///   `Codable` and human-inspectable — MCP tool *schemas* never contain raw
    ///   data, so in practice this only fires for the rare tool result that
    ///   inlines binary content, where a base64 string is the reasonable
    ///   textual stand-in.
    public static func jsonValue(from value: MCP.Value) -> JSONValue {
        switch value {
        case .null:
            return .null
        case .bool(let bool):
            return .bool(bool)
        case .int(let int):
            return .int(int)
        case .double(let double):
            return .double(double)
        case .string(let string):
            return .string(string)
        case .data(_, let data):
            return .string(data.base64EncodedString())
        case .array(let array):
            return .array(array.map(jsonValue(from:)))
        case .object(let object):
            return .object(object.mapValues(jsonValue(from:)))
        }
    }

    // MARK: - macMLX JSONValue → MCP.Value

    /// Convert a macMLX `JSONValue` into an `MCP.Value` — used to build the
    /// `arguments` dictionary handed to `MCPClientPool.callTool`. Total: macMLX
    /// has no case without an MCP counterpart.
    public static func mcpValue(from json: JSONValue) -> MCP.Value {
        switch json {
        case .null:
            return .null
        case .bool(let bool):
            return .bool(bool)
        case .int(let int):
            return .int(int)
        case .double(let double):
            return .double(double)
        case .string(let string):
            return .string(string)
        case .array(let array):
            return .array(array.map(mcpValue(from:)))
        case .object(let object):
            return .object(object.mapValues(mcpValue(from:)))
        }
    }

    // MARK: - MLXLMCommon.JSONValue → macMLX JSONValue

    /// Convert an upstream `MLXLMCommon.JSONValue` (a parsed `ToolCall`'s
    /// argument value) into macMLX's `JSONValue`. The two enums share the same
    /// seven cases, so this is total and lossless.
    public static func jsonValue(from upstream: MLXLMCommon.JSONValue) -> JSONValue {
        switch upstream {
        case .null:
            return .null
        case .bool(let bool):
            return .bool(bool)
        case .int(let int):
            return .int(int)
        case .double(let double):
            return .double(double)
        case .string(let string):
            return .string(string)
        case .array(let array):
            return .array(array.map(jsonValue(from:)))
        case .object(let object):
            return .object(object.mapValues(jsonValue(from:)))
        }
    }

    // MARK: - MCP.Tool → OpenAI function spec

    /// Convert an MCP `Tool` into the OpenAI "function" tool spec shape that a
    /// chat template expects in `GenerateRequest.tools`:
    ///
    /// ```json
    /// { "type": "function",
    ///   "function": { "name": …, "description": …, "parameters": <inputSchema> } }
    /// ```
    ///
    /// The tool's `inputSchema` (a JSON Schema expressed as `MCP.Value`) becomes
    /// the `parameters` object verbatim. A `nil` description becomes an empty
    /// string so the rendered spec always carries the key.
    public static func openAIToolSpec(from tool: MCP.Tool) -> JSONValue {
        .object([
            "type": .string("function"),
            "function": .object([
                "name": .string(tool.name),
                "description": .string(tool.description ?? ""),
                "parameters": jsonValue(from: tool.inputSchema),
            ]),
        ])
    }
}
