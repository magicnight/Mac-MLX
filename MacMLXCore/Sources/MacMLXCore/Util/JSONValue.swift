// JSONValue.swift
// MacMLXCore
//
// A small, fully-Codable JSON value tree. Introduced for per-model
// chat-template kwargs (v0.5.1): free-form values a user pins to a
// model (e.g. `{"enable_thinking": true}` for Qwen3) that need to
// round-trip through `ModelParameters` persistence and then reach the
// Jinja chat template as `additionalContext`.
//
// Codable is hand-written (single-value container, type-probing decode)
// so the enum round-trips through `JSONEncoder`/`JSONDecoder` unchanged.

import Foundation

/// A JSON value: string, integer, double, bool, null, array, or object.
///
/// `Codable` so it persists inside `ModelParameters`; `Hashable` so the
/// containing structs keep their synthesised conformances; `Sendable`
/// for strict-concurrency crossing into the engine actors.
public enum JSONValue: Codable, Hashable, Sendable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case null
    case array([JSONValue])
    case object([String: JSONValue])

    // MARK: - Decoding

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
            return
        }
        // Probe concrete types in an order that avoids Foundation's
        // number/bool ambiguity: `Bool` first (JSON `true`/`false` only —
        // numbers throw), then `Int` (JSON `1.5` throws), then `Double`.
        if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .int(value)
        } else if let value = try? container.decode(Double.self) {
            self = .double(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else if let value = try? container.decode([String: JSONValue].self) {
            self = .object(value)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Unsupported JSON value"
            )
        }
    }

    // MARK: - Encoding

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .int(let value): try container.encode(value)
        case .double(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        case .null: try container.encodeNil()
        case .array(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        }
    }

    // MARK: - Bridging

    /// Unwrap to the plain `Sendable` shape the tokenizer's
    /// `additionalContext` expects (`String`/`Int`/`Double`/`Bool`/
    /// `NSNull`, or nested `[any Sendable]` / `[String: any Sendable]`).
    public func toSendable() -> any Sendable {
        switch self {
        case .string(let value): return value
        case .int(let value): return value
        case .double(let value): return value
        case .bool(let value): return value
        case .null: return NSNull()
        case .array(let value): return value.map { $0.toSendable() }
        case .object(let value): return value.mapValues { $0.toSendable() }
        }
    }
}
