// JSONValue+Jinja.swift
// MacMLXCore
//
// A `JSONValue` as the chat template's own value type.

import Jinja
import OrderedCollections

extension JSONValue {
    /// This value as a `Jinja.Value`, every object with its keys in order.
    ///
    /// swift-transformers turns what it is handed into `Jinja.Value` with
    /// `Value(any:)`, which sorts the keys of a Swift dictionary and passes a
    /// `Jinja.Value` through as it is. So this is the shape in which a tool
    /// schema reaches `tojson` in the order the client sent it, the order
    /// Python prints it in. A JSON null is `Value.null`.
    func toJinjaValue() -> Jinja.Value {
        switch self {
        case .string(let value): return .string(value)
        case .int(let value): return .int(value)
        case .double(let value): return .double(value)
        case .bool(let value): return .boolean(value)
        case .null: return .null
        case .array(let value): return .array(value.map { $0.toJinjaValue() })
        case .object(let value):
            var members = OrderedDictionary<ObjectKey, Jinja.Value>(minimumCapacity: value.count)
            for (key, member) in value {
                members[.string(key)] = member.toJinjaValue()
            }
            return .object(members)
        }
    }
}
