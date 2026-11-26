import Foundation

@testable import MacMLXCore

import enum MacMLXCore.JSONValue

extension JSONValue {
    /// This value written out with every object's keys in the order the
    /// object holds them, for a test to compare key order as text. Strings are
    /// not quoted or escaped; it is for reading, not for parsing.
    var orderedDescription: String {
        switch self {
        case .string(let value): return value
        case .int(let value): return String(value)
        case .double(let value): return String(value)
        case .bool(let value): return String(value)
        case .null: return "null"
        case .array(let value): return "[" + value.map(\.orderedDescription).joined(separator: ",") + "]"
        case .object(let value):
            return "{" + value.map { "\($0.key):\($0.value.orderedDescription)" }.joined(separator: ",") + "}"
        }
    }
}
