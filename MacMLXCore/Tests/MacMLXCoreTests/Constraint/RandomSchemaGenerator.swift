// Copyright © 2026 macMLX. English comments only.

@testable import MacMLXCore

/// Seeded schema and document generator for the schema automaton's property
/// tests (the trap search and the differential test against
/// ``ReferenceSchemaValidator``). Deterministic for a given seed, so a failure
/// reproduces.
struct RandomSchemaGenerator {

    /// SplitMix64: a tiny, well-mixed, seedable generator.
    struct SplitMix64: RandomNumberGenerator {
        var state: UInt64

        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }

    /// Property names: shared prefixes (`a`, `ab`, `abc`) and names that are
    /// also schema keywords (`type`, `items`).
    static let names = ["a", "ab", "b", "type", "items", "abc"]

    /// Scalar types, including enums whose values share a prefix and the empty
    /// string as an enum value.
    static let scalars: [SchemaValueType] = [
        .string, .number, .integer, .boolean,
        .stringEnum(["x", "xy", "y"]), .stringEnum([""]), .stringEnum(["a"]),
    ]

    var rng: SplitMix64

    init(seed: UInt64) {
        rng = SplitMix64(state: seed)
    }

    mutating func int(in range: ClosedRange<Int>) -> Int {
        Int.random(in: range, using: &rng)
    }

    mutating func bool() -> Bool {
        Bool.random(using: &rng)
    }

    /// A uniformly chosen element of a non-empty array.
    mutating func pick<T>(_ items: [T]) -> T {
        items[Int.random(in: 0..<items.count, using: &rng)]
    }

    /// A random object schema with one to four properties and a random
    /// `required` subset. With `nested`, a property may be an object or an
    /// array (with random bounds, possibly unbounded) down to depth 3;
    /// otherwise every property is a scalar.
    mutating func object(depth: Int = 1, nested: Bool = false) -> JSONSchemaObject {
        var names = Self.names
        names.shuffle(using: &rng)
        let count = int(in: 1...4)
        var properties: [JSONSchemaObject.Property] = []
        for name in names.prefix(count) {
            properties.append(.init(name: name, type: valueType(depth: depth, nested: nested)))
        }
        var required: [String] = []
        for property in properties where bool() {
            required.append(property.name)
        }
        return JSONSchemaObject(properties: properties, required: required)
    }

    mutating func valueType(depth: Int, nested: Bool) -> SchemaValueType {
        if !nested || depth >= 3 || int(in: 0...9) < 6 {
            return pick(Self.scalars)
        }
        if bool() {
            return .object(object(depth: depth + 1, nested: true))
        }
        let minItems = int(in: 0...2)
        let maxItems: Int? = bool() ? nil : minItems + int(in: 0...2)
        return .array(items: valueType(depth: depth + 1, nested: true), minItems: minItems, maxItems: maxItems)
    }

    // MARK: Documents

    /// A random document conforming to `type`, with random whitespace between
    /// tokens. Strings exercise escapes, a surrogate pair, raw UTF-8 and
    /// structural characters; an unbounded array gets up to three items more
    /// than its minimum.
    mutating func document(for type: SchemaValueType) -> String {
        switch type {
        case .string:
            return "\"" + pick(["", "hi", "a\\nb", "\\u00e9", "\\uD83D\\uDE00", "q\\\"q", "é", "{,}"]) + "\""
        case .number:
            return pick(["0", "-0", "3.14", "1e10", "-2.5E-3", "42", "0.5"])
        case .integer:
            return pick(["0", "-7", "42", "100"])
        case .boolean:
            return bool() ? "true" : "false"
        case .stringEnum(let values):
            return "\"" + pick(values) + "\""
        case .object(let object):
            var members: [JSONSchemaObject.Property] = []
            for property in object.properties where object.required.contains(property.name) || bool() {
                members.append(property)
            }
            members.shuffle(using: &rng)
            var parts: [String] = []
            for member in members {
                let value = document(for: member.type)
                parts.append(whitespace() + "\"\(member.name)\"" + whitespace() + ":" + whitespace() + value + whitespace())
            }
            return "{" + (parts.isEmpty ? whitespace() : parts.joined(separator: ",")) + "}"
        case .array(let items, let minItems, let maxItems):
            let count = int(in: minItems...(maxItems ?? (minItems + 3)))
            var parts: [String] = []
            for _ in 0..<count {
                let item = document(for: items)
                parts.append(whitespace() + item + whitespace())
            }
            return "[" + (parts.isEmpty ? whitespace() : parts.joined(separator: ",")) + "]"
        }
    }

    /// Usually nothing; sometimes a space, newline, tab or two spaces.
    mutating func whitespace() -> String {
        int(in: 0...3) == 0 ? pick([" ", "\n", "\t", "  "]) : ""
    }

    /// One to three random byte edits of `document`: a deletion, an insertion
    /// or a replacement from a JSON-ish alphabet, or a duplicated span.
    mutating func mutate(_ document: String) -> [UInt8] {
        var bytes = Array(document.utf8)
        let alphabet = Array("{}[],:\"0123456789.-eEtrufalsn abxy\\".utf8)
        for _ in 0..<int(in: 1...3) {
            switch int(in: 0...3) {
            case 0 where !bytes.isEmpty:
                bytes.remove(at: int(in: 0...(bytes.count - 1)))
            case 1:
                bytes.insert(pick(alphabet), at: int(in: 0...bytes.count))
            case 2 where !bytes.isEmpty:
                bytes[int(in: 0...(bytes.count - 1))] = pick(alphabet)
            default:
                if bytes.count > 2 {
                    let lower = int(in: 0...(bytes.count - 2))
                    let upper = int(in: lower...(bytes.count - 1))
                    bytes.insert(contentsOf: bytes[lower...upper], at: upper + 1)
                }
            }
        }
        return bytes
    }
}
