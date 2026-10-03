// Copyright © 2026 macMLX. English comments only.

@testable import MacMLXCore

/// Seeded schema generator for the schema automaton's property tests (the
/// trap search). Deterministic for a given seed, so a failure reproduces.
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

    /// A random object schema with one to four scalar properties and a random
    /// `required` subset.
    mutating func object() -> JSONSchemaObject {
        var names = Self.names
        names.shuffle(using: &rng)
        let count = int(in: 1...4)
        var properties: [JSONSchemaObject.Property] = []
        for name in names.prefix(count) {
            properties.append(.init(name: name, type: pick(Self.scalars)))
        }
        var required: [String] = []
        for property in properties where bool() {
            required.append(property.name)
        }
        return JSONSchemaObject(properties: properties, required: required)
    }
}
