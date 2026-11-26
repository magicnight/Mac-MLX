// JSONObject.swift
// MacMLXCore
//
// The payload of `JSONValue.object`: a JSON object that remembers the order
// of its keys.

import Foundation

/// A JSON object whose keys keep the order they were given in.
///
/// A chat template prints a tool schema with `tojson`, and Python prints the
/// keys in the order the client sent them, so that order is part of the prompt
/// a model was trained on. A Swift `Dictionary` has no order, which is why this
/// type exists.
///
/// The order is what a reader of the keys sees (`keys`, `values`, iteration);
/// it is not part of the value. Two objects with the same members are equal and
/// hash alike whatever their key order, as two `Dictionary` values would.
///
/// Where the order comes from:
/// - a dictionary literal keeps the order it is written in;
/// - decoding keeps the order of the JSON text when the decoder was given one
///   (see `JSONKeyOrder`), and sorts the keys otherwise;
/// - `init(_:)` over a `Dictionary` sorts the keys, so the result does not
///   depend on the dictionary's hash seed.
///
/// Encoding does not promise an order: `JSONEncoder` writes a keyed container
/// in its own order. It also goes through a keyed container where a
/// `[String: JSONValue]` did not, so a coder with a key strategy
/// (`convertFromSnakeCase`) would rewrite the keys. Nothing here sets one; do
/// not decode or encode a `JSONValue` with one.
public struct JSONObject: Sendable {
    /// The keys, in order. Every key is in `storage`, once.
    public private(set) var keys: [String]
    private var storage: [String: JSONValue]

    /// An empty object.
    public init() {
        keys = []
        storage = [:]
    }

    /// The members of `dictionary`, keys sorted.
    public init(_ dictionary: [String: JSONValue]) {
        keys = dictionary.keys.sorted()
        storage = dictionary
    }

    /// The pairs of `members` in their order. A key that repeats keeps its
    /// first position and its last value.
    public init(_ members: some Sequence<(String, JSONValue)>) {
        self.init()
        for (key, value) in members {
            self[key] = value
        }
    }

    public var count: Int { keys.count }
    public var isEmpty: Bool { keys.isEmpty }

    /// The values, in key order.
    public var values: [JSONValue] { keys.compactMap { storage[$0] } }

    /// The members as a plain dictionary, for a caller that has no use for the
    /// order.
    public var dictionary: [String: JSONValue] { storage }

    /// Reads, sets or removes a member. A new key goes last; a key that is
    /// already there keeps its position.
    public subscript(key: String) -> JSONValue? {
        get { storage[key] }
        set {
            if let newValue {
                if storage.updateValue(newValue, forKey: key) == nil {
                    keys.append(key)
                }
            } else if storage.removeValue(forKey: key) != nil {
                keys.removeAll { $0 == key }
            }
        }
    }

    /// Removes a member and returns its value, `nil` when there was none.
    @discardableResult
    public mutating func removeValue(forKey key: String) -> JSONValue? {
        let removed = storage[key]
        self[key] = nil
        return removed
    }

    /// The same keys in the same order, each value transformed.
    public func mapValues(_ transform: (JSONValue) throws -> JSONValue) rethrows -> JSONObject {
        var result = self
        result.storage = try storage.mapValues(transform)
        return result
    }

    /// The values transformed into a plain dictionary. The order is dropped:
    /// this is for the plain shapes `JSONSerialization` and the tokenizer take.
    public func mapValues<T>(_ transform: (JSONValue) throws -> T) rethrows -> [String: T] {
        try storage.mapValues(transform)
    }

    /// The members that satisfy `isIncluded`, in order.
    public func filter(_ isIncluded: ((key: String, value: JSONValue)) throws -> Bool) rethrows -> JSONObject {
        var result = JSONObject()
        for member in self where try isIncluded(member) {
            result[member.key] = member.value
        }
        return result
    }
}

// MARK: - Sequence

extension JSONObject: Sequence {
    /// Iterates the members in key order.
    public func makeIterator() -> AnyIterator<(key: String, value: JSONValue)> {
        var index = keys.startIndex
        return AnyIterator {
            while index < keys.endIndex {
                let key = keys[index]
                index += 1
                if let value = storage[key] {
                    return (key: key, value: value)
                }
            }
            return nil
        }
    }
}

// MARK: - Literal

extension JSONObject: ExpressibleByDictionaryLiteral {
    /// The pairs in the order they are written.
    public init(dictionaryLiteral elements: (String, JSONValue)...) {
        self.init(elements)
    }
}

// MARK: - Equality

extension JSONObject: Hashable {
    public static func == (lhs: JSONObject, rhs: JSONObject) -> Bool {
        lhs.storage == rhs.storage
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(storage)
    }
}

// MARK: - Codable

extension JSONObject: Codable {
    private struct MemberKey: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }

        init(_ string: String) { stringValue = string }
        init?(stringValue: String) { self.init(stringValue) }
        init?(intValue: Int) { nil }
    }

    /// Whether the value `decoder` is positioned at is a JSON object.
    static func isObject(at decoder: any Decoder) -> Bool {
        (try? decoder.container(keyedBy: MemberKey.self)) != nil
    }

    /// Decodes a JSON object. The keys take the order of the JSON text when the
    /// decoder carries a `JSONKeyOrder` that knows this object, and sorted
    /// order otherwise.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: MemberKey.self)
        let present = Set(container.allKeys.map(\.stringValue))
        let recorded = JSONKeyOrder.keys(at: decoder)?.filter(present.contains) ?? []
        // A key the recorded order does not name (there is no order, or the
        // scan and the decoder disagree about this object) goes after the
        // recorded ones, sorted.
        let ordered = recorded + present.subtracting(recorded).sorted()
        self.init()
        for key in ordered {
            self[key] = try container.decode(JSONValue.self, forKey: MemberKey(key))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: MemberKey.self)
        for (key, value) in self {
            try container.encode(value, forKey: MemberKey(key))
        }
    }
}
