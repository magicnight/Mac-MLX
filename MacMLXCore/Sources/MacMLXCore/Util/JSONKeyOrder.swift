// JSONKeyOrder.swift
// MacMLXCore
//
// The key order of every object in a JSON text, for `JSONObject` to decode in.

import Foundation

/// The order of the keys of every object in one JSON text.
///
/// `JSONDecoder` does not report it: a keyed container's `allKeys` come out of
/// a dictionary, in a different order on every run. So the text is scanned once
/// before it is decoded, and the scan rides along in the decoder's `userInfo`,
/// where `JSONObject.init(from:)` finds the order of the object it is decoding
/// by its coding path.
///
/// ```swift
/// let request = try JSONKeyOrder.decode(ChatCompletionRequest.self, from: body)
/// ```
///
/// The scan only records; it never rejects. On text it cannot follow it stops,
/// the objects it did not reach decode with sorted keys, and reporting the
/// malformed text is left to the decoder.
public struct JSONKeyOrder: Sendable {
    /// One step from a container to a child: a member of an object or an
    /// element of an array.
    private enum Component: Hashable {
        case member(String)
        case element(Int)
    }

    /// The step `component` taken from the container numbered `parent`.
    private struct Edge: Hashable {
        let parent: Int
        let component: Component
    }

    /// The number of the text's top-level value.
    private static let root = 0

    /// Object keys in text order, by the number of the object that has them.
    /// An object with fewer than two keys is left out: it has one order.
    private let orders: [Int: [String]]
    /// The numbers of the containers that lead to an object in `orders`, by
    /// the step that reaches them. A container is a number and one step from
    /// its parent, never a whole path, so what the scan keeps grows with the
    /// text and not with how deep the text nests.
    private let children: [Edge: Int]

    /// Scans `data`, a JSON text in UTF-8.
    public init(scanning data: Data) {
        (orders, children) = data.withUnsafeBytes { raw in
            var scanner = Scanner(bytes: raw.bindMemory(to: UInt8.self))
            scanner.skipByteOrderMark()
            scanner.scan()
            return (scanner.orders, scanner.children)
        }
    }

    /// Decodes `type` from `data` with every `JSONObject` in it keeping the key
    /// order of the text.
    public static func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
        let decoder = JSONDecoder()
        if let userInfoKey {
            decoder.userInfo[userInfoKey] = JSONKeyOrder(scanning: data)
        }
        return try decoder.decode(type, from: data)
    }

    /// The recorded keys of the object `decoder` is positioned at, `nil` when
    /// the decoder carries no scan or the scan does not know the object.
    static func keys(at decoder: any Decoder) -> [String]? {
        guard let userInfoKey, let order = decoder.userInfo[userInfoKey] as? JSONKeyOrder else { return nil }
        var container = root
        for key in decoder.codingPath {
            // An array element's key, as `JSONDecoder` builds it. A key of any
            // other origin has no `intValue` or a `stringValue` that is the
            // object key itself, so an object key is never read as an index.
            let component: Component
            if let index = key.intValue, key.stringValue == "Index \(index)" {
                component = .element(index)
            } else {
                component = .member(key.stringValue)
            }
            guard let child = order.children[Edge(parent: container, component: component)] else { return nil }
            container = child
        }
        return order.orders[container]
    }

    private static let userInfoKey = CodingUserInfoKey(rawValue: "MacMLXCore.JSONKeyOrder")

    // MARK: - Scanner

    /// Walks the text with its own stack of open containers and no recursion:
    /// the scan runs on whatever thread decodes the request, and a text is
    /// free to nest as deep as it likes.
    private struct Scanner {
        let bytes: UnsafeBufferPointer<UInt8>
        var index = 0
        var orders: [Int: [String]] = [:]
        var children: [Edge: Int] = [:]

        /// A container the scan is inside of.
        private struct Open {
            /// The step that reached it, `nil` for the top-level value.
            let edge: Edge?
            let number: Int
            let isObject: Bool
            /// Whether `children` has its edge yet. An edge is only kept once
            /// an object at or below it is recorded, so a text of many
            /// containers with nothing to record keeps nothing.
            var kept: Bool
            /// Whether a first member or element has been looked for.
            var started = false
            /// An object's keys so far, in order and once each.
            var keys: [String] = []
            var seen: Set<String> = []
            /// An array's elements so far.
            var count = 0
        }

        /// The containers the scan is inside of, outermost first.
        private var open: [Open] = []
        private var nextNumber = JSONKeyOrder.root + 1

        /// `JSONDecoder` refuses deeper nesting than this, so the scan stops
        /// there too.
        private static let maximumDepth = 512

        init(bytes: UnsafeBufferPointer<UInt8>) {
            self.bytes = bytes
        }

        private var current: UInt8? { index < bytes.count ? bytes[index] : nil }

        /// `JSONDecoder` accepts a UTF-8 byte order mark in front of the text.
        mutating func skipByteOrderMark() {
            if bytes.count >= 3, bytes[0] == 0xEF, bytes[1] == 0xBB, bytes[2] == 0xBF {
                index = 3
            }
        }

        private mutating func skipWhitespace() {
            while let byte = current, byte == 0x20 || byte == 0x0A || byte == 0x0D || byte == 0x09 {
                index += 1
            }
        }

        private mutating func consume(_ byte: UInt8) -> Bool {
            guard current == byte else { return false }
            index += 1
            return true
        }

        /// Scans the text. It stops, with what it has recorded so far, where
        /// the text stops making sense or nests deeper than `maximumDepth`.
        mutating func scan() {
            guard begin(reachedBy: nil) else { return }
            // Each turn finishes the innermost open container or begins its
            // next member or element.
            while let innermost = open.indices.last {
                skipWhitespace()
                let isObject = open[innermost].isObject
                let closing = isObject ? UInt8(ascii: "}") : UInt8(ascii: "]")
                if open[innermost].started {
                    if consume(closing) {
                        close()
                        continue
                    }
                    guard consume(UInt8(ascii: ",")) else { return }
                    skipWhitespace()
                } else {
                    open[innermost].started = true
                    if consume(closing) {
                        close()
                        continue
                    }
                }
                if isObject {
                    guard let key = scanString(wanted: true) else { return }
                    skipWhitespace()
                    guard consume(UInt8(ascii: ":")) else { return }
                    if open[innermost].seen.insert(key).inserted {
                        open[innermost].keys.append(key)
                    }
                    guard begin(reachedBy: nextIsContainer() ? .member(key) : nil) else { return }
                } else {
                    let element = open[innermost].count
                    open[innermost].count += 1
                    guard begin(reachedBy: nextIsContainer() ? .element(element) : nil) else { return }
                }
            }
        }

        /// Whether the value about to be scanned is an object or an array.
        /// Only those are numbered, so a scalar child costs nothing.
        private mutating func nextIsContainer() -> Bool {
            skipWhitespace()
            return current == UInt8(ascii: "{") || current == UInt8(ascii: "[")
        }

        /// Begins one value: steps over a scalar, or opens a container, which
        /// `component` reaches from the innermost open one (`nil` for the
        /// top-level value). `false` ends the scan.
        private mutating func begin(reachedBy component: Component?) -> Bool {
            skipWhitespace()
            switch current {
            case UInt8(ascii: "{"), UInt8(ascii: "["):
                guard open.count < Self.maximumDepth else { return false }
                let isObject = current == UInt8(ascii: "{")
                index += 1
                if let component, let parent = open.last?.number {
                    let edge = Edge(parent: parent, component: component)
                    open.append(Open(edge: edge, number: nextNumber, isObject: isObject, kept: false))
                    nextNumber += 1
                } else {
                    open.append(Open(edge: nil, number: JSONKeyOrder.root, isObject: isObject, kept: true))
                }
                return true
            case UInt8(ascii: "\""):
                return scanString(wanted: false) != nil
            case nil:
                return false
            default:
                // A number, `true`, `false` or `null`: up to the next delimiter.
                while let byte = current, !Self.endsScalar(byte) {
                    index += 1
                }
                return true
            }
        }

        private static func endsScalar(_ byte: UInt8) -> Bool {
            switch byte {
            case UInt8(ascii: ","), UInt8(ascii: "}"), UInt8(ascii: "]"), 0x20, 0x0A, 0x0D, 0x09: return true
            default: return false
            }
        }

        /// Closes the innermost open container. An object with more than one
        /// key has its keys recorded, and the edges that lead to it kept.
        private mutating func close() {
            guard let closed = open.last else { return }
            if closed.isObject, closed.keys.count > 1 {
                var index = open.count - 1
                while index >= 0, !open[index].kept {
                    if let edge = open[index].edge {
                        children[edge] = open[index].number
                    }
                    open[index].kept = true
                    index -= 1
                }
                orders[closed.number] = closed.keys
            }
            open.removeLast()
        }

        /// Scans a string, `nil` when it is not one. Returns its value when
        /// that is `wanted` (a key) and an empty string otherwise (a value,
        /// which only has to be stepped over).
        private mutating func scanString(wanted: Bool) -> String? {
            guard consume(UInt8(ascii: "\"")) else { return nil }
            let start = index
            var escaped = false
            while let byte = current {
                if byte == UInt8(ascii: "\"") {
                    let content = UnsafeBufferPointer(rebasing: bytes[start..<index])
                    index += 1
                    guard wanted else { return "" }
                    return escaped ? Self.unescaping(content) : String(decoding: content, as: UTF8.self)
                }
                if byte == UInt8(ascii: "\\") {
                    // The escaped character is never the closing quote. The
                    // four digits of a `\u` escape are skipped as plain bytes.
                    escaped = true
                    index += 1
                }
                index += 1
            }
            return nil
        }

        /// The value of a string that has escapes in it. Rare for a key, so the
        /// work goes to `JSONSerialization`.
        private static func unescaping(_ content: UnsafeBufferPointer<UInt8>) -> String? {
            var quoted = Data([UInt8(ascii: "\"")])
            quoted.append(content)
            quoted.append(UInt8(ascii: "\""))
            return (try? JSONSerialization.jsonObject(with: quoted, options: .fragmentsAllowed)) as? String
        }
    }
}
