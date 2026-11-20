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
    /// Object keys in text order, by the path of the object that has them.
    /// An object with fewer than two keys is left out: it has one order.
    private let orders: [String: [String]]

    /// Scans `data`, a JSON text in UTF-8.
    public init(scanning data: Data) {
        orders = data.withUnsafeBytes { raw in
            var scanner = Scanner(bytes: raw.bindMemory(to: UInt8.self))
            scanner.skipByteOrderMark()
            _ = scanner.scanValue(path: "", depth: 0)
            return scanner.orders
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
        var path = ""
        for component in decoder.codingPath {
            // An array element's key, as `JSONDecoder` builds it. A key of any
            // other origin has no `intValue` or a `stringValue` that is the
            // object key itself, so an object key is never read as an index.
            if let index = component.intValue, component.stringValue == "Index \(index)" {
                path += Self.element(index)
            } else {
                path += Self.member(component.stringValue)
            }
        }
        return order.orders[path]
    }

    private static let userInfoKey = CodingUserInfoKey(rawValue: "MacMLXCore.JSONKeyOrder")

    // A path is its components joined, each with a NUL and a tag in front, so a
    // member named `i0` and the element 0 differ.
    private static func member(_ key: String) -> String { "\u{0}k" + key }
    private static func element(_ index: Int) -> String { "\u{0}i\(index)" }

    // MARK: - Scanner

    private struct Scanner {
        let bytes: UnsafeBufferPointer<UInt8>
        var index = 0
        var orders: [String: [String]] = [:]

        /// `JSONDecoder` refuses deeper nesting than this.
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

        /// The path of a child that is about to be scanned. Only an object or
        /// an array has a use for it, so a scalar child costs no string.
        private mutating func childPath(_ path: String, _ component: @autoclosure () -> String) -> String {
            skipWhitespace()
            switch current {
            case UInt8(ascii: "{"), UInt8(ascii: "["): return path + component()
            default: return ""
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

        /// Scans one value. `false` means the text stopped making sense and the
        /// scan is over.
        mutating func scanValue(path: String, depth: Int) -> Bool {
            guard depth <= Self.maximumDepth else { return false }
            skipWhitespace()
            switch current {
            case UInt8(ascii: "{"): return scanObject(path: path, depth: depth)
            case UInt8(ascii: "["): return scanArray(path: path, depth: depth)
            case UInt8(ascii: "\""): return scanString(wanted: false) != nil
            case nil: return false
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

        private mutating func scanObject(path: String, depth: Int) -> Bool {
            index += 1
            var keys: [String] = []
            var seen: Set<String> = []
            skipWhitespace()
            if consume(UInt8(ascii: "}")) { return true }
            while true {
                skipWhitespace()
                guard let key = scanString(wanted: true) else { return false }
                skipWhitespace()
                guard consume(UInt8(ascii: ":")) else { return false }
                let child = childPath(path, JSONKeyOrder.member(key))
                guard scanValue(path: child, depth: depth + 1) else { return false }
                if seen.insert(key).inserted {
                    keys.append(key)
                }
                skipWhitespace()
                if consume(UInt8(ascii: ",")) { continue }
                guard consume(UInt8(ascii: "}")) else { return false }
                if keys.count > 1 {
                    orders[path] = keys
                }
                return true
            }
        }

        private mutating func scanArray(path: String, depth: Int) -> Bool {
            index += 1
            skipWhitespace()
            if consume(UInt8(ascii: "]")) { return true }
            var element = 0
            while true {
                let child = childPath(path, JSONKeyOrder.element(element))
                guard scanValue(path: child, depth: depth + 1) else { return false }
                element += 1
                skipWhitespace()
                if consume(UInt8(ascii: ",")) { continue }
                return consume(UInt8(ascii: "]"))
            }
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
