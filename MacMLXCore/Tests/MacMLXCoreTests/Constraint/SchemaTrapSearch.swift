// Copyright © 2026 macMLX. English comments only.

@testable import MacMLXCore

/// Trap search over the schema automaton: breadth-first exploration of every
/// state reachable over a byte alphabet, then a backward co-reachability pass
/// from the complete states.
///
/// A *trap* is an explored state from which no sequence of alphabet bytes
/// reaches a complete document. At a trap the constraint processor finds no
/// legal token and has to force EOS on a truncated output, so a correct
/// automaton has none. States left unexpanded when the cap is hit count as
/// live: the cap can hide a trap, never invent one.
enum SchemaTrapSearch {

    struct Result {
        /// States discovered (expanded or on the frontier).
        let explored: Int
        let traps: [SchemaConstraintState]
    }

    /// Structural bytes, a space, `\u` escapes reaching the surrogate range
    /// (`\`, `u`, hex digits, `d`, `c`), every digit (a bounded number's legal
    /// digits depend on its bounds), and number and literal bytes.
    static let baseAlphabet = Array("{}[],:\" \\u0123456789dc.-eEtrufalsn".utf8)

    /// ``baseAlphabet`` plus every byte of the schema's keys and enum values,
    /// raw and as `\u` escapes, without duplicates.
    static func alphabet(for schema: JSONSchemaObject) -> [UInt8] {
        alphabet(for: .object(schema))
    }

    static func alphabet(for root: SchemaValueType) -> [UInt8] {
        var bytes = baseAlphabet
        collectBytes(of: root, into: &bytes)
        var seen = Set<UInt8>()
        return bytes.filter { seen.insert($0).inserted }
    }

    private static func collectBytes(of schema: JSONSchemaObject, into bytes: inout [UInt8]) {
        for property in schema.properties {
            collectBytes(ofLiteral: property.name, into: &bytes)
            collectBytes(of: property.type, into: &bytes)
        }
    }

    /// The literal's UTF-8, and the hex digits of its scalars' `\u` escapes
    /// (surrogate halves above the BMP).
    private static func collectBytes(ofLiteral literal: String, into bytes: inout [UInt8]) {
        bytes.append(contentsOf: literal.utf8)
        for scalar in literal.unicodeScalars {
            var units = [scalar.value]
            if scalar.value > 0xFFFF {
                let offset = scalar.value - 0x10000
                units = [0xD800 + (offset >> 10), 0xDC00 + (offset & 0x3FF)]
            }
            for unit in units {
                let hex = String(unit, radix: 16)
                bytes.append(contentsOf: (String(repeating: "0", count: 4 - hex.count) + hex).utf8)
            }
        }
    }

    private static func collectBytes(of type: SchemaValueType, into bytes: inout [UInt8]) {
        switch type {
        case .stringEnum(let values):
            for value in values { collectBytes(ofLiteral: value, into: &bytes) }
        case .object(let object):
            collectBytes(of: object, into: &bytes)
        case .array(let items, _, _):
            collectBytes(of: items, into: &bytes)
        case .string, .number, .integer, .boolean, .boundedInteger, .boundedNumber:
            break
        }
    }

    static func run(from start: SchemaConstraintState, alphabet: [UInt8], limit: Int) -> Result {
        var index: [SchemaConstraintState: Int] = [start: 0]
        var states: [SchemaConstraintState] = [start]
        var successors: [[Int]] = [[]]
        var expanded: [Bool] = [false]
        var head = 0
        while head < states.count, states.count < limit {
            let current = head
            head += 1
            expanded[current] = true
            for byte in alphabet {
                guard let next = states[current].advanced(over: byte) else { continue }
                if let known = index[next] {
                    successors[current].append(known)
                    continue
                }
                let id = states.count
                index[next] = id
                states.append(next)
                successors.append([])
                expanded.append(false)
                successors[current].append(id)
            }
        }

        var predecessors = Array(repeating: [Int](), count: states.count)
        for (from, targets) in successors.enumerated() {
            for target in targets { predecessors[target].append(from) }
        }
        var live = states.indices.map { states[$0].isComplete || !expanded[$0] }
        var work = live.indices.filter { live[$0] }
        while let state = work.popLast() {
            for from in predecessors[state] where !live[from] {
                live[from] = true
                work.append(from)
            }
        }
        let traps = states.indices.filter { !live[$0] }.map { states[$0] }
        return Result(explored: states.count, traps: traps)
    }
}
