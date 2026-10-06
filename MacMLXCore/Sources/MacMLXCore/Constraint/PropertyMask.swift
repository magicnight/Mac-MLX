// Copyright © 2026 macMLX. English comments only.

/// A set of small non-negative integers — the member indices of one object
/// node, or the candidate indices of an enum — used by the schema automaton
/// for emitted keys and for candidates narrowed byte by byte.
///
/// Members `0..<64` live in one inline word, so the common object needs no
/// allocation and copying a state is a plain copy; larger indices spill into
/// ``high``.
@usableFromInline
struct PropertyMask: Hashable, Sendable {
    /// Members `0..<64`.
    @usableFromInline var low: UInt64
    /// Words for members `>= 64`. Kept trimmed (no trailing zero word), so the
    /// synthesized `==` and `hash` are semantic.
    @usableFromInline var high: [UInt64]

    @inlinable
    init(low: UInt64 = 0, high: [UInt64] = []) {
        self.low = low
        self.high = high
    }

    @usableFromInline static let empty = PropertyMask()

    /// The members `0..<count`.
    @inlinable
    static func all(count: Int) -> PropertyMask {
        var mask = PropertyMask()
        for member in 0..<count { mask.insert(member) }
        return mask
    }

    @inlinable
    var isEmpty: Bool { low == 0 && high.isEmpty }

    @inlinable
    func contains(_ member: Int) -> Bool {
        if member < 64 { return low & (1 << UInt64(member)) != 0 }
        let word = (member - 64) >> 6
        return word < high.count && high[word] & (1 << UInt64((member - 64) & 63)) != 0
    }

    @inlinable
    mutating func insert(_ member: Int) {
        if member < 64 {
            low |= 1 << UInt64(member)
            return
        }
        let word = (member - 64) >> 6
        if word >= high.count {
            high.append(contentsOf: repeatElement(0, count: word - high.count + 1))
        }
        high[word] |= 1 << UInt64((member - 64) & 63)
    }

    /// `self` minus `other`.
    @inlinable
    func subtracting(_ other: PropertyMask) -> PropertyMask {
        var result = PropertyMask(low: low & ~other.low, high: high)
        if !result.high.isEmpty {
            for word in 0..<Swift.min(result.high.count, other.high.count) {
                result.high[word] &= ~other.high[word]
            }
            while let last = result.high.last, last == 0 { result.high.removeLast() }
        }
        return result
    }

    @inlinable
    func isSubset(of other: PropertyMask) -> Bool { subtracting(other).isEmpty }

    /// The members for which `keep` is true.
    @inlinable
    func filtered(_ keep: (Int) -> Bool) -> PropertyMask {
        var result = PropertyMask()
        var bits = low
        while bits != 0 {
            let member = bits.trailingZeroBitCount
            bits &= bits &- 1
            if keep(member) { result.low |= 1 << UInt64(member) }
        }
        for (word, value) in high.enumerated() {
            var bits = value
            while bits != 0 {
                let member = 64 + word * 64 + bits.trailingZeroBitCount
                bits &= bits &- 1
                if keep(member) { result.insert(member) }
            }
        }
        return result
    }

    /// The smallest member, if any.
    @inlinable
    var first: Int? { first(where: { _ in true }) }

    /// The smallest member satisfying `predicate`, if any.
    @inlinable
    func first(where predicate: (Int) -> Bool) -> Int? {
        var bits = low
        while bits != 0 {
            let member = bits.trailingZeroBitCount
            bits &= bits &- 1
            if predicate(member) { return member }
        }
        for (word, value) in high.enumerated() {
            var bits = value
            while bits != 0 {
                let member = 64 + word * 64 + bits.trailingZeroBitCount
                bits &= bits &- 1
                if predicate(member) { return member }
            }
        }
        return nil
    }
}
