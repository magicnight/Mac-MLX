// Copyright © 2026 macMLX. English comments only.

import Foundation
import Testing

@testable import MacMLXCore

/// The decimal arithmetic under bounded numbers: ``SchemaDecimal`` parses,
/// normalises and compares exactly; ``NumberRange`` admits exactly the
/// prefixes that can still become a value in range.
@Suite("SchemaDecimal and NumberRange")
struct NumberRangeTests {

    // The helpers record an issue instead of throwing: `#expect` evaluates
    // each operand in its own autoclosure, where a `try` does not reach.

    private func dec(_ text: String) -> SchemaDecimal {
        guard let value = SchemaDecimal(parsing: text) else {
            Issue.record("not a decimal: \(text)")
            return SchemaDecimal(0)
        }
        return value
    }

    private func integers(_ minimum: Int?, _ maximum: Int?) -> NumberRange {
        guard let bounds = SchemaIntegerBounds(minimum: minimum, maximum: maximum) else {
            Issue.record("no integer between \(String(describing: minimum)) and \(String(describing: maximum))")
            return integers(0, 0)
        }
        return NumberRange(bounds)
    }

    private func numbers(_ minimum: String?, _ maximum: String?, openBelow: Bool = false, openAbove: Bool = false) -> NumberRange {
        guard let bounds = SchemaNumberBounds(
            minimum: minimum.map(dec), minimumIsExclusive: openBelow,
            maximum: maximum.map(dec), maximumIsExclusive: openAbove)
        else {
            Issue.record("no number between \(String(describing: minimum)) and \(String(describing: maximum))")
            return numbers("0", "1")
        }
        return NumberRange(bounds)
    }

    @Test
    func parsesAndNormalises() throws {
        #expect(dec("0.5") == dec("0.50"))
        #expect(dec("-0") == dec("0"))
        #expect(dec("-0.000") == SchemaDecimal(0))
        #expect(dec("1e2") == SchemaDecimal(100))
        #expect(dec("1E+2") == SchemaDecimal(100))
        #expect(dec("12.50e-1") == dec("1.25"))
        #expect(dec("0.000000000000000001e18") == SchemaDecimal(1))
        #expect(dec(String(repeating: "9", count: 19)) == SchemaDecimal.largest(scale: 0))
        for bad in ["", "+1", "abc", "1.2.3", "1e", "1e-20", "0.00000000000000000001", String(repeating: "9", count: 20), "1e19", "1e99999", "-"] {
            #expect(SchemaDecimal(parsing: bad) == nil, "\(bad)")
        }
        #expect(SchemaDecimal(0.1) == dec("0.1"))
        #expect(SchemaDecimal(-12.5) == dec("-12.5"))
        #expect(SchemaDecimal(100.0) == SchemaDecimal(100))
        #expect(SchemaDecimal(1e-5) == dec("0.00001"))
        #expect(SchemaDecimal(1e25) == nil)
        #expect(SchemaDecimal(Double.infinity) == nil)
        #expect(SchemaDecimal(Double.nan) == nil)
        #expect(SchemaDecimal(Int.min).description == "-9223372036854775808")
        #expect(SchemaDecimal(Int.max).description == "9223372036854775807")
        #expect(dec("0.05").description == "0.05")
        #expect(dec("-1.250").description == "-1.25")
        #expect(dec("100").description == "100")
        #expect(dec("0.0000000000000000001").description == "0.0000000000000000001")
        #expect(dec("123.456").description == "123.456")
        #expect(SchemaDecimal(9.3e18)?.significantDigits == 2)
        #expect(dec("0.3333333333333333").significantDigits == 16)
        #expect(dec("1234567890123456789").significantDigits == 19)
        #expect(SchemaDecimal(0).significantDigits == 0)
    }

    @Test
    func comparesExactlyAndRoundsToIntegers() throws {
        #expect(dec("0.3") > dec("0.29999999999999998"))
        #expect(dec("-0.25") > dec("-1.5"))
        #expect(dec("9999999999999999999") > dec("999999999999999999.9"))
        #expect(dec("0.1") < dec("0.10000000000000001"))
        #expect(dec("1.10") == dec("1.1"))
        #expect(SchemaDecimal(-1) < SchemaDecimal(0))
        #expect(SchemaDecimal(0) < SchemaDecimal(1))
        #expect(dec("1.5").floor == 1)
        #expect(dec("1.5").ceiling == 2)
        #expect(dec("-1.5").floor == -2)
        #expect(dec("-1.5").ceiling == -1)
        #expect(dec("2").floor == 2)
        #expect(dec("2").ceiling == 2)
        #expect(dec("-0.5").ceiling == 0)
        #expect(dec("2.0").integerValue == 2)
        #expect(dec("2.5").integerValue == nil)
        #expect(dec("9223372036854775808").integerValue == nil)
        #expect(dec("9223372036854775808").floor == nil)
        #expect(dec("-9223372036854775808").integerValue == Int.min)
        #expect(dec("-9223372036854775808").floor == Int.min)
        #expect(dec("-9223372036854775808").ceiling == Int.min)
        #expect(dec("-9223372036854775809").floor == nil)
        #expect(dec("-9223372036854775809").ceiling == nil)
    }

    @Test
    func admitsExactlyThePrefixesThatCanStillFit() throws {
        let range = integers(-12, 35)
        #expect(range.admits(negative: false, mantissa: 3, scale: 0, phase: .integerDigits))
        #expect(range.admits(negative: false, mantissa: 35, scale: 0, phase: .integerDigits))
        #expect(range.admits(negative: false, mantissa: 4, scale: 0, phase: .integerDigits), "4 itself is in range")
        #expect(!range.admits(negative: false, mantissa: 40, scale: 0, phase: .integerDigits))
        #expect(!range.admits(negative: false, mantissa: 36, scale: 0, phase: .integerDigits))
        let tens = integers(10, 35)
        #expect(!tens.admits(negative: false, mantissa: 4, scale: 0, phase: .integerDigits), "nothing in [10, 35] starts with 4")
        #expect(tens.admits(negative: false, mantissa: 1, scale: 0, phase: .integerDigits))
        #expect(!tens.contains(dec("1")))
        #expect(range.admits(negative: false, mantissa: 0, scale: 0, phase: .loneZero))
        #expect(range.admits(negative: true, mantissa: 1, scale: 0, phase: .integerDigits))
        #expect(range.admits(negative: true, mantissa: 12, scale: 0, phase: .integerDigits))
        #expect(!range.admits(negative: true, mantissa: 13, scale: 0, phase: .integerDigits))
        #expect(range.admits(negative: true, mantissa: 2, scale: 0, phase: .integerDigits), "-2 itself is in range")
        #expect(!range.admits(negative: true, mantissa: 20, scale: 0, phase: .integerDigits), "-2x is below -12")
        #expect(range.admits(negative: true, mantissa: 0, scale: 0, phase: .loneZero), "-0")
        #expect(range.admitsNegativeSign)
        #expect(range.contains(dec("35")))
        #expect(!range.contains(dec("36")))
        #expect(range.contains(dec("-12")))
        #expect(try !integers(1, 10).admitsNegativeSign)
        #expect(integers(0, 10).admitsNegativeSign, "-0 is 0")

        // A single value: only its digits, in order.
        let hundred = integers(100, 100)
        #expect(hundred.admits(negative: false, mantissa: 1, scale: 0, phase: .integerDigits))
        #expect(hundred.admits(negative: false, mantissa: 10, scale: 0, phase: .integerDigits))
        #expect(!hundred.admits(negative: false, mantissa: 2, scale: 0, phase: .integerDigits))
        #expect(!hundred.admits(negative: false, mantissa: 101, scale: 0, phase: .integerDigits))
        #expect(!hundred.admits(negative: false, mantissa: 0, scale: 0, phase: .loneZero))

        // An open range: 0 itself is out, but "0" can still become 0.5; "1" cannot become anything in (0, 1).
        let open = numbers("0", "1", openBelow: true, openAbove: true)
        #expect(open.admits(negative: false, mantissa: 0, scale: 0, phase: .loneZero))
        #expect(!open.contains(dec("0")))
        #expect(open.contains(dec("0.5")))
        #expect(!open.contains(dec("1")))
        #expect(open.admits(negative: false, mantissa: 0, scale: 1, phase: .fraction), "0.0 can still become 0.01")
        #expect(!open.admits(negative: false, mantissa: 1, scale: 0, phase: .integerDigits))
        #expect(!open.admits(negative: false, mantissa: 10, scale: 1, phase: .fraction), "1.0x is never below 1")
        #expect(!open.admitsNegativeSign, "no value at or below zero is in (0, 1)")
        #expect(numbers("0", "1").admitsNegativeSign, "-0 is in [0, 1]")

        // Ties: an open range end against a closed interval end is empty; closed ends meet.
        let below = numbers(nil, "0.3", openAbove: true)
        #expect(below.admits(negative: false, mantissa: 2, scale: 1, phase: .fraction), "[0.2, 0.3) meets (-inf, 0.3)")
        #expect(!below.admits(negative: false, mantissa: 3, scale: 1, phase: .fraction), "[0.3, 0.4) does not")
        #expect(numbers(nil, "0.3").admits(negative: false, mantissa: 3, scale: 1, phase: .fraction), "0.3 itself is in [.., 0.3]")
        #expect(below.admits(negative: true, mantissa: 5, scale: 0, phase: .integerDigits))
        #expect(below.admits(negative: false, mantissa: 0, scale: 0, phase: .loneZero))

        // The dot: [m, m+1) must meet the range.
        let narrow = numbers("2.5", "2.75")
        #expect(narrow.admits(negative: false, mantissa: 2, scale: 0, phase: .afterDot))
        #expect(!narrow.admits(negative: false, mantissa: 3, scale: 0, phase: .integerDigits))
        #expect(!narrow.admits(negative: false, mantissa: 2, scale: 1, phase: .fraction), "2.0x stays below 2.5")
        #expect(narrow.admits(negative: false, mantissa: 25, scale: 1, phase: .fraction))
        #expect(!narrow.admits(negative: false, mantissa: 28, scale: 1, phase: .fraction))

        // Negative ranges mirror.
        let negative = numbers("-1.5", "-0.25")
        #expect(negative.admits(negative: true, mantissa: 1, scale: 0, phase: .integerDigits))
        #expect(negative.admits(negative: true, mantissa: 0, scale: 0, phase: .loneZero), "-0.x can reach -0.25")
        #expect(!negative.admits(negative: true, mantissa: 2, scale: 0, phase: .integerDigits))
        #expect(negative.admits(negative: true, mantissa: 2, scale: 1, phase: .fraction), "-0.2 can still become -0.25")
        #expect(!negative.admits(negative: true, mantissa: 24, scale: 2, phase: .fraction), "-0.24x is above -0.25")
        #expect(negative.admits(negative: true, mantissa: 25, scale: 2, phase: .fraction))
        #expect(!negative.admits(negative: false, mantissa: 0, scale: 0, phase: .loneZero), "nothing at or above 0")
        #expect(!negative.contains(dec("0")))

        // The 19-digit limit bounds a prefix's reach: with a lower bound just
        // above 9e18, a 1 can never be completed, a 9 can.
        let huge = integers(9_000_000_000_000_000_001, nil)
        #expect(!huge.admits(negative: false, mantissa: 1, scale: 0, phase: .integerDigits))
        #expect(!huge.admits(negative: false, mantissa: 8, scale: 0, phase: .integerDigits))
        #expect(huge.admits(negative: false, mantissa: 9, scale: 0, phase: .integerDigits))
        #expect(huge.admits(negative: false, mantissa: 9_000_000_000_000_000_001, scale: 0, phase: .integerDigits))
        #expect(!huge.admitsNegativeSign)

        // The limits are part of what a prefix can become: after the last
        // digit the cell is a single value, and an open bound on it is out.
        let positive = numbers("0", nil, openBelow: true)
        #expect(positive.admits(negative: false, mantissa: 0, scale: 18, phase: .fraction), "0.000000000000000000 can still become 0.0000000000000000001")
        #expect(!positive.admits(negative: false, mantissa: 0, scale: 19, phase: .fraction), "the last digit could only spell 0")
        #expect(!positive.admits(negative: false, mantissa: 1_000_000_000_000_000_000, scale: 0, phase: .afterDot), "no digit fits after 19 digits")
        #expect(positive.admits(negative: false, mantissa: 100_000_000_000_000_000, scale: 0, phase: .afterDot))
        let aboveFive = numbers("5", "10", openBelow: true)
        #expect(aboveFive.admits(negative: false, mantissa: 500_000_000_000_000_000, scale: 17, phase: .fraction))
        #expect(!aboveFive.admits(negative: false, mantissa: 5_000_000_000_000_000_000, scale: 18, phase: .fraction))
        #expect(numbers("5", "10").admits(negative: false, mantissa: 5_000_000_000_000_000_000, scale: 18, phase: .fraction), "closed at 5, 5.000000000000000000 is in")
    }

    /// Whether the range holds a value the automaton can spell, as the
    /// bounds' initialiser decides it: open bounds on two neighbouring values
    /// of the grid hold nothing.
    @Test
    func rangesHoldAValueOfTheGrid() {
        #expect(numbers("0", "1").holdsSomeValue)
        #expect(numbers("0", nil, openBelow: true).holdsSomeValue)
        #expect(numbers(nil, "0", openAbove: true).holdsSomeValue)
        #expect(integers(nil, nil).holdsSomeValue)
        #expect(SchemaNumberBounds(minimum: dec("0"), minimumIsExclusive: true, maximum: dec("0.0000000000000000002"), maximumIsExclusive: true) != nil)
        #expect(SchemaNumberBounds(minimum: dec("0"), minimumIsExclusive: true, maximum: dec("0.0000000000000000001"), maximumIsExclusive: true) == nil)
        #expect(SchemaNumberBounds(minimum: dec("0.5"), minimumIsExclusive: true, maximum: dec("0.5000000000000000001"), maximumIsExclusive: true) == nil)
        #expect(SchemaNumberBounds(minimum: dec("-0.0000000000000000001"), minimumIsExclusive: true, maximum: dec("0"), maximumIsExclusive: true) == nil)
        #expect(SchemaNumberBounds(minimum: dec("100000000000000000.2"), minimumIsExclusive: true, maximum: dec("100000000000000000.3"), maximumIsExclusive: true) == nil, "a value between would need 20 digits")
        #expect(SchemaNumberBounds(minimum: dec("100000000000000000.2"), minimumIsExclusive: true, maximum: dec("100000000000000000.8"), maximumIsExclusive: true) != nil, "100000000000000000.3 lies between")
        #expect(SchemaNumberBounds(minimum: dec("100000000000000000.2"), maximum: dec("100000000000000000.3")) != nil, "the bounds themselves are values")
        #expect(SchemaNumberBounds(minimum: dec("1"), maximum: dec("1")) != nil)
        #expect(SchemaNumberBounds(minimum: dec("1"), minimumIsExclusive: true, maximum: dec("1")) == nil)
        #expect(SchemaNumberBounds(minimum: dec("2"), maximum: dec("1")) == nil)
        // At the top of the grid: 9000000000000000001 lies above 9e18, nothing lies above 19 nines.
        #expect(SchemaNumberBounds(minimum: dec("9000000000000000000"), minimumIsExclusive: true, maximum: nil) != nil)
        #expect(SchemaNumberBounds(minimum: dec("9999999999999999999"), minimumIsExclusive: true, maximum: nil) == nil)
        #expect(SchemaNumberBounds(minimum: dec("9999999999999999999"), maximum: nil) != nil)
        #expect(SchemaNumberBounds(minimum: nil, maximum: dec("-9999999999999999999"), maximumIsExclusive: true) == nil)
    }

    /// Decoding checks what the initialisers check: a decimal is normalised
    /// and within the limits, bounds hold a value.
    @Test
    func decodesOnlyWhatTheInitialisersAccept() throws {
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        let value = dec("-12.5")
        #expect(try decoder.decode(SchemaDecimal.self, from: encoder.encode(value)) == value)
        let bounds = SchemaNumberBounds(minimum: dec("0"), minimumIsExclusive: true, maximum: dec("2.75"))
        #expect(try decoder.decode(SchemaNumberBounds.self, from: encoder.encode(bounds)) == bounds)
        let integerBounds = SchemaIntegerBounds(minimum: -12, maximum: 35)
        #expect(try decoder.decode(SchemaIntegerBounds.self, from: encoder.encode(integerBounds)) == integerBounds)
        for bad in [
            #"{"negative":true,"mantissa":0,"scale":0}"#,
            #"{"negative":false,"mantissa":50,"scale":1}"#,
            #"{"negative":false,"mantissa":1,"scale":25}"#,
            #"{"negative":false,"mantissa":10000000000000000000,"scale":0}"#,
        ] {
            #expect(throws: DecodingError.self, "\(bad)") { try decoder.decode(SchemaDecimal.self, from: Data(bad.utf8)) }
        }
        #expect(throws: DecodingError.self) { try decoder.decode(SchemaIntegerBounds.self, from: Data(#"{"minimum":5,"maximum":3}"#.utf8)) }
        let empty = #"{"minimum":{"negative":false,"mantissa":1,"scale":0},"minimumIsExclusive":true,"maximum":{"negative":false,"mantissa":1,"scale":0},"maximumIsExclusive":false}"#
        #expect(throws: DecodingError.self) { try decoder.decode(SchemaNumberBounds.self, from: Data(empty.utf8)) }
    }

    // MARK: An exact model of the grid, near the bounds

    /// The same grid model written over Foundation's `Decimal`: the values a
    /// prefix can still become, and whether one lies in the range.
    private struct GridModel {
        let lower: Decimal?, lowerOpen: Bool, upper: Decimal?, upperOpen: Bool, integersOnly: Bool

        init(_ range: NumberRange) {
            lower = range.lower.flatMap { Decimal(string: $0.description) }
            lowerOpen = range.lowerOpen
            upper = range.upper.flatMap { Decimal(string: $0.description) }
            upperOpen = range.upperOpen
            integersOnly = range.integersOnly
        }

        static func power(_ n: Int) -> Decimal { Decimal(string: "1" + String(repeating: "0", count: n)) ?? 1 }
        static func tenth(_ n: Int) -> Decimal { n == 0 ? 1 : (Decimal(string: "0." + String(repeating: "0", count: n - 1) + "1") ?? 1) }

        func contains(_ value: Decimal) -> Bool {
            if let lower, value < lower || (value == lower && lowerOpen) { return false }
            if let upper, value > upper || (value == upper && upperOpen) { return false }
            return true
        }

        /// Whether the closed interval `[low, high]`, mirrored when negative, holds a value in the range.
        func meets(_ low: Decimal, _ high: Decimal, negative: Bool) -> Bool {
            var l = negative ? -high : low, lOpen = false
            var h = negative ? -low : high, hOpen = false
            if let lower {
                if lower > l { l = lower; lOpen = lowerOpen } else if lower == l { lOpen = lOpen || lowerOpen }
            }
            if let upper {
                if upper < h { h = upper; hOpen = upperOpen } else if upper == h { hOpen = hOpen || upperOpen }
            }
            return l < h || (l == h && !lOpen && !hOpen)
        }

        /// `(admitted, completeHere)` for a prefix as the model would spell it, or `nil` when it is not a prefix at all.
        func judge(_ text: String) -> (admitted: Bool, complete: Bool)? {
            var rest = Substring(text)
            var negative = false
            if rest.first == "-" { negative = true; rest = rest.dropFirst() }
            if rest.isEmpty {
                guard negative else { return nil }
                let any = judge("-0")?.admitted == true || (1...9).contains { judge("-\($0)")?.admitted == true }
                return (any, false)
            }
            let integerPart = rest.prefix { $0.isNumber }
            rest = rest.dropFirst(integerPart.count)
            guard !integerPart.isEmpty, integerPart.allSatisfy({ $0.isASCII }) else { return nil }
            if integerPart.count > 1, integerPart.first == "0" { return (false, false) }
            var dot = false
            var fraction = Substring("")
            if rest.first == "." {
                dot = true
                rest = rest.dropFirst()
                fraction = rest.prefix { $0.isNumber }
                rest = rest.dropFirst(fraction.count)
            }
            guard rest.isEmpty else { return nil }
            if integersOnly, dot { return (false, false) }
            let allDigits = String(integerPart) + String(fraction)
            let significant = String(allDigits.drop { $0 == "0" })
            let mantissa = Decimal(string: significant.isEmpty ? "0" : significant) ?? 0
            let d = significant.count
            let s = fraction.count
            if d > 19 || s > 19 { return (false, false) }
            let value = mantissa * Self.tenth(s)
            if !dot, integerPart == "0" {
                // The lone zero: 0 itself, or any fraction, or nothing for integers.
                let admitted = integersOnly ? meets(0, 0, negative: negative) : meets(0, (Self.power(19) - 1) * Self.tenth(19), negative: negative)
                return (admitted, contains(negative ? -value : value))
            }
            if dot {
                let remaining = Swift.min(19 - s, 19 - d)
                if fraction.isEmpty, remaining < 1 { return (false, false) }
                let low = value
                let high = (mantissa + 1) * Self.tenth(s) - Self.tenth(s + remaining)
                let admitted = meets(low, high, negative: negative)
                return (admitted, !fraction.isEmpty && admitted && contains(negative ? -value : value))
            }
            var admitted = false
            for k in 0...(19 - d) {
                let low = mantissa * Self.power(k)
                let high: Decimal = integersOnly
                    ? (mantissa + 1) * Self.power(k) - 1
                    : (mantissa + 1) * Self.power(k) - Self.tenth(19 - d - k)
                if meets(low, high, negative: negative) { admitted = true; break }
            }
            return (admitted, admitted && contains(negative ? -value : value))
        }
    }

    /// Near every bound of the pooled ranges — the bound's own spelling, each
    /// prefix of it, its last digit moved, padded with zeros and nines to the
    /// digit limit, with and without a point, both signs — the automaton
    /// admits a prefix exactly when the model says a value in range can still
    /// follow, and is complete exactly when the model says the value is in.
    @Test
    func agreesWithTheGridModelNearTheBounds() {
        var pool = RandomSchemaGenerator.boundedScalars
        if let forty = SchemaIntegerBounds(minimum: 40, maximum: 40) { pool.append(.boundedInteger(forty)) }
        if let huge = SchemaIntegerBounds(minimum: 9_000_000_000_000_000_001, maximum: nil) { pool.append(.boundedInteger(huge)) }
        for text in ["0.3", "0.125", "99.5", "100.25", "1000000000000000000", "0.0000000000000000001", "123456789012345678.9"] {
            if let a = SchemaDecimal(parsing: text) {
                if let b = SchemaNumberBounds(minimum: a, minimumIsExclusive: true, maximum: nil) { pool.append(.boundedNumber(b)) }
                if let b = SchemaNumberBounds(minimum: nil, maximum: a, maximumIsExclusive: true) { pool.append(.boundedNumber(b)) }
                if let b = SchemaNumberBounds(minimum: a, maximum: a) { pool.append(.boundedNumber(b)) }
                if let b = SchemaNumberBounds(minimum: a.negated, maximum: nil) { pool.append(.boundedNumber(b)) }
            }
        }
        var compared = 0
        var disagreements: [String] = []
        for root in pool {
            let range: NumberRange
            switch root {
            case .boundedInteger(let bounds): range = NumberRange(bounds)
            case .boundedNumber(let bounds): range = NumberRange(bounds)
            default: continue
            }
            let model = GridModel(range)
            let start = SchemaConstraintState(root: root)
            for prefix in Self.prefixes(near: range) {
                guard let verdict = model.judge(prefix) else { continue }
                let walked = start.walk(Array(prefix.utf8))
                compared += 1
                if (walked != nil) != verdict.admitted {
                    disagreements.append("admitted: automaton \(walked != nil), model \(verdict.admitted): \(prefix) for \(root)")
                } else if let walked, walked.isComplete != verdict.complete {
                    disagreements.append("complete: automaton \(walked.isComplete), model \(verdict.complete): \(prefix) for \(root)")
                }
            }
        }
        #expect(disagreements.isEmpty, "\(disagreements.count) of \(compared): \(disagreements.prefix(5).joined(separator: "; "))")
        #expect(compared > 20_000, "\(compared) prefixes compared")
    }

    /// Prefixes around a range's bounds (and around 0 and 1 for a missing bound).
    private static func prefixes(near range: NumberRange) -> [String] {
        var seeds: [String] = []
        for bound in [range.lower, range.upper] {
            if let bound { seeds.append(bound.description) }
        }
        if range.lower == nil { seeds += ["-1", "-0.5"] }
        if range.upper == nil { seeds += ["1", "0.5"] }
        seeds += ["0", "9"]
        var out: Set<String> = []
        func add(_ text: String) {
            out.insert(text)
            out.insert("-" + text)
            if text.hasPrefix("-") { out.insert(String(text.dropFirst())) }
        }
        for seed in seeds {
            let bare = seed.hasPrefix("-") ? String(seed.dropFirst()) : seed
            var spellings = [bare]
            // The last digit moved by one, when it stays a digit.
            if let last = bare.last, let digit = last.wholeNumberValue {
                if digit > 0 { spellings.append(String(bare.dropLast()) + String(digit - 1)) }
                if digit < 9 { spellings.append(String(bare.dropLast()) + String(digit + 1)) }
            }
            for spelling in spellings {
                for length in 1...spelling.count {
                    add(String(spelling.prefix(length)))
                }
                let digits = spelling.filter(\.isNumber).count
                for pad in 1...max(1, 21 - digits) {
                    for filler in ["0", "9"] {
                        let padding = String(repeating: filler, count: pad)
                        add(spelling + padding)
                        if !spelling.contains(".") { add(spelling + "." + padding) }
                    }
                }
                if !spelling.contains(".") { add(spelling + ".") }
            }
        }
        return out.sorted()
    }
}
