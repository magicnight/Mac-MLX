// Copyright © 2026 macMLX. English comments only.

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
    }
}
