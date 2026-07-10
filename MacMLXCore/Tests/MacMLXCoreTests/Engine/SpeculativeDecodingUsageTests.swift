import Testing
@testable import MacMLXCore

/// Track F: `SpeculativeDecodingUsage.acceptancePercent`, the derived value
/// the GUI's chat message footer displays ("X% draft accepted").

@Test
func acceptancePercentRoundsToNearestWholeNumber() {
    let usage = SpeculativeDecodingUsage(proposedTokens: 3, acceptedTokens: 2)
    // 2/3 = 66.67% → rounds to 67.
    #expect(usage.acceptancePercent == 67)
}

@Test
func acceptancePercentIsHundredWhenAllAccepted() {
    let usage = SpeculativeDecodingUsage(proposedTokens: 10, acceptedTokens: 10)
    #expect(usage.acceptancePercent == 100)
}

@Test
func acceptancePercentIsZeroWhenNoneAccepted() {
    let usage = SpeculativeDecodingUsage(proposedTokens: 10, acceptedTokens: 0)
    #expect(usage.acceptancePercent == 0)
}

@Test
func acceptancePercentIsNilWhenNoTokensProposed() {
    let usage = SpeculativeDecodingUsage(proposedTokens: 0, acceptedTokens: 0)
    #expect(usage.acceptancePercent == nil)
}
