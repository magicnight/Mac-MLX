/// Speculative decoding acceptance counters surfaced on the terminal
/// `GenerateChunk` of a generation that used the classic draft-model path
/// (D1). Mirrors mlx-swift-lm's `SpeculativeDecodingTelemetry`
/// (`Evaluate.swift` / `SpeculativeDecoding.swift`) but is macMLX's own
/// `Codable` value type — upstream's telemetry struct isn't `Codable`, so it
/// can't cross the HTTP boundary directly.
public struct SpeculativeDecodingUsage: Codable, Hashable, Sendable {
    /// Tokens proposed by the draft model across every speculative round.
    public let proposedTokens: Int
    /// Tokens the target model accepted across every speculative round.
    public let acceptedTokens: Int

    public init(proposedTokens: Int, acceptedTokens: Int) {
        self.proposedTokens = proposedTokens
        self.acceptedTokens = acceptedTokens
    }
}
