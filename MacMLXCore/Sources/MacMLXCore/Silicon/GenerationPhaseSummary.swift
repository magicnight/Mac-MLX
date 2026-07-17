// Copyright © 2026 macMLX. English comments only.

/// The terminal summary of one generation, handed to a `SiliconEngineObserver`
/// when the token stream ends.
///
/// Sourced from the engine's completion info (mlx-swift-lm's
/// `GenerateCompletionInfo`). It carries the phase-split token counts and
/// timings so an observer can compute prefill vs. decode throughput after the
/// fact — the honest place to read tokens/second, since it divides real counts
/// by real elapsed time rather than sampling a noisy live rate.
///
/// HONESTY: throughput fields are `nil` when their elapsed time was zero (a
/// degenerate or extremely short window), rather than reporting a divide-by-zero
/// infinity as if it were a real rate.
public struct GenerationPhaseSummary: Sendable, Equatable {

    /// The generation configuration this run used.
    public let config: EngineGenerationConfig

    /// Prompt tokens processed during prefill.
    public let promptTokenCount: Int

    /// Tokens produced during decode.
    public let generationTokenCount: Int

    /// Seconds spent in prefill, when reported.
    public let promptSeconds: Double?

    /// Seconds spent in decode, when reported.
    public let generateSeconds: Double?

    /// Prefill throughput (prompt tokens / prompt seconds), or `nil` when the
    /// prompt time was zero.
    public var prefillTokensPerSecond: Double? {
        guard let promptSeconds, promptSeconds > 0 else { return nil }
        return Double(promptTokenCount) / promptSeconds
    }

    /// Decode throughput (generation tokens / generate seconds), or `nil` when
    /// the generate time was zero.
    public var decodeTokensPerSecond: Double? {
        guard let generateSeconds, generateSeconds > 0 else { return nil }
        return Double(generationTokenCount) / generateSeconds
    }

    public init(
        config: EngineGenerationConfig,
        promptTokenCount: Int,
        generationTokenCount: Int,
        promptSeconds: Double?,
        generateSeconds: Double?
    ) {
        self.config = config
        self.promptTokenCount = promptTokenCount
        self.generationTokenCount = generationTokenCount
        self.promptSeconds = promptSeconds
        self.generateSeconds = generateSeconds
    }
}
