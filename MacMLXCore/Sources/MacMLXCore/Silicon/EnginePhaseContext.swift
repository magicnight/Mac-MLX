// Copyright © 2026 macMLX. English comments only.

/// The engine-side context that pairs with each hardware `SiliconSample` the
/// bottleneck classifier consumes.
///
/// W1 answers "what is the silicon doing?"; this answers "what is the engine
/// doing while the silicon does it?". Fusing the two is the whole point of W2:
/// the same GPU/bandwidth reading is a healthy prefill or an unhealthy decode
/// depending on `phase`, and the actionable advice depends on `config`.
///
/// `tokensPerSecond` is optional on purpose. Live per-token throughput is not
/// cheaply available without adding timing to the decode hot path, so during a
/// running generation it is usually `nil`; it is populated from the engine's
/// terminal completion info once a generation finishes. The classifier therefore
/// treats it as supplementary colour for advice, never as a load-bearing input
/// to the bottleneck decision.
public struct EnginePhaseContext: Sendable, Equatable {

    /// Which phase the engine is in for this sample window.
    public let phase: InferencePhase

    /// Generation throughput in tokens/second, when known (typically only at
    /// completion). `nil` while a generation is mid-flight.
    public let tokensPerSecond: Double?

    /// The in-process generation knobs, used to make advice concrete.
    public let config: EngineGenerationConfig

    public init(
        phase: InferencePhase,
        tokensPerSecond: Double?,
        config: EngineGenerationConfig
    ) {
        self.phase = phase
        self.tokensPerSecond = tokensPerSecond
        self.config = config
    }
}
