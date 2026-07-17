// Copyright © 2026 macMLX. English comments only.

/// A sink the inference engine notifies at generation phase-transition points,
/// so a bottleneck classifier (or any monitor) can align hardware samples with
/// the engine's own prefill/decode timeline.
///
/// This is the non-invasive seam W2 adds to the engine. An engine holds an
/// OPTIONAL observer; when none is attached (the default) every call site is a
/// no-op and the generation path is byte-for-byte what shipped before W2. That
/// is the contract the equivalence tests protect: attaching an observer must not
/// change a single generated token, only surface the phase timeline alongside.
///
/// The methods fire only at PHASE BOUNDARIES, never per token:
///   * `engineDidBeginPrefill` — once, as the prompt starts processing.
///   * `engineDidBeginDecode`  — once, on the first output token.
///   * exactly one TERMINAL event ends every generation that began:
///       - `engineDidCompleteGeneration` when the stream runs to its end, or
///       - `engineDidAbortGeneration` when it is cancelled or the consumer
///         abandons the stream (hitting "stop" in the UI is this path, not an
///         edge case).
/// So a `begin` is always balanced by exactly one terminal call — a stateful
/// observer that pairs them for lifecycle (start/stop a sampler, mark a
/// generation active) is never left believing a finished generation is still
/// running. Keeping the seam off the per-token hot path is deliberate: the
/// decode loop must stay tight, so nothing here runs inside it.
///
/// `Sendable` because the engine is an actor and calls these from its isolated
/// context; a conforming observer must be safe to invoke across that boundary.
public protocol SiliconEngineObserver: Sendable {

    /// Prefill has begun for a new generation with the given configuration.
    func engineDidBeginPrefill(config: EngineGenerationConfig)

    /// The first output token was produced, so decode has begun.
    func engineDidBeginDecode(config: EngineGenerationConfig)

    /// The generation ran to its end; `summary` carries the phase-split token
    /// counts and timings (or the best available when the stream ended without a
    /// completion record). Fires at most once, and never after an abort.
    func engineDidCompleteGeneration(summary: GenerationPhaseSummary)

    /// The generation ended without completing — cancelled, or the consumer
    /// abandoned the stream — so no terminal summary is available. Fires at most
    /// once, and never after a completion. This is the terminal event for the
    /// common "stop generating" path.
    func engineDidAbortGeneration()
}
