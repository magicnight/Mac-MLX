// Copyright © 2026 macMLX. English comments only.

/// The engine's adoption of `SiliconEngineObserver`: a tiny value that each
/// token-generation path (LLM, speculative, VLM) drives so all three report an
/// identical prefill → decode → complete timeline through one shared latch.
///
/// It exists so the phase-notification logic is written once, not copy-pasted
/// into three decode loops, and so the once-only "first token = decode begins"
/// rule can never drift between paths.
///
/// ZERO-OVERHEAD WHEN UNOBSERVED: the engine only builds a reporter when an
/// observer is actually attached (`observer.map { … }`), so with no observer
/// every call site is `nil?.method()` — a single nil-check, no allocation, no
/// behaviour change. The reporter never touches the token stream or the
/// continuation; it only reads the phase timeline, so it cannot alter a single
/// generated token. That is the equivalence contract the seam tests protect.
struct GenerationPhaseReporter {

    private let observer: SiliconEngineObserver
    /// The generation config the observer receives; read by the engine to build
    /// the terminal summary only when a reporter actually exists.
    let config: EngineGenerationConfig
    /// Latch so `engineDidBeginDecode` fires exactly once, on the first token.
    private var didBeginDecode = false
    /// Latch so exactly one terminal event (complete OR abort) fires. The engine
    /// calls `complete` on the normal path and `abortIfUnfinished` via `defer`;
    /// whichever runs first wins and the other becomes a no-op.
    private var didFinish = false

    init(observer: SiliconEngineObserver, config: EngineGenerationConfig) {
        self.observer = observer
        self.config = config
    }

    /// Call once, immediately before consuming the token stream. Prefill starts
    /// as the stream is first pulled, so this marks the prefill boundary.
    func begin() {
        observer.engineDidBeginPrefill(config: config)
    }

    /// Call on every `.token` event. Fires `engineDidBeginDecode` exactly once,
    /// on the first token; every later call is a cheap latched no-op, keeping
    /// the per-token cost to a single branch.
    mutating func noteTokenGenerated() {
        guard !didBeginDecode else { return }
        didBeginDecode = true
        observer.engineDidBeginDecode(config: config)
    }

    /// Call once after the stream runs to its end, with the terminal summary.
    /// Idempotent and mutually exclusive with `abortIfUnfinished`.
    mutating func complete(summary: GenerationPhaseSummary) {
        guard !didFinish else { return }
        didFinish = true
        observer.engineDidCompleteGeneration(summary: summary)
    }

    /// Terminal event for an aborted stream (cancel / consumer abandonment).
    /// Safe to call unconditionally from a `defer`: a no-op once a terminal
    /// event has already fired, so a normal `complete(summary:)` suppresses it.
    mutating func abortIfUnfinished() {
        guard !didFinish else { return }
        didFinish = true
        observer.engineDidAbortGeneration()
    }
}
