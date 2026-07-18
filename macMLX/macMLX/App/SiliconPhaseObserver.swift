// SiliconPhaseObserver.swift
// macMLX
//
// The Sendable adapter that lets a main-actor SiliconMonitor conform, indirectly,
// to the engine's SiliconEngineObserver seam.
//
// SiliconEngineObserver's methods are synchronous and are invoked from the engine
// actor's isolation; a @MainActor @Observable class cannot satisfy them directly.
// This tiny value bridges the gap: each callback simply yields onto an ordered
// AsyncStream continuation that the monitor drains on the main actor. Yielding is
// non-blocking and never touches main-actor state, so it is safe to call from the
// engine's context and keeps the engine's decode loop untouched.

import MacMLXCore

/// Forwards the engine's phase-boundary callbacks onto an ordered channel the
/// monitor consumes on the main actor. `Sendable` because it holds only an
/// `AsyncStream.Continuation` (itself `Sendable`), so the engine can call it from
/// its own isolation domain.
struct SiliconPhaseObserver: SiliconEngineObserver {

    private let continuation: AsyncStream<SiliconPhaseEvent>.Continuation

    init(continuation: AsyncStream<SiliconPhaseEvent>.Continuation) {
        self.continuation = continuation
    }

    func engineDidBeginPrefill(config: EngineGenerationConfig) {
        continuation.yield(.beganPrefill(config))
    }

    func engineDidBeginDecode(config: EngineGenerationConfig) {
        continuation.yield(.beganDecode(config))
    }

    func engineDidCompleteGeneration(summary: GenerationPhaseSummary) {
        continuation.yield(.completed(summary))
    }

    func engineDidAbortGeneration() {
        continuation.yield(.aborted)
    }
}
