// SiliconPhaseEvent.swift
// macMLX
//
// The internal, ordered channel that carries the engine's SiliconEngineObserver
// callbacks from the engine's isolation domain onto the main actor, in order.
//
// The engine calls the observer synchronously from its actor context; the SwiftUI
// SiliconMonitor lives on the main actor. Rather than spawn one detached hop per
// callback (which gives no ordering guarantee between a "begin" and its terminal
// event), the observer adapter yields these events into an AsyncStream and the
// monitor drains them in FIFO order — so a completion can never be applied before
// the decode transition that preceded it.

import MacMLXCore

/// One engine phase-timeline event, carried across the actor boundary in order.
enum SiliconPhaseEvent: Sendable {
    case beganPrefill(EngineGenerationConfig)
    case beganDecode(EngineGenerationConfig)
    case completed(GenerationPhaseSummary)
    case aborted
}
