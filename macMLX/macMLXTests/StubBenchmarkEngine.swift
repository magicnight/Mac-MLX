// StubBenchmarkEngine.swift
// macMLXTests
//
// Scriptable stand-in for `EngineCoordinator` on the benchmark seam. It never
// touches MLX: each generation replays a canned chunk sequence, optionally
// parked behind an `AsyncGate` and optionally ending in a throw, so a test can
// decide exactly when a run makes progress and how it ends.
//
// Gates and the failure flag are captured PER CALL, at the moment
// `generate(_:)` is invoked. That is what lets one test script two overlapping
// runs differently: run A's first generation takes gate A (and, say, the
// failure), then the test re-arms the stub before starting run B, whose first
// generation takes gate B and succeeds.

import Foundation
import MacMLXCore
@testable import macMLX

final class StubBenchmarkEngine: BenchmarkEngineProviding {

    /// Error a scripted generation throws. Distinguishable from a cancellation
    /// so tests can tell "the run failed" from "the run was abandoned".
    enum Failure: Error, Equatable {
        case generationFailed
    }

    // MARK: - Seam state

    var currentModel: LocalModel?
    var status: EngineStatus = .idle
    var engineID: EngineID = .mlxSwift
    var engineVersion: String = "stub-engine 1.0"

    // MARK: - Script

    /// Gates handed to the next generations, in call order. A generation made
    /// when the queue is empty runs straight through.
    var gateQueue: [AsyncGate] = []

    /// Whether the NEXT `generate(_:)` call produces a failing stream. Read and
    /// captured at call time, so re-arming it afterwards only affects later
    /// generations, never one already in flight.
    var nextGenerationFails = false

    /// Chunks a successful generation emits. The terminal chunk carries the
    /// `finishReason` + `usage` the runner uses for its token accounting.
    var chunks: [GenerateChunk] = [
        GenerateChunk(text: "hello"),
        GenerateChunk(
            text: "",
            finishReason: .stop,
            usage: TokenUsage(promptTokens: 32, completionTokens: 8)
        ),
    ]

    // MARK: - Recorded calls

    private(set) var loadCount = 0
    private(set) var generateCount = 0

    // MARK: - BenchmarkEngineProviding

    @discardableResult
    func load(_ model: LocalModel, adapter: LocalAdapter?) async -> Result<Void, Error> {
        loadCount += 1
        currentModel = model
        status = .ready(model: model.id)
        return .success(())
    }

    func generate(_ request: GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error> {
        generateCount += 1
        let gate = gateQueue.isEmpty ? nil : gateQueue.removeFirst()
        let fails = nextGenerationFails
        let scripted = chunks
        return AsyncThrowingStream { continuation in
            let task = Task { @MainActor in
                if let gate { await gate.wait() }
                if fails {
                    continuation.finish(throwing: Failure.generationFailed)
                    return
                }
                for chunk in scripted { continuation.yield(chunk) }
                continuation.finish()
            }
            // Mirrors the production streams: a consumer that stops listening
            // (Cancel) must tear the producer down rather than leave it running.
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
