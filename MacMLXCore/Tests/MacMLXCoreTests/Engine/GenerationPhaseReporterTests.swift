// Copyright © 2026 macMLX. English comments only.

import Foundation
import Testing

@testable import MacMLXCore

/// Coverage for the W2 engine phase seam: the `GenerationPhaseReporter` forwards
/// a correct prefill → decode → complete timeline (decode fired exactly once),
/// and attaching a `SiliconEngineObserver` to `MLXSwiftEngine` does not perturb
/// the generation path — the nil-default equivalence contract.
struct GenerationPhaseReporterTests {

    /// Thread-safe recording observer. `@unchecked Sendable` with an explicit
    /// lock because the engine (an actor) may call it across its boundary.
    private final class SpyObserver: SiliconEngineObserver, @unchecked Sendable {
        private let lock = NSLock()
        private var _tags: [String] = []
        private var _lastConfig: EngineGenerationConfig?
        private var _lastSummary: GenerationPhaseSummary?

        var tags: [String] { lock.withLock { _tags } }
        var lastConfig: EngineGenerationConfig? { lock.withLock { _lastConfig } }
        var lastSummary: GenerationPhaseSummary? { lock.withLock { _lastSummary } }

        func engineDidBeginPrefill(config: EngineGenerationConfig) {
            lock.withLock { _tags.append("prefill"); _lastConfig = config }
        }
        func engineDidBeginDecode(config: EngineGenerationConfig) {
            lock.withLock { _tags.append("decode"); _lastConfig = config }
        }
        func engineDidCompleteGeneration(summary: GenerationPhaseSummary) {
            lock.withLock { _tags.append("complete"); _lastSummary = summary }
        }
        func engineDidAbortGeneration() {
            lock.withLock { _tags.append("abort") }
        }
    }

    private func summary() -> GenerationPhaseSummary {
        GenerationPhaseSummary(
            config: config, promptTokenCount: 10, generationTokenCount: 5,
            promptSeconds: 0.5, generateSeconds: 1.0)
    }

    private let config = EngineGenerationConfig(
        kvBits: 8, kvGroupSize: 32, quantizedKVStart: 0, batchSize: 2)

    // MARK: - Reporter timeline

    @Test("Reporter emits prefill, a single decode, then complete — in order")
    func reporterEmitsOrderedTimelineWithSingleDecode() {
        let spy = SpyObserver()
        var reporter = GenerationPhaseReporter(observer: spy, config: config)

        reporter.begin()
        // Three tokens, but decode must be reported exactly once (on the first).
        reporter.noteTokenGenerated()
        reporter.noteTokenGenerated()
        reporter.noteTokenGenerated()
        reporter.complete(
            summary: GenerationPhaseSummary(
                config: config,
                promptTokenCount: 10,
                generationTokenCount: 5,
                promptSeconds: 0.5,
                generateSeconds: 1.0))

        #expect(spy.tags == ["prefill", "decode", "complete"])
        #expect(spy.lastConfig == config)
        #expect(spy.lastSummary?.generationTokenCount == 5)
    }

    @Test("A prefill-only generation (no tokens) never reports a decode boundary")
    func reporterWithoutTokensReportsNoDecode() {
        let spy = SpyObserver()
        var reporter = GenerationPhaseReporter(observer: spy, config: config)
        reporter.begin()
        reporter.complete(
            summary: GenerationPhaseSummary(
                config: config, promptTokenCount: 3, generationTokenCount: 0,
                promptSeconds: 0.1, generateSeconds: 0))
        #expect(spy.tags == ["prefill", "complete"])
    }

    // MARK: - Terminal event on abort (the cancel/abandon path)

    @Test("An aborted generation (no complete) still emits exactly one terminal event")
    func abortEmitsTerminalWhenNotCompleted() {
        let spy = SpyObserver()
        var reporter = GenerationPhaseReporter(observer: spy, config: config)
        reporter.begin()
        reporter.noteTokenGenerated()
        // The engine returned early (cancel / consumer abandonment) without a
        // complete; the `defer`-guarded abortIfUnfinished must still fire.
        reporter.abortIfUnfinished()
        #expect(spy.tags == ["prefill", "decode", "abort"])
    }

    @Test("abortIfUnfinished after a normal complete is a no-op — one terminal event")
    func abortIsSuppressedAfterComplete() {
        let spy = SpyObserver()
        var reporter = GenerationPhaseReporter(observer: spy, config: config)
        reporter.begin()
        reporter.noteTokenGenerated()
        reporter.complete(summary: summary())
        // The `defer` still runs on the normal exit; it must not double-fire.
        reporter.abortIfUnfinished()
        #expect(spy.tags == ["prefill", "decode", "complete"])
    }

    @Test("complete after an abort is a no-op — the first terminal event wins")
    func completeIsSuppressedAfterAbort() {
        let spy = SpyObserver()
        var reporter = GenerationPhaseReporter(observer: spy, config: config)
        reporter.begin()
        reporter.abortIfUnfinished()
        reporter.complete(summary: summary())
        #expect(spy.tags == ["prefill", "abort"])
    }

    @Test("A begin with an immediate abort (prefill-only cancel) emits prefill then abort")
    func abortDuringPrefillEmitsPrefillThenAbort() {
        let spy = SpyObserver()
        var reporter = GenerationPhaseReporter(observer: spy, config: config)
        reporter.begin()
        // Cancelled before the first token: no decode boundary, but a terminal
        // event still fires so a stateful observer is not left hanging.
        reporter.abortIfUnfinished()
        #expect(spy.tags == ["prefill", "abort"])
    }

    // MARK: - Engine nil-default equivalence

    @Test("Attaching an observer leaves the no-model generate path unchanged")
    func attachingObserverDoesNotPerturbNoModelPath() async {
        let spy = SpyObserver()
        let engine = MLXSwiftEngine(siliconObserver: spy)
        let request = GenerateRequest(
            model: "x", messages: [ChatMessage(role: .user, content: "hi")])

        do {
            for try await _ in engine.generate(request) { /* drain */ }
            Issue.record("Expected stream to throw, but it finished normally")
        } catch let error as EngineError {
            #expect(error == .modelNotLoaded)
        } catch {
            Issue.record("Expected EngineError.modelNotLoaded, got \(type(of: error)): \(error)")
        }

        // Generation never ran, so the observer was never notified — attaching
        // it did not change the (failing) control flow.
        #expect(spy.tags.isEmpty)
    }

    @Test("An engine constructed with an observer still starts idle")
    func engineWithObserverStartsIdle() async {
        let engine = MLXSwiftEngine(siliconObserver: SpyObserver())
        let status = await engine.status
        let model = await engine.loadedModel
        #expect(status == .idle)
        #expect(model == nil)
    }
}
