// BenchmarkViewModelRunLifecycleTests.swift
// macMLXTests
//
// Run-lifecycle invariants for `BenchmarkViewModel` — the generation-token
// semantics introduced when Cancel-then-restart was found to corrupt the
// restarted run's state (PR #105), which until this target existed could only
// be argued for in review.
//
// The shape the ordering tests use: park run A at a chosen suspension point,
// hand the state to run B, then release A and look at what its epilogue did to
// a view model it no longer owns. Where A is parked is the interesting part —
// inside a generation for the state-clobbering invariants, inside the failure
// log for the error one, because that is the only point at which a run can have
// already failed with a real error and still be retired before it writes.
// Without the gates those windows are races; with them they are exact.

import Foundation
import Testing
import MacMLXCore
@testable import macMLX

@Suite("BenchmarkViewModel run lifecycle")
@MainActor
final class BenchmarkViewModelRunLifecycleTests {

    private let temporaryDirectory: URL
    private let engine = StubBenchmarkEngine()
    private let monitor = RecordingSiliconSource()
    private let logs = GatedBenchmarkLog()
    private let store: BenchmarkStore
    private let viewModel: BenchmarkViewModel
    private let model = LocalModel(
        id: "mlx-community/stub-model-4bit",
        displayName: "Stub Model",
        directory: URL(fileURLWithPath: "/nonexistent/stub-model"),
        sizeBytes: 1024,
        format: .mlx,
        quantization: "4bit",
        parameterCount: "1B",
        architecture: "llama"
    )

    init() {
        temporaryDirectory = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("macMLXTests-\(UUID().uuidString)", isDirectory: true)
        store = BenchmarkStore(directory: temporaryDirectory)
        viewModel = BenchmarkViewModel(
            coordinator: engine,
            library: ModelLibraryManager(),
            store: store,
            logs: logs,
            siliconMonitor: monitor
        )
        viewModel.selectedModel = model
        // One measured iteration keeps every run to exactly two generations
        // (un-counted warm-up + one measured), so the scripts stay readable.
        viewModel.runs = 1
    }

    deinit {
        try? FileManager.default.removeItem(at: temporaryDirectory)
    }

    // MARK: - Invariant 1: a retired run must not clobber the live one

    @Test("A cancelled run finishing later leaves the restarted run's state alone")
    func retiredRunDoesNotClobberRestartedRun() async {
        let gateA = AsyncGate()
        let gateB = AsyncGate()

        engine.gateQueue = [gateA]
        viewModel.start()
        await waitUntil("run A to park in its first generation") { gateA.waiterCount == 1 }

        viewModel.cancel()
        #expect(viewModel.isRunning == false)

        engine.gateQueue = [gateB]
        viewModel.start()
        await waitUntil("run B to park in its first generation") { gateB.waiterCount == 1 }
        #expect(viewModel.isRunning)

        // Let the retired run wind all the way down. Its `defer` releases
        // sampling as the very last thing it does, so seeing that release is
        // proof its epilogue has already run — and therefore that everything
        // asserted below survived it.
        gateA.open()
        await waitUntil("run A to finish winding down") { self.monitor.deactivateCount == 1 }

        #expect(viewModel.isRunning, "the retired run cleared the live run's isRunning")
        #expect(viewModel.statusMessage.isEmpty == false, "the retired run cleared the live run's status")
        #expect(viewModel.errorMessage == nil)
        #expect(viewModel.lastResult == nil, "the retired run published its result")
        #expect(viewModel.history.isEmpty, "the retired run wrote to history")
        #expect(await storedResultCount() == 0, "the retired run persisted a result")

        // And the live run still completes normally afterwards.
        gateB.open()
        await waitUntil("run B to finish") { self.viewModel.isRunning == false }

        #expect(viewModel.lastResult != nil)
        #expect(viewModel.history.count == 1)
        #expect(viewModel.statusMessage.isEmpty)
        #expect(await storedResultCount() == 1)
    }

    // MARK: - Invariant 2: `guard !isRunning` still rejects a restart's successor

    @Test("start() is rejected while the run that replaced a cancelled one is live")
    func startIsRejectedWhileRestartedRunIsLive() async {
        let gateA = AsyncGate()
        let gateB = AsyncGate()

        engine.gateQueue = [gateA]
        viewModel.start()
        await waitUntil("run A to park") { gateA.waiterCount == 1 }

        viewModel.cancel()

        engine.gateQueue = [gateB]
        viewModel.start()
        await waitUntil("run B to park") { gateB.waiterCount == 1 }

        // The third start(). An accepted one is visible the instant it returns,
        // because `start()` writes its state synchronously: it would stamp
        // "Preparing…" over the live run's status and hand `runTask` to a run
        // Cancel can no longer reach.
        let statusUnderRunB = viewModel.statusMessage
        #expect(statusUnderRunB.isEmpty == false)
        viewModel.start()
        #expect(
            viewModel.statusMessage == statusUnderRunB,
            "a third run took over the live run's state"
        )

        gateA.open()
        gateB.open()
        await waitUntil("exactly two runs to wind down") {
            self.viewModel.isRunning == false && self.monitor.deactivateCount == 2
        }

        // Run A: one warm-up generation, then Cancel stops it before the
        // measured loop starts a second. Run B: warm-up + one measured. A third
        // run would add two more.
        #expect(engine.generateCount == 3, "a third run generated: \(engine.generateCount) generations, expected 3")
        #expect(monitor.activateCount == 2, "a third run activated sampling")
        // Run B found the model already loaded by run A, so only run A paid a load.
        #expect(engine.loadCount == 1)
        #expect(await storedResultCount() == 1, "more than one run persisted a result")
    }

    // MARK: - Invariant 3: a retired run's error must not reach the UI

    @Test("A retired run's error never lands in the live run's banner")
    func retiredRunErrorIsNotSurfaced() async {
        // Park run A inside the log call it makes on its way to the error
        // banner. That await is the whole reason the ownership guard sits where
        // it does — and the only point at which a run can already have failed
        // with a real (non-cancellation) error and still be retired before it
        // writes. Cancelling any earlier makes the run exit as cancelled
        // instead, which never reaches the guard.
        let failureLogGate = AsyncGate()
        logs.nextCallGate = failureLogGate
        engine.nextGenerationFails = true

        viewModel.start()
        await waitUntil("run A to reach its failure log") { failureLogGate.waiterCount == 1 }
        #expect(logs.messages.contains { $0.hasPrefix("Benchmark failed") })

        // Hand the state to a healthy run while A is still suspended there.
        viewModel.cancel()
        engine.nextGenerationFails = false
        viewModel.start()

        failureLogGate.open()
        await waitUntil("both runs to wind down") {
            self.viewModel.isRunning == false && self.monitor.deactivateCount == 2
        }

        #expect(
            viewModel.errorMessage == nil,
            "a retired run's error was shown: \(viewModel.errorMessage ?? "")"
        )
        // The live run's own outcome is intact.
        #expect(viewModel.lastResult != nil)
        #expect(await storedResultCount() == 1)
    }

    @Test("A live run's own error IS surfaced and resets the run state")
    func liveRunErrorIsSurfaced() async {
        engine.nextGenerationFails = true

        viewModel.start()
        await waitUntil("the run to fail") { self.viewModel.isRunning == false }

        #expect(viewModel.errorMessage != nil, "a live run's failure was swallowed")
        #expect(viewModel.statusMessage.isEmpty)
        #expect(viewModel.lastResult == nil)
        #expect(monitor.balance == 0, "sampling leaked on the error path")
    }

    // MARK: - Invariant 4: sampling activate/deactivate pair on every exit path

    @Test("Sampling is released exactly once when a run completes")
    func samplingIsBalancedOnCompletion() async {
        viewModel.start()
        await waitUntil("the run to finish") { self.viewModel.isRunning == false }

        #expect(monitor.activateCount == 1)
        #expect(monitor.deactivateCount == 1)
        #expect(monitor.balance == 0)
    }

    @Test("Sampling is released exactly once when a run is cancelled mid-generation")
    func samplingIsBalancedOnCancellation() async {
        let gate = AsyncGate()
        engine.gateQueue = [gate]

        viewModel.start()
        await waitUntil("the run to park") { gate.waiterCount == 1 }
        #expect(monitor.activateCount == 1)

        viewModel.cancel()
        await waitUntil("the cancelled run to wind down") { self.monitor.deactivateCount == 1 }

        #expect(monitor.balance == 0, "sampling leaked on the cancel path")
        // Cancel reset the UI immediately, and the abandoned run published nothing.
        #expect(viewModel.isRunning == false)
        #expect(viewModel.statusMessage.isEmpty)
        #expect(viewModel.errorMessage == nil)
        #expect(viewModel.lastResult == nil)
        #expect(await storedResultCount() == 0)

        gate.open()
    }

    @Test("Sampling is balanced across a cancel-and-restart pair")
    func samplingIsBalancedAcrossRestart() async {
        let gateA = AsyncGate()
        engine.gateQueue = [gateA]

        viewModel.start()
        await waitUntil("run A to park") { gateA.waiterCount == 1 }
        viewModel.cancel()
        gateA.open()

        viewModel.start()
        await waitUntil("run B to finish") { self.viewModel.isRunning == false }
        await waitUntil("both runs to release sampling") { self.monitor.deactivateCount == 2 }

        #expect(monitor.activateCount == 2)
        #expect(monitor.balance == 0)
    }

    // MARK: - Entry guards

    @Test("start() without a selected model reports it and starts nothing")
    func startWithoutModelDoesNothing() async {
        viewModel.selectedModel = nil

        viewModel.start()

        #expect(viewModel.isRunning == false)
        #expect(viewModel.errorMessage != nil)
        #expect(engine.loadCount == 0)
        #expect(engine.generateCount == 0)
        #expect(monitor.activateCount == 0, "sampling was activated for a run that never started")
    }

    @Test("An already-loaded model is not re-loaded, and reports no cold-load time")
    func alreadyLoadedModelSkipsLoad() async {
        engine.currentModel = model
        engine.status = .ready(model: model.id)

        viewModel.start()
        await waitUntil("the run to finish") { self.viewModel.isRunning == false }

        #expect(engine.loadCount == 0, "an already-loaded model was loaded again")
        #expect(viewModel.lastResult?.modelLoadTimeS == 0)
        #expect(viewModel.lastResult?.engineID == engine.engineID)
        #expect(viewModel.lastResult?.engineVersion == engine.engineVersion)
    }

    // MARK: - Helpers

    /// Poll the main actor until `condition` holds. Everything under test hands
    /// off on the main actor, so polling it is enough to observe every
    /// transition; the bound turns a hang into a readable failure.
    private func waitUntil(
        _ description: String,
        limit: Int = 3000,
        _ condition: () -> Bool
    ) async {
        for _ in 0..<limit {
            if condition() { return }
            try? await Task.sleep(for: .milliseconds(1))
        }
        #expect(Bool(false), "timed out waiting for \(description)")
    }

    /// Results actually written to disk, read back through the real store —
    /// the view model's in-memory `history` cannot vouch for what was persisted.
    private func storedResultCount() async -> Int {
        ((try? await store.list()) ?? []).count
    }
}
