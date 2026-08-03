import Testing
import Foundation
@testable import MacMLXCore

// MARK: - Median helper

@Test
func medianEmptyReturnsZero() {
    #expect(BenchmarkRunner.median([]) == 0)
}

@Test
func medianSingleValue() {
    #expect(BenchmarkRunner.median([42.0]) == 42.0)
}

@Test
func medianOddCountPicksMiddle() {
    #expect(BenchmarkRunner.median([3.0, 1.0, 2.0]) == 2.0)
}

@Test
func medianEvenCountAveragesTwoMiddles() {
    // Sorted: [1, 2, 3, 4] → (2 + 3) / 2 = 2.5
    #expect(BenchmarkRunner.median([4.0, 2.0, 3.0, 1.0]) == 2.5)
}

// MARK: - End-to-end with canned stream

/// Builds a canned stream that emits `chunks` text pieces, then a
/// terminal chunk with a TokenUsage so the runner can pick up authoritative
/// token counts. The chunks are emitted with a small delay between them so
/// the runner's first-token / last-token timing measures something > 0.
private func cannedStream(
    chunks: [String],
    promptTokens: Int,
    delayBetweenChunksMs: UInt64 = 5
) -> AsyncThrowingStream<GenerateChunk, Error> {
    AsyncThrowingStream { continuation in
        Task {
            // Initial tiny delay so TTFT > 0 even on the warm-up path.
            try? await Task.sleep(nanoseconds: 1_000_000)
            for text in chunks {
                continuation.yield(GenerateChunk(text: text))
                try? await Task.sleep(nanoseconds: delayBetweenChunksMs * 1_000_000)
            }
            continuation.yield(
                GenerateChunk(
                    text: "",
                    finishReason: .stop,
                    usage: TokenUsage(
                        promptTokens: promptTokens,
                        completionTokens: chunks.count
                    )
                )
            )
            continuation.finish()
        }
    }
}

@Test
func runProducesResultWithExpectedProvenance() async throws {
    let runner = BenchmarkRunner { _ in
        cannedStream(
            chunks: ["a", "b", "c", "d", "e"],
            promptTokens: 128
        )
    }

    let result = try await runner.run(
        modelID: "test-model",
        engineID: .mlxSwift,
        engineVersion: "mock 0.0.0",
        promptTokens: 128,
        generationTokens: 5,
        runs: 2,
        modelLoadTimeS: 3.5,
        notes: "smoke"
    )

    #expect(result.modelID == "test-model")
    #expect(result.engineID == .mlxSwift)
    #expect(result.engineVersion == "mock 0.0.0")
    #expect(result.runs == 2)
    #expect(result.modelLoadTimeS == 3.5)
    #expect(result.notes == "smoke")
    #expect(result.promptTokens == 128)
    #expect(result.completionTokens == 5)

    // Timing-derived metrics must be positive.
    #expect(result.ttftMs > 0)
    #expect(result.generationTPS > 0)
    #expect(result.promptTPS >= 0)

    // Provenance pulled from HardwareInfo — chip name non-empty on any
    // working host, macOS version looks right.
    #expect(!result.system.chip.isEmpty)
    #expect(result.system.ramGB > 0)
    #expect(result.system.macOSVersion.contains("."))
}

@Test
func runRequiresAtLeastOneRun() async {
    let runner = BenchmarkRunner { _ in
        cannedStream(chunks: ["x"], promptTokens: 16)
    }
    // precondition(runs > 0) → expected runtime trap on runs=0.
    // We don't assert the trap (Swift Testing lacks expectTrap); we just
    // document that runs=0 is unsupported. A call with runs=1 must work.
    let result = try? await runner.run(
        modelID: "m",
        engineID: .mlxSwift,
        engineVersion: "",
        promptTokens: 8,
        generationTokens: 1,
        runs: 1
    )
    #expect(result != nil)
}

// MARK: - Warm-up boundary signal

/// Thread-safe ordered event log. `@unchecked Sendable`: the mutable array is
/// guarded by an `NSLock`, because the runner drives it from its own actor
/// (through the `@Sendable` generate closure and the warm-up callback) while
/// the test reads it back afterwards.
private final class PhaseEventLog: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [String] = []

    func record(_ event: String) {
        lock.lock()
        defer { lock.unlock() }
        events.append(event)
    }

    var recorded: [String] {
        lock.lock()
        defer { lock.unlock() }
        return events
    }
}

@Test
func warmupCompleteFiresOnceBetweenWarmupAndMeasuredIterations() async throws {
    let log = PhaseEventLog()
    let runner = BenchmarkRunner { _ in
        log.record("generate")
        return cannedStream(chunks: ["a", "b"], promptTokens: 32)
    }

    _ = try await runner.run(
        modelID: "m",
        engineID: .mlxSwift,
        engineVersion: "",
        promptTokens: 32,
        generationTokens: 2,
        runs: 3,
        onWarmupComplete: { log.record("warmupComplete") }
    )

    // One un-counted warm-up generate, then the boundary signal exactly once,
    // then exactly `runs` measured generates. Anything observing the machine
    // from the signal onwards therefore sees only measured iterations.
    #expect(
        log.recorded == [
            "generate", "warmupComplete", "generate", "generate", "generate",
        ]
    )
}

@Test
func runWithoutWarmupCallbackKeepsPreviousBehaviour() async throws {
    let log = PhaseEventLog()
    let runner = BenchmarkRunner { _ in
        log.record("generate")
        return cannedStream(chunks: ["a", "b"], promptTokens: 32)
    }

    // Call shape every pre-existing caller uses: no `onWarmupComplete` at all.
    let result = try await runner.run(
        modelID: "m",
        engineID: .mlxSwift,
        engineVersion: "",
        promptTokens: 32,
        generationTokens: 2,
        runs: 2
    )

    #expect(log.recorded == ["generate", "generate", "generate"])
    #expect(result.runs == 2)
    #expect(result.generationTPS > 0)
}

@Test
func warmupCompleteDoesNotFireWhenWarmupThrows() async {
    struct StreamError: Error {}
    let log = PhaseEventLog()
    let runner = BenchmarkRunner { _ in
        AsyncThrowingStream { continuation in
            continuation.finish(throwing: StreamError())
        }
    }

    do {
        _ = try await runner.run(
            modelID: "m",
            engineID: .mlxSwift,
            engineVersion: "",
            promptTokens: 8,
            generationTokens: 1,
            runs: 1,
            onWarmupComplete: { log.record("warmupComplete") }
        )
        Issue.record("Expected run() to rethrow the warm-up stream's error")
    } catch {
        // A run that never reached a measured iteration signals no boundary.
        #expect(log.recorded.isEmpty)
    }
}

@Test
func runPropagatesStreamError() async {
    struct StreamError: Error {}
    let runner = BenchmarkRunner { _ in
        AsyncThrowingStream { continuation in
            continuation.finish(throwing: StreamError())
        }
    }
    do {
        _ = try await runner.run(
            modelID: "m",
            engineID: .mlxSwift,
            engineVersion: "",
            promptTokens: 8,
            generationTokens: 1,
            runs: 1
        )
        Issue.record("Expected run() to rethrow the stream's error")
    } catch is StreamError {
        // expected
    } catch {
        Issue.record("Unexpected error type: \(error)")
    }
}
