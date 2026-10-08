import Foundation
import Testing
import os

@testable import MacMLXCore

// MARK: - MLXSwiftEngine.awaitWorker (#136)
//
// mlx-swift-lm's token worker is a Task feeding an AsyncStream; the stream
// cancels the worker only when the stream is dropped or a consumer is
// cancelled while waiting on it. `runGeneration` keeps the stream alive in
// its frame, so when its loop leaves the stream early the worker has to be
// cancelled by hand, or it decodes to EOS or maxTokens holding the model.
// These cases wire a stream and a worker the way `generateLoopTask` does in
// 3.32.3 (the worker checks `Task.isCancelled` once per step; the stream's
// termination handler cancels it) and take each exit through `awaitWorker`.

@Suite("MLXSwiftEngine worker wait (#136)")
struct MLXSwiftEngineWorkerTests {

    private struct Worker {
        let stream: AsyncStream<Int>
        let task: Task<Void, Never>
        let steps: OSAllocatedUnfairLock<Int>
        let finished: OSAllocatedUnfairLock<Bool>
    }

    /// A step the way the real one behaves: a cancel cannot interrupt it (the
    /// real step is a synchronous `next()` on the generation worker's queue),
    /// so a cancelled worker still finishes the step it is in.
    private func uninterruptibleStep() async {
        await withCheckedContinuation { (resume: CheckedContinuation<Void, Never>) in
            DispatchQueue.global().asyncAfter(deadline: .now() + .milliseconds(5)) { resume.resume() }
        }
    }

    /// A worker of `total` steps, each an uninterruptible wait then a yield,
    /// checking for cancellation between steps like the real one.
    private func worker(steps total: Int) -> Worker {
        let (stream, continuation) = AsyncStream<Int>.makeStream()
        let steps = OSAllocatedUnfairLock<Int>(initialState: 0)
        let finished = OSAllocatedUnfairLock<Bool>(initialState: false)
        let task = Task {
            for i in 0..<total {
                if Task.isCancelled { break }
                steps.withLock { $0 += 1 }
                await uninterruptibleStep()
                _ = continuation.yield(i)
            }
            finished.withLock { $0 = true }
            continuation.finish()
        }
        continuation.onTermination = { termination in
            if case .cancelled = termination { task.cancel() }
        }
        return Worker(stream: stream, task: task, steps: steps, finished: finished)
    }

    @Test("a loop that leaves the stream early stops the worker instead of letting it run out")
    func earlyExitStopsTheWorker() async {
        let w = worker(steps: 2_000)
        var received = 0
        consume: for await _ in w.stream {
            received += 1
            if received == 3 { break consume }
        }
        await MLXSwiftEngine.awaitWorker(w.task)
        #expect(w.finished.withLock { $0 }, "the wait returns only once the worker is done")
        #expect(
            w.steps.withLock { $0 } < 1_000,
            "the worker stopped at its next cancellation check, not after all 2,000 steps")
        // The stream stays alive across the wait, as it does in `runGeneration`.
        withExtendedLifetime(w.stream) {}
    }

    @Test("a loop that read the stream to its end finds the worker done")
    func naturalEndFindsTheWorkerDone() async {
        let w = worker(steps: 20)
        var received = 0
        for await _ in w.stream { received += 1 }
        await MLXSwiftEngine.awaitWorker(w.task)
        #expect(received == 20)
        #expect(w.finished.withLock { $0 })
        #expect(w.steps.withLock { $0 } == 20, "every step ran; the cancel found nothing to stop")
    }

    @Test("a loop cancelled while waiting on the stream has already stopped the worker")
    func cancelledConsumerStopsTheWorkerThroughTheStream() async {
        let w = worker(steps: 2_000)
        let consumer = Task {
            var received = 0
            for await _ in w.stream { received += 1 }
            await MLXSwiftEngine.awaitWorker(w.task)
            return received
        }
        try? await Task.sleep(for: .milliseconds(40))
        consumer.cancel()
        let received = await consumer.value
        #expect(w.finished.withLock { $0 })
        #expect(w.steps.withLock { $0 } < 1_000, "stopped by the stream's cancellation, not after all 2,000 steps")
        #expect(received < 1_000)
        withExtendedLifetime(w.stream) {}
    }
}
