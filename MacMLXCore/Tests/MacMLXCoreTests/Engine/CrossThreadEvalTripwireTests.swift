import Testing
import Foundation
import MLX
@testable import MacMLXCore

/// A regression guard for `ml-explore/mlx-swift#457`, the reason the fork sat
/// on core v0.31.1 for as long as it did.
///
/// mlx-core made `CommandEncoder` and `default_stream()` thread-local in
/// `ml-explore/mlx#3348`, first released in **core v0.31.2**. mlx-swift 0.31.x
/// predated that model: `Stream.gpu` and `Stream.cpu` were process-global
/// `static let`s materialized on whichever thread first touched MLX, so
/// evaluating from any other OS thread died with
///
///     Fatal error: There is no Stream(gpu, 0) in current thread.
///
/// a process abort, not a throw. macMLX is squarely in the blast radius: a
/// Hummingbird server answering concurrent requests, a SwiftUI app, and Swift
/// Concurrency tasks that hop cooperative-pool threads all evaluate from
/// threads that are not the one that booted MLX.
///
/// mlx-swift 0.32.3, the base since 2026-10-02, creates every stream through
/// `mlx_stream_new_thread_unsafe` (mlx-c `ebc88f1`, `ml-explore/mlx-c#122`),
/// which registers it globally rather than per thread, and resolves the
/// default stream from a `@TaskLocal` with a process-wide fallback. A thread
/// that never set one gets the global pair. Upstream closed #457 on that basis.
///
/// ## What a failure looks like
///
/// Not a red assertion. The process aborts inside `MLX.eval` on the detached
/// thread and takes the test runner with it. A crashed run of *this* suite,
/// after a fork bump, means the bump regressed the stream model — look at
/// whether the runner survived, not at the summary. The timeout branch below
/// only catches the milder shape where the thread hangs instead.
///
/// ## What this test does NOT prove
///
/// That a thread-unsafe stream shared across threads is safe under concurrent
/// *graph construction*. mlx-swift serialises `eval` behind `evalLock`; ops
/// that build the graph do not take it, which is the same shape as before this
/// move. This test exercises one thread at a time. It was first observed to
/// pass, rather than merely held green, against 0.32.3 — the old base could
/// not have run it at all.
@Suite(
    "Cross-thread eval tripwire",
    .enabled(if: mlxMetallibIsAvailable, "Requires default.metallib (run under xcodebuild)"))
struct CrossThreadEvalTripwireTests {

    /// Carries the detached thread's result back. `@unchecked Sendable` is
    /// sound here because the semaphore establishes the ordering: the writer
    /// signals only after its last write, and the reader reads only after the
    /// wait returns.
    private final class Outcome: @unchecked Sendable {
        var sum: Int32?
    }

    @Test("an array evaluated on a second OS thread does not abort")
    func evaluatingOnASecondThreadDoesNotAbort() {
        // Materialize Stream.gpu HERE first. That is the precondition for the
        // failure: the process-global stream has to belong to some other
        // thread before the detached one asks for it.
        MLX.eval(MLXArray(0 ..< 16).sum())

        let outcome = Outcome()
        let finished = DispatchSemaphore(value: 0)

        Thread.detachNewThread {
            let total = MLXArray(0 ..< 16).sum()
            MLX.eval(total)
            outcome.sum = total.item(Int32.self)
            finished.signal()
        }

        let timedOut = finished.wait(timeout: .now() + 60) == .timedOut
        #expect(
            !timedOut,
            """
            An eval on a second OS thread did not finish within 60s. This is \
            the shape of ml-explore/mlx-swift#457: the default stream belongs \
            to the thread that first touched MLX. mlx-swift 0.32.3 creates \
            streams with new_thread_unsafe_stream; a bump that regressed that \
            lands here (or aborts the runner outright).
            """)
        #expect(
            outcome.sum == 120,
            "Cross-thread eval returned \(outcome.sum.map(String.init) ?? "nothing"), expected 120")
    }
}
