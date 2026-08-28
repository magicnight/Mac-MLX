import Testing
import Foundation
import MLX
@testable import MacMLXCore

/// A tripwire for `ml-explore/mlx-swift#457`, which is the reason this project
/// cannot bump its vendored MLX by simply moving the pin forward.
///
/// mlx-core made `CommandEncoder` and `default_stream()` thread-local in
/// `ml-explore/mlx#3348`, first released in **core v0.31.2**. Each OS thread
/// that asks for a default stream now gets its own, whose command encoder is
/// registered only in that thread's registry. mlx-swift's Swift layer predates
/// that model: `Stream.gpu` and `Stream.cpu` are process-global `static let`s,
/// materialized on whichever thread first touches MLX, and handed to every op
/// through `Device.defaultStream`. Evaluating from any other OS thread then
/// dies with
///
///     Fatal error: There is no Stream(gpu, 0) in current thread.
///
/// which is a process abort, not a throw. macMLX is squarely in the blast
/// radius: a Hummingbird server answering concurrent requests, a SwiftUI app,
/// and Swift Concurrency tasks that hop cooperative-pool threads all evaluate
/// from threads that are not the one that booted MLX.
///
/// Our pin — mlx-core v0.31.1 — is the last release before that change, which
/// is luck rather than design. This test is the tripwire for the moment that
/// luck runs out.
///
/// ## What a failure looks like
///
/// Not a red assertion. The process aborts inside `MLX.eval` on the detached
/// thread and takes the test runner with it. A crashed run of *this* suite,
/// after a fork bump, means the bump carried the thread-local stream model
/// without the mlx-c side fix (`new_thread_unsafe_stream` for the two default
/// stream entry points — bound upstream in `ml-explore/mlx-c#122`). The
/// timeout branch below only catches the milder shape where the thread hangs
/// instead.
///
/// ## What this test does NOT prove
///
/// It has not been observed to fail. Demonstrating that would mean building
/// against core >= 0.31.2, which is the whole of the bump this guards. It is a
/// faithful transcription of the reproduction in mlx-swift#457, held green
/// against the current pin — not an empirically inverted guard like the parity
/// suites.
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
            An eval on a second OS thread did not finish within 60s. If the \
            vendored core is now >= 0.31.2, this is ml-explore/mlx-swift#457 — \
            Stream.gpu belongs to the thread that first touched MLX, and the \
            mlx-c default stream entry points need new_thread_unsafe_stream.
            """)
        #expect(
            outcome.sum == 120,
            "Cross-thread eval returned \(outcome.sum.map(String.init) ?? "nothing"), expected 120")
    }
}
