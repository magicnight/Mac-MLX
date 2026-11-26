import Dispatch

/// The actor for a test body that keeps its thread for seconds without
/// suspending: a long synchronous computation, or a blocking wait. Its executor
/// is a serial queue of its own, for the reason `MLXTestActor` gives: on the
/// cooperative pool such a body takes one of a 3-core runner's three threads
/// away from every other test for as long as it runs.
///
/// A queue apart from `MLXTestActor`'s, so that neither kind waits for the
/// other. Bodies on this actor run one at a time.
@globalActor
actor BlockingTestActor {
    static let shared = BlockingTestActor()

    static let queue = DispatchSerialQueue(label: "macmlx.tests.blocking", qos: .userInitiated, autoreleaseFrequency: .workItem)

    nonisolated var unownedExecutor: UnownedSerialExecutor {
        Self.queue.asUnownedSerialExecutor()
    }
}
