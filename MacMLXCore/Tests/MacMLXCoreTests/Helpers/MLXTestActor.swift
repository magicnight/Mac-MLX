/// The actor every MLX-gated Swift Testing suite is isolated to. Its executor
/// is a thread of its own at utility QoS (`UtilityThreadExecutor`), so the
/// bodies themselves never run on the cooperative pool, and on a busy runner
/// the other tests' work comes first.
///
/// MLX work is synchronous. `eval` blocks its thread until the GPU is done, the
/// first use of a kernel compiles its shader inline, and mlx-swift takes one
/// process-wide lock (`evalLock`) around every evaluation. Swift Testing runs
/// test bodies on the cooperative pool, which has one thread per core and does
/// not grow when a thread blocks. On the 3-core CI runner the MLX bodies held
/// those threads, one evaluating and the rest waiting for the lock, and every
/// other test of the run waited in the queue behind them: a test that awaits
/// nothing reported 22.7–62.7 s in the "Metal Tests (xcodebuild)" job. A
/// deadline computed before that wait is spent on it (run 44844150764, where
/// a 60 s poll expired).
///
/// The cost is that the MLX bodies now run one at a time. Evaluation was
/// already serialised by the lock; graph building, weight init and sanitize
/// used to overlap and no longer do. The bodies need no particular thread: at the pinned mlx-swift every stream is registered process-wide
/// (`mlx_stream_new_thread_unsafe`), the default stream and random state fall
/// back to process-wide values unless a task-local one is set, and on the pool
/// a body ran on whichever thread was free.
///
/// What this does not cover: work an MLX test hands to another actor, such as
/// `MLXSwiftEngine`, runs on that actor's executor, which is the pool.
///
/// Put `@MLXTestActor` on every suite or test that carries
/// `.enabled(if: mlxMetallibIsAvailable, …)`; `OffPoolTestActorTests` fails when
/// one is missing.
@globalActor
actor MLXTestActor {
    static let shared = MLXTestActor()

    static let threadName = "macmlx.tests.mlx"
    static let executor = UtilityThreadExecutor(name: threadName)

    nonisolated var unownedExecutor: UnownedSerialExecutor {
        Self.executor.asUnownedSerialExecutor()
    }
}
