import Foundation

/// A serial executor that runs its jobs, one at a time, on a thread of its own
/// at utility quality of service.
///
/// Why not a serial dispatch queue: Swift raises the QoS a queue runs a job at
/// to the job's priority, so a test body on a `.utility` queue still ran at the
/// test task's user-initiated QoS (measured: `qos_class_self()` 25 on a
/// `.utility` queue). A thread of our own keeps the QoS it was given. That
/// matters on the 3-core CI runner, where CPU-heavy bodies at the tests' own
/// priority took the cores from every other test: a stall watchdog set to 1 s
/// answered after 10.2 s while the schema search ran.
final class UtilityThreadExecutor: SerialExecutor, @unchecked Sendable {
    /// Guards `jobs`; the thread waits on it while there is nothing to run.
    private let condition = NSCondition()
    private var jobs: [UnownedJob] = []

    init(name: String) {
        let thread = Thread { [self] in run() }
        thread.name = name
        thread.qualityOfService = .utility
        thread.start()
    }

    func enqueue(_ job: consuming ExecutorJob) {
        let job = UnownedJob(job)
        condition.lock()
        jobs.append(job)
        condition.signal()
        condition.unlock()
    }

    func asUnownedSerialExecutor() -> UnownedSerialExecutor {
        UnownedSerialExecutor(ordinary: self)
    }

    private func run() {
        while true {
            condition.lock()
            while jobs.isEmpty { condition.wait() }
            let job = jobs.removeFirst()
            condition.unlock()
            // Metal objects created outside MLX's own pools are freed per job,
            // not when the thread ends (it never does).
            autoreleasepool {
                job.runSynchronously(on: asUnownedSerialExecutor())
            }
        }
    }
}
