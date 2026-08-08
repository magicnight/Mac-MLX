// GatedBenchmarkLog.swift
// macMLXTests
//
// Test double for the benchmark's logging seam. It records messages and — the
// reason it exists — can park the caller inside `log(_:level:category:)`.
//
// That suspension is the only window in which a run can be retired between
// raising an error and writing it to the UI, so parking there is the only way
// to reach the ownership guard that keeps a dead run's error out of the live
// run's banner. Without it, a Cancel always beats the error to the punch and
// the run exits down the `catch is CancellationError` branch instead, leaving
// that guard untested.
//
// `log` is `nonisolated` to match the seam (the real witness is `LogManager`,
// a plain actor) and hops to the main actor for its own state, which is where
// the tests read and arm it.

import Foundation
import MacMLXCore
@testable import macMLX

final class GatedBenchmarkLog: BenchmarkLogging {

    /// Gate handed to the NEXT `log` call, captured and cleared at call time so
    /// re-arming afterwards cannot park a later, unrelated log.
    @MainActor var nextCallGate: AsyncGate?

    @MainActor private(set) var messages: [String] = []

    nonisolated func log(_ message: String, level: LogLevel, category: LogCategory) async {
        let gate = await MainActor.run { () -> AsyncGate? in
            messages.append(message)
            let gate = nextCallGate
            nextCallGate = nil
            return gate
        }
        await gate?.wait()
    }
}
