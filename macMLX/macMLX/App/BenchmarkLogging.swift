// BenchmarkLogging.swift
// macMLX
//
// The one logging call a benchmark run makes, as an injection seam.
//
// Same rationale as `BenchmarkEngineProviding` — `AppState` still passes the
// app's single `LogManager`, and the call forwards unchanged. What the seam
// buys is a suspension point the tests can hold: the ownership guard that
// keeps a retired run's error out of the live run's banner sits AFTER this
// await precisely because the await is where a Cancel-and-restart can slip in,
// so a test can only reach that guard by parking a failing run inside its
// failure log.
//
// Not `@MainActor`: `LogManager` is a plain actor, and the requirement is
// `async` so its isolated method witnesses it from the actor's own executor —
// exactly as the direct call did.

import Foundation
import MacMLXCore

nonisolated protocol BenchmarkLogging: AnyObject, Sendable {
    func log(_ message: String, level: LogLevel, category: LogCategory) async
}

extension LogManager: BenchmarkLogging {}
