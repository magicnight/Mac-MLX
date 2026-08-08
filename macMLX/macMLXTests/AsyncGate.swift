// AsyncGate.swift
// macMLXTests
//
// A one-shot rendezvous the tests use to park an in-flight benchmark run at a
// known point and hold it there.
//
// The run-lifecycle invariants under test are all about ORDER — a retired run
// finishing while a newer one owns the state. Sleeping for "long enough" would
// make that ordering a race; a gate makes it exact: the test parks run A inside
// its first generation, starts run B, then releases A and observes what A's
// epilogue does while B is demonstrably still live.
//
// `@MainActor` because the whole benchmark path (view model, stub engine, the
// tests themselves) is, so a gate's state needs no further synchronisation.

import Foundation

@MainActor
final class AsyncGate {

    private var pending: [CheckedContinuation<Void, Never>] = []
    private var isOpen = false

    /// Number of callers currently parked in `wait()`. Tests poll this to know
    /// a run has actually reached the gate rather than guessing with a sleep.
    private(set) var waiterCount = 0

    /// Suspend until `open()` is called. Returns immediately once open, so a
    /// gate that is opened before anyone waits does not deadlock.
    func wait() async {
        if isOpen { return }
        waiterCount += 1
        await withCheckedContinuation { continuation in
            pending.append(continuation)
        }
        waiterCount -= 1
    }

    /// Release everyone parked in `wait()`, and let later waiters through.
    func open() {
        isOpen = true
        let waiting = pending
        pending = []
        for continuation in waiting { continuation.resume() }
    }
}
