// Copyright © 2026 macMLX. English comments only.

/// A reference count for "who currently needs hardware sampling running".
///
/// The W3 silicon monitor samples IOReport on a timer, but more than one surface
/// can need that loop at once — the Activity panel while it is visible, and a
/// benchmark run while it measures its bottleneck. If each surface owned a bare
/// start/stop, whichever finished first would stop sampling out from under the
/// other. This is the shared gate: each surface `activate`s when it needs sampling
/// and `deactivate`s when it no longer does, and the loop runs while the count is
/// positive.
///
/// It is a pure value type so the counting semantics are unit-testable without the
/// `@MainActor` app shell (which has no test target). The shell holds one of these
/// and starts/stops its timer on the `true` edges the two calls return.
public struct SamplingActivation: Sendable, Equatable {

    /// Number of outstanding activations. Sampling should run while this is `> 0`.
    public private(set) var count = 0

    /// Whether sampling should currently be running.
    public var isActive: Bool { count > 0 }

    public init() {}

    /// Register one more consumer of sampling.
    ///
    /// - Returns: `true` iff this call transitioned the gate from inactive to
    ///   active (count `0 → 1`), i.e. the caller should start the sampling loop.
    ///   `false` when sampling was already active.
    @discardableResult
    public mutating func activate() -> Bool {
        count += 1
        return count == 1
    }

    /// Release one consumer of sampling. Never underflows below zero, so a stray
    /// unbalanced `deactivate` cannot wedge the count negative and keep sampling
    /// pinned on forever.
    ///
    /// - Returns: `true` iff this call transitioned the gate from active to
    ///   inactive (count `1 → 0`), i.e. the caller should stop the sampling loop.
    ///   `false` when other consumers still need sampling, or it was already off.
    @discardableResult
    public mutating func deactivate() -> Bool {
        guard count > 0 else { return false }
        count -= 1
        return count == 0
    }
}
