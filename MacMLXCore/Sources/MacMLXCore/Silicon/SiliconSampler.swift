// Copyright © 2026 macMLX. English comments only.

/// A source of `SiliconSample` snapshots.
///
/// The single seam W2 injects against: the bottleneck classifier depends on this
/// protocol, so tests can feed canned or scripted samples through a mock while
/// production uses `IOReportSiliconSampler`. Kept intentionally minimal — sampling is
/// the whole contract; interpretation belongs to the consumer.
///
/// `sample()` returns the metrics for the window since the previous call. Every call
/// yields a usable reading: the sampler captures a baseline when it opens, so even the
/// first `sample()` covers a (short) window and reports delta-derived fields. Callers
/// wanting steady rates should poll on a regular cadence and may discard the first,
/// short-window reading. If a transient IOReport miss occurs, delta-derived fields are
/// `nil` and `intervalSeconds` is 0; instantaneous fields (thermal, memory pressure)
/// are always populated.
public protocol SiliconSampler: Sendable {
    func sample() async -> SiliconSample
}
