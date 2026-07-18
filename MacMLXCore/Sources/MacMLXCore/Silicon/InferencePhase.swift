// Copyright © 2026 macMLX. English comments only.

/// The two phases of an autoregressive generation, as the bottleneck classifier
/// needs to tell them apart.
///
/// WHY THE CLASSIFIER CARES
/// ------------------------
/// The two phases have DIFFERENT healthy bottlenecks, so the same hardware
/// reading means different things depending on which one is running:
///   * `prefill` — the whole prompt is processed in one (or a few) large
///     matmul-heavy steps. It is compute-bound in the normal, healthy case:
///     the GPU is pinned and the memory bus has headroom. That is expected, not
///     a problem to fix.
///   * `decode` — tokens are produced one at a time, each step re-reading the
///     model weights and KV cache from unified memory. It is memory-bandwidth-
///     bound in the normal, healthy case: throughput is gated by how fast
///     weights stream from DRAM, not by GPU math. That too is expected.
///
/// A verdict that ignored the phase would flag healthy decode as "bandwidth-
/// bound trouble" and healthy prefill as "compute-bound trouble". Carrying the
/// phase lets the classifier phrase its advice honestly ("decode is bandwidth-
/// bound → expected; a lower-bit quantization can help" vs. "prefill is
/// compute-bound → this is expected").
/// The `String` raw values are pinned explicitly (not derived from the case names)
/// because this is a persisted form — a benchmark's bottleneck attribution stores a
/// phase (see `BenchmarkBottleneck`). Pinning them means a future case rename cannot
/// silently change the on-disk value and orphan old history. Keep them stable;
/// nothing switches on the raw value.
public enum InferencePhase: String, Sendable, Equatable, CaseIterable, Codable {
    /// Processing the input prompt, before the first output token.
    case prefill = "prefill"
    /// Producing output tokens one at a time, after the first token.
    case decode = "decode"
}
